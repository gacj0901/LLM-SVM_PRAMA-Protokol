"""Causal operator-geometry diagnostics for numeric LLM trajectories."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


MODES = ("EXPANSION", "CONSERVATION", "RELEASE")


@dataclass(frozen=True)
class OperatorGeometryConfig:
    """Model-independent parameters fixed before geometry reprojection."""

    window_size_tokens: int = 16
    geometry_window: int = 16
    minimum_geometry_points: int = 8
    recurrence_lag_exclusion: int = 2
    recurrence_radius: float = 0.10
    gap_scale: float = 1.0
    mode_deadband: float = 0.02
    activity_path_length_threshold: float = 0.50
    recirculation_efficiency_threshold: float = 0.25
    recirculation_rate_threshold: float = 0.30
    recirculation_dwell_threshold: int = 4

    def __post_init__(self) -> None:
        if self.window_size_tokens <= 0 or self.geometry_window < 3:
            raise ValueError("window sizes are invalid")
        if not 3 <= self.minimum_geometry_points <= self.geometry_window:
            raise ValueError("minimum_geometry_points is invalid")
        if not 0 <= self.recurrence_lag_exclusion < self.geometry_window - 1:
            raise ValueError("recurrence_lag_exclusion is invalid")
        for name in (
            "recurrence_radius",
            "gap_scale",
            "mode_deadband",
            "activity_path_length_threshold",
            "recirculation_efficiency_threshold",
            "recirculation_rate_threshold",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.recirculation_efficiency_threshold >= 1.0:
            raise ValueError("efficiency threshold must be below one")
        if self.recirculation_rate_threshold >= 1.0:
            raise ValueError("recurrence threshold must be below one")
        if self.recirculation_dwell_threshold <= 0:
            raise ValueError("dwell threshold must be positive")

    @classmethod
    def from_contract(cls, contract: Mapping[str, Any]) -> "OperatorGeometryConfig":
        if contract.get("schema") != "LLM-SVM-CoCC-operator-geometry-contract/1":
            raise ValueError("unexpected operator-geometry contract schema")
        if contract.get("model_specific_parameters") is not False:
            raise ValueError("operator geometry must forbid model-specific parameters")
        if contract.get("requires_external_calibration") is not False:
            raise ValueError("operator geometry must not require calibration")
        source = contract["input"]
        geometry = contract["geometry"]
        modes = contract["modes"]
        classification = contract["classification"]
        return cls(
            window_size_tokens=int(source["window_size_tokens"]),
            geometry_window=int(geometry["trailing_window"]),
            minimum_geometry_points=int(geometry["minimum_points"]),
            recurrence_lag_exclusion=int(geometry["recurrence_lag_exclusion"]),
            recurrence_radius=float(geometry["recurrence_radius"]),
            gap_scale=float(source["gap_scale"]),
            mode_deadband=float(modes["load_change_deadband"]),
            activity_path_length_threshold=float(classification["minimum_path_length"]),
            recirculation_efficiency_threshold=float(
                classification["maximum_transport_efficiency"]
            ),
            recirculation_rate_threshold=float(
                classification["minimum_recurrence_rate"]
            ),
            recirculation_dwell_threshold=int(
                classification["minimum_recurrent_dwell"]
            ),
        )


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    """RMS Euclidean distance; bounded by one for unit-cube states."""

    if len(left) != len(right) or not left:
        raise ValueError("state vectors must have the same nonzero dimension")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)) / len(left))


def _token_entropy_norm(token: Mapping[str, Any]) -> float:
    candidates = token.get("top_logprobs") or []
    count = max(2, len(candidates))
    denominator = math.log(count)
    return _clamp01(float(token.get("entropy") or 0.0) / denominator)


def _window_state(
    tokens: Sequence[Mapping[str, Any]], trajectory: Mapping[str, Any], config: OperatorGeometryConfig
) -> tuple[list[float], dict[str, float]]:
    if not tokens:
        raise ValueError("operator geometry received an empty token window")
    entropy_norm = sum(_token_entropy_norm(token) for token in tokens) / len(tokens)
    gap_confidence = sum(
        max(0.0, float(token.get("gap") or 0.0))
        / (max(0.0, float(token.get("gap") or 0.0)) + config.gap_scale)
        for token in tokens
    ) / len(tokens)
    rigidity = (1.0 - entropy_norm) * gap_confidence
    uncertainty = entropy_norm * (1.0 - gap_confidence)
    delta = _clamp01(float(trajectory["delta"]))
    theta = float(trajectory["theta"])
    xi = float(trajectory["xi"])
    xi_occupancy = _clamp01(xi / theta) if theta > 0.0 else 0.0
    state = [entropy_norm, gap_confidence, rigidity, uncertainty, delta, xi_occupancy]
    load = sum(
        (entropy_norm, 1.0 - gap_confidence, uncertainty, delta, xi_occupancy)
    ) / 5.0
    return state, {
        "entropy_norm": entropy_norm,
        "gap_confidence": gap_confidence,
        "rigidity": rigidity,
        "uncertainty": uncertainty,
        "xi_occupancy": xi_occupancy,
        "operator_load": load,
    }


def _mode_entropy(modes: Sequence[str]) -> float:
    if not modes:
        return 0.0
    counts = Counter(modes)
    entropy = -sum(
        (count / len(modes)) * math.log(count / len(modes))
        for count in counts.values()
    )
    return entropy / math.log(len(MODES))


def observe_operator_geometry(
    tokens: Sequence[Mapping[str, Any]],
    trajectory: Sequence[Mapping[str, Any]],
    config: OperatorGeometryConfig,
) -> list[dict[str, Any]]:
    """Emit causal geometry records aligned to an existing PRAMA trajectory."""

    windows = [
        tokens[start : start + config.window_size_tokens]
        for start in range(0, len(tokens), config.window_size_tokens)
    ]
    if len(windows) != len(trajectory):
        raise ValueError(
            f"token/trajectory window mismatch: {len(windows)} != {len(trajectory)}"
        )
    states: list[list[float]] = []
    components: list[dict[str, float]] = []
    loads: list[float] = []
    modes: list[str] = []
    records: list[dict[str, Any]] = []
    recurrent_dwell = 0
    mode_dwell = 0
    for index, (members, source) in enumerate(zip(windows, trajectory, strict=True)):
        state, component = _window_state(members, source, config)
        states.append(state)
        components.append(component)
        loads.append(component["operator_load"])
        if index == 0:
            mode = "CONSERVATION"
        else:
            change = loads[-1] - loads[-2]
            if change > config.mode_deadband:
                mode = "EXPANSION"
            elif change < -config.mode_deadband:
                mode = "RELEASE"
            else:
                mode = "CONSERVATION"
        if modes and modes[-1] == mode:
            mode_dwell += 1
        else:
            mode_dwell = 1
        modes.append(mode)

        start = max(0, index - config.geometry_window + 1)
        local_states = states[start : index + 1]
        local_modes = modes[start : index + 1]
        steps = [
            _distance(local_states[offset], local_states[offset - 1])
            for offset in range(1, len(local_states))
        ]
        path_length = sum(steps)
        net_displacement = (
            _distance(local_states[-1], local_states[0])
            if len(local_states) >= 2
            else 0.0
        )
        transport_efficiency = (
            net_displacement / path_length if path_length > 0.0 else 0.0
        )

        recurrent_pairs = 0
        admissible_pairs = 0
        for left in range(len(local_states)):
            for right in range(left + 1, len(local_states)):
                if right - left <= config.recurrence_lag_exclusion:
                    continue
                admissible_pairs += 1
                if _distance(local_states[left], local_states[right]) < config.recurrence_radius:
                    recurrent_pairs += 1
        recurrence_rate = recurrent_pairs / admissible_pairs if admissible_pairs else 0.0
        current_recurrent = any(
            _distance(local_states[-1], local_states[prior]) < config.recurrence_radius
            for prior in range(
                0, max(0, len(local_states) - config.recurrence_lag_exclusion - 1)
            )
        )
        recurrent_dwell = recurrent_dwell + 1 if current_recurrent else 0
        dominant_mode = Counter(local_modes).most_common(1)[0][0]
        ready = len(local_states) >= config.minimum_geometry_points
        recirculating = bool(
            ready
            and path_length > config.activity_path_length_threshold
            and transport_efficiency < config.recirculation_efficiency_threshold
            and recurrence_rate > config.recirculation_rate_threshold
            and recurrent_dwell >= config.recirculation_dwell_threshold
        )
        records.append(
            {
                "window_index": int(source["window_index"]),
                "geometry_ready": ready,
                "state_vector": state,
                **component,
                "path_length": path_length,
                "net_displacement": net_displacement,
                "transport_efficiency": transport_efficiency,
                "recurrence_rate": recurrence_rate,
                "current_state_recurrent": current_recurrent,
                "recurrent_dwell": recurrent_dwell,
                "mode": mode,
                "mode_dwell": mode_dwell,
                "mode_entropy": _mode_entropy(local_modes),
                "dominant_mode": dominant_mode,
                "structural_condition": (
                    "CRYSTALLIZED_RECIRCULATION" if recirculating else "UNCLASSIFIED"
                ),
            }
        )
    return records
