"""Causal, model-independent observation layer for window-scale PRAMA."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class DynamicObserverConfig:
    """Frozen algorithmic parameters; never fitted to a model or outcome."""

    window_size_tokens: int = 16
    warmup_windows: int = 8
    location_update_alpha: float = 1.0 / 64.0
    scale_update_alpha: float = 1.0 / 64.0
    scale_floor: float = 0.001
    relative_scale_floor: float = 0.1
    mad_consistency: float = 1.4826
    compression_constant: float = 8.0
    warmup_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.window_size_tokens <= 0:
            raise ValueError("window_size_tokens must be positive")
        if self.warmup_windows < 2:
            raise ValueError("warmup_windows must be at least 2")
        for name in ("location_update_alpha", "scale_update_alpha"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        for name in (
            "scale_floor",
            "relative_scale_floor",
            "mad_consistency",
            "compression_constant",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.warmup_delta < 1.0:
            raise ValueError("warmup_delta must be in [0, 1)")

    @classmethod
    def from_contract(cls, contract: Mapping[str, Any]) -> "DynamicObserverConfig":
        if contract.get("schema") != "LLM-SVM-CoCC-dynamic-observer-contract/1":
            raise ValueError("unexpected dynamic observer contract schema")
        if contract.get("model_specific_parameters") is not False:
            raise ValueError("dynamic observer must forbid model-specific parameters")
        if contract.get("requires_external_calibration") is not False:
            raise ValueError("dynamic observer must not require external calibration")
        source = contract["input"]
        state = contract["causal_state"]
        mapping = contract["observation_map"]
        return cls(
            window_size_tokens=int(source["window_size_tokens"]),
            warmup_windows=int(state["warmup_windows"]),
            location_update_alpha=float(state["location_update_alpha"]),
            scale_update_alpha=float(state["scale_update_alpha"]),
            scale_floor=float(state["scale_floor"]),
            compression_constant=float(mapping["compression_constant"]),
            warmup_delta=float(mapping["warmup_delta"]),
        )


def _initial_state(
    values: Sequence[float], config: DynamicObserverConfig
) -> tuple[float, float]:
    location = statistics.median(values)
    mad = statistics.median(abs(value - location) for value in values)
    scale = max(
        config.mad_consistency * mad,
        config.relative_scale_floor * abs(location),
        config.scale_floor,
    )
    return location, scale


def observe(values: Sequence[float], config: DynamicObserverConfig) -> list[dict[str, Any]]:
    """Map nonnegative window means to Δ using only present and past values."""

    transformed = []
    for value in values:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError("window mean surprisal must be finite and nonnegative")
        transformed.append(math.log1p(numeric))
    if not transformed:
        raise ValueError("observer requires at least one window")

    records: list[dict[str, Any]] = []
    location: float | None = None
    scale: float | None = None
    for index, value in enumerate(transformed):
        ready = index >= config.warmup_windows
        if ready:
            assert location is not None and scale is not None
            location_before = location
            scale_before = scale
            deviation = abs(value - location_before) / scale_before
            delta = deviation / (deviation + config.compression_constant)
            residual = abs(value - location_before)
            location = (
                (1.0 - config.location_update_alpha) * location_before
                + config.location_update_alpha * value
            )
            scale = max(
                config.scale_floor,
                (1.0 - config.scale_update_alpha) * scale_before
                + config.scale_update_alpha * residual,
            )
        else:
            location_before = None
            scale_before = None
            deviation = None
            delta = config.warmup_delta
            if index + 1 == config.warmup_windows:
                location, scale = _initial_state(transformed[: index + 1], config)
        records.append(
            {
                "observer_ready": ready,
                "transformed_mean_surprisal": value,
                "location_before_update": location_before,
                "scale_before_update": scale_before,
                "standardized_deviation": deviation,
                "delta": delta,
            }
        )
    return records
