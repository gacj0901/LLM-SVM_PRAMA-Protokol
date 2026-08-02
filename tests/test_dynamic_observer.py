import json
import math
from pathlib import Path
from hashlib import sha256

import pytest

from aptadynamic_llm.dynamic_observer import DynamicObserverConfig, observe


CONTRACT = Path("config/cocc_dynamic_observer_contract_v1.json")
FREEZE = Path("config/cocc_dynamic_observer_contract_v1.freeze.json")


def config():
    return DynamicObserverConfig.from_contract(
        json.loads(CONTRACT.read_text(encoding="utf-8"))
    )


def test_contract_is_model_independent_and_calibration_free():
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert value["model_specific_parameters"] is False
    assert value["requires_external_calibration"] is False
    assert value["contains_outcome_labels"] is False
    assert value["contains_prompt_or_answer"] is False


def test_freeze_binds_exact_contract_bytes():
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert frozen["status"] == "FROZEN"
    assert frozen["observer_contract_sha256"] == sha256(CONTRACT.read_bytes()).hexdigest()


def test_warmup_is_explicit_then_observer_becomes_ready():
    records = observe([0.1] * 8 + [0.5], config())
    assert all(record["delta"] == 0.0 for record in records[:8])
    assert all(record["observer_ready"] is False for record in records[:8])
    assert records[8]["observer_ready"] is True
    assert 0.0 < records[8]["delta"] < 1.0


def test_observer_is_causal():
    prefix = [0.1] * 8 + [0.3, 0.2]
    first = observe(prefix, config())
    extended = observe(prefix + [100.0, 0.01], config())
    assert first == extended[: len(prefix)]


def test_observer_is_invariant_to_model_identity_because_none_is_accepted():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["model_specific_parameters"] = True
    with pytest.raises(ValueError, match="model-specific"):
        DynamicObserverConfig.from_contract(contract)


def test_outputs_are_finite_and_bounded():
    records = observe([0.0, 0.01, 0.2, 0.4, 0.1, 0.3, 0.2, 0.1, 1e6], config())
    assert all(math.isfinite(record["delta"]) for record in records)
    assert all(0.0 <= record["delta"] < 1.0 for record in records)
