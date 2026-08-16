from types import SimpleNamespace

import numpy as np
import pytest

import inference
from config.settings import NeuralNetConfig
from inference import BuildingInference
from utils.artifact_contract import current_artifact_contract


def test_missing_experiment_does_not_fall_back_to_latest(tmp_path, monkeypatch):
    latest = tmp_path / "latest_but_wrong"
    latest.mkdir()
    monkeypatch.setattr(inference.paths, "EXPERIMENTS_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="não encontrado"):
        BuildingInference("requested_experiment")


def _inference_stub():
    runner = BuildingInference.__new__(BuildingInference)
    runner.feature_pipeline = SimpleNamespace(
        artifact_contract=current_artifact_contract()
    )
    return runner


def test_runtime_feature_vector_rejects_extra_feature():
    runner = _inference_stub()
    values = [0.0] * (NeuralNetConfig.INPUT_SIZE + 1)

    with pytest.raises(RuntimeError, match="inferência cancelada"):
        runner._validate_feature_vector(values)


def test_runtime_feature_vector_rejects_non_finite_value():
    runner = _inference_stub()
    values = [0.0] * NeuralNetConfig.INPUT_SIZE
    values[3] = np.inf

    with pytest.raises(RuntimeError, match="NaN ou infinito"):
        runner._validate_feature_vector(values)


def test_runtime_feature_vector_accepts_exact_contract():
    runner = _inference_stub()
    values = [float(i) for i in range(NeuralNetConfig.INPUT_SIZE)]

    assert runner._validate_feature_vector(values) == values
