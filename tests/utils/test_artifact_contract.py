import joblib
import numpy as np
import pytest
import torch

from config.settings import NeuralNetConfig
from models.dnnmodel import SimpleNN
from models.nn_manager import NeuralNetworkManager
from utils.artifact_contract import current_artifact_contract
from utils.feature_pipeline import FeaturePipeline


def _fitted_pipeline():
    pipeline = FeaturePipeline()
    X = np.arange(4 * NeuralNetConfig.INPUT_SIZE, dtype=np.float32).reshape(4, -1)
    y = np.arange(4, dtype=np.float32).reshape(-1, 1)
    pipeline.fit(X, y)
    return pipeline


def test_pipeline_round_trip_preserves_exact_contract(tmp_path):
    path = tmp_path / "feature_pipeline.pkl"
    pipeline = _fitted_pipeline()

    pipeline.save(path)
    loaded = FeaturePipeline()
    loaded.load(path)

    assert loaded.artifact_contract == current_artifact_contract()
    assert loaded.scaler_X.n_features_in_ == NeuralNetConfig.INPUT_SIZE
    assert loaded.scaler_y.n_features_in_ == NeuralNetConfig.OUTPUT_SIZE


def test_pipeline_rejects_legacy_artifact_without_contract(tmp_path):
    path = tmp_path / "legacy_pipeline.pkl"
    pipeline = _fitted_pipeline()
    joblib.dump(
        {"scaler_X": pipeline.scaler_X, "scaler_y": pipeline.scaler_y},
        path,
    )

    with pytest.raises(RuntimeError, match="missing required contract fields"):
        FeaturePipeline().load(path)


def test_pipeline_rejects_changed_feature_order(tmp_path):
    path = tmp_path / "changed_pipeline.pkl"
    pipeline = _fitted_pipeline()
    pipeline.save(path)
    payload = joblib.load(path)
    payload["feature_names"][0], payload["feature_names"][1] = (
        payload["feature_names"][1],
        payload["feature_names"][0],
    )
    joblib.dump(payload, path)

    with pytest.raises(RuntimeError, match="contract is incompatible"):
        FeaturePipeline().load(path)


def _trained_manager_stub():
    manager = NeuralNetworkManager()
    manager.model = SimpleNN(
        NeuralNetConfig.INPUT_SIZE,
        NeuralNetConfig.OUTPUT_SIZE,
        NeuralNetConfig.HIDDEN_LAYERS,
        NeuralNetConfig.DROPOUT_RATE,
    ).to(manager.device)
    manager.is_trained = True
    manager.best_epoch = 7
    manager.best_val_loss = 0.125
    return manager


def test_model_round_trip_preserves_exact_contract(tmp_path):
    path = tmp_path / "trained_model.pth"
    manager = _trained_manager_stub()
    manager.save_model(path)

    loaded = NeuralNetworkManager()

    assert loaded.load_model(path) is True
    assert loaded.artifact_contract == current_artifact_contract()
    assert loaded.best_epoch == 7
    assert loaded.best_val_loss == pytest.approx(0.125)


def test_model_rejects_changed_schema(tmp_path):
    path = tmp_path / "changed_model.pth"
    manager = _trained_manager_stub()
    manager.save_model(path)
    payload = torch.load(path, weights_only=True)
    payload["feature_schema_version"] = 1
    torch.save(payload, path)

    loaded = NeuralNetworkManager()

    with pytest.raises(RuntimeError, match="legado ou incompatível"):
        loaded.load_model(path)
    assert loaded.is_trained is False


def test_untrained_model_cannot_be_saved(tmp_path):
    with pytest.raises(RuntimeError, match="modelo não treinado"):
        NeuralNetworkManager().save_model(tmp_path / "model.pth")
