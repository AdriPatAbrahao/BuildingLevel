import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config.settings import NeuralNetConfig
from models.nn_manager import NeuralNetworkManager


def _valid_arrays(train_rows=8, val_rows=4):
    X_train = np.zeros((train_rows, NeuralNetConfig.INPUT_SIZE), dtype=np.float32)
    y_train = np.zeros((train_rows, NeuralNetConfig.OUTPUT_SIZE), dtype=np.float32)
    X_val = np.zeros((val_rows, NeuralNetConfig.INPUT_SIZE), dtype=np.float32)
    y_val = np.zeros((val_rows, NeuralNetConfig.OUTPUT_SIZE), dtype=np.float32)
    return X_train, y_train, X_val, y_val


def test_training_contract_rejects_wrong_feature_count():
    manager = NeuralNetworkManager()
    X_train, y_train, X_val, y_val = _valid_arrays()
    X_train = X_train[:, :-1]
    X_val = X_val[:, :-1]

    with pytest.raises(ValueError, match="INPUT_SIZE"):
        manager._validate_training_arrays(X_train, y_train, X_val, y_val)


def test_training_contract_rejects_non_finite_values():
    manager = NeuralNetworkManager()
    X_train, y_train, X_val, y_val = _valid_arrays()
    X_val[0, 0] = np.nan

    with pytest.raises(ValueError, match="X_val contains NaN"):
        manager._validate_training_arrays(X_train, y_train, X_val, y_val)


def test_train_loader_avoids_single_sample_batch_without_dropping_data():
    manager = NeuralNetworkManager()
    X_train, y_train, X_val, y_val = _valid_arrays(train_rows=33)

    train_loader, _ = manager._create_dataloaders(
        X_train, y_train, X_val, y_val
    )
    batch_sizes = [len(batch_X) for batch_X, _ in train_loader]

    assert sum(batch_sizes) == 33
    assert batch_sizes == [31, 2]
    assert 1 not in batch_sizes


def test_eval_loss_is_weighted_by_number_of_samples():
    manager = NeuralNetworkManager()
    manager.model = nn.Identity()
    X = torch.tensor([[0.0], [0.0], [3.0]], dtype=torch.float32)
    y = torch.zeros_like(X)
    loader = DataLoader(TensorDataset(X, y), batch_size=2, shuffle=False)

    loss = manager._run_eval_epoch(loader, nn.MSELoss())

    assert loss == pytest.approx(3.0)


def test_train_loss_is_weighted_by_number_of_samples():
    manager = NeuralNetworkManager()
    manager.model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        manager.model.weight.fill_(1.0)
    X = torch.tensor([[0.0], [0.0], [3.0]], dtype=torch.float32)
    y = torch.zeros_like(X)
    loader = DataLoader(TensorDataset(X, y), batch_size=2, shuffle=False)
    optimizer = torch.optim.SGD(manager.model.parameters(), lr=0.0)

    loss = manager._run_train_epoch(loader, nn.MSELoss(), optimizer)

    assert loss == pytest.approx(3.0)


def test_prediction_rejects_wrong_feature_count():
    manager = NeuralNetworkManager()
    manager.model = nn.Identity()
    manager.is_trained = True
    X = np.zeros((1, NeuralNetConfig.INPUT_SIZE - 1), dtype=np.float32)

    with pytest.raises(ValueError, match="feature count"):
        manager.predict(X)


def test_short_training_records_and_restores_best_epoch():
    manager = NeuralNetworkManager()
    manager.num_epochs = 2
    manager.early_stopping_patience = 2
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(33, NeuralNetConfig.INPUT_SIZE)).astype(np.float32)
    y_train = rng.normal(size=(33, NeuralNetConfig.OUTPUT_SIZE)).astype(np.float32)
    X_val = rng.normal(size=(8, NeuralNetConfig.INPUT_SIZE)).astype(np.float32)
    y_val = rng.normal(size=(8, NeuralNetConfig.OUTPUT_SIZE)).astype(np.float32)

    manager.train(X_train, y_train, X_val, y_val)

    assert manager.is_trained is True
    assert manager.best_epoch in {1, 2}
    assert manager.best_val_loss is not None
    assert np.isfinite(manager.best_val_loss)
