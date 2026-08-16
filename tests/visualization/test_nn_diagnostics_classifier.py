import json

import numpy as np

import visualization.nn_diagnostics as diagnostics
from utils.plot_data_store import PlotDataStore


class _FakeClassifier:
    classes_ = np.array([0, 1])

    def __init__(self, invalid_probability):
        self.invalid_probability = np.asarray(invalid_probability, dtype=float)

    def predict_proba(self, X):
        assert len(X) == len(self.invalid_probability)
        return np.column_stack(
            [self.invalid_probability, 1.0 - self.invalid_probability]
        )


def test_roc_plot_prefers_saved_test_curve_over_validation(tmp_path, monkeypatch):
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    test_payload = {
        "fpr": [0.0, 0.2, 1.0],
        "tpr": [0.0, 0.9, 1.0],
        "thresholds": [None, 0.7, 0.0],
        "split": "test",
    }
    validation_payload = {
        "fpr": [0.0, 0.8, 1.0],
        "tpr": [0.0, 0.2, 1.0],
        "thresholds": [None, 0.4, 0.0],
        "split": "validation",
    }
    (metrics / "roc_curve_test.json").write_text(json.dumps(test_payload))
    (metrics / "roc_curve.json").write_text(json.dumps(validation_payload))

    captured = {}
    real_auc = diagnostics.auc

    def _capture_auc(fpr, tpr):
        captured["fpr"] = np.asarray(fpr).tolist()
        captured["tpr"] = np.asarray(tpr).tolist()
        return real_auc(fpr, tpr)

    monkeypatch.setattr(diagnostics, "auc", _capture_auc)
    output = diagnostics.NNDiagnosticsPlotter(tmp_path).plot_roc_auc()

    assert output.exists()
    assert captured["fpr"] == test_payload["fpr"]
    assert captured["tpr"] == test_payload["tpr"]


def test_full_diagnostics_uses_calibrated_invalidity_rule(tmp_path, monkeypatch):
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    (metrics / "validity_threshold.json").write_text(
        json.dumps({"threshold": 0.7}),
        encoding="utf-8",
    )

    captured = {}
    plotter = diagnostics.NNDiagnosticsPlotter
    monkeypatch.setattr(plotter, "plot_learning_curves", lambda self: None)
    monkeypatch.setattr(plotter, "plot_gradient_norms", lambda self: None)
    monkeypatch.setattr(plotter, "plot_speedup_comparison", lambda self: None)
    monkeypatch.setattr(
        plotter,
        "plot_confusion_matrix",
        lambda self, y_true, y_pred, **kwargs: captured.update(
            {
                "confusion_true": np.asarray(y_true).tolist(),
                "confusion_pred": np.asarray(y_pred).tolist(),
                "threshold": kwargs["invalid_threshold"],
            }
        ),
    )
    monkeypatch.setattr(
        plotter,
        "plot_roc_auc",
        lambda self, **kwargs: captured.update(
            {
                "roc_true": np.asarray(kwargs["y_true"]).tolist(),
                "roc_score": np.asarray(kwargs["y_score"]).tolist(),
                "roc_split": kwargs["split_label"],
            }
        ),
    )
    monkeypatch.setattr(plotter, "plot_permutation_importance", lambda self, **kwargs: None)

    diagnostics.run_full_diagnostics(
        experiment_dir=tmp_path,
        feature_names=["feature"],
        classifier=_FakeClassifier([0.8, 0.6, 0.2]),
        y_test_valid=np.array([0, 0, 1]),
        X_test_clf=np.array([[1.0], [2.0], [3.0]]),
    )

    assert captured["confusion_true"] == [0, 0, 1]
    assert captured["confusion_pred"] == [0, 1, 1]
    assert captured["threshold"] == 0.7
    assert captured["roc_true"] == [1, 1, 0]
    assert captured["roc_score"] == [0.8, 0.6, 0.2]
    assert captured["roc_split"] == "test"


def test_regeneration_uses_saved_predictions_without_model(tmp_path, monkeypatch):
    store = PlotDataStore(tmp_path)
    store.save_regression_test([1000.0, 1200.0], [990.0, 1230.0])
    store.save_classifier_test(
        [0, 1],
        [0.85, 0.10],
        [0, 1],
        invalid_threshold=0.65,
    )

    captured = {}
    plotter = diagnostics.NNDiagnosticsPlotter
    monkeypatch.setattr(plotter, "plot_learning_curves", lambda self: None)
    monkeypatch.setattr(plotter, "plot_gradient_norms", lambda self: None)
    monkeypatch.setattr(plotter, "plot_speedup_comparison", lambda self: None)
    monkeypatch.setattr(
        plotter,
        "plot_scatter_and_residuals",
        lambda self, actual, predicted, **kwargs: captured.update(
            regression_actual=np.asarray(actual).tolist(),
            regression_predicted=np.asarray(predicted).tolist(),
        ),
    )
    monkeypatch.setattr(plotter, "plot_residuals_vs_predicted", lambda *args, **kwargs: None)
    monkeypatch.setattr(plotter, "plot_qq_residuals", lambda *args, **kwargs: None)
    monkeypatch.setattr(plotter, "plot_error_histogram", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        plotter,
        "plot_confusion_matrix",
        lambda self, actual, predicted, **kwargs: captured.update(
            classifier_actual=np.asarray(actual).tolist(),
            classifier_predicted=np.asarray(predicted).tolist(),
            threshold=kwargs["invalid_threshold"],
        ),
    )
    monkeypatch.setattr(
        plotter,
        "plot_roc_auc",
        lambda self, **kwargs: captured.update(
            invalid_event=np.asarray(kwargs["y_true"]).tolist(),
            invalid_probability=np.asarray(kwargs["y_score"]).tolist(),
        ),
    )

    diagnostics.regenerate_training_figures(tmp_path)

    assert captured["regression_actual"] == [1000.0, 1200.0]
    assert captured["regression_predicted"] == [990.0, 1230.0]
    assert captured["classifier_actual"] == [0, 1]
    assert captured["classifier_predicted"] == [0, 1]
    assert captured["threshold"] == 0.65
    assert captured["invalid_event"] == [1, 0]
    assert captured["invalid_probability"] == [0.85, 0.1]
