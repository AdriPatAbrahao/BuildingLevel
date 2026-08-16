import csv
import json

import numpy as np

from utils.plot_data_store import PLOT_DATA_FORMAT_VERSION, PlotDataStore


def test_plot_data_store_merges_sections_and_writes_auditable_csv(tmp_path):
    store = PlotDataStore(tmp_path)
    store.save_regression_test(
        [1000.0, 1200.0],
        [980.0, 1260.0],
        sample_indices=[7, 11],
    )
    store.save_classifier_test(
        [0, 1],
        [0.8, 0.2],
        [0, 1],
        invalid_threshold=0.6,
        X_test=np.array([[1.0, 2.0], [3.0, 4.0]]),
        feature_names=["a", "b"],
    )

    arrays = store.load()
    assert arrays["regression_test_indices"].tolist() == [7, 11]
    assert arrays["regression_residual_steel_kgf"].tolist() == [20.0, -60.0]
    assert arrays["classifier_invalid_probability"].tolist() == [0.8, 0.2]
    assert arrays["classifier_X_test"].shape == (2, 2)

    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    assert manifest["plot_data_format_version"] == PLOT_DATA_FORMAT_VERSION
    assert set(manifest["sections"]) == {"regression_test", "classifier_test"}
    assert manifest["sections"]["regression_test"]["metadata"]["unit"] == "kgf"
    assert manifest["sections"]["classifier_test"]["metadata"]["feature_names"] == [
        "a",
        "b",
    ]

    with open(
        tmp_path / "metrics" / "regression_test_predictions.csv",
        newline="",
        encoding="utf-8",
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["sample_index"] == "7"
    assert float(rows[1]["absolute_error_kgf"]) == 60.0


def test_plot_data_store_rejects_misaligned_predictions(tmp_path):
    store = PlotDataStore(tmp_path)

    try:
        store.save_regression_test([1.0, 2.0], [1.0])
    except ValueError as exc:
        assert "aligned" in str(exc)
    else:
        raise AssertionError("Expected misaligned regression arrays to be rejected.")
