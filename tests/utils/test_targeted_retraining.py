import numpy as np
import pytest

from utils.targeted_retraining import (
    array_sha256,
    merge_retraining_arrays,
    validate_feature_isolation,
)


def _fixture():
    X = np.arange(48, dtype=np.float32).reshape(8, 6)
    y = np.arange(8, dtype=np.float32).reshape(-1, 1) + 100.0
    source = {
        "feature_vectors": X.tolist(),
        "output_values": y.tolist(),
        "classifier_features": X.tolist(),
        "classifier_labels": [0, 1, 1, 1, 0, 1, 1, 1],
    }
    arrays = {
        "X_train": X[[0, 1, 2, 3]],
        "y_train": y[[0, 1, 2, 3]],
        "X_val": X[[4, 5]],
        "y_val": y[[4, 5]],
        "X_test": X[[6, 7]],
        "y_test": y[[6, 7]],
        "train_indices": np.array([0, 1, 2, 3]),
        "validation_indices": np.array([4, 5]),
        "test_indices": np.array([6, 7]),
    }
    classifier_manifest = {
        "train": {"indices": [0, 1, 2, 3]},
        "validation": {"indices": [4, 5]},
        "test": {"indices": [6, 7]},
    }
    targeted_X = np.arange(48, 60, dtype=np.float32).reshape(2, 6)
    targeted_y = np.array([80.0, 90.0], dtype=np.float32)
    targeted_labels = np.array([1, 1], dtype=np.int64)
    return source, arrays, classifier_manifest, targeted_X, targeted_y, targeted_labels


def test_targeted_rows_are_appended_only_to_training_and_holdouts_are_frozen():
    source, arrays, manifest, target_X, target_y, target_labels = _fixture()
    reg, clf, checks = merge_retraining_arrays(
        source_checkpoint=source,
        source_arrays=arrays,
        classifier_split_manifest=manifest,
        targeted_features=target_X,
        targeted_targets=target_y,
        targeted_labels=target_labels,
    )

    assert reg["X_train"].shape == (6, 6)
    assert clf["X_train"].shape == (6, 6)
    np.testing.assert_array_equal(reg["X_train"][-2:], target_X)
    np.testing.assert_array_equal(clf["X_train"][-2:], target_X)
    np.testing.assert_array_equal(reg["X_val"], arrays["X_val"])
    np.testing.assert_array_equal(reg["X_test"], arrays["X_test"])
    assert checks["regression_validation_frozen"]
    assert checks["regression_test_frozen"]
    assert checks["classifier_validation_frozen"]
    assert checks["classifier_test_frozen"]
    assert checks["targeted_label_counts"] == {"1": 2}


def test_feature_isolation_rejects_cross_partition_duplicates():
    train = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    test = np.array([[3.0, 4.0]], dtype=np.float32)
    with pytest.raises(RuntimeError, match="split leakage"):
        validate_feature_isolation({"train": train, "test": test})


def test_array_hash_includes_shape_and_dtype():
    values = np.array([[1, 2], [3, 4]], dtype=np.float32)
    assert array_sha256(values) == array_sha256(values.copy())
    assert array_sha256(values) != array_sha256(values.astype(np.float64))
    assert array_sha256(values) != array_sha256(values.reshape(1, 4))
