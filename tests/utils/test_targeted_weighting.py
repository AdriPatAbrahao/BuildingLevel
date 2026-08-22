import numpy as np

from utils.targeted_weighting import (
    expanded_weighted_training,
    select_targeted_weight,
    stratified_fold_ids,
)


def test_stratified_folds_place_each_band_in_every_fold():
    labels = np.repeat(["a", "b", "c", "d"], 10)
    folds = stratified_fold_ids(labels, n_splits=5, seed=42)
    assert sorted(np.unique(folds).tolist()) == [0, 1, 2, 3, 4]
    for fold in range(5):
        assert set(labels[folds == fold]) == {"a", "b", "c", "d"}


def test_integer_weight_replication_does_not_modify_base_rows():
    base_X = np.array([[1.0], [2.0]], dtype=np.float32)
    base_y = np.array([[10.0], [20.0]], dtype=np.float32)
    target_X = np.array([[3.0], [4.0]], dtype=np.float32)
    target_y = np.array([[30.0], [40.0]], dtype=np.float32)
    X, y = expanded_weighted_training(
        base_X, base_y, target_X, target_y, np.array([1]), targeted_weight=3
    )
    np.testing.assert_array_equal(X[:2], base_X)
    np.testing.assert_array_equal(y[:2], base_y)
    np.testing.assert_array_equal(X[2:], np.array([[4.0], [4.0], [4.0]]))


def test_selection_rejects_global_degradation_and_uses_lowest_oof_mae():
    def result(weight, targeted_mae, global_mae, global_rmse):
        return {
            "targeted_weight": weight,
            "oof_targeted": {"mean_absolute_error_kgf": targeted_mae},
            "global_validation_mean": {
                "mean_absolute_error_kgf": global_mae,
                "rmse_kgf": global_rmse,
            },
        }

    selection = select_targeted_weight(
        [result(2, 18.0, 17.0, 22.0), result(4, 16.0, 19.0, 24.0)],
        source_targeted_mae_kgf=22.0,
        source_global_validation_mae_kgf=17.0,
        source_global_validation_rmse_kgf=22.0,
    )
    assert selection["selected_targeted_weight"] == 2
    assert not selection["assessed_weights"][1]["eligible"]
