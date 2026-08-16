import numpy as np
import pytest

from utils.data_split import (
    classification_train_validation_test_split,
    regression_train_validation_test_split,
)


def _split(targets, prefix=230):
    return regression_train_validation_test_split(
        targets,
        test_ratio=0.15,
        validation_ratio_of_development=0.20,
        random_state=42,
        preused_development_prefix=prefix,
        max_stratification_bins=10,
    )


def test_final_2500_split_has_expected_sizes_and_clean_test():
    targets = np.linspace(250.0, 1100.0, 2500)

    split = _split(targets)

    assert len(split.train_indices) == 1700
    assert len(split.validation_indices) == 425
    assert len(split.test_indices) == 375
    assert np.all(split.test_indices >= 230)
    development = np.concatenate(
        [split.train_indices, split.validation_indices]
    )
    assert set(range(230)).issubset(set(development.tolist()))


def test_split_is_reproducible_and_complete():
    targets = np.random.default_rng(7).normal(size=2500)

    first = _split(targets)
    second = _split(targets)

    assert np.array_equal(first.train_indices, second.train_indices)
    assert np.array_equal(first.validation_indices, second.validation_indices)
    assert np.array_equal(first.test_indices, second.test_indices)
    combined = np.concatenate(
        [first.train_indices, first.validation_indices, first.test_indices]
    )
    assert sorted(combined.tolist()) == list(range(2500))


def test_target_rank_stratification_covers_full_range_in_each_split():
    targets = np.linspace(0.0, 1.0, 2500)

    split = _split(targets)

    for indices in (split.train_indices, split.validation_indices):
        values = targets[indices]
        assert values.min() < 0.05
        assert values.max() > 0.95

    # The artificial target is ordered, so protecting the first 230 samples
    # deliberately removes the global low tail from test eligibility. The test
    # must still cover the range of the newly collected eligible population.
    test_values = targets[split.test_indices]
    assert test_values.min() < 0.15
    assert test_values.max() > 0.95


def test_training_is_blocked_when_only_preused_samples_exist():
    with pytest.raises(ValueError, match="not enough new samples"):
        _split(np.arange(230, dtype=float))


def test_manifest_declares_no_preused_test_leakage():
    targets = np.linspace(250.0, 1100.0, 2500)
    split = _split(targets)

    manifest = split.as_manifest(targets)

    assert manifest["preused_development_prefix"] == 230
    assert manifest["test_contains_preused_samples"] is False
    assert manifest["test"]["count"] == 375


def test_classification_split_protects_preused_prefix_and_stratifies():
    labels = np.asarray(([1] * 9 + [0]) * 40, dtype=int)
    split = classification_train_validation_test_split(
        labels,
        test_ratio=0.20,
        validation_ratio_of_development=0.25,
        random_state=42,
        preused_development_prefix=100,
    )

    assert not np.any(split.test_indices < 100)
    combined = np.concatenate(
        [split.train_indices, split.validation_indices, split.test_indices]
    )
    assert sorted(combined.tolist()) == list(range(len(labels)))
    for indices in (
        split.train_indices,
        split.validation_indices,
        split.test_indices,
    ):
        assert set(labels[indices].tolist()) == {0, 1}
    assert split.as_manifest(labels)["test_contains_preused_samples"] is False


def test_classification_split_rejects_prefix_that_leaves_no_final_test():
    labels = np.asarray(([1] * 9 + [0]) * 10, dtype=int)

    with pytest.raises(ValueError, match="not enough new classifier samples"):
        classification_train_validation_test_split(
            labels,
            test_ratio=0.20,
            validation_ratio_of_development=0.25,
            random_state=42,
            preused_development_prefix=85,
        )
