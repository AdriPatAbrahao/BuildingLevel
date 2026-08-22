import numpy as np
import pytest

from utils.targeted_sampling import (
    TargetedSamplingConfig,
    assign_protected_roles,
    discretized_latin_hypercube,
    nearest_training_distances,
    segments_hash,
    select_stratified_candidates,
    targeted_design_vectors,
)


def _records(per_bin=10):
    rows = []
    for bin_index, (left, right) in enumerate(((220, 260), (260, 300))):
        for index in range(per_bin):
            rows.append(
                {
                    "candidate_id": f"{bin_index}-{index:02d}",
                    "predicted_steel_kgf": left + (right - left) * (index + 0.5) / per_bin,
                    "predicted_invalid_probability": index / per_bin,
                    "nearest_training_distance": 0.1 + index,
                }
            )
    return rows


def test_latin_hypercube_is_bounded_discrete_and_reproducible():
    kwargs = dict(
        lower_bounds=[20.0, 30.0],
        upper_bounds=[40.0, 60.0],
        pool_size=30,
        random_seed=7,
        step_cm=5.0,
    )
    first = discretized_latin_hypercube(**kwargs)
    second = discretized_latin_hypercube(**kwargs)

    np.testing.assert_array_equal(first, second)
    assert np.all(first >= [20.0, 30.0])
    assert np.all(first <= [40.0, 60.0])
    assert np.allclose(first / 5.0, np.round(first / 5.0))


def test_segment_hash_matches_json_semantics_for_lists_and_tuples():
    with_tuple = [{"start": (0.0, 1.0), "length": 20.0}]
    with_list = [{"start": [0.0, 1.0], "length": 20.0}]
    assert segments_hash(with_tuple) == segments_hash(with_list)


def test_nearest_training_distance_returns_expected_values():
    distances = nearest_training_distances(
        np.array([[0.0, 0.0], [3.0, 4.0]]),
        np.array([[0.0, 0.0], [0.0, 4.0]]),
    )
    np.testing.assert_allclose(distances, [0.0, 3.0])


def test_targeted_vectors_include_anchor_and_preserve_bounds():
    vectors = targeted_design_vectors(
        [20.0, 10.0],
        [120.0, 140.0],
        [20.0, 30.0],
        pool_size=100,
        random_seed=42,
        step_cm=5.0,
    )
    assert any(np.array_equal(row, [20.0, 30.0]) for row in vectors)
    assert np.all(vectors >= [20.0, 10.0])
    assert np.all(vectors <= [120.0, 140.0])
    assert np.allclose(vectors / 5.0, np.round(vectors / 5.0))


def test_stratified_selection_balances_bins_and_preassigns_roles():
    selected, diagnostics = select_stratified_candidates(
        _records(),
        sample_size=10,
        steel_bin_edges_kgf=(220.0, 260.0, 300.0),
        invalid_threshold=0.6,
        boundary_half_width=0.2,
    )
    assigned = assign_protected_roles(selected, protected_fraction=0.2)

    assert len(selected) == 10
    assert [row["selected"] for row in diagnostics] == [5, 5]
    assert sum(row["role"] == "protected_evaluation" for row in assigned) == 2
    assert len({row["candidate_id"] for row in assigned}) == 10


def test_stratified_selection_refuses_underfilled_bin():
    with pytest.raises(RuntimeError, match="Insufficient candidates"):
        select_stratified_candidates(
            _records(per_bin=2),
            sample_size=10,
            steel_bin_edges_kgf=(220.0, 260.0, 300.0),
            invalid_threshold=0.6,
            boundary_half_width=0.2,
        )


def test_configuration_rejects_invalid_ranges():
    with pytest.raises(ValueError, match="strictly increasing"):
        TargetedSamplingConfig(steel_bin_edges_kgf=(220.0, 220.0)).validate()
