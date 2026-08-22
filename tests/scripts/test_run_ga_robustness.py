from scripts.run_ga_robustness import _select_candidates


def _record(seed, geometry_hash, vector, cost, margin):
    return {
        "ga_seed": seed,
        "solution_csv_sha256": geometry_hash,
        "vector_cm": vector,
        "predicted": {
            "material_cost_brl": cost,
            "classifier_margin_to_invalid": margin,
        },
    }


def test_candidate_selection_groups_equivalent_effective_geometries():
    records = [
        _record(1, "geometry-a", [10.0, 15.0], 100.0, 0.10),
        _record(2, "geometry-a", [10.0, 20.0], 100.0, 0.10),
        _record(3, "geometry-b", [15.0, 20.0], 101.0, 0.30),
        _record(4, "geometry-c", [20.0, 20.0], 103.0, 0.40),
    ]

    selected = _select_candidates(records)

    assert len(selected) == 3
    assert selected[0]["effective_geometry_sha256"] == "geometry-a"
    assert selected[0]["frequency"] == 2
    assert selected[0]["observed_in_seeds"] == [1, 2]
    assert selected[0]["selection_roles"] == ["lowest_predicted_cost", "modal_design"]
    assert selected[1]["effective_geometry_sha256"] == "geometry-b"
    assert selected[1]["selection_roles"] == [
        "largest_classifier_margin_within_2pct_of_best"
    ]
    assert selected[2]["effective_geometry_sha256"] == "geometry-c"
    assert selected[2]["selection_roles"] == ["next_lowest_cost_distinct_design"]
