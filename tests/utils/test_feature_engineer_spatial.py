import pytest
from shapely.geometry import box

from utils.feature_engineer import FeatureEngineer


def _rectangle(center_x, center_y, width, height):
    return box(
        center_x - width / 2.0,
        center_y - height / 2.0,
        center_x + width / 2.0,
        center_y + height / 2.0,
    )


def _features_by_name(columns):
    engineer = FeatureEngineer(columns, [])
    values = engineer.extract_features()
    names = FeatureEngineer.feature_names()
    assert len(values) == len(names) == 23
    return dict(zip(names, values)), engineer.get_spatial_diagnostics()


def test_fixed_reference_spatial_features_are_symmetric_and_unbiased():
    columns = [
        _rectangle(100.0, 100.0, 20.0, 20.0),
        _rectangle(620.0, 100.0, 20.0, 20.0),
        _rectangle(100.0, 720.0, 20.0, 20.0),
        _rectangle(620.0, 720.0, 20.0, 20.0),
        _rectangle(360.0, 410.0, 20.0, 20.0),
    ]

    features, diagnostics = _features_by_name(columns)

    assert diagnostics["column_area_offset_x_norm"] == pytest.approx(0.0)
    assert diagnostics["column_area_offset_y_norm"] == pytest.approx(0.0)
    assert diagnostics["column_area_coupling_xy_norm"] == pytest.approx(0.0)
    assert diagnostics["max_quadrant_area_ratio_fixed"] == pytest.approx(0.25)
    assert features["column_area_spread_x_norm"] > 0.0
    assert features["column_area_spread_y_norm"] > 0.0


def test_spatial_spread_distinguishes_center_from_corners():
    centered = [_rectangle(360.0, 410.0, 20.0, 20.0) for _ in range(4)]
    corners = [
        _rectangle(10.0, 10.0, 20.0, 20.0),
        _rectangle(710.0, 10.0, 20.0, 20.0),
        _rectangle(10.0, 810.0, 20.0, 20.0),
        _rectangle(710.0, 810.0, 20.0, 20.0),
    ]

    center_features, _ = _features_by_name(centered)
    corner_features, _ = _features_by_name(corners)

    assert center_features["column_area_spread_x_norm"] == pytest.approx(0.0)
    assert center_features["column_area_spread_y_norm"] == pytest.approx(0.0)
    assert corner_features["column_area_spread_x_norm"] > 0.0
    assert corner_features["column_area_spread_y_norm"] > 0.0


def test_stiffness_eccentricity_preserves_column_rotation_effect():
    columns = [
        _rectangle(100.0, 410.0, 20.0, 100.0),
        _rectangle(620.0, 410.0, 100.0, 20.0),
    ]

    features, diagnostics = _features_by_name(columns)

    assert diagnostics["column_area_offset_x_norm"] == pytest.approx(0.0)
    assert diagnostics["stiffness_ecc_x_norm"] > 0.0
    assert diagnostics["stiffness_ecc_y_norm"] == pytest.approx(0.0)


def test_constant_metrics_are_diagnostics_not_model_features():
    engineer = FeatureEngineer(
        [_rectangle(360.0, 410.0, 20.0, 100.0)],
        [],
    )
    engineer.extract_features()
    diagnostics = engineer.get_diagnostics()
    model_names = set(engineer.feature_names())

    assert diagnostics["columns_count"] == pytest.approx(1.0)
    assert diagnostics["columns_mean_area_cm2"] == pytest.approx(2000.0)
    assert diagnostics["vol_columns_m3"] == pytest.approx(0.6)
    assert diagnostics["columns_total_perimeter_cm"] == pytest.approx(240.0)
    assert diagnostics["columns_mean_perimeter_cm"] == pytest.approx(240.0)
    assert diagnostics["columns_std_perimeter_cm"] == pytest.approx(0.0)
    assert diagnostics["mean_radius_gyration_min"] == pytest.approx(20.0 / 12**0.5)
    assert diagnostics["min_radius_gyration_global"] == pytest.approx(20.0 / 12**0.5)
    assert "columns_count" not in model_names
    assert "columns_mean_area_cm2" not in model_names
    assert "vol_columns_m3" not in model_names
    assert "columns_total_perimeter_cm" not in model_names
    assert "columns_mean_perimeter_cm" not in model_names
    assert "columns_std_perimeter_cm" not in model_names
    assert "mean_radius_gyration_min" not in model_names
    assert "min_radius_gyration_global" not in model_names


def test_continuous_beam_is_split_into_physical_clear_spans():
    columns = [
        _rectangle(0.0, 0.0, 20.0, 20.0),
        _rectangle(100.0, 0.0, 20.0, 20.0),
        _rectangle(200.0, 0.0, 20.0, 20.0),
    ]
    beams = [{"node_1": (-10.0, 0.0), "node_2": (210.0, 0.0)}]
    engineer = FeatureEngineer(columns, beams)
    features = dict(zip(engineer.feature_names(), engineer.extract_features()))
    diagnostics = engineer.get_diagnostics()

    assert diagnostics["beams_total_clear_length_x_cm"] == pytest.approx(160.0)
    assert diagnostics["beams_total_clear_length_y_cm"] == pytest.approx(0.0)
    assert features["beams_std_clear_span_x_cm"] == pytest.approx(0.0)
    assert features["beams_max_clear_span_x_cm"] == pytest.approx(80.0)
    assert diagnostics["beam_definition_count"] == pytest.approx(1.0)
    assert diagnostics["clear_span_count_x"] == pytest.approx(2.0)
    assert diagnostics["clear_span_count_y"] == pytest.approx(0.0)
    assert diagnostics["beams_mean_clear_span_x_cm"] == pytest.approx(80.0)
    assert diagnostics["beams_p95_clear_span_x_cm"] == pytest.approx(80.0)
    assert diagnostics["vol_beams_m3"] == pytest.approx(0.128)


def test_radius_balance_preserves_section_direction_without_redundant_means():
    vertical_features, _ = _features_by_name(
        [_rectangle(360.0, 410.0, 20.0, 100.0)]
    )
    horizontal_features, _ = _features_by_name(
        [_rectangle(360.0, 410.0, 100.0, 20.0)]
    )
    name = "columns_mean_radius_gyration_directional_balance"

    assert vertical_features[name] == pytest.approx(-2.0 / 3.0)
    assert horizontal_features[name] == pytest.approx(2.0 / 3.0)


def test_redundant_beam_totals_and_mean_radii_are_diagnostics_only():
    engineer = FeatureEngineer(
        [_rectangle(360.0, 410.0, 20.0, 100.0)],
        [],
    )
    engineer.extract_features()
    diagnostics = engineer.get_diagnostics()
    model_names = set(engineer.feature_names())

    for name in (
        "beams_total_clear_length_x_cm",
        "beams_total_clear_length_y_cm",
        "mean_radius_gyration_x",
        "mean_radius_gyration_y",
    ):
        assert name in diagnostics
        assert name not in model_names


def test_log_aspect_descriptors_are_symmetric_under_rotation():
    vertical_features, _ = _features_by_name(
        [_rectangle(360.0, 410.0, 20.0, 100.0)]
    )
    horizontal_features, _ = _features_by_name(
        [_rectangle(360.0, 410.0, 100.0, 20.0)]
    )

    mean_name = "columns_mean_log_aspect_ratio"
    std_name = "columns_std_log_aspect_ratio"
    max_name = "columns_max_abs_log_aspect_ratio"
    assert horizontal_features[mean_name] == pytest.approx(-vertical_features[mean_name])
    assert vertical_features[std_name] == pytest.approx(horizontal_features[std_name])
    assert vertical_features[max_name] == pytest.approx(horizontal_features[max_name])
    assert vertical_features[max_name] == pytest.approx(vertical_features[mean_name] * -1.0)


def test_raw_aspect_summary_names_are_not_model_features():
    model_names = set(FeatureEngineer.feature_names())
    assert "mean_col_aspect_ratio" not in model_names
    assert "std_col_aspect_ratio" not in model_names
    assert "max_col_aspect_ratio" not in model_names


def test_stiffness_spreads_distinguish_where_rotated_sections_are_located():
    configuration_a = [
        _rectangle(100.0, 410.0, 100.0, 20.0),
        _rectangle(360.0, 100.0, 20.0, 100.0),
    ]
    configuration_b = [
        _rectangle(100.0, 410.0, 20.0, 100.0),
        _rectangle(360.0, 100.0, 100.0, 20.0),
    ]
    features_a, _ = _features_by_name(configuration_a)
    features_b, _ = _features_by_name(configuration_b)
    stiffness_names = {
        "columns_stiffness_spread_x_response_norm",
        "columns_stiffness_spread_y_response_norm",
    }

    for name in set(features_a) - stiffness_names:
        assert features_a[name] == pytest.approx(features_b[name])
    for name in stiffness_names:
        assert features_b[name] > features_a[name]


def test_mean_inertias_are_diagnostics_not_model_features():
    engineer = FeatureEngineer(
        [_rectangle(360.0, 410.0, 20.0, 100.0)],
        [],
    )
    engineer.extract_features()
    diagnostics = engineer.get_diagnostics()
    model_names = set(engineer.feature_names())

    assert diagnostics["inertia_mean_Ix"] > 0.0
    assert diagnostics["inertia_mean_Iy"] > 0.0
    assert "inertia_mean_Ix" not in model_names
    assert "inertia_mean_Iy" not in model_names


def test_compactness_is_diagnostic_and_shape_factors_are_named_correctly():
    engineer = FeatureEngineer(
        [_rectangle(360.0, 410.0, 20.0, 100.0)],
        [],
    )
    engineer.extract_features()
    diagnostics = engineer.get_diagnostics()
    model_names = set(engineer.feature_names())

    assert diagnostics["columns_mean_compactness"] > 0.0
    assert "columns_mean_compactness" not in model_names
    assert "columns_mean_shape_factor" in model_names
    assert "columns_p95_shape_factor" in model_names
    assert "pillars_mean_slenderness" not in model_names
    assert "pillars_p95_slenderness" not in model_names
