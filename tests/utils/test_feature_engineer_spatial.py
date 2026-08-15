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
    assert len(values) == len(names) == 28
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
