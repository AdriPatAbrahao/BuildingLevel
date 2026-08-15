import pytest
from shapely.affinity import rotate
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient

from utils.feature_engineer import calculate_centroidal_moment_of_inertia


def test_rectangle_inertia_is_independent_of_vertex_order():
    rectangle = box(0.0, 0.0, 20.0, 100.0)
    expected_ix = 20.0 * 100.0**3 / 12.0
    expected_iy = 100.0 * 20.0**3 / 12.0

    for polygon in (orient(rectangle, 1.0), orient(rectangle, -1.0)):
        ix, iy = calculate_centroidal_moment_of_inertia(polygon)
        assert ix == pytest.approx(expected_ix)
        assert iy == pytest.approx(expected_iy)


def test_physical_rotation_swaps_directional_inertias():
    rectangle = box(0.0, 0.0, 20.0, 100.0)
    rotated = rotate(rectangle, 90.0, origin="centroid")
    original_ix, original_iy = calculate_centroidal_moment_of_inertia(rectangle)
    rotated_ix, rotated_iy = calculate_centroidal_moment_of_inertia(rotated)

    assert rotated_ix == pytest.approx(original_iy)
    assert rotated_iy == pytest.approx(original_ix)


def test_polygon_hole_is_subtracted_from_inertia():
    hollow = Polygon(
        [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
        holes=[[(40.0, 40.0), (60.0, 40.0), (60.0, 60.0), (40.0, 60.0)]],
    )
    expected = (100.0 * 100.0**3 - 20.0 * 20.0**3) / 12.0
    ix, iy = calculate_centroidal_moment_of_inertia(hollow)

    assert ix == pytest.approx(expected)
    assert iy == pytest.approx(expected)
