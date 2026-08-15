import numpy as np

from geometry.geometry_utils import GeometryProcessor
from geometry.length_input_processor import LengthProcessor
from optimization.design_space import DesignSpace


def _offset_corner_segments():
    """A 20 x 20 seed column whose centrelines have different starts."""
    return [
        {
            "start": (100.0, 210.0),
            "end": (120.0, 210.0),
            "length": 20.0,
            "maxlength": 100.0,
            "binary": 1,
            "group_id": "horizontal",
        },
        {
            "start": (110.0, 200.0),
            "end": (110.0, 220.0),
            "length": 20.0,
            "maxlength": 120.0,
            "binary": 1,
            "group_id": "vertical",
        },
    ]


def test_detects_offset_orthogonal_groups_by_physical_overlap():
    pairs = GeometryProcessor.find_orthogonal_group_pairs(
        _offset_corner_segments(),
        segment_total_thickness=20.0,
    )

    assert pairs == [("horizontal", "vertical")]


def test_does_not_pair_distinct_groups_that_only_touch_at_boundary():
    segments = [
        {
            "start": (0.0, 0.0),
            "end": (20.0, 0.0),
            "group_id": "horizontal",
        },
        {
            "start": (30.0, 10.0),
            "end": (30.0, 30.0),
            "group_id": "vertical",
        },
    ]

    pairs = GeometryProcessor.find_orthogonal_group_pairs(
        segments,
        segment_total_thickness=20.0,
    )

    assert pairs == []


def test_design_space_keeps_only_one_growth_axis_for_offset_corner(tmp_path):
    csv_path = tmp_path / "offset_corner.csv"
    csv_path.write_text(
        "x;y;dx;dy;length;maxlength;group_id\n"
        "100;210;1;0;20;100;horizontal\n"
        "110;200;0;1;20;120;vertical\n",
        encoding="utf-8",
    )

    design_space = DesignSpace(csv_path)
    group_to_index = {
        group: index for index, group in enumerate(design_space.group_keys)
    }
    expected_pair = (
        group_to_index["horizontal"],
        group_to_index["vertical"],
    )
    assert design_space._rect_constraints == [expected_pair]

    requested = np.array(
        [
            100.0 if group == "horizontal" else 120.0
            for group in design_space.group_keys
        ]
    )
    segments = design_space.segments_from_vector(requested)
    lengths_by_group = {
        segment["group_id"]: segment["length"] for segment in segments
    }

    # Vertical grows more, so the horizontal axis returns to its seed length.
    assert lengths_by_group == {
        "horizontal": 20.0,
        "vertical": 120.0,
    }

    columns, _ = LengthProcessor().process_segments(segments)
    assert len(columns) == 1
    rectangle = columns[0].minimum_rotated_rectangle
    assert np.isclose(columns[0].area / rectangle.area, 1.0)


def test_training_variation_uses_same_offset_corner_constraint(monkeypatch):
    segments = _offset_corner_segments()
    monkeypatch.setattr(
        "geometry.length_input_processor.random.random",
        lambda: 0.0,
    )
    monkeypatch.setattr(
        "geometry.length_input_processor.random.randint",
        lambda lower, upper: upper,
    )

    varied = LengthProcessor().generate_variation(segments)
    lengths_by_group = {
        segment["group_id"]: segment["length"] for segment in varied
    }

    assert lengths_by_group == {
        "horizontal": 20.0,
        "vertical": 120.0,
    }

