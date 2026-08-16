import pytest

from scripts.analyze_feature_importance import feature_groups
from utils.feature_engineer import FeatureEngineer


def test_feature_importance_groups_cover_current_schema_once():
    names = FeatureEngineer.feature_names()
    groups = feature_groups(names)
    flattened = [index for indices in groups.values() for index in indices]

    assert len(groups) == 7
    assert sorted(flattened) == list(range(len(names)))
    assert len(flattened) == len(set(flattened))


def test_feature_importance_groups_reject_unknown_schema():
    with pytest.raises(ValueError, match="do not match"):
        feature_groups(FeatureEngineer.feature_names() + ["unknown"])
