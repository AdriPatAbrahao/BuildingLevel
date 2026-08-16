"""Versioned feature/target contract shared by training and inference."""

from __future__ import annotations

from typing import Any, Mapping

from config.settings import NeuralNetConfig
from utils.feature_engineer import FeatureEngineer


ARTIFACT_FORMAT_VERSION = 2
TARGET_NAMES = ["column_steel_weight"]
TARGET_UNITS = ["kgf"]


def current_artifact_contract() -> dict[str, Any]:
    """Return the exact semantic contract expected by the current code."""
    return {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "feature_schema_version": int(NeuralNetConfig.FEATURE_SCHEMA_VERSION),
        "feature_names": FeatureEngineer.feature_names(),
        "target_names": list(TARGET_NAMES),
        "target_units": list(TARGET_UNITS),
        "input_size": int(NeuralNetConfig.INPUT_SIZE),
        "output_size": int(NeuralNetConfig.OUTPUT_SIZE),
    }


def validate_artifact_contract(
    artifact: Mapping[str, Any],
    *,
    artifact_label: str,
) -> dict[str, Any]:
    """Reject legacy, incomplete or semantically incompatible artifacts."""
    expected = current_artifact_contract()
    missing = [key for key in expected if key not in artifact]
    if missing:
        raise RuntimeError(
            f"{artifact_label} is missing required contract fields: {missing}. "
            "Legacy artifacts must be retrained with the current pipeline."
        )

    mismatches = {
        key: {"artifact": artifact[key], "expected": expected_value}
        for key, expected_value in expected.items()
        if artifact[key] != expected_value
    }
    if mismatches:
        raise RuntimeError(
            f"{artifact_label} contract is incompatible with the current code: "
            f"{mismatches}."
        )
    return expected
