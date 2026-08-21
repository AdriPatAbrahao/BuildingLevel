import pytest

from tqs_interface.tqs_errors import TQSErrorReader


def _reader_without_dlls():
    reader = TQSErrorReader.__new__(TQSErrorReader)
    reader._ngererro = None
    reader._nmsgerro = None
    reader._dll_dir = None
    return reader


def test_pillar_only_scope_is_accepted_without_changing_default_contract():
    reader = _reader_without_dlls()

    assert reader.get_critical_errors(
        building_name="AnyBuilding",
        target_scopes=("PILAR",),
    ) == []


def test_unknown_error_scope_is_rejected():
    reader = _reader_without_dlls()

    with pytest.raises(ValueError, match="Unsupported TQS error scope"):
        reader.get_critical_errors(
            building_name="AnyBuilding",
            target_scopes=("FOUNDATION",),
        )
