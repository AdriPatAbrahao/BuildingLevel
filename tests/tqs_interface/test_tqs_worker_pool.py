import pytest

from tqs_interface.tqs_worker_pool import _evaluate_structural_validity


class FakeErrorReader:
    def __init__(self, available=True, errors=None, exception=None):
        self.available = available
        self.errors = [] if errors is None else errors
        self.exception = exception
        self.calls = 0

    def _dlls_available(self):
        return self.available

    def get_critical_errors(self, building_name=None, strict=False):
        self.calls += 1
        assert strict is True
        if self.exception is not None:
            raise self.exception
        return self.errors


def test_optional_validity_check_does_not_call_reader():
    reader = FakeErrorReader(available=False)

    assert _evaluate_structural_validity(reader, "Slot_01", required=False)
    assert reader.calls == 0


def test_required_validity_check_fails_when_dll_is_unavailable():
    reader = FakeErrorReader(available=False)

    with pytest.raises(RuntimeError, match="DLLs are unavailable"):
        _evaluate_structural_validity(reader, "Slot_01", required=True)


def test_required_validity_check_fails_when_reader_raises():
    reader = FakeErrorReader(exception=OSError("reader failure"))

    with pytest.raises(RuntimeError, match="validity check failed"):
        _evaluate_structural_validity(reader, "Slot_01", required=True)


@pytest.mark.parametrize(
    ("errors", "expected"),
    [([], True), ([object()], False)],
)
def test_required_validity_check_uses_critical_errors(errors, expected):
    reader = FakeErrorReader(errors=errors)

    assert _evaluate_structural_validity(reader, "Slot_01", required=True) is expected
    assert reader.calls == 1
