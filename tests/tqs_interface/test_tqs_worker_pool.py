import pytest

from tqs_interface.tqs_worker_pool import (
    TQSWorkerPool,
    _evaluate_structural_validity,
    _safe_writef,
)


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


def test_multiple_workers_require_explicit_simultaneous_tqs_gate():
    with pytest.raises(ValueError, match="allow_simultaneous_tqs=True"):
        TQSWorkerPool(num_workers=2)


class _Latin1OnlyTQSUtil:
    """Mimics TQSUtil.writef's real behaviour: raises UnicodeEncodeError for
    any character outside latin-1, exactly like the live TQS DLL wrapper."""

    def __init__(self):
        self.received = []

    def writef(self, text):
        text.encode("latin-1")  # raises UnicodeEncodeError, same as TQSUtil
        self.received.append(text)


def test_safe_writef_survives_non_latin1_characters():
    """Regression test: a 2026-08-17 canary run crashed both worker
    subprocesses outright because the TimeoutError message contained an
    em dash, which TQSUtil.writef cannot encode as latin-1. That turned one
    recoverable job timeout into a permanently dead worker slot instead of
    a WorkerResult with an error. _safe_writef must never raise."""
    tqs_util = _Latin1OnlyTQSUtil()

    _safe_writef(tqs_util, "slot killed — TQS process killed.")  # em dash

    assert len(tqs_util.received) == 1
    assert tqs_util.received[0].encode("latin-1")  # sanitized, now encodable


def test_safe_writef_passes_plain_text_through_unchanged():
    tqs_util = _Latin1OnlyTQSUtil()

    _safe_writef(tqs_util, "plain ascii message")

    assert tqs_util.received == ["plain ascii message"]
