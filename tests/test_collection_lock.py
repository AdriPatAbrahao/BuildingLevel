import json
import os

import pytest

from main import _acquire_collection_lock, _release_collection_lock


def test_collection_lock_blocks_an_active_owner(tmp_path):
    lock = tmp_path / "collection.lock"
    lock.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Another collection process"):
        _acquire_collection_lock(lock, tmp_path)


def test_collection_lock_replaces_stale_owner_and_releases(tmp_path):
    lock = tmp_path / "collection.lock"
    lock.write_text(json.dumps({"pid": 999_999_999}), encoding="utf-8")

    _acquire_collection_lock(lock, tmp_path)
    payload = json.loads(lock.read_text(encoding="utf-8"))
    assert payload["pid"] == os.getpid()

    _release_collection_lock(lock)
    assert not lock.exists()
