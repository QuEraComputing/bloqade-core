"""Helpers for bloqade's local StorageBackend shapes.

The dict schema used by `DictStorage`/`SQLiteStorage` is owned by bloqade and
has nothing to do with the qlam wire format — keep these helpers separate from
`remote.py`. The `storage` pytest fixture below is the single place where the
in-memory vs. SQLite parametrize lives; tests that re-roll their own copy of
this fixture should switch to importing it.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

import numpy as np
import pytest

from bloqade.core.device.local_storage import (
    DictStorage,
    ShotResult,
    SQLiteStorage,
)

# Shared anchor; previously duplicated across test_storage / test_future / test_result.
CREATION_TIME = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def make_shot(
    *,
    task_id: str = "task-1",
    shot_index: int = 0,
    subtask_index: int = 0,
    subtask_shot_index: int = 0,
    frame_type: str = "DETECTED",
    bitstring: Iterable[bool] = (True, False),
) -> ShotResult:
    """Build a `ShotResult` row for the local storage backend.

    `frame_type` defaults to upper-case `"DETECTED"` because bloqade normalizes
    frame type on ingest (see `future._fetch_subtask_page`). Stored rows are
    always upper-case; use this helper when asserting on storage contents.
    """
    return ShotResult(
        task_id=task_id,
        shot_index=shot_index,
        subtask_index=subtask_index,
        subtask_shot_index=subtask_shot_index,
        frame_type=frame_type,
        bitstring=np.array(list(bitstring)),
    )


def assert_shot_equal(actual: ShotResult, expected: ShotResult) -> None:
    assert actual.task_id == expected.task_id
    assert actual.shot_index == expected.shot_index
    assert actual.subtask_index == expected.subtask_index
    assert actual.subtask_shot_index == expected.subtask_shot_index
    assert actual.frame_type == expected.frame_type
    np.testing.assert_array_equal(actual.bitstring, expected.bitstring)


def _sort_key(shot: ShotResult):
    return (shot.task_id, shot.shot_index, shot.frame_type)


def assert_shots_equal(actual: list[ShotResult], expected: list[ShotResult]) -> None:
    """Compare two shot lists order-insensitively (sorted by id/index/frame)."""
    actual = sorted(actual, key=_sort_key)
    expected = sorted(expected, key=_sort_key)
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert_shot_equal(a, e)


@pytest.fixture(params=["dict", "sqlite"])
def storage(request, tmp_path):
    """Parametrize `DictStorage` and a fresh `SQLiteStorage` for each test.

    Use as a normal pytest fixture: `def test_x(storage): ...`. Tests that
    want to opt out of the SQLite case can override locally.
    """
    if request.param == "dict":
        yield DictStorage()
        return

    sqlite_storage = SQLiteStorage(str(tmp_path / "shots.sqlite"))
    try:
        yield sqlite_storage
    finally:
        sqlite_storage.close()
