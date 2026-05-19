from datetime import datetime, timezone

import numpy as np
import pytest
from qlam_core.plugins.tasks.api.tasks_models import (
    Program,
    Subtask,
    TaskMetadata,
    TaskDefinition,
)

from bloqade.core.device.local_storage import (
    ShotFilter,
    ShotResult,
    DictStorage,
    SQLiteStorage,
    StorageFilter,
)

CREATION_TIME = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


@pytest.fixture(params=["dict", "sqlite"])
def storage(request, tmp_path):
    if request.param == "dict":
        yield DictStorage()
        return

    sqlite_storage = SQLiteStorage(str(tmp_path / "shots.sqlite"))
    try:
        yield sqlite_storage
    finally:
        sqlite_storage.close()


def make_shot(
    *,
    task_id: str = "task-1",
    shot_index: int = 0,
    subtask_index: int = 0,
    subtask_shot_index: int = 0,
    frame_type: str = "RAW",
    bitstring: tuple[bool, ...] = (True, False),
):
    return ShotResult(
        task_id=task_id,
        shot_index=shot_index,
        subtask_index=subtask_index,
        subtask_shot_index=subtask_shot_index,
        frame_type=frame_type,
        bitstring=np.array(bitstring),
    )


def sort_key(shot: ShotResult):
    return (shot.task_id, shot.shot_index, shot.frame_type)


def assert_shot_equal(actual: ShotResult, expected: ShotResult):
    assert actual.task_id == expected.task_id
    assert actual.shot_index == expected.shot_index
    assert actual.subtask_index == expected.subtask_index
    assert actual.subtask_shot_index == expected.subtask_shot_index
    assert actual.frame_type == expected.frame_type
    np.testing.assert_array_equal(actual.bitstring, expected.bitstring)


def assert_shots_equal(actual: list[ShotResult], expected: list[ShotResult]):
    actual = sorted(actual, key=sort_key)
    expected = sorted(expected, key=sort_key)
    assert len(actual) == len(expected)
    for actual_shot, expected_shot in zip(actual, expected):
        assert_shot_equal(actual_shot, expected_shot)


def test_storage_get_shots_without_filter_yields_stored_shots(storage):
    shots = [
        make_shot(task_id="task-1", shot_index=0),
        make_shot(task_id="task-1", shot_index=1, bitstring=(False, True)),
    ]

    storage.add_shots(shots)

    assert_shots_equal(list(storage.get_shots(shot_filter=None)), shots)


def test_storage_get_shots_with_task_filter_yields_matching_shots(storage):
    matching_shot = make_shot(task_id="task-1", shot_index=0)
    other_shot = make_shot(task_id="task-2", shot_index=0, bitstring=(False, True))

    storage.add_shots([matching_shot, other_shot])

    assert_shots_equal(
        list(storage.get_shots(shot_filter=ShotFilter(task_ids=("task-1",)))),
        [matching_shot],
    )


def test_storage_get_shots_with_multiple_task_filter_yields_matching_shots(storage):
    first_matching_shot = make_shot(task_id="task-1", shot_index=0)
    other_shot = make_shot(task_id="task-2", shot_index=0, bitstring=(False, True))
    second_matching_shot = make_shot(task_id="task-3", shot_index=0)

    storage.add_shots([first_matching_shot, other_shot, second_matching_shot])

    assert_shots_equal(
        list(storage.get_shots(shot_filter=ShotFilter(task_ids=("task-1", "task-3")))),
        [first_matching_shot, second_matching_shot],
    )


def test_storage_get_shots_with_subtask_filter_yields_matching_shots(storage):
    matching_shot = make_shot(
        shot_index=0,
        subtask_index=2,
        subtask_shot_index=0,
    )
    other_shot = make_shot(
        shot_index=1,
        subtask_index=3,
        subtask_shot_index=0,
        bitstring=(False, True),
    )

    storage.add_shots([matching_shot, other_shot])

    assert_shots_equal(
        list(storage.get_shots(shot_filter=ShotFilter(subtask_indices=(2,)))),
        [matching_shot],
    )


def test_storage_get_shots_with_frame_type_filter_yields_matching_shots(storage):
    matching_shot = make_shot(shot_index=0, frame_type="RAW")
    other_shot = make_shot(
        shot_index=0,
        frame_type="CALIBRATION",
        bitstring=(False, True),
    )

    storage.add_shots([matching_shot, other_shot])

    assert_shots_equal(
        list(storage.get_shots(shot_filter=ShotFilter(frame_type="raw"))),
        [matching_shot],
    )


def test_storage_get_shots_with_multiple_subtask_indices(storage):
    s0 = make_shot(shot_index=0, subtask_index=0)
    s1 = make_shot(shot_index=1, subtask_index=1, bitstring=(False, True))
    s2 = make_shot(shot_index=2, subtask_index=2)

    storage.add_shots([s0, s1, s2])

    assert_shots_equal(
        list(storage.get_shots(shot_filter=ShotFilter(subtask_indices=(0, 2)))),
        [s0, s2],
    )


def test_storage_get_shots_with_combined_filters(storage):
    matching = make_shot(
        task_id="task-1", shot_index=0, subtask_index=2, frame_type="RAW"
    )
    wrong_task = make_shot(
        task_id="task-2",
        shot_index=1,
        subtask_index=2,
        frame_type="RAW",
        bitstring=(False, True),
    )
    wrong_subtask = make_shot(
        task_id="task-1", shot_index=2, subtask_index=3, frame_type="RAW"
    )
    wrong_frame = make_shot(
        task_id="task-1", shot_index=3, subtask_index=2, frame_type="DETECTED"
    )

    storage.add_shots([matching, wrong_task, wrong_subtask, wrong_frame])

    assert_shots_equal(
        list(
            storage.get_shots(
                shot_filter=ShotFilter(
                    task_ids=("task-1",),
                    subtask_indices=(2,),
                    frame_type="raw",
                )
            )
        ),
        [matching],
    )


def test_storage_get_shots_returns_empty_when_no_match(storage):
    storage.add_shots([make_shot(task_id="task-1")])

    assert (
        list(storage.get_shots(shot_filter=ShotFilter(task_ids=("task-other",)))) == []
    )


def test_storage_get_shots_returns_empty_when_storage_empty(storage):
    assert list(storage.get_shots(shot_filter=None)) == []


def make_task_definition(
    *,
    program_language: str | None = "flair.v1",
    programs: list[Program] | None = None,
    subtasks: list[Subtask] | None = None,
) -> TaskDefinition:
    if programs is None:
        programs = [Program(content="program-0")]
    if subtasks is None:
        subtasks = [Subtask(program_index=0, num_shots=10)]
    return TaskDefinition(
        program_language=program_language,
        programs=programs,
        subtasks=subtasks,
    )


def add_task_definition(
    storage,
    task_id: str,
    task_definition: TaskDefinition,
    creation_time: datetime = CREATION_TIME,
):
    storage.add_task_definition(task_id, task_definition, creation_time)


def test_storage_task_ids_empty(storage):
    assert storage.task_ids() == set()


def test_storage_task_ids_returns_added_definitions(storage):
    add_task_definition(storage, "task-1", make_task_definition())
    add_task_definition(storage, "task-2", make_task_definition())

    assert storage.task_ids() == {"task-1", "task-2"}


def test_storage_get_programs_empty(storage):
    assert storage.get_programs() == []


def test_storage_get_programs_returns_program_records(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(programs=[Program(content="p1"), Program(content="p2")]),
    )
    add_task_definition(
        storage,
        "task-2",
        make_task_definition(programs=[Program(content="p3")]),
    )

    programs = storage.get_programs()
    keyed = sorted((p["task_id"], p["program_index"], p["content"]) for p in programs)
    assert keyed == [
        ("task-1", 0, "p1"),
        ("task-1", 1, "p2"),
        ("task-2", 0, "p3"),
    ]


def test_storage_get_programs_with_task_filter(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(programs=[Program(content="p1")]),
    )
    add_task_definition(
        storage,
        "task-2",
        make_task_definition(programs=[Program(content="p2")]),
    )

    programs = storage.get_programs(task_ids=("task-1",))
    assert len(programs) == 1
    assert programs[0]["task_id"] == "task-1"
    assert programs[0]["program_index"] == 0
    assert programs[0]["content"] == "p1"


def test_storage_get_programs_returns_independent_copies(storage):
    """Mutating the returned dicts must not affect subsequent reads."""
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(programs=[Program(content="p1")]),
    )

    programs = storage.get_programs()
    programs[0]["content"] = "MUTATED"

    refetched = storage.get_programs()
    assert refetched[0]["content"] == "p1"


def test_storage_get_subtasks_returns_independent_copies(storage):
    """Mutating the returned dicts must not affect subsequent reads —
    GeminiResult.subtasks() relies on this when popping task_id/metadata."""
    add_task_definition(storage, "task-1", make_task_definition())

    subtasks = storage.get_subtasks()
    subtasks[0].pop("task_id", None)

    refetched = storage.get_subtasks()
    assert refetched[0]["task_id"] == "task-1"


def test_storage_get_subtasks_empty(storage):
    assert storage.get_subtasks() == []


def test_storage_get_subtasks_returns_added_subtasks_as_dicts(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            subtasks=[
                Subtask(
                    program_index=0,
                    num_shots=10,
                    arguments={"a": 1.5, "b": 2.0},
                    subtask_metadata=TaskMetadata(user_metadata="hello"),
                ),
                Subtask(program_index=0, num_shots=5),
            ],
        ),
    )

    subtasks = sorted(storage.get_subtasks(), key=lambda s: s["subtask_index"])
    assert len(subtasks) == 2

    assert subtasks[0] == {
        "task_id": "task-1",
        "subtask_index": 0,
        "program_index": 0,
        "num_shots": 10,
        "arguments": {"a": 1.5, "b": 2.0},
        "metadata": {"user_metadata": "hello", "system_metadata": None},
        "completed_date": None,
    }
    assert subtasks[1] == {
        "task_id": "task-1",
        "subtask_index": 1,
        "program_index": 0,
        "num_shots": 5,
        "arguments": None,
        "metadata": None,
        "completed_date": None,
    }


def test_storage_get_subtasks_filtered_by_task(storage):
    add_task_definition(storage, "task-1", make_task_definition())
    add_task_definition(storage, "task-2", make_task_definition())

    subtasks = storage.get_subtasks(storage_filter=StorageFilter(task_ids=("task-1",)))

    assert [s["task_id"] for s in subtasks] == ["task-1"]


def test_storage_get_subtasks_filtered_by_subtask_indices(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            subtasks=[
                Subtask(program_index=0, num_shots=1),
                Subtask(program_index=0, num_shots=2),
                Subtask(program_index=0, num_shots=3),
            ],
        ),
    )

    subtasks = storage.get_subtasks(
        storage_filter=StorageFilter(subtask_indices=(0, 2))
    )

    assert sorted(s["subtask_index"] for s in subtasks) == [0, 2]


def test_storage_get_subtasks_filtered_by_task_and_subtask(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            subtasks=[
                Subtask(program_index=0, num_shots=1),
                Subtask(program_index=0, num_shots=2),
            ],
        ),
    )
    add_task_definition(
        storage,
        "task-2",
        make_task_definition(
            subtasks=[
                Subtask(program_index=0, num_shots=3),
                Subtask(program_index=0, num_shots=4),
            ],
        ),
    )

    subtasks = storage.get_subtasks(
        storage_filter=StorageFilter(task_ids=("task-1",), subtask_indices=(1,))
    )

    assert len(subtasks) == 1
    assert subtasks[0]["task_id"] == "task-1"
    assert subtasks[0]["subtask_index"] == 1


def test_storage_get_task_definition_round_trip(storage):
    task_def = TaskDefinition(
        program_language="flair.v1",
        programs=[Program(content="p0"), Program(content="p1")],
        subtasks=[
            Subtask(
                program_index=0,
                num_shots=10,
                arguments={"a": 1.5, "b": 2.0},
                subtask_metadata=TaskMetadata(user_metadata="hello"),
            ),
            Subtask(program_index=1, num_shots=5),
        ],
    )

    add_task_definition(storage, "task-1", task_def)

    assert storage.get_task_definition("task-1") == task_def


def test_storage_get_task_definition_isolates_by_task_id(storage):
    task_def_1 = TaskDefinition(
        program_language="flair.v1",
        programs=[Program(content="p1")],
        subtasks=[Subtask(program_index=0, num_shots=1)],
    )
    task_def_2 = TaskDefinition(
        program_language="qasm",
        programs=[Program(content="p2a"), Program(content="p2b")],
        subtasks=[
            Subtask(program_index=0, num_shots=2),
            Subtask(program_index=1, num_shots=3),
        ],
    )

    add_task_definition(storage, "task-1", task_def_1)
    add_task_definition(storage, "task-2", task_def_2)

    assert storage.get_task_definition("task-1") == task_def_1
    assert storage.get_task_definition("task-2") == task_def_2


def test_storage_get_program_language_returns_stored_language(storage):
    add_task_definition(
        storage, "task-1", make_task_definition(program_language="flair.v1")
    )
    add_task_definition(
        storage, "task-2", make_task_definition(program_language="qasm")
    )

    assert storage.get_program_language("task-1") == "flair.v1"
    assert storage.get_program_language("task-2") == "qasm"


def test_storage_get_task_creation_time_round_trip(storage):
    add_task_definition(storage, "task-1", make_task_definition())

    assert storage.get_task_creation_time("task-1") == CREATION_TIME


def test_storage_update_subtasks_completed_date_round_trip(storage):
    first_completed_date = datetime(2026, 1, 2, 3, 4, 6, tzinfo=timezone.utc)
    second_completed_date = datetime(2026, 1, 2, 3, 4, 7, tzinfo=timezone.utc)
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            subtasks=[
                Subtask(program_index=0, num_shots=1),
                Subtask(program_index=0, num_shots=1),
            ],
        ),
    )

    storage.update_subtasks_completed_date(
        "task-1",
        [
            {"subtask_index": 0, "completed_date": first_completed_date},
            {"subtask_index": 1, "completed_date": second_completed_date.isoformat()},
        ],
    )

    subtasks = sorted(storage.get_subtasks(), key=lambda s: s["subtask_index"])
    assert subtasks[0]["completed_date"] == first_completed_date
    assert subtasks[1]["completed_date"] == second_completed_date


def test_sqlite_storage_persists_data_across_connections(tmp_path):
    db_path = str(tmp_path / "shots.sqlite")
    shot = make_shot(task_id="task-1", shot_index=0)

    with SQLiteStorage(db_path) as store:
        add_task_definition(store, "task-1", make_task_definition())
        store.add_shots([shot])

    with SQLiteStorage(db_path) as store:
        assert store.task_ids() == {"task-1"}
        assert_shots_equal(list(store.get_shots(shot_filter=None)), [shot])
        assert len(store.get_subtasks()) == 1
