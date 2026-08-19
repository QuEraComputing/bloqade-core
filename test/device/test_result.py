import json

import numpy as np
import pytest
from qlam_core.plugins.tasks.api.tasks_models import Subtask, TaskDefinition

from bloqade.core.device.local_storage import (
    DictStorage,
    ShotFilter,
    StorageFilter,
)
from bloqade.core.device.result import Result

from .fixtures import local, remote

CREATION_TIME = local.CREATION_TIME


def make_task_definition(subtasks: list[Subtask]) -> TaskDefinition:
    return remote.make_task_definition(subtasks=subtasks)


def add_task(storage: DictStorage, task_id: str, subtasks: list[Subtask]):
    storage.add_task_definition(task_id, make_task_definition(subtasks), CREATION_TIME)


def make_metadata(value: dict):
    return remote.make_task_metadata(user_metadata=json.dumps(value))


def make_shot(*, frame_type: str = "DETECTED", **kwargs):
    return local.make_shot(frame_type=frame_type, **kwargs)


def add_compatible_tasks(storage: DictStorage):
    add_task(
        storage,
        "task-1",
        [
            remote.make_subtask(
                num_shots=2,
                arguments={"theta": 1.0},
                subtask_metadata=make_metadata({"task": 1}),
            ),
            remote.make_subtask(num_shots=1),
        ],
    )
    add_task(
        storage,
        "task-2",
        [
            remote.make_subtask(
                num_shots=3,
                arguments={"theta": 1.0},
                subtask_metadata=make_metadata({"task": 2}),
            ),
            remote.make_subtask(num_shots=4),
        ],
    )
    storage.add_shots(
        [
            make_shot(task_id="task-1", shot_index=0, subtask_index=0),
            make_shot(
                task_id="task-1",
                shot_index=1,
                subtask_index=0,
                bitstring=(False, True),
            ),
            make_shot(
                task_id="task-1",
                shot_index=2,
                subtask_index=1,
                bitstring=(True, True),
            ),
            make_shot(
                task_id="task-2",
                shot_index=0,
                subtask_index=0,
                bitstring=(True, True),
            ),
            make_shot(
                task_id="task-2",
                shot_index=1,
                subtask_index=1,
                bitstring=(False, False),
            ),
        ]
    )


def test_result_default_shot_filter_selects_detected_frame():
    result = Result(storage=DictStorage())

    assert result.shot_filter.frame_type == "DETECTED"


def test_result_storage_filter_drops_shot_only_fields():
    result = Result(
        storage=DictStorage(),
        shot_filter=ShotFilter(
            task_ids=("task-1",),
            subtask_indices=(1,),
            task_subtask_pairs=(("task-1", 1),),
            frame_type="raw",
            task_shot_pairs=(("task-1", 3),),
        ),
    )

    assert result.storage_filter == StorageFilter(
        task_ids=("task-1",),
        subtask_indices=(1,),
        task_subtask_pairs=(("task-1", 1),),
    )


def test_result_subtasks_merges_compatible_task_ids_and_sums_shots():
    storage = DictStorage()
    add_compatible_tasks(storage)
    result = Result(
        storage=storage,
        shot_filter=ShotFilter(task_ids=("task-1", "task-2"), frame_type="DETECTED"),
    )

    subtasks = result.subtasks()

    assert subtasks == [
        {
            "subtask_index": 0,
            "program_index": 0,
            "num_shots": 5,
            "arguments": {"theta": 1.0},
            "completed_date": None,
        },
        {
            "subtask_index": 1,
            "program_index": 0,
            "num_shots": 5,
            "arguments": None,
            "completed_date": None,
        },
    ]


def test_result_shot_results_returns_bitstrings_grouped_by_subtask():
    storage = DictStorage()
    add_compatible_tasks(storage)
    result = Result(
        storage=storage,
        shot_filter=ShotFilter(task_ids=("task-1", "task-2"), frame_type="DETECTED"),
    )

    shot_results = result.shot_results()

    assert len(shot_results) == 2
    np.testing.assert_array_equal(
        shot_results[0],
        np.array(
            [
                [True, False],
                [False, True],
                [True, True],
            ]
        ),
    )
    np.testing.assert_array_equal(
        shot_results[1],
        np.array(
            [
                [True, True],
                [False, False],
            ]
        ),
    )


@pytest.mark.parametrize(
    ("second_subtask", "message"),
    [
        (
            remote.make_subtask(program_index=1, num_shots=1, arguments={"theta": 1.0}),
            "program_index",
        ),
        (
            remote.make_subtask(num_shots=1, arguments={"theta": 2.0}),
            "arguments",
        ),
    ],
)
def test_result_validate_rejects_incompatible_task_ids(second_subtask, message):
    storage = DictStorage()
    add_task(
        storage,
        "task-1",
        [remote.make_subtask(num_shots=1, arguments={"theta": 1.0})],
    )
    add_task(storage, "task-2", [second_subtask])
    result = Result(
        storage=storage,
        shot_filter=ShotFilter(task_ids=("task-1", "task-2")),
    )

    with pytest.raises(ValueError, match=message):
        result.validate()


def test_result_validate_treats_missing_and_empty_arguments_as_compatible():
    storage = DictStorage()
    add_task(storage, "task-1", [remote.make_subtask(num_shots=1)])
    add_task(storage, "task-2", [remote.make_subtask(num_shots=1, arguments={})])
    result = Result(
        storage=storage,
        shot_filter=ShotFilter(task_ids=("task-1", "task-2")),
    )

    result.validate()

    assert result._is_valid is True


def test_result_full_views_and_task_ids_respect_filter_scope():
    storage = DictStorage()
    add_compatible_tasks(storage)
    result = Result(
        storage=storage,
        shot_filter=ShotFilter(task_ids=("task-1",), frame_type="DETECTED"),
    )

    assert result.task_ids() == {"task-1"}
    assert result.arguments() == [{"theta": 1.0}, None]
    assert result.full_arguments() == [{"theta": 1.0}, None]
    assert [subtask["task_id"] for subtask in result.full_subtasks()] == [
        "task-1",
        "task-1",
    ]


def test_result_where_methods_return_narrowed_results():
    storage = DictStorage()
    add_task(
        storage,
        "task-1",
        [
            remote.make_subtask(
                num_shots=2,
                arguments={"theta": 1.0},
                subtask_metadata=make_metadata({"keep": False}),
            ),
            remote.make_subtask(
                num_shots=3,
                arguments={"theta": 2.0},
                subtask_metadata=make_metadata({"keep": True}),
            ),
        ],
    )
    storage.add_shots(
        [
            make_shot(task_id="task-1", shot_index=0, subtask_index=0),
            make_shot(
                task_id="task-1",
                shot_index=1,
                subtask_index=1,
                bitstring=(False, False),
            ),
        ]
    )
    result = Result(
        storage=storage,
        shot_filter=ShotFilter(task_ids=("task-1",), frame_type="DETECTED"),
    )

    by_argument = result.where_arguments(
        lambda arguments: arguments is not None and arguments["theta"] > 1.5
    )
    by_metadata = result.where_metadata(
        lambda metadata: metadata is not None and metadata["keep"]
    )
    by_subtask = result.where_subtasks(lambda subtask: subtask["num_shots"] == 3)
    by_shot = result.where_shots(lambda shot: bool(shot.bitstring.any()))

    assert by_argument.shot_filter.task_subtask_pairs == (("task-1", 1),)
    assert by_metadata.shot_filter.task_subtask_pairs == (("task-1", 1),)
    assert by_subtask.shot_filter.task_subtask_pairs == (("task-1", 1),)
    assert by_shot.shot_filter.task_shot_pairs == (("task-1", 0),)
