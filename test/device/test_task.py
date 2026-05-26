import importlib
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from bloqade.core.device.future import ApiFetchOptions
from bloqade.core.device.local_storage import DictStorage
from bloqade.core.device.task import (
    KernelBatchTask,
    ParameterScanTask,
    SingleKernelTask,
)

task_mod = importlib.import_module("bloqade.core.device.task")

CREATION_TIME = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


class FakeKernel:
    def __init__(self, sym_name: str):
        self.sym_name = sym_name


class SerializableSingleKernelTask(SingleKernelTask):
    def serialize_kernel(self, kernel):
        return f"serialized:{kernel.sym_name}"


class SerializableParameterScanTask(ParameterScanTask):
    def serialize_kernel(self, kernel):
        return f"serialized:{kernel.sym_name}"


class SerializableKernelBatchTask(KernelBatchTask):
    def serialize_kernel(self, kernel):
        return f"serialized:{kernel.sym_name}"


class RecordingFuture:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__dict__.update(kwargs)


def test_single_kernel_task_creates_task_definition_with_arguments_and_metadata():
    task = SerializableSingleKernelTask(
        context_name="ctx",
        program_language="flair.v1",
        kernel=FakeKernel("main"),
        arguments={"theta": 1.5},
        metadata={"purpose": "unit-test"},
        num_shots=23,
    )

    task_definition = task.create_task_definition()

    assert task_definition.program_language == "flair.v1"
    assert [program.content for program in task_definition.programs] == [
        "serialized:main"
    ]
    assert len(task_definition.subtasks) == 1
    subtask = task_definition.subtasks[0]
    assert subtask.program_index == 0
    assert subtask.num_shots == 23
    assert subtask.arguments == {"theta": 1.5}
    assert subtask.subtask_metadata is not None
    assert subtask.subtask_metadata.user_metadata == json.dumps(
        {"purpose": "unit-test"}
    )


def test_single_kernel_task_omits_arguments_and_metadata_when_unset():
    task = SerializableSingleKernelTask(
        context_name="ctx",
        program_language="squin",
        kernel=FakeKernel("main"),
        num_shots=1,
    )

    assert task.get_arguments() is None
    assert task.get_metadata() is None

    subtask = task.create_task_definition().subtasks[0]
    assert subtask.arguments is None
    assert subtask.subtask_metadata is None


def test_parameter_scan_reuses_one_program_for_all_argument_sets():
    task = SerializableParameterScanTask(
        context_name="ctx",
        program_language="squin",
        kernel=FakeKernel("scan"),
        arguments=[{"x": 1.0}, {"x": 2.0}, {"x": 3.0}],
        metadata=[{"i": 0}, {"i": 1}, {"i": 2}],
        num_shots=7,
    )

    task_definition = task.create_task_definition()

    assert [program.content for program in task_definition.programs] == [
        "serialized:scan"
    ]
    assert [subtask.program_index for subtask in task_definition.subtasks] == [0, 0, 0]
    assert [subtask.num_shots for subtask in task_definition.subtasks] == [7, 7, 7]
    assert [subtask.arguments for subtask in task_definition.subtasks] == [
        {"x": 1.0},
        {"x": 2.0},
        {"x": 3.0},
    ]
    assert [
        subtask.subtask_metadata.user_metadata
        for subtask in task_definition.subtasks
        if subtask.subtask_metadata is not None
    ] == [json.dumps({"i": 0}), json.dumps({"i": 1}), json.dumps({"i": 2})]


def test_kernel_batch_task_maps_each_kernel_to_its_own_program():
    task = SerializableKernelBatchTask(
        context_name="ctx",
        program_language="squin",
        kernels=[FakeKernel("first"), FakeKernel("second")],
        arguments=[{"x": 1.0}, {"x": 2.0}],
        num_shots=[3, 5],
    )

    task_definition = task.create_task_definition()

    assert [program.content for program in task_definition.programs] == [
        "serialized:first",
        "serialized:second",
    ]
    assert [subtask.program_index for subtask in task_definition.subtasks] == [0, 1]
    assert [subtask.num_shots for subtask in task_definition.subtasks] == [3, 5]
    assert [subtask.arguments for subtask in task_definition.subtasks] == [
        {"x": 1.0},
        {"x": 2.0},
    ]


def test_validate_arguments_rejects_metadata_length_mismatch():
    task = SerializableParameterScanTask(
        context_name="ctx",
        program_language="squin",
        kernel=FakeKernel("scan"),
        arguments=[{"x": 1.0}, {"x": 2.0}],
        metadata=[{"only": "one"}],
        num_shots=7,
    )

    with pytest.raises(ValueError, match="1 sets of metadata for 2 subtasks"):
        task.validate_arguments()


def test_validate_arguments_rejects_shot_count_length_mismatch():
    task = SerializableKernelBatchTask(
        context_name="ctx",
        program_language="squin",
        kernels=[FakeKernel("first"), FakeKernel("second")],
        num_shots=[3],
    )

    with pytest.raises(ValueError, match="1 shot counts for 2 subtasks"):
        task.validate_arguments()


def test_run_async_dry_run_prints_summary_and_does_not_submit(monkeypatch, capsys):
    task = SerializableSingleKernelTask(
        context_name="ctx",
        program_language="squin",
        kernel=FakeKernel("main"),
        num_shots=1,
    )

    def fail_if_submitted(**kwargs):
        raise AssertionError("dry runs should not submit")

    monkeypatch.setattr(task, "submit_task_definition", fail_if_submitted)

    assert task.run_async(dry_run=True, storage=DictStorage()) is None

    output = capsys.readouterr().out
    assert "DRY RUN -- NO PROGRAM WAS ACTUALLY SUBMITTED FOR EXECUTION" in output
    assert "main(" in output


def test_run_async_submits_created_task_definition(monkeypatch):
    task = SerializableSingleKernelTask(
        context_name="ctx",
        program_language="squin",
        kernel=FakeKernel("main"),
        num_shots=1,
    )
    storage = DictStorage()
    fetch_options = ApiFetchOptions(shots_per_fetch=5)
    submitted = {}
    sentinel = object()

    def submit_task_definition(**kwargs):
        submitted.update(kwargs)
        return sentinel

    monkeypatch.setattr(task, "submit_task_definition", submit_task_definition)

    assert (
        task.run_async(
            dry_run=False,
            storage=storage,
            fetch_options=fetch_options,
        )
        is sentinel
    )
    assert submitted["storage"] is storage
    assert submitted["fetch_options"] is fetch_options
    assert submitted["task_definition"].programs[0].content == "serialized:main"


def test_submit_task_definition_stores_definition_and_returns_future(monkeypatch):
    task = SerializableSingleKernelTask(
        context_name="ctx",
        program_language="squin",
        kernel=FakeKernel("main"),
        num_shots=1,
        future_cls=RecordingFuture,
    )
    storage = DictStorage()
    fetch_options = ApiFetchOptions(subtasks_per_fetch=2)
    task_definition = task.create_task_definition()
    calls = {"authenticated": False}
    created_bodies = []

    class FakeTasksClient:
        def __init__(self, app_context):
            self.app_context = app_context

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def create(self, body):
            created_bodies.append(body)
            return SimpleNamespace(id="task-created", created_date=CREATION_TIME)

    monkeypatch.setattr(task, "authenticate", lambda: calls.update(authenticated=True))
    monkeypatch.setattr(task_mod, "TasksClient", FakeTasksClient)

    future = task.submit_task_definition(
        task_definition=task_definition,
        storage=storage,
        fetch_options=fetch_options,
    )

    assert calls["authenticated"] is True
    assert created_bodies[0].root == task_definition
    assert storage.get_task_definition("task-created") == task_definition
    assert storage.get_task_creation_time("task-created") == CREATION_TIME
    assert future.task_id == "task-created"
    assert future.storage is storage
    assert future.fetch_options is fetch_options
    assert future.context_name == "ctx"


def test_run_async_defaults_storage_to_fresh_dict_storage(monkeypatch):
    task = SingleKernelTask(
        context_name="ctx",
        program_language="squin",
        kernel=main,
        num_shots=1,
    )
    submitted = {}
    sentinel = object()

    def submit_task_definition(**kwargs):
        submitted.update(kwargs)
        return sentinel

    monkeypatch.setattr(task, "submit_task_definition", submit_task_definition)

    assert task.run_async(dry_run=False) is sentinel
    assert submitted["storage"] is None


def test_submit_task_definition_defaults_to_fresh_dict_storage(monkeypatch):
    task = SingleKernelTask(
        context_name="ctx",
        program_language="squin",
        kernel=main,
        num_shots=1,
        future_cls=RecordingFuture,  # type: ignore
    )
    task_definition = task.create_task_definition()

    class FakeTasksClient:
        def __init__(self, app_context):
            self.app_context = app_context

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def create(self, body):
            return SimpleNamespace(id="task-created", created_date=CREATION_TIME)

    monkeypatch.setattr(task, "authenticate", lambda: None)
    monkeypatch.setattr(task_mod, "TasksClient", FakeTasksClient)

    future = task.submit_task_definition(task_definition=task_definition)

    assert isinstance(future.storage, DictStorage)
    assert future.storage.get_task_definition("task-created") == task_definition
    assert future.storage.get_task_creation_time("task-created") == CREATION_TIME


def test_submit_task_definition_rejects_missing_created_task_id(monkeypatch):
    task = SerializableSingleKernelTask(
        context_name="ctx",
        program_language="squin",
        kernel=FakeKernel("main"),
        num_shots=1,
    )

    class FakeTasksClient:
        def __init__(self, app_context):
            self.app_context = app_context

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def create(self, body):
            return SimpleNamespace(id=None, created_date=CREATION_TIME)

    monkeypatch.setattr(task, "authenticate", lambda: None)
    monkeypatch.setattr(task_mod, "TasksClient", FakeTasksClient)

    with pytest.raises(ValueError, match="Couldn't get id of created task"):
        task.submit_task_definition(
            task_definition=task.create_task_definition(),
            storage=DictStorage(),
        )
