import base64
import importlib
import json
from typing import cast

import pytest
from kirin.prelude import basic_no_opt
from kirin.serialization import JSONSerializer
from kirin.serialization.bsonserializer import CompressedBSONSerializer
from qlam_core.errors import APIError
from qlam_core.plugins.tasks.api.tasks_models import TaskStatus

from bloqade.core.device.future import ApiFetchOptions
from bloqade.core.device.local_storage import DictStorage
from bloqade.core.device.task import (
    KernelBatchTask,
    KernelSerializer,
    ParameterScanTask,
    SingleKernelTask,
)

from .fixtures import local, remote

task_mod = importlib.import_module("bloqade.core.device.task")
mixins_mod = importlib.import_module("bloqade.core.device.mixins")

CREATION_TIME = local.CREATION_TIME


@basic_no_opt
def main():
    return


@basic_no_opt
def scan():
    return


@basic_no_opt
def first():
    return


@basic_no_opt
def second():
    return


class RecordingFuture:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__dict__.update(kwargs)


def test_single_kernel_task_creates_task_definition_with_arguments_and_metadata():
    task = SingleKernelTask(
        context_name="ctx",
        program_language="flair",
        language_version="1",
        kernel=main,
        arguments={"theta": 1.5},
        metadata={"purpose": "unit-test"},
    )

    task_definition = task.create_task_definition(num_shots=23)

    assert task_definition.program_language == "flair.v1"
    assert len(task_definition.subtasks) == 1
    subtask = task_definition.subtasks[0]
    assert subtask.program_index == 0
    assert subtask.num_shots == 23
    assert subtask.arguments == {"theta": 1.5}
    assert subtask.subtask_metadata is not None
    assert subtask.subtask_metadata.user_metadata == json.dumps(
        {"purpose": "unit-test"}
    )


def test_create_task_definition_requires_shot_count():
    task = SingleKernelTask(
        context_name="ctx",
        program_language="squin",
        kernel=main,
    )

    with pytest.raises(TypeError, match="num_shots"):
        task.create_task_definition()  # type: ignore[call-arg]


def test_single_kernel_task_omits_arguments_and_metadata_when_unset():
    task = SingleKernelTask(
        context_name="ctx",
        program_language="squin",
        kernel=main,
    )

    assert task.get_arguments() is None
    assert task.get_metadata() is None

    subtask = task.create_task_definition(num_shots=1).subtasks[0]
    assert subtask.arguments is None
    assert subtask.subtask_metadata is None


def test_single_kernel_task_serializes_kernel_with_json_by_default():
    task = SingleKernelTask(
        context_name="ctx",
        program_language="squin",
        kernel=main,
    )

    content = task.create_task_definition(num_shots=1).programs[0].content
    encoded_module = main.dialects.encode(main, version=task.program_language_version)

    assert content == JSONSerializer().encode(encoded_module)
    assert json.loads(content)


def test_single_kernel_task_base64_encodes_binary_serializer_output():
    task = SingleKernelTask(
        context_name="ctx",
        program_language="flair",
        kernel=main,
        kernel_serializer=CompressedBSONSerializer(),
    )

    content = task.create_task_definition(num_shots=1).programs[0].content
    decoded_module = CompressedBSONSerializer().decode(
        base64.b64decode(content, validate=True)
    )
    restored_main = main.dialects.decode(decoded_module)

    assert restored_main.code.is_structurally_equal(main.code)


def test_single_kernel_task_rejects_non_string_serializer_output():
    class DictSerializer:
        def encode(self, _encoded_module):
            return {"content": "not valid"}

    task = SingleKernelTask(
        context_name="ctx",
        program_language="squin",
        kernel=main,
        kernel_serializer=cast(KernelSerializer, DictSerializer()),
    )

    with pytest.raises(TypeError, match="kernel_serializer.encode must return"):
        task.serialize_kernel(main)


def test_parameter_scan_reuses_one_program_for_all_argument_sets():
    task = ParameterScanTask(
        context_name="ctx",
        program_language="squin",
        kernel=scan,
        arguments=[{"x": 1.0}, {"x": 2.0}, {"x": 3.0}],
        metadata=[{"i": 0}, {"i": 1}, {"i": 2}],
    )

    task_definition = task.create_task_definition(num_shots=7)

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
    task = KernelBatchTask(
        context_name="ctx",
        program_language="squin",
        kernels=[first, second],
        arguments=[{"x": 1.0}, {"x": 2.0}],
    )

    task_definition = task.create_task_definition(num_shots=[3, 5])

    assert [subtask.program_index for subtask in task_definition.subtasks] == [0, 1]
    assert [subtask.num_shots for subtask in task_definition.subtasks] == [3, 5]
    assert [subtask.arguments for subtask in task_definition.subtasks] == [
        {"x": 1.0},
        {"x": 2.0},
    ]


def test_validate_arguments_rejects_metadata_length_mismatch():
    task = ParameterScanTask(
        context_name="ctx",
        program_language="squin",
        kernel=scan,
        arguments=[{"x": 1.0}, {"x": 2.0}],
        metadata=[{"only": "one"}],
    )

    with pytest.raises(ValueError, match="1 sets of metadata for 2 subtasks"):
        task.validate_arguments()


def test_create_task_definition_rejects_shot_count_length_mismatch():
    task = KernelBatchTask(
        context_name="ctx",
        program_language="squin",
        kernels=[first, second],
    )

    with pytest.raises(ValueError, match="1 shot counts for 2 subtasks"):
        task.create_task_definition(num_shots=[3])


def test_run_async_dry_run_prints_summary_and_does_not_submit(monkeypatch, capsys):
    task = SingleKernelTask(
        context_name="ctx",
        program_language="squin",
        kernel=main,
    )

    def fail_if_submitted(**kwargs):
        raise AssertionError("dry runs should not submit")

    monkeypatch.setattr(task, "submit_task_definition", fail_if_submitted)

    assert task.run_async(dry_run=True, storage=DictStorage()) is None

    output = capsys.readouterr().out
    assert "DRY RUN -- NO PROGRAM WAS ACTUALLY SUBMITTED FOR EXECUTION" in output
    assert "main(" in output


def test_run_async_defaults_to_dry_run_with_one_shot(monkeypatch, capsys):
    task = SingleKernelTask(
        context_name="ctx",
        program_language="squin",
        kernel=main,
    )

    monkeypatch.setattr(
        task,
        "submit_task_definition",
        lambda **kwargs: pytest.fail("dry runs should not submit"),
    )

    assert task.run_async() is None
    assert "1 shots" in capsys.readouterr().out


def test_run_async_defaults_to_one_shot_on_submission(monkeypatch):
    task = SingleKernelTask(
        context_name="ctx",
        program_language="squin",
        kernel=main,
    )
    submitted = {}
    sentinel = object()

    def submit_task_definition(**kwargs):
        submitted.update(kwargs)
        return sentinel

    monkeypatch.setattr(task, "submit_task_definition", submit_task_definition)

    assert task.run_async(dry_run=False) is sentinel
    assert submitted["task_definition"].subtasks[0].num_shots == 1


def test_run_async_submits_unconfigured_task_with_shots(monkeypatch):
    task = SingleKernelTask(
        context_name="ctx",
        program_language="squin",
        kernel=main,
    )
    submitted = {}
    sentinel = object()

    def submit_task_definition(**kwargs):
        submitted.update(kwargs)
        return sentinel

    monkeypatch.setattr(task, "submit_task_definition", submit_task_definition)

    assert task.run_async(dry_run=False, shots=17) is sentinel
    assert submitted["task_definition"].subtasks[0].num_shots == 17


def test_run_async_dry_run_summary_uses_shot_override(monkeypatch, capsys):
    task = SingleKernelTask(
        context_name="ctx",
        program_language="squin",
        kernel=main,
        arguments={"theta": 1.5},
    )

    monkeypatch.setattr(
        task,
        "submit_task_definition",
        lambda **kwargs: pytest.fail("dry runs should not submit"),
    )

    assert task.run_async(dry_run=True, shots=17) is None

    output = capsys.readouterr().out
    assert "theta=1.5" in output
    assert "17 shots" in output
    assert "1 shots" in task.summary()


def test_parameter_scan_dry_run_uses_default_summary_for_task_definition(
    monkeypatch, capsys
):
    task = ParameterScanTask(
        context_name="ctx",
        program_language="squin",
        kernel=scan,
        arguments=[{"x": 1.0}, {"x": 2.0}],
    )

    monkeypatch.setattr(
        task,
        "submit_task_definition",
        lambda **kwargs: pytest.fail("dry runs should not submit"),
    )

    assert task.run_async(dry_run=True, shots=[17, 19]) is None

    assert "parameter sets" in capsys.readouterr().out


def test_batch_task_dry_run_summary_uses_per_subtask_shot_override(monkeypatch, capsys):
    task = KernelBatchTask(
        context_name="ctx",
        program_language="squin",
        kernels=[first, second],
        arguments=[{"theta": 1.5}, {"phi": 0.5}],
    )

    monkeypatch.setattr(
        task,
        "submit_task_definition",
        lambda **kwargs: pytest.fail("dry runs should not submit"),
    )

    assert task.run_async(dry_run=True, shots=[17, 19]) is None

    output = capsys.readouterr().out
    assert "17 shots" in output
    assert "19 shots" in output
    assert "1 shots" in task.summary()


def test_run_async_submits_created_task_definition(monkeypatch):
    task = SingleKernelTask(
        context_name="ctx",
        program_language="squin",
        kernel=main,
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
            shots=1,
            storage=storage,
            fetch_options=fetch_options,
        )
        is sentinel
    )
    assert submitted["storage"] is storage
    assert submitted["fetch_options"] is fetch_options


def test_run_async_default_shots_broadcast_to_every_batch_subtask(monkeypatch):
    task = KernelBatchTask(
        context_name="ctx",
        program_language="squin",
        kernels=[first, second],
    )
    submitted = {}
    sentinel = object()

    def submit_task_definition(**kwargs):
        submitted.update(kwargs)
        return sentinel

    monkeypatch.setattr(task, "submit_task_definition", submit_task_definition)

    assert task.run_async(dry_run=False) is sentinel
    assert [subtask.num_shots for subtask in submitted["task_definition"].subtasks] == [
        1,
        1,
    ]


def test_run_async_default_shots_broadcast_to_every_parameter_scan_subtask(
    monkeypatch,
):
    task = ParameterScanTask(
        context_name="ctx",
        program_language="squin",
        kernel=scan,
        arguments=[{"x": 1.0}, {"x": 2.0}],
    )
    submitted = {}
    sentinel = object()

    def submit_task_definition(**kwargs):
        submitted.update(kwargs)
        return sentinel

    monkeypatch.setattr(task, "submit_task_definition", submit_task_definition)

    assert task.run_async(dry_run=False) is sentinel
    assert [subtask.num_shots for subtask in submitted["task_definition"].subtasks] == [
        1,
        1,
    ]


def test_run_async_supports_create_task_definition_override(monkeypatch):
    class CustomSingleKernelTask(SingleKernelTask):
        def create_task_definition(self, *, num_shots: int | list[int]):
            return super().create_task_definition(num_shots=num_shots)

    task = CustomSingleKernelTask(
        context_name="ctx",
        program_language="squin",
        kernel=main,
    )
    submitted = {}
    sentinel = object()

    def submit_task_definition(**kwargs):
        submitted.update(kwargs)
        return sentinel

    monkeypatch.setattr(task, "submit_task_definition", submit_task_definition)

    assert task.run_async(dry_run=False, shots=17) is sentinel
    assert submitted["task_definition"].subtasks[0].num_shots == 17


def test_run_async_accepts_per_subtask_shot_counts(monkeypatch):
    task = KernelBatchTask(
        context_name="ctx",
        program_language="squin",
        kernels=[first, second],
    )
    submitted = {}
    sentinel = object()

    def submit_task_definition(**kwargs):
        submitted.update(kwargs)
        return sentinel

    monkeypatch.setattr(task, "submit_task_definition", submit_task_definition)

    assert task.run_async(dry_run=False, shots=[17, 19]) is sentinel
    assert [subtask.num_shots for subtask in submitted["task_definition"].subtasks] == [
        17,
        19,
    ]


def test_run_async_rejects_shot_override_length_mismatch(monkeypatch):
    task = KernelBatchTask(
        context_name="ctx",
        program_language="squin",
        kernels=[first, second],
    )

    monkeypatch.setattr(
        task,
        "submit_task_definition",
        lambda **kwargs: pytest.fail("invalid overrides should not submit"),
    )

    with pytest.raises(ValueError, match="1 shot counts for 2 subtasks"):
        task.run_async(dry_run=False, shots=[17])


def test_submit_task_definition_stores_definition_and_returns_future(monkeypatch):
    task = SingleKernelTask(
        context_name="ctx",
        program_language="squin",
        kernel=main,
        future_cls=RecordingFuture,  # type: ignore
    )
    storage = DictStorage()
    fetch_options = ApiFetchOptions(subtasks_per_fetch=2)
    task_definition = task.create_task_definition(num_shots=1)
    calls = {"authenticated": False}

    created_task = remote.make_task(
        id="task-created",
        task_status=TaskStatus.CREATED,
        created_date=CREATION_TIME,
    )
    client = remote.FakeTasksClient(create_return=created_task)

    monkeypatch.setattr(task, "authenticate", lambda: calls.update(authenticated=True))
    monkeypatch.setattr(task_mod, "TasksClient", lambda app_context: client)

    future = task.submit_task_definition(
        task_definition=task_definition,
        storage=storage,
        fetch_options=fetch_options,
    )

    assert calls["authenticated"] is True
    assert len(client.calls) == 1
    name, kwargs = client.calls[0]
    assert name == "create"
    assert kwargs["body"].root == task_definition
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
    )
    submitted = {}
    sentinel = object()

    def submit_task_definition(**kwargs):
        submitted.update(kwargs)
        return sentinel

    monkeypatch.setattr(task, "submit_task_definition", submit_task_definition)

    assert task.run_async(dry_run=False, shots=1) is sentinel
    assert submitted["storage"] is None


def test_submit_task_definition_defaults_to_fresh_dict_storage(monkeypatch):
    task = SingleKernelTask(
        context_name="ctx",
        program_language="squin",
        kernel=main,
        future_cls=RecordingFuture,  # type: ignore
    )
    task_definition = task.create_task_definition(num_shots=1)

    created_task = remote.make_task(
        id="task-created",
        task_status=TaskStatus.CREATED,
        created_date=CREATION_TIME,
    )
    client = remote.FakeTasksClient(create_return=created_task)

    monkeypatch.setattr(task, "authenticate", lambda: None)
    monkeypatch.setattr(task_mod, "TasksClient", lambda app_context: client)

    future = task.submit_task_definition(task_definition=task_definition)

    assert isinstance(future.storage, DictStorage)
    assert future.storage.get_task_definition("task-created") == task_definition
    assert future.storage.get_task_creation_time("task-created") == CREATION_TIME


def test_submit_task_definition_rejects_missing_created_task_id(monkeypatch):
    task = SingleKernelTask(
        context_name="ctx",
        program_language="squin",
        kernel=main,
    )

    created_task = remote.make_task(
        id=None,
        task_status=TaskStatus.CREATED,
        created_date=CREATION_TIME,
    )
    client = remote.FakeTasksClient(create_return=created_task)

    monkeypatch.setattr(task, "authenticate", lambda: None)
    monkeypatch.setattr(task_mod, "TasksClient", lambda app_context: client)

    with pytest.raises(ValueError, match="Couldn't get id of created task"):
        task.submit_task_definition(
            task_definition=task.create_task_definition(num_shots=1),
            storage=DictStorage(),
        )


def test_submit_task_definition_retries_on_403_after_refresh(monkeypatch):
    task = SingleKernelTask(
        context_name="ctx",
        program_language="squin",
        kernel=main,
        future_cls=RecordingFuture,  # type: ignore
    )
    task_definition = task.create_task_definition(num_shots=1)
    created_task = remote.make_task(
        id="task-x",
        task_status=TaskStatus.CREATED,
        created_date=CREATION_TIME,
    )

    invocations = []

    def create_return(body):
        invocations.append(body)
        if len(invocations) == 1:
            raise APIError(message="permission denied", status_code=403)
        return created_task

    client = remote.FakeTasksClient(create_return=create_return)
    auth_client = remote.FakeAuthClient(refresh_result={"qlam": True})

    monkeypatch.setattr(task, "authenticate", lambda: None)
    monkeypatch.setattr(task_mod, "TasksClient", lambda app_context: client)
    monkeypatch.setattr(mixins_mod, "AuthClient", lambda app_context: auth_client)

    future = task.submit_task_definition(
        task_definition=task_definition,
        storage=DictStorage(),
    )

    assert future.task_id == "task-x"
    assert len(invocations) == 2
    assert [name for name, _ in client.calls] == ["create", "create"]
    assert [name for name, _ in auth_client.calls] == ["refresh_credentials"]
