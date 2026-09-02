import importlib
from uuid import UUID

import pytest
from kirin.prelude import basic_no_opt
from qlam_core.plugins.tasks.api.tasks_models import TaskStatus

from bloqade.core.device.device import Device
from bloqade.core.device.future import ApiFetchOptions
from bloqade.core.device.local_storage import DictStorage
from bloqade.core.device.task_builder import TaskBuilder

from .fixtures import local, remote

device_mod = importlib.import_module("bloqade.core.device.device")


@basic_no_opt
def builder_kernel(value: float):
    return


class RecordingSerializer:
    def encode(self, encoded_module):
        return f"serialized:{encoded_module.version}"


class RecordingFuture:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def make_builder() -> TaskBuilder:
    builder = TaskBuilder()
    builder.add_subtask(builder_kernel, 11, {"kind": "test"}, value=1.25)
    return builder


def test_device_dry_run_prints_without_submission_and_keeps_builder_editable(
    monkeypatch, capsys
):
    device = Device(
        context_name="ctx",
        program_language="squin",
        language_version="3",
        kernel_serializer=RecordingSerializer(),
    )
    builder = make_builder()
    before_dry_run = builder.copy()

    def fail_if_submitted(**kwargs):
        raise AssertionError("a dry run must not submit")

    monkeypatch.setattr(device, "submit_task_definition", fail_if_submitted)

    assert device.run_async(builder, dry_run=True) is None
    output = capsys.readouterr().out
    assert (
        "builder_kernel, Program 0, Args {'value': 1.25}, "
        "Metadata {'kind': 'test'} -> 11 shots"
    ) in output
    assert builder == before_dry_run
    assert builder.add_subtask(builder_kernel, 3, value=2.5) == 1


def test_device_run_async_passes_finalized_definition_to_submitter(monkeypatch):
    device = Device(
        context_name="ctx",
        program_language="squin",
        language_version="3",
        kernel_serializer=RecordingSerializer(),
    )
    storage = DictStorage()
    fetch_options = ApiFetchOptions(shots_per_fetch=4)
    submitted = {}
    sentinel = object()

    def record_submission(**kwargs):
        submitted.update(kwargs)
        return sentinel

    monkeypatch.setattr(device, "submit_task_definition", record_submission)

    result = device.run_async(
        make_builder(),
        dry_run=False,
        group="research",
        storage=storage,
        fetch_options=fetch_options,
    )

    assert result is sentinel
    assert submitted["task_definition"].program_language == "squin.v3"
    assert submitted["task_definition"].subtasks[0].num_shots == 11
    assert submitted["group"] == "research"
    assert submitted["storage"] is storage
    assert submitted["fetch_options"] is fetch_options


def test_device_submit_resolves_group_stores_definition_and_returns_future(
    monkeypatch,
):
    group_id = UUID("11111111-1111-1111-1111-111111111111")
    created_task = remote.make_task(
        id="builder-task",
        task_status=TaskStatus.CREATED,
        created_date=local.CREATION_TIME,
    )
    tasks_client = remote.FakeTasksClient(create_return=created_task)
    groups_client = remote.FakeGroupsClient(resolve_id_return=group_id)
    device = Device(
        context_name="ctx",
        program_language="squin",
        kernel_serializer=RecordingSerializer(),
        future_cls=RecordingFuture,  # type: ignore[arg-type]
    )
    fetch_options = ApiFetchOptions(subtasks_per_fetch=2)

    monkeypatch.setattr(device, "authenticate", lambda: None)
    monkeypatch.setattr(device_mod, "TasksClient", lambda app_context: tasks_client)
    monkeypatch.setattr(device_mod, "GroupsClient", lambda app_context: groups_client)

    future = device.run_async(
        make_builder(),
        dry_run=False,
        group="research",
        fetch_options=fetch_options,
    )

    assert future.task_id == "builder-task"
    assert future.context_name == "ctx"
    assert future.fetch_options is fetch_options
    assert isinstance(future.storage, DictStorage)
    stored = future.storage.get_task_definition("builder-task")
    assert stored.group_id == group_id
    assert future.storage.get_task_creation_time("builder-task") == (
        local.CREATION_TIME
    )
    assert groups_client.calls == [("resolve_id", {"group": "research"})]
    assert tasks_client.calls[0][1]["body"].root.group_id == group_id


def test_device_configured_group_uses_task_plugin_before_defaults(
    write_qsh_config,
):
    write_qsh_config(
        defaults_group="default-group",
        tasks_plugin_group="tasks-group",
    )
    device = Device(context_name="ctx")

    assert device._configured_group() == "tasks-group"
    assert device._configured_group("explicit-group") == "explicit-group"


def test_device_configured_group_falls_back_to_context_default(write_qsh_config):
    write_qsh_config(defaults_group="default-group")

    assert Device(context_name="ctx")._configured_group() == "default-group"


def test_device_submit_preserves_definition_group_id(monkeypatch):
    group_id = UUID("22222222-2222-2222-2222-222222222222")
    created_task = remote.make_task(id="existing-group-task")
    tasks_client = remote.FakeTasksClient(create_return=created_task)
    device = Device(
        context_name="ctx",
        future_cls=RecordingFuture,  # type: ignore[arg-type]
    )
    definition = remote.make_task_definition(group_id=group_id)

    monkeypatch.setattr(device, "authenticate", lambda: None)
    monkeypatch.setattr(device_mod, "TasksClient", lambda app_context: tasks_client)
    monkeypatch.setattr(
        device,
        "_resolve_group_id",
        lambda group: pytest.fail("an existing group ID must not be replaced"),
    )

    future = device.submit_task_definition(
        task_definition=definition,
        group="ignored-group",
    )

    assert future.storage.get_task_definition("existing-group-task").group_id == (
        group_id
    )


def test_device_submit_rejects_missing_task_id(monkeypatch):
    created_task = remote.make_task(id=None)
    tasks_client = remote.FakeTasksClient(create_return=created_task)
    device = Device(context_name="ctx")

    monkeypatch.setattr(device, "authenticate", lambda: None)
    monkeypatch.setattr(device_mod, "TasksClient", lambda app_context: tasks_client)

    with pytest.raises(ValueError, match="Couldn't get id of created task"):
        device.submit_task_definition(
            task_definition=remote.make_task_definition(
                group_id=UUID("33333333-3333-3333-3333-333333333333")
            )
        )
