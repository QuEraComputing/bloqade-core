from uuid import UUID

from kirin.prelude import basic_no_opt

import bloqade.core.device.device as device_mod
from bloqade.core.device.device import Device
from bloqade.core.device.future import Future
from bloqade.core.device.task import (
    KernelBatchTask,
    ParameterScanTask,
    SingleKernelTask,
)

from .fixtures import remote


class CustomFuture(Future):
    pass


class RecordingSerializer:
    def encode(self, encoded_module):
        return f"serialized:{encoded_module.version}"


def test_device_task_builds_single_kernel_task_with_device_defaults():
    @basic_no_opt
    def main():
        return

    kernel = main
    device = Device(context_name="ctx", future_cls=CustomFuture)

    task = device.task(
        kernel,
        num_shots=17,
        arguments={"theta": 1.5},
        metadata={"label": "calibration"},
        program_language="flair.v1",
    )

    assert isinstance(task, SingleKernelTask)
    assert task.context_name == "ctx"
    assert task.future_cls is CustomFuture
    assert task.kernel is kernel
    assert task.num_shots == 17
    assert task.arguments == {"theta": 1.5}
    assert task.metadata == {"label": "calibration"}
    assert task.program_language == "flair.v1"
    assert task.kernel_serializer is device.kernel_serializer


def test_device_batch_task_builds_kernel_batch_task():
    @basic_no_opt
    def first():
        return

    @basic_no_opt
    def second():
        return

    kernels = [first, second]
    device = Device(context_name="ctx", future_cls=CustomFuture)

    task = device.batch_task(
        kernels,
        arguments=[{"alpha": 0.25}, {"alpha": 0.5}],
        metadata=[{"name": "a"}, {"name": "b"}],
        num_shots=[3, 5],
        program_language="squin",
    )

    assert isinstance(task, KernelBatchTask)
    assert task.context_name == "ctx"
    assert task.future_cls is CustomFuture
    assert task.kernels == kernels
    assert task.arguments == [{"alpha": 0.25}, {"alpha": 0.5}]
    assert task.metadata == [{"name": "a"}, {"name": "b"}]
    assert task.num_shots == [3, 5]
    assert task.program_language == "squin"
    assert task.kernel_serializer is device.kernel_serializer


def test_device_parameter_scan_builds_parameter_scan_task():
    @basic_no_opt
    def scan():
        return

    kernel = scan
    device = Device(context_name="ctx", future_cls=CustomFuture)

    task = device.parameter_scan(
        kernel,
        arguments=[{"detuning": -1.0}, {"detuning": 1.0}],
        metadata=[{"point": "left"}, {"point": "right"}],
        num_shots=11,
        program_language="flair.v2",
    )

    assert isinstance(task, ParameterScanTask)
    assert task.context_name == "ctx"
    assert task.future_cls is CustomFuture
    assert task.kernel is kernel
    assert task.arguments == [{"detuning": -1.0}, {"detuning": 1.0}]
    assert task.metadata == [{"point": "left"}, {"point": "right"}]
    assert task.num_shots == 11
    assert task.program_language == "flair.v2"
    assert task.kernel_serializer is device.kernel_serializer


def test_device_uses_configured_kernel_serializer_for_all_task_shapes():
    @basic_no_opt
    def first():
        return

    @basic_no_opt
    def second():
        return

    serializer = RecordingSerializer()
    device = Device(context_name="ctx", kernel_serializer=serializer)

    assert device.task(first).kernel_serializer is serializer
    assert device.batch_task([first, second]).kernel_serializer is serializer
    assert device.parameter_scan(first, arguments=[{}]).kernel_serializer is serializer


def test_device_kernel_serializer_override_wins_over_device_default():
    @basic_no_opt
    def first():
        return

    @basic_no_opt
    def second():
        return

    default_serializer = RecordingSerializer()
    override_serializer = RecordingSerializer()
    device = Device(context_name="ctx", kernel_serializer=default_serializer)

    assert (
        device.task(first, kernel_serializer=override_serializer).kernel_serializer
        is override_serializer
    )
    assert (
        device.batch_task(
            [first, second], kernel_serializer=override_serializer
        ).kernel_serializer
        is override_serializer
    )
    assert (
        device.parameter_scan(
            first, arguments=[{}], kernel_serializer=override_serializer
        ).kernel_serializer
        is override_serializer
    )


def test_device_group_id_is_inherited_and_task_override_wins():
    @basic_no_opt
    def first():
        return

    @basic_no_opt
    def second():
        return

    default_group_id = UUID("11111111-1111-1111-1111-111111111111")
    override_group_id = UUID("22222222-2222-2222-2222-222222222222")
    device = Device(context_name="ctx", group_id=default_group_id)

    assert device.task(first).group_id == default_group_id
    assert device.batch_task([first, second]).group_id == default_group_id
    assert device.parameter_scan(first, arguments=[{}]).group_id == default_group_id

    task = device.task(first, group_id=override_group_id)
    assert task.group_id == override_group_id
    assert task.create_task_definition().group_id == override_group_id


def test_device_lists_and_gets_groups(monkeypatch):
    first_group = remote.make_group()
    second_group = remote.make_group(
        id=UUID("11111111-1111-1111-1111-111111111111"),
        name="second-group",
    )
    groups_client = remote.FakeGroupsClient(
        list_all_return=[first_group, second_group], get_return=second_group
    )
    device = Device(context_name="ctx")

    monkeypatch.setattr(device_mod.AuthMixin, "authenticate", lambda self: None)
    monkeypatch.setattr(device_mod, "GroupsClient", lambda app_context: groups_client)

    assert device.list_groups() == [first_group, second_group]
    assert device.get_group(second_group.id) == second_group
    assert groups_client.calls == [
        ("list_all", {}),
        ("get", {"id": second_group.id}),
    ]
