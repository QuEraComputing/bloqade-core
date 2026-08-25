from kirin.prelude import basic_no_opt

from bloqade.core.device.device import Device
from bloqade.core.device.future import Future
from bloqade.core.device.task import (
    KernelBatchTask,
    ParameterScanTask,
    SingleKernelTask,
)


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
        arguments={"theta": 1.5},
        metadata={"label": "calibration"},
        program_language="flair.v1",
    )

    assert isinstance(task, SingleKernelTask)
    assert task.context_name == "ctx"
    assert task.future_cls is CustomFuture
    assert task.kernel is kernel
    assert not hasattr(task, "num_shots")
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
        program_language="squin",
    )

    assert isinstance(task, KernelBatchTask)
    assert task.context_name == "ctx"
    assert task.future_cls is CustomFuture
    assert task.kernels == kernels
    assert task.arguments == [{"alpha": 0.25}, {"alpha": 0.5}]
    assert task.metadata == [{"name": "a"}, {"name": "b"}]
    assert not hasattr(task, "num_shots")
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
        program_language="flair.v2",
    )

    assert isinstance(task, ParameterScanTask)
    assert task.context_name == "ctx"
    assert task.future_cls is CustomFuture
    assert task.kernel is kernel
    assert task.arguments == [{"detuning": -1.0}, {"detuning": 1.0}]
    assert task.metadata == [{"point": "left"}, {"point": "right"}]
    assert not hasattr(task, "num_shots")
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


def test_device_group_is_passed_to_each_task_shape():
    @basic_no_opt
    def first():
        return

    @basic_no_opt
    def second():
        return

    group = "qec-experiments"
    device = Device(context_name="ctx")

    assert device.task(first, group=group).group == group
    assert device.batch_task([first, second], group=group).group == group
    assert device.parameter_scan(first, arguments=[{}], group=group).group == group
