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
