from dataclasses import dataclass, field
from typing import Any, Generic, cast

from kirin import ir

from .future import Future, FutureType
from .mixins import AuthMixin
from .task import KernelBatchTask, ParameterScanTask, SingleKernelTask


@dataclass(kw_only=True)
class Device(Generic[FutureType], AuthMixin):
    """Factory for tasks.

    The device does not submit work directly. Instead, it builds task objects
    that can be dry-run or submitted asynchronously.
    """

    # NOTE: for python 3.10, we need the future_cls to be Future, not Future[Result]
    # in order to keep the typing correct, we use cast and set a default on the
    # FutureType TypeVar
    future_cls: type[FutureType] = cast(type[FutureType], Future)

    # NOTE: we also need to cast these, otherwise the return type annotations
    # give type errors in the task creating methods below
    single_kernel_task_cls: type[SingleKernelTask[FutureType]] = field(
        default=cast(type[SingleKernelTask[FutureType]], SingleKernelTask),
        init=False,
    )
    kernel_batch_task_cls: type[KernelBatchTask[FutureType]] = field(
        default=cast(type[KernelBatchTask[FutureType]], KernelBatchTask),
        init=False,
    )
    parameter_scan_task_cls: type[ParameterScanTask[FutureType]] = field(
        default=cast(type[ParameterScanTask[FutureType]], ParameterScanTask),
        init=False,
    )

    def task(
        self,
        kernel: ir.Method,
        num_shots: int = 1,
        arguments: dict | None = None,
        metadata: dict | None = None,
        program_language: str = "squin",
    ) -> SingleKernelTask[FutureType]:
        """Create a task for one kernel.

        Args:
            kernel (ir.Method): The kernel to execute.
            num_shots (int): Number of shots to run for the kernel. Defaults to 1.
            arguments (dict | None): Argument dictionary for the kernel.
                Defaults to None.
            metadata (dict | None): Metadata for the single subtask. When
                provided, it is wrapped in a one-element list to match the
                task API. Defaults to None.
            program_language (str): Program language to store in the task
                definition. Defaults to "squin".

        Returns:
            SingleKernelTask[FutureType]: A task object ready for dry-run or submission.
        """

        return self.single_kernel_task_cls(
            context_name=self.context_name,
            kernel=kernel,
            num_shots=num_shots,
            arguments=arguments,
            metadata=metadata,
            program_language=program_language,
            future_cls=self.future_cls,
        )

    def batch_task(
        self,
        kernels: list[ir.Method],
        arguments: list[dict] | None = None,
        metadata: list[dict[str, Any]] | None = None,
        num_shots: list[int] | int = 1,
        program_language: str = "squin",
    ) -> KernelBatchTask[FutureType]:
        """Create a task containing one subtask per kernel.

        Args:
            kernels (list[ir.Method]): Kernels to execute.
            arguments (list[dict] | None): Per-kernel argument dictionaries.
                Defaults to None.
            metadata (list[dict[str, Any]] | None): Per-kernel metadata
                dictionaries. Defaults to None.
            num_shots (list[int] | int): Shot count for each kernel, or one
                value to broadcast to every kernel. Defaults to 1.
            program_language (str): Program language to store in the task
                definition. Defaults to "squin".

        Returns:
            KernelBatchTask[FutureType]: A batch task object ready for dry-run or
                submission.
        """

        return self.kernel_batch_task_cls(
            context_name=self.context_name,
            kernels=kernels,
            arguments=arguments,
            num_shots=num_shots,
            metadata=metadata,
            program_language=program_language,
            future_cls=self.future_cls,
        )

    def parameter_scan(
        self,
        kernel: ir.Method,
        arguments: list[dict],
        metadata: list[dict] | None = None,
        num_shots: list[int] | int = 1,
        program_language: str = "squin",
    ) -> ParameterScanTask[FutureType]:
        """Create a parameter-scan task for one kernel.

        Args:
            kernel (ir.Method): Kernel to execute for each parameter set.
            arguments (list[dict]): Argument dictionaries, one per subtask.
            metadata (list[dict] | None): Metadata dictionaries, one per
                subtask. Defaults to None.
            num_shots (list[int] | int): Shot count for each parameter set, or
                one value to broadcast to every subtask. Defaults to 1.
            program_language (str): Program language to store in the task
                definition. Defaults to "squin".

        Returns:
            ParameterScanTask[FutureType]: A parameter-scan task object ready for
                dry-run or submission.
        """

        return self.parameter_scan_task_cls(
            context_name=self.context_name,
            kernel=kernel,
            num_shots=num_shots,
            arguments=arguments,
            metadata=metadata,
            program_language=program_language,
            future_cls=self.future_cls,
        )
