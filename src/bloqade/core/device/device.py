from dataclasses import dataclass, field
from typing import Any, Generic, cast
from uuid import UUID

from kirin import ir
from kirin.serialization import JSONSerializer

from .future import Future, FutureType
from .mixins import AuthMixin
from .task import (
    KernelBatchTask,
    KernelSerializer,
    ParameterScanTask,
    SingleKernelTask,
)


@dataclass(kw_only=True)
class Device(Generic[FutureType], AuthMixin):
    """Factory for tasks.

    The device does not submit work directly. Instead, it builds task objects
    that can be dry-run or submitted asynchronously.

    Attributes:
        future_cls (type[FutureType]): Future class used by tasks created from
            this device. Defaults to `Future`.
        kernel_serializer (KernelSerializer): Default serializer passed to
            created tasks. The serializer is used by
            `TaskABC.serialize_kernel`; it should provide an `encode` method
            for Kirin serialization modules.
            Defaults to `kirin.serialization.JSONSerializer`.
        group_id (UUID | None): Default QLAM group applied to task definitions
            created by this device. A task-specific value takes precedence.
            When neither is set, the `~/.qsh` config group
            (`plugins.tasks.group`, then `defaults.group`) is applied at
            submission time; when that is also unset, QLAM selects the backend
            default group. Defaults to None.
    """

    # NOTE: for python 3.10, we need the future_cls to be Future, not Future[Result]
    # in order to keep the typing correct, we use cast and set a default on the
    # FutureType TypeVar
    future_cls: type[FutureType] = cast(type[FutureType], Future)
    kernel_serializer: KernelSerializer = field(default_factory=JSONSerializer)
    group_id: UUID | None = None

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

    def _resolve_kernel_serializer(
        self, kernel_serializer: KernelSerializer | None
    ) -> KernelSerializer:
        if kernel_serializer is None:
            return self.kernel_serializer

        return kernel_serializer

    def _resolve_group_id(self, group_id: UUID | None) -> UUID | None:
        """Return a task-specific group, or this device's default group."""
        if group_id is None:
            return self.group_id

        return group_id

    def task(
        self,
        kernel: ir.Method,
        num_shots: int = 1,
        arguments: dict | None = None,
        metadata: dict | None = None,
        program_language: str = "squin",
        language_version: str = "0.1.0",
        kernel_serializer: KernelSerializer | None = None,
        group_id: UUID | None = None,
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
            language_version (str): Semantic version of the program language to
                store in the task definition. Defaults to "0.1.0".
            kernel_serializer (KernelSerializer | None): Serializer for this
                task's kernel. When None, the device's `kernel_serializer` is
                used. Defaults to None.
            group_id (UUID | None): QLAM group for this task definition. When
                None, the device's default group is used. Defaults to None.

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
            language_version=language_version,
            future_cls=self.future_cls,
            kernel_serializer=self._resolve_kernel_serializer(kernel_serializer),
            group_id=self._resolve_group_id(group_id),
        )

    def batch_task(
        self,
        kernels: list[ir.Method],
        arguments: list[dict] | None = None,
        metadata: list[dict[str, Any]] | None = None,
        num_shots: list[int] | int = 1,
        program_language: str = "squin",
        language_version: str = "0.1.0",
        kernel_serializer: KernelSerializer | None = None,
        group_id: UUID | None = None,
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
            language_version (str): Semantic version of the program language to
                store in the task definition. Defaults to "0.1.0".
            kernel_serializer (KernelSerializer | None): Serializer for this
                task's kernels. When None, the device's `kernel_serializer` is
                used. Defaults to None.
            group_id (UUID | None): QLAM group for this task definition. When
                None, the device's default group is used. Defaults to None.

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
            language_version=language_version,
            future_cls=self.future_cls,
            kernel_serializer=self._resolve_kernel_serializer(kernel_serializer),
            group_id=self._resolve_group_id(group_id),
        )

    def parameter_scan(
        self,
        kernel: ir.Method,
        arguments: list[dict],
        metadata: list[dict] | None = None,
        num_shots: list[int] | int = 1,
        program_language: str = "squin",
        language_version: str = "0.1.0",
        kernel_serializer: KernelSerializer | None = None,
        group_id: UUID | None = None,
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
            language_version (str): Semantic version of the program language to
                store in the task definition. Defaults to "0.1.0".
            kernel_serializer (KernelSerializer | None): Serializer for the
                scanned kernel. When None, the device's `kernel_serializer` is
                used. Defaults to None.
            group_id (UUID | None): QLAM group for this task definition. When
                None, the device's default group is used. Defaults to None.

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
            language_version=language_version,
            future_cls=self.future_cls,
            kernel_serializer=self._resolve_kernel_serializer(kernel_serializer),
            group_id=self._resolve_group_id(group_id),
        )
