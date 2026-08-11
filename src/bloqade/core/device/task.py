import base64
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, Protocol, cast, overload
from uuid import UUID

from kirin import ir
from kirin.serialization import JSONSerializer
from qlam_core.plugins.tasks.api.client import TasksClient
from qlam_core.plugins.tasks.api.tasks_models import (
    Program,
    Subtask,
    Task,
    TaskCreationRequest,
    TaskDefinition,
    TaskMetadata,
)

from .future import ApiFetchOptions, Future, FutureType
from .local_storage import DictStorage, StorageBackend
from .log_info import logger
from .mixins import AuthMixin


class KernelSerializer(Protocol):
    """Structural interface for kernel serializers."""

    def encode(self, encoded_module: Any, /) -> str | bytes:
        """Encode a Kirin serialization module for `Program.content`."""
        ...


@dataclass(kw_only=True)
class TaskABC(Generic[FutureType], AuthMixin, ABC):
    """Abstract base class for kernel tasks.

    A task collects one or more kernels and per-subtask metadata into a
    `TaskDefinition` that can be dry-run or submitted to the backend.

    Attributes:
        program_language (str): Program language identifier stored on the
            task definition and used when serializing kernels.
        language_version (str): Program language version stored on the task
            definition and used when serializing kernels. Must be a semantic
            version. Set this directly for a static version, or override the
            `program_language_version` property if the version needs
            additional logic. Defaults to "0.1.0".
        kernel_serializer (KernelSerializer): Serializer used by the default
            `serialize_kernel` implementation. It must provide an `encode`
            method compatible with the value returned by
            `kernel.dialects.encode(...)`. If `encode` returns bytes, the
            bytes are base64-encoded before being stored in `Program.content`;
            if it returns str, the value is used unchanged. Defaults to
            `kirin.serialization.JSONSerializer`.
        future_cls (type[FutureType]): Future class used to construct the
            return value of `submit_task_definition`. Defaults to `Future`.
        group_id (UUID | None): QLAM group for the task definition. Defaults
            to None, allowing QLAM to select the backend default group.
    """

    program_language: str
    language_version: str = "0.1.0"
    kernel_serializer: KernelSerializer = field(default_factory=JSONSerializer)
    group_id: UUID | None = None

    # NOTE: bound to subclasses of future, so need to ignore the typing issue here
    future_cls: type[FutureType] = Future  # type: ignore

    @property
    def program_language_version(self) -> str:
        """Program language version recorded when serializing kernels.

        Defaults to the `language_version` attribute. Override this property
        in a subclass if the version needs to be computed with additional
        logic. The value must be a semantic version.

        Returns:
            str: Semantic version string.
        """
        return self.language_version

    def serialize_kernel(self, kernel: ir.Method) -> str:
        """Serialize a kernel into content suitable for the backend.

        The default implementation first converts the kernel to a Kirin
        serialization module using `program_language_version`, then passes
        that module to `kernel_serializer.encode`. Binary serializer output is
        base64-encoded so it can travel through the API's string-valued
        `Program.content` field. String serializer output is returned as
        produced by the serializer.

        Args:
            kernel (ir.Method): Kernel to serialize.

        Returns:
            str: Serialized kernel content for the submitted `Program`.
        """

        encoded_module = kernel.dialects.encode(
            kernel, version=self.program_language_version
        )
        payload = self.kernel_serializer.encode(encoded_module)

        if isinstance(payload, bytes):
            return base64.b64encode(payload).decode("ascii")

        if isinstance(payload, str):
            return payload

        raise TypeError(
            "kernel_serializer.encode must return str or bytes, "
            f"got {type(payload).__name__}"
        )

    @property
    @abstractmethod
    def num_subtasks(self) -> int:
        """Number of subtasks in this task's definition."""
        ...

    def summary(self) -> str:
        """Return a human-readable summary printed on dry-run.

        Returns:
            str: Summary describing what would be submitted.
        """
        return f"Would now submit {self.num_subtasks} subtasks"

    def validate_arguments(self) -> None:
        """Validate that argument and metadata lengths match subtask count.

        Raises:
            ValueError: If arguments or metadata length differs from
                `num_subtasks`.
        """
        arguments = self.get_arguments()
        if arguments is not None and len(arguments) != self.num_subtasks:
            raise ValueError(
                f"Length mismatch: got {len(arguments)} sets of arguments for {self.num_subtasks} subtasks!"
            )

        metadata = self.get_metadata()
        if metadata is not None and len(metadata) != self.num_subtasks:
            raise ValueError(
                f"Length mismatch: got {len(metadata)} sets of metadata for {self.num_subtasks} subtasks!"
            )

        num_shots = self.get_num_shots()
        if len(num_shots) != self.num_subtasks:
            raise ValueError(
                f"Length mismatch: got {len(num_shots)} shot counts for {self.num_subtasks} subtasks!"
            )

    @abstractmethod
    def get_kernels(self) -> list[ir.Method]:
        """Return the kernels used to build the task's programs.

        Returns:
            list[ir.Method]: Kernels in program-index order.
        """
        ...

    @abstractmethod
    def get_arguments(self) -> list[dict] | None:
        """Return per-subtask argument dictionaries.

        Returns:
            list[dict] | None: One argument dictionary per subtask, or None
                when no arguments are set.
        """
        ...

    @abstractmethod
    def get_metadata(self) -> list[dict] | None:
        """Return per-subtask metadata dictionaries.

        Returns:
            list[dict] | None: One metadata dictionary per subtask, or None
                when no metadata is set.
        """
        ...

    @abstractmethod
    def get_num_shots(self) -> list[int]:
        """Return the per-subtask shot counts.

        Returns:
            list[int]: Shot count for each subtask, in subtask order.
        """
        ...

    def programs(self) -> list[Program]:
        """Build the program list for the task definition.

        Returns:
            list[Program]: One `Program` per kernel returned by
                `get_kernels`, serialized via `serialize_kernel`.
        """
        kernels = self.get_kernels()
        programs = []
        for kernel in kernels:
            programs.append(Program(content=self.serialize_kernel(kernel)))
        return programs

    def program_index_for_subtask(self, i: int) -> int:
        """Return the program index used by subtask `i`.

        The default implementation maps each subtask to its own program.
        Parameter-scan tasks override this to reuse a single program.

        Args:
            i (int): Subtask index.

        Returns:
            int: Program index.
        """
        return i

    def create_task_definition(self) -> TaskDefinition:
        """Build a `TaskDefinition` from this task's kernels and subtasks.

        Override this method directly if your use-case doesn't fit the API
        contract.

        Returns:
            TaskDefinition: Definition ready to be submitted.
        """
        programs = self.programs()

        num_shots = self.get_num_shots()

        subtasks = []
        arguments = self.get_arguments()
        metadata = self.get_metadata()
        for i in range(self.num_subtasks):
            if arguments is not None:
                args = arguments[i]
            else:
                args = None

            if metadata is not None:
                subtask_metadata = TaskMetadata(user_metadata=json.dumps(metadata[i]))
            else:
                subtask_metadata = None

            subtasks.append(
                Subtask(
                    program_index=self.program_index_for_subtask(i),
                    num_shots=num_shots[i],
                    arguments=args,
                    subtask_metadata=subtask_metadata,
                )
            )

        program_language_with_version = f"{self.program_language}.v{self.program_language_version.removeprefix('v')}"
        return TaskDefinition(
            program_language=program_language_with_version,
            programs=programs,
            subtasks=subtasks,
            group_id=self.group_id,
        )

    @overload
    def run_async(
        self,
        *,
        dry_run: Literal[True],
        storage: StorageBackend | None = None,
        fetch_options: ApiFetchOptions = ApiFetchOptions(),
    ) -> None: ...

    @overload
    def run_async(
        self,
        *,
        dry_run: Literal[False],
        storage: StorageBackend | None = None,
        fetch_options: ApiFetchOptions = ApiFetchOptions(),
    ) -> FutureType: ...

    def run_async(
        self,
        *,
        dry_run: bool,
        storage: StorageBackend | None = None,
        fetch_options: ApiFetchOptions = ApiFetchOptions(),
    ) -> FutureType | None:
        """Validate the task and either dry-run or submit it.

        Keyword Args:
            dry_run (bool): When True, print a summary and return None.
                When False, submit the task and return a future.
            storage (StorageBackend | None): Storage backend that will receive
                the task definition and later fetched shots. When None, a fresh
                `DictStorage` is used (in-memory; not persisted across
                processes). Defaults to None.
            fetch_options (ApiFetchOptions): Pagination and polling options
                attached to the returned future. Defaults to
                `ApiFetchOptions()`.

        Returns:
            FutureType | None: Future attached to the submitted task when
                `dry_run` is False; otherwise None.

        Raises:
            ValueError: If argument or metadata lengths do not match
                `num_subtasks`.
        """
        self.validate_arguments()

        task_def = self.create_task_definition()

        if dry_run:
            print(self.summary())
            return

        return self.submit_task_definition(
            task_definition=task_def,
            storage=storage,
            fetch_options=fetch_options,
        )

    def submit_task_definition(
        self,
        *,
        task_definition: TaskDefinition,
        storage: StorageBackend | None = None,
        fetch_options: ApiFetchOptions = ApiFetchOptions(),
    ) -> FutureType:
        """Submit a prepared task definition and return a future.

        Keyword Args:
            task_definition (TaskDefinition): Task definition to submit.
            storage (StorageBackend | None): Storage backend that will receive
                the task definition. When None, a fresh `DictStorage` is used
                (in-memory; not persisted across processes). Defaults to None.
            fetch_options (ApiFetchOptions): Pagination and polling options
                attached to the returned future. Defaults to
                `ApiFetchOptions()`.

        Returns:
            FutureType: Future attached to the created task ID.

        Raises:
            ValueError: If the backend response is missing a task ID.
        """
        if storage is None:
            storage = DictStorage()

        self.authenticate()

        task_request = TaskCreationRequest(root=task_definition)
        with TasksClient(self.app_context) as tasks_client:
            created_task = cast(
                Task,
                self.call_with_auth_refresh(
                    lambda: tasks_client.create(body=task_request)  # type: ignore
                ),
            )

        task_id = created_task.id

        if not task_id:
            raise ValueError(
                f"Couldn't get id of created task {created_task}. Please report this issue!"
            )

        logger.info(f"Submitted task with ID: {task_id}")

        storage.add_task_definition(task_id, task_definition, created_task.created_date)

        return self.future_cls(
            task_id=task_id,
            fetch_options=fetch_options,
            storage=storage,
            context_name=self.context_name,
        )


@dataclass(kw_only=True)
class ParameterScanTask(TaskABC[FutureType]):
    """Task that runs one kernel against multiple argument sets.

    Each entry in `arguments` becomes a subtask. The same program is reused
    for every subtask.

    Attributes:
        kernel (ir.Method): Kernel executed for each parameter set.
        arguments (list[dict]): Argument dictionaries, one per subtask.
        num_shots (int | list[int]): Shot count per subtask, or one value
            broadcast to every subtask.
        metadata (list[dict] | None): Per-subtask metadata. Defaults to
            None.
    """

    kernel: ir.Method
    arguments: list[dict]
    num_shots: int | list[int]
    metadata: list[dict] | None = None

    @property
    def num_subtasks(self) -> int:
        assert self.arguments is not None
        return len(self.arguments)

    def get_kernels(self) -> list[ir.Method]:
        return [self.kernel]

    def get_arguments(self) -> list[dict]:
        return self.arguments

    def get_num_shots(self) -> list[int]:
        if isinstance(self.num_shots, int):
            return [self.num_shots] * self.num_subtasks
        return self.num_shots

    def get_metadata(self) -> list[dict] | None:
        return self.metadata

    def program_index_for_subtask(self, i: int) -> int:
        return 0

    def summary(self) -> str:
        msg = "=" * 60 + "\n"
        msg += "DRY RUN -- NO PROGRAM WAS ACTUALLY SUBMITTED FOR EXECUTION\n"
        msg += f"Would now submit a task containing {self.num_subtasks} subtasks.\n"
        msg += f"These subtasks correspond to parameter sets of the kernel {self.kernel.sym_name}.\n"
        msg += "Set dry_run=False to actually execute the parameter scan.\n"
        msg += "=" * 60
        return msg


@dataclass(kw_only=True)
class SingleKernelTask(TaskABC[FutureType]):
    """Task that runs a single kernel with one set of arguments.

    Attributes:
        kernel (ir.Method): Kernel to execute.
        arguments (dict | None): Arguments for the kernel. Defaults to
            None.
        num_shots (int): Shot count for the kernel.
        metadata (dict | None): Metadata for the single subtask. Defaults
            to None.
    """

    kernel: ir.Method
    arguments: dict | None = None
    num_shots: int
    metadata: dict | None = None

    @property
    def num_subtasks(self) -> int:
        return 1

    def get_kernels(self) -> list[ir.Method]:
        return [self.kernel]

    def get_arguments(self) -> list[dict] | None:
        if self.arguments is not None:
            return [self.arguments]

    def get_num_shots(self) -> list[int]:
        return [self.num_shots]

    def get_metadata(self) -> list[dict] | None:
        if self.metadata is not None:
            return [self.metadata]

    def summary(self) -> str:
        msg = "=" * 60 + "\n"
        msg += "DRY RUN -- NO PROGRAM WAS ACTUALLY SUBMITTED FOR EXECUTION\n"
        msg += "Would now submit a task containing a single subtask for the kernel:\n"
        if self.arguments is not None:
            formatted_arguments = ", ".join(
                f"{key}={value}" for key, value in self.arguments.items()
            )
        else:
            formatted_arguments = ""
        kernel_print = f"{self.kernel.sym_name}({formatted_arguments})"
        shots = self.num_shots if isinstance(self.num_shots, int) else self.num_shots[0]
        msg += f"  * {kernel_print} - {shots} shots\n"
        msg += "Set dry_run=False to actually execute this kernel.\n"
        msg += "=" * 60
        return msg


@dataclass(kw_only=True)
class KernelBatchTask(TaskABC[FutureType]):
    """Task that runs multiple kernels, one subtask per kernel.

    Attributes:
        kernels (list[ir.Method]): Kernels to execute.
        arguments (list[dict] | None): Per-kernel argument dictionaries.
            Defaults to None.
        num_shots (int | list[int]): Shot count per kernel, or one value
            broadcast to every kernel.
        metadata (list[dict] | None): Per-kernel metadata. Defaults to
            None.
    """

    kernels: list[ir.Method]
    arguments: list[dict] | None = None
    num_shots: int | list[int]
    metadata: list[dict] | None = None

    @property
    def num_subtasks(self) -> int:
        return len(self.kernels)

    def get_kernels(self) -> list[ir.Method]:
        return self.kernels

    def get_arguments(self) -> list[dict] | None:
        return self.arguments

    def get_num_shots(self) -> list[int]:
        if isinstance(self.num_shots, int):
            return [self.num_shots] * self.num_subtasks
        return self.num_shots

    def get_metadata(self) -> list[dict] | None:
        return self.metadata

    def summary(self) -> str:
        msg = "=" * 60 + "\n"
        msg += "DRY RUN -- NO PROGRAM WAS ACTUALLY SUBMITTED FOR EXECUTION\n"
        msg += f"Would now submit a task containing {len(self.kernels)} programs:\n"
        for i, kernel in enumerate(self.kernels):
            kernel_print = f"{kernel.sym_name}("
            if self.arguments is not None:
                for arg in self.arguments[i]:
                    kernel_print += f"{arg}, "
            kernel_print += ")"
            shots = (
                self.num_shots if isinstance(self.num_shots, int) else self.num_shots[i]
            )
            msg += f"  * {kernel_print} - {shots} shots\n"
        msg += "Set dry_run=False to actually execute the programs.\n"
        msg += "=" * 60 + "\n"
        return msg
