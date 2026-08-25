import base64
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, Protocol, overload
from uuid import UUID

from kirin import ir
from kirin.serialization import JSONSerializer
from qlam_core.plugins.groups.api.client import GroupsClient
from qlam_core.plugins.tasks.api.client import TasksClient
from qlam_core.plugins.tasks.api.tasks_models import (
    Program,
    Subtask,
    TaskCreationRequest,
    TaskDefinition,
    TaskMetadata,
)

from .future import DEFAULT_FETCH_OPTIONS, ApiFetchOptions, Future, FutureType
from .local_storage import DictStorage, StorageBackend
from .log_info import logger
from .mixins import AuthMixin


class KernelSerializer(Protocol):
    """Structural interface for kernel serializers."""

    def encode(self, encoded_module: Any, /) -> str | bytes:
        """Encode a Kirin serialization module for `Program.content`."""
        ...


@dataclass(kw_only=True)
class TaskABC(AuthMixin, ABC, Generic[FutureType]):
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
        group (str | None): Name of the QLAM group for the task definition.
            When None,
            the `~/.qsh` config group (`plugins.tasks.group`, then
            `defaults.group`) is applied at submission time; when that is also
            unset, QLAM selects the backend default group. Defaults to None.
    """

    program_language: str
    language_version: str = "0.1.0"
    kernel_serializer: KernelSerializer = field(default_factory=JSONSerializer)
    group: str | None = None

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

    def summary_for_task_definition(self, task_definition: TaskDefinition) -> str:
        """Return the dry-run summary for a prepared task definition.

        Subclasses whose summary includes shot counts can override this method
        to read the effective values from ``task_definition``. The default
        preserves the existing ``summary`` extension point.
        """
        return self.summary()

    def validate_arguments(self) -> None:
        """Validate that task field lengths match the subtask count.

        Raises:
            ValueError: If argument or metadata length differs from
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

    def create_task_definition(self, *, num_shots: int | list[int]) -> TaskDefinition:
        """Build a `TaskDefinition` from this task's kernels and subtasks.

        Every QLAM subtask requires a concrete shot count, so callers must
        provide one value to broadcast or one value per subtask. Override this
        method directly if your use-case doesn't fit the API contract.

        Keyword Args:
            num_shots (int | list[int]): Shot count for every subtask, or one
                count per subtask.

        Returns:
            TaskDefinition: Definition ready to be submitted.
        """
        programs = self.programs()

        if isinstance(num_shots, int):
            shot_counts = [num_shots] * self.num_subtasks
        else:
            shot_counts = num_shots

        if len(shot_counts) != self.num_subtasks:
            raise ValueError(
                f"Length mismatch: got {len(shot_counts)} shot counts for {self.num_subtasks} subtasks!"
            )

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
                    num_shots=shot_counts[i],
                    arguments=args,
                    subtask_metadata=subtask_metadata,
                )
            )

        program_language_with_version = f"{self.program_language}.v{self.program_language_version.removeprefix('v')}"
        return TaskDefinition(
            program_language=program_language_with_version,
            programs=programs,
            subtasks=subtasks,
            group_id=None,
        )

    def _configured_group(self) -> str | None:
        """Return the configured group reference for submissions, if set.

        Mirrors the qsh CLI precedence for submission commands:
        `plugins.tasks.group` first, then `defaults.group`.

        Returns:
            str | None: The configured group name or UUID string, or None
                when the config does not set a group.
        """
        if self.group is not None:
            return self.group
        config = self.app_context.config
        plugin_config = config.get_plugin_config("tasks")
        group = plugin_config.group if plugin_config is not None else None
        if group is None:
            defaults = config.current_context.defaults
            group = defaults.group if defaults is not None else None
        return group

    def _resolve_group_id(self, group: str) -> UUID:
        """Resolve a configured or task-level group name to its UUID."""
        with GroupsClient(self.app_context) as groups_client:
            return self.call_with_auth_refresh(lambda: groups_client.resolve_id(group))

    @overload
    def run_async(
        self,
        *,
        dry_run: Literal[True] = True,
        shots: int | list[int] = 1,
        storage: StorageBackend | None = None,
        fetch_options: ApiFetchOptions = DEFAULT_FETCH_OPTIONS,
    ) -> None: ...

    @overload
    def run_async(
        self,
        *,
        dry_run: Literal[False],
        shots: int | list[int] = 1,
        storage: StorageBackend | None = None,
        fetch_options: ApiFetchOptions = DEFAULT_FETCH_OPTIONS,
    ) -> FutureType: ...

    def run_async(
        self,
        *,
        dry_run: bool = True,
        shots: int | list[int] = 1,
        storage: StorageBackend | None = None,
        fetch_options: ApiFetchOptions = DEFAULT_FETCH_OPTIONS,
    ) -> FutureType | None:
        """Validate the task and either dry-run or submit it.

        Keyword Args:
            dry_run (bool): When True, print a summary and return None.
                Defaults to True. When False, submit the task and return a
                future.
            shots (int | list[int]): Shot count applied to every
                subtask, or one count per subtask, in the submitted QLAM task
                definition. Defaults to 1.
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
            ValueError: If argument, metadata, or shot-count lengths do not
                match `num_subtasks`.
        """
        self.validate_arguments()
        task_def = self.create_task_definition(num_shots=shots)

        if dry_run:
            print(self.summary_for_task_definition(task_def))
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
        fetch_options: ApiFetchOptions = DEFAULT_FETCH_OPTIONS,
    ) -> FutureType:
        """Submit a prepared task definition and return a future.

        When the definition does not set a group ID, a task-level group name
        takes precedence over the `~/.qsh` config group (`plugins.tasks.group`,
        then `defaults.group`). The selected name is resolved before
        submission. When neither is set, the group is omitted and QLAM selects
        the backend default group.

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

        if task_definition.group_id is None:
            group = self._configured_group()
            if group is not None:
                task_definition = task_definition.model_copy(
                    update={"group_id": self._resolve_group_id(group)}
                )

        task_request = TaskCreationRequest(root=task_definition)
        with TasksClient(self.app_context) as tasks_client:
            created_task = self.call_with_auth_refresh(
                lambda: tasks_client.create(body=task_request)
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
        metadata (list[dict] | None): Per-subtask metadata. Defaults to
            None.
    """

    kernel: ir.Method
    arguments: list[dict]
    metadata: list[dict] | None = None

    @property
    def num_subtasks(self) -> int:
        assert self.arguments is not None
        return len(self.arguments)

    def get_kernels(self) -> list[ir.Method]:
        return [self.kernel]

    def get_arguments(self) -> list[dict]:
        return self.arguments

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
        metadata (dict | None): Metadata for the single subtask. Defaults
            to None.
    """

    kernel: ir.Method
    arguments: dict | None = None
    metadata: dict | None = None

    @property
    def num_subtasks(self) -> int:
        return 1

    def get_kernels(self) -> list[ir.Method]:
        return [self.kernel]

    def get_arguments(self) -> list[dict] | None:
        if self.arguments is not None:
            return [self.arguments]

    def get_metadata(self) -> list[dict] | None:
        if self.metadata is not None:
            return [self.metadata]

    def _summary(self, shots: int | str) -> str:
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
        msg += f"  * {kernel_print} - {shots} shots\n"
        msg += "Set dry_run=False to actually execute this kernel.\n"
        msg += "=" * 60
        return msg

    def summary(self) -> str:
        return self._summary(1)

    def summary_for_task_definition(self, task_definition: TaskDefinition) -> str:
        return self._summary(task_definition.subtasks[0].num_shots)


@dataclass(kw_only=True)
class KernelBatchTask(TaskABC[FutureType]):
    """Task that runs multiple kernels, one subtask per kernel.

    Attributes:
        kernels (list[ir.Method]): Kernels to execute.
        arguments (list[dict] | None): Per-kernel argument dictionaries.
            Defaults to None.
        metadata (list[dict] | None): Per-kernel metadata. Defaults to
            None.
    """

    kernels: list[ir.Method]
    arguments: list[dict] | None = None
    metadata: list[dict] | None = None

    @property
    def num_subtasks(self) -> int:
        return len(self.kernels)

    def get_kernels(self) -> list[ir.Method]:
        return self.kernels

    def get_arguments(self) -> list[dict] | None:
        return self.arguments

    def get_metadata(self) -> list[dict] | None:
        return self.metadata

    def _summary(self, shot_counts: list[int] | None) -> str:
        msg = "=" * 60 + "\n"
        msg += "DRY RUN -- NO PROGRAM WAS ACTUALLY SUBMITTED FOR EXECUTION\n"
        msg += f"Would now submit a task containing {len(self.kernels)} programs:\n"
        for i, kernel in enumerate(self.kernels):
            kernel_print = f"{kernel.sym_name}("
            if self.arguments is not None:
                for arg in self.arguments[i]:
                    kernel_print += f"{arg}, "
            kernel_print += ")"
            shots = shot_counts[i] if shot_counts is not None else "unspecified"
            msg += f"  * {kernel_print} - {shots} shots\n"
        msg += "Set dry_run=False to actually execute the programs.\n"
        msg += "=" * 60 + "\n"
        return msg

    def summary(self) -> str:
        return self._summary([1] * self.num_subtasks)

    def summary_for_task_definition(self, task_definition: TaskDefinition) -> str:
        return self._summary(
            [subtask.num_shots for subtask in task_definition.subtasks]
        )
