from dataclasses import dataclass, field
from typing import Any, Generic, Literal, cast, overload
from uuid import UUID

from kirin import ir
from kirin.serialization import JSONSerializer
from kirin.validation import ValidationSuite
from qlam_core.plugins.groups import GroupsClient
from qlam_core.plugins.tasks import TasksClient
from qlam_core.plugins.tasks.api import TaskCreationRequest
from qlam_core.plugins.tasks.api.tasks_models import TaskDefinition

from .future import DEFAULT_FETCH_OPTIONS, ApiFetchOptions, Future, FutureType
from .local_storage import DictStorage, StorageBackend
from .log_info import logger
from .mixins import AuthMixin
from .task import (
    KernelBatchTask,
    KernelSerializer,
    ParameterScanTask,
    SingleKernelTask,
)
from .task_builder import FinalizeContext, TaskBuilder


# NOTE: 'Device' is like the 'master class' for communicating with the server.
@dataclass(kw_only=True)
class Device(AuthMixin, Generic[FutureType]):
    """Create legacy tasks and finalize or submit task builders.

    Legacy factory methods return task objects that submit themselves. For the
    builder API, the device supplies serialization and validation context and
    owns dry-run and submission through :meth:`run_async`.

    Attributes:
        program_language (str): Language name placed on builder-generated task
            definitions. Defaults to ``"squin"``.
        language_version (str): Language version used when serializing builder
            kernels. Defaults to ``"0.1.0"``.
        future_cls (type[FutureType]): Future class used by tasks created from
            this device. Defaults to `Future`.
        kernel_serializer (KernelSerializer): Default serializer passed to
            created tasks. The serializer is used by
            `TaskABC.serialize_kernel`; it should provide an `encode` method
            for Kirin serialization modules.
            Defaults to `kirin.serialization.JSONSerializer`.
    """

    # NOTE: for python 3.10, we need the future_cls to be Future, not Future[Result]
    # in order to keep the typing correct, we use cast and set a default on the
    # FutureType TypeVar
    # TODO: need to add impl of GeminiLogicalDevice w/ these fields in bloqade-lanes.
    # NOTE: adding defaults here just in case for backwards compatibility
    program_language: str = "squin"
    language_version: str = "0.1.0"
    future_cls: type[FutureType] = Future  # type: ignore[assignment]
    kernel_serializer: KernelSerializer = field(default_factory=JSONSerializer)
    validation_suite: ValidationSuite | None = None
    dialect_group: ir.DialectGroup | None = None

    # legacy
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

    def _finalize_context(self) -> FinalizeContext:
        """Return the device-owned inputs needed to finalize a builder."""
        return FinalizeContext(
            program_language=self.program_language,
            language_version=self.language_version,
            # TODO: what if you need to "override" a kernel serializer? just create a new Device class with the field?
            kernel_serializer=self.kernel_serializer,
            dialect_group=self.dialect_group,
            validation_suite=self.validation_suite,
        )

    @overload
    def run_async(
        self,
        task_builder: TaskBuilder,
        *,
        dry_run: Literal[True],
        group: str | None = None,
        storage: StorageBackend | None = None,
        fetch_options: ApiFetchOptions = DEFAULT_FETCH_OPTIONS,
    ) -> None: ...

    @overload
    def run_async(
        self,
        task_builder: TaskBuilder,
        *,
        dry_run: Literal[False],
        group: str | None = None,
        storage: StorageBackend | None = None,
        fetch_options: ApiFetchOptions = DEFAULT_FETCH_OPTIONS,
    ) -> FutureType: ...

    def run_async(
        self,
        task_builder: TaskBuilder,
        *,
        dry_run: bool,
        group: str | None = None,
        storage: StorageBackend | None = None,
        fetch_options: ApiFetchOptions = DEFAULT_FETCH_OPTIONS,
    ) -> FutureType | None:
        """Validate and finalize a builder, optionally submitting it.

        A dry run performs the same kernel validation and serialization as a
        submission, prints the builder summary, and returns ``None`` without
        authenticating, writing storage, or calling QLAM. Finalization does not
        mutate the builder, so it remains editable after a dry run.

        Args:
            task_builder (TaskBuilder): Incrementally assembled task to
                finalize.

        Keyword Args:
            dry_run (bool): When true, print a preview without submitting.
                When false, submit and return a future.
            group (str | None): Optional QLAM group name or UUID string for
                this submission. Defaults to configured group precedence.
            storage (StorageBackend | None): Storage for the submitted task
                definition. Ignored during a dry run.
            fetch_options (ApiFetchOptions): Fetch configuration attached to
                the returned future. Ignored during a dry run.

        Returns:
            FutureType | None: ``None`` for a dry run; otherwise the future
                associated with the submitted QLAM task.
        """
        finalize_ctx = self._finalize_context()
        task_definition = task_builder._finalize(finalize_ctx)
        if dry_run:
            print(task_builder.summary())
            return
        return self.submit_task_definition(
            task_definition=task_definition,
            storage=storage,
            fetch_options=fetch_options,
            group=group,
        )

    def _configured_group(self, group: str | None = None) -> str | None:
        """Return the configured group reference for submissions, if set.

        Mirrors the qsh CLI precedence for submission commands:
        `plugins.tasks.group` first, then `defaults.group`.

        Returns:
            str | None: The configured group name or UUID string, or None
                when the config does not set a group.
        """
        if group is not None:
            return group
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

    def submit_task_definition(
        self,
        *,
        task_definition: TaskDefinition,
        storage: StorageBackend | None = None,
        fetch_options: ApiFetchOptions = DEFAULT_FETCH_OPTIONS,
        group: str | None = None,
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
            group = self._configured_group(group)
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

    def _resolve_kernel_serializer(
        self, kernel_serializer: KernelSerializer | None
    ) -> KernelSerializer:
        if kernel_serializer is None:
            return self.kernel_serializer

        return kernel_serializer

    # Legacy methods
    def task(
        self,
        kernel: ir.Method,
        num_shots: int = 1,
        arguments: dict | None = None,
        metadata: dict | None = None,
        program_language: str = "squin",
        language_version: str = "0.1.0",
        kernel_serializer: KernelSerializer | None = None,
        group: str | None = None,
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
            group (str | None): Name of the QLAM group for this task
                definition. When None, the configured group is used at
                submission time.

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
            group=group,
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
        group: str | None = None,
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
            group (str | None): Name of the QLAM group for this task
                definition. When None, the configured group is used at
                submission time.

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
            group=group,
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
        group: str | None = None,
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
            group (str | None): Name of the QLAM group for this task
                definition. When None, the configured group is used at
                submission time.

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
            group=group,
        )
