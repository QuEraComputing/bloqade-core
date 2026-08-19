import time
from dataclasses import dataclass, field
from typing import Any, Generic, Self
from warnings import warn

import numpy as np
from qlam_core.plugins.compilations.api import CompilationsClient
from qlam_core.plugins.definitions.api.client import DefinitionsClient
from qlam_core.plugins.results.api.client import ResultsClient
from qlam_core.plugins.tasks.api.client import TasksClient
from qlam_core.plugins.tasks.api.tasks_models import (
    Task,
    TaskDefinition,
    TaskStatus,
)
from typing_extensions import TypeVar

from .local_storage import (
    DictStorage,
    ShotFilter,
    ShotResult,
    StorageBackend,
)
from .log_info import logger
from .mixins import AuthMixin
from .result import Result, ResultType


@dataclass(frozen=True)
class ApiFetchOptions:
    """Options controlling task polling and result pagination.

    Attributes:
        subtasks_per_fetch (int): Number of subtasks to request per results
            page. Defaults to 10.
        shots_per_fetch (int): Number of shots to request per subtask shot
            page. Defaults to 100.
        poll_interval_initial (float): Initial delay, in seconds, between
            status polls. Defaults to 0.5.
        poll_interval_max (float): Maximum polling delay, in seconds. Defaults
            to 30.0.
        poll_interval_factor (float): Multiplier applied to the polling delay
            after each non-terminal status. Defaults to 2.0.
    """

    subtasks_per_fetch: int = 10
    shots_per_fetch: int = 100
    poll_interval_initial: float = 0.5  # seconds before first retry
    poll_interval_max: float = 30.0  # cap for backoff
    poll_interval_factor: float = 2.0  # multiplier per iteration


DEFAULT_FETCH_OPTIONS = ApiFetchOptions()


@dataclass(kw_only=True)
class Future(AuthMixin, Generic[ResultType]):
    """Future for a submitted task.

    A future can poll task status, fetch available shot results into storage,
    and construct result views over that storage using ``result_cls``.

    Attributes:
        task_id (str): Backend task ID.
        storage (StorageBackend): Storage backend used for fetched shots and
            task metadata. Defaults to a fresh `DictStorage` (in-memory; not
            persisted across processes).
        fetch_options (ApiFetchOptions): Pagination and polling options.
    """

    task_id: str
    storage: StorageBackend = field(default_factory=DictStorage)

    # API polling behavior
    fetch_options: ApiFetchOptions = ApiFetchOptions()

    EXIT_STATUS = (
        TaskStatus.CANCELLED,
        TaskStatus.FAILED,
        TaskStatus.PAYLOAD_PROCESSING_ERROR,
        TaskStatus.COMPLETED,
    )

    result_cls: type[ResultType] = Result  # type: ignore

    # internal things
    # TODO: this is not persisted across sessions and will refetch all results
    # when coming from a new python session using the from_task_id or from_storage methods
    _first_incomplete_subtask_page: int = field(init=False, default=0)

    def get_task(self) -> "Task":
        """Fetch the current task record from the backend.

        Returns:
            Task: The task object returned by the backend.
        """
        self.authenticate()
        with TasksClient(self.app_context) as client:
            task = self.call_with_auth_refresh(lambda: client.get(id=self.task_id))
            logger.info(
                f"Fetched task with id {self.task_id}. Current status: {task.task_status}"
            )
            return task

    def get_compilation(self, compilation_id: str | None = None):
        """Fetch the compilation record associated with this task.

        Args:
            compilation_id (str | None): ID of the compilation to fetch. When
                `None`, the compilation ID is retrieved from the task record.
                Defaults to `None`.

        Returns:
            The compilation object returned by the backend.
        """
        self.authenticate()

        if compilation_id is None:
            compilation_id = self.get_task().compilation_id

        with CompilationsClient(self.app_context) as client:
            return self.call_with_auth_refresh(lambda: client.get(id=compilation_id))

    def fetch(self) -> None:
        """Fetch currently available shot results into this future's storage.

        Results are requested from the first known incomplete subtask page and
        paginated by both subtask and shot page. Repeated calls are safe because
        storage backends de-duplicate rows by task ID, shot index, and frame
        type.
        """
        self.authenticate()
        subtask_page = self._first_incomplete_subtask_page
        done = False

        with ResultsClient(self.app_context) as client:
            while not done:
                done = self.call_with_auth_refresh(
                    lambda page=subtask_page: self._fetch_subtask_page(
                        client=client,
                        subtask_page=page,
                    )
                )

                subtask_page += 1

    def done(self) -> bool:
        """Return whether the task has reached a terminal status.

        Returns:
            bool: `True` if the task status is one of `EXIT_STATUS`
                (completed, cancelled, failed, or payload processing error).
        """
        return self.status() in self.EXIT_STATUS

    def status(self) -> TaskStatus:
        """Return the current status of the task.

        Returns:
            TaskStatus: The task status as reported by the backend.
        """
        return self.get_task().task_status

    def cancel(self):
        """Attempt to cancel the execution of the task.

        Returns:
            The backend cancellation response when cancellation is submitted;
            otherwise None if cancellation raises and a warning is emitted.
        """

        self.authenticate()
        with TasksClient(self.app_context) as client:
            try:
                return self.call_with_auth_refresh(
                    lambda: client.cancel(id=self.task_id)
                )
            except Exception as e:  # noqa: BLE001
                warn(
                    f"Exception encountered when trying to cancel task with ID {self.task_id}: {repr(e)!s}"
                )

    def cancelled(self) -> bool:
        """Return whether the task was cancelled.

        Returns:
            bool: `True` if the task status is `CANCELLED`.
        """
        return self.status() == TaskStatus.CANCELLED

    def result(
        self,
        *,
        timeout: float | None = None,
    ) -> ResultType:
        """Wait for completion, fetch results, and return a result view.

        Keyword Args:
            timeout (float | None): Maximum number of seconds to wait for a
                terminal task status. If None, wait indefinitely. Defaults to
                None.

        Returns:
            ResultType: A result view scoped to this future's task ID and the
                DETECTED frame type.

        Raises:
            TimeoutError: If `timeout` elapses before a terminal status is
                reached.
            ValueError: If the task was cancelled or failed.
        """
        self._wait_for_completion(timeout=timeout)
        self.fetch()

        return self.results_from_storage()

    def partial_result(self) -> ResultType:
        """Fetch currently available results and return a result view.

        Unlike `result`, this method does not wait for the task to finish.

        Returns:
            ResultType: A result view scoped to this future's task ID and the
                DETECTED frame type.
        """
        self.fetch()
        return self.results_from_storage()

    def results_from_storage(self, shot_filter: ShotFilter | None = None) -> ResultType:
        """Build a result view over this future's storage.

        Args:
            shot_filter (ShotFilter | None): Filter to apply to the result view.
                When None, the view is scoped to this future's task ID and the
                DETECTED frame type. Defaults to None.

        Returns:
            ResultType: A result view over the storage backend.
        """
        if shot_filter is None:
            shot_filter = ShotFilter(task_ids=(self.task_id,), frame_type="DETECTED")
        return self.result_cls(
            storage=self.storage,
            shot_filter=shot_filter,
        )

    def export_to(
        self,
        storage: StorageBackend,
        chunk_size: int = 1000,
        shot_filter: ShotFilter | None = None,
    ):
        """Copy stored shots and task definitions to another storage backend.

        Args:
            storage (StorageBackend): Destination storage backend.
            chunk_size (int): Maximum number of shots to write per batch.
                Defaults to 1000.
            shot_filter (ShotFilter | None): Optional shot filter. When the
                filter includes `task_ids`, only those task definitions are
                copied. Otherwise, all task definitions from this future's
                storage are copied. Defaults to None.
        """
        chunk = []
        for shot in self.storage.get_shots(shot_filter=shot_filter):
            chunk.append(shot)
            if len(chunk) >= chunk_size:
                storage.add_shots(chunk)
                chunk = []
        if chunk:
            storage.add_shots(chunk)

        if shot_filter is not None and shot_filter.task_ids is not None:
            task_ids = shot_filter.task_ids
        else:
            task_ids = self.storage.task_ids()

        for task_id in task_ids:
            task_def = self.storage.get_task_definition(task_id)
            creation_time = self.storage.get_task_creation_time(task_id)
            storage.add_task_definition(task_id, task_def, creation_time)

    def fetch_and_export_to(self, storage: StorageBackend, chunk_size: int = 1000):
        """Fetch available results and export this future's storage.

        Args:
            storage (StorageBackend): Destination storage backend.
            chunk_size (int): Maximum number of shots to write per batch.
                Defaults to 1000.
        """
        self.fetch()
        self.export_to(storage, chunk_size=chunk_size)

    @classmethod
    def _resolve_context_name(cls, context_name: str | None) -> str:
        """Return `context_name` when set, otherwise the class-level default.

        Args:
            context_name (str | None): Caller-supplied context name. When
                None, the class-level `context_name` attribute is used.

        Returns:
            str: The resolved context name.

        Raises:
            ValueError: If `context_name` is None and `cls` has no
                class-level `context_name` default.
        """
        if context_name is not None:
            return context_name
        resolved = getattr(cls, "context_name", None)
        if resolved is None:
            raise ValueError(
                f'{cls.__name__} has no default context_name; please pass in `context_name="my-context"` explicitly.'
            )
        return resolved

    @classmethod
    def from_storage(
        cls,
        *,
        storage: StorageBackend,
        new_storage: StorageBackend | None = None,
        task_id: str | None = None,
        fetch_options: ApiFetchOptions = DEFAULT_FETCH_OPTIONS,
        context_name: str | None = None,
    ) -> Self:
        """Create a future from task metadata already present in storage.

        Args:
            storage (StorageBackend): Storage used to discover and validate the
                task ID.
            new_storage (StorageBackend | None): Storage backend to attach to
                the returned future. When None, `storage` is reused. Defaults to
                None.
            task_id (str | None): Task ID to attach to. Required when `storage`
                contains more than one task ID. Defaults to None.
            fetch_options (ApiFetchOptions): Pagination and polling options.
                Defaults to `ApiFetchOptions()`.
            context_name (str | None): Name of the qlam context to attach to
                the returned future. When None, the class-level default on
                `cls` is used. Defaults to None.

        Returns:
            Self: A future attached to the selected task ID.

        Raises:
            ValueError: If storage has no task IDs, the requested task ID is not
                present, multiple task IDs are present without an explicit
                `task_id`, or `context_name` is None and `cls` has no
                class-level default.
        """
        context_name = cls._resolve_context_name(context_name)

        if new_storage is None:
            new_storage = storage

        task_ids_at_storage = storage.task_ids()

        if not task_ids_at_storage:
            raise ValueError("Found no task IDs in storage.")

        if task_id is not None and task_id not in task_ids_at_storage:
            raise ValueError(f"Task with ID {task_id} not found at storage {storage}")

        if len(task_ids_at_storage) > 1 and task_id is None:
            msg = "More than one task ID found! Please specify a task_id. Candidates are:\n"
            for candidate_task_id in task_ids_at_storage:
                creation_time = storage.get_task_creation_time(candidate_task_id)
                msg += f"  * {candidate_task_id}, created at {creation_time}\n"
            raise ValueError(msg)

        if task_id is None:
            task_id = task_ids_at_storage.pop()

        return cls(
            task_id=task_id,
            storage=new_storage,
            fetch_options=fetch_options,
            result_cls=cls.result_cls,
            context_name=context_name,
        )

    @classmethod
    def from_task_id(
        cls,
        *,
        task_id: str,
        storage: StorageBackend | None = None,
        fetch_options: ApiFetchOptions = DEFAULT_FETCH_OPTIONS,
        context_name: str | None = None,
    ) -> Self:
        """Create a future from a backend task ID.

        The task record and task definition are fetched from the backend, and
        the task definition is stored in `storage` before the future is
        returned.

        Args:
            task_id (str): Backend task ID.
            storage (StorageBackend | None): Storage backend that will receive
                the task definition and later fetched shots. When None, a fresh
                `DictStorage` is used (in-memory; not persisted across
                processes). Defaults to None.
            fetch_options (ApiFetchOptions): Pagination and polling options.
                Defaults to `ApiFetchOptions()`.
            context_name (str | None): Name of the qlam context used to fetch
                the task and attached to the returned future. When None, the
                class-level default on `cls` is used. Defaults to None.

        Returns:
            Self: A future attached to `task_id`.

        Raises:
            ValueError: If `context_name` is None and `cls` has no
                class-level default.
        """
        if storage is None:
            storage = DictStorage()

        context_name = cls._resolve_context_name(context_name)
        auth = AuthMixin(context_name=context_name)
        auth.authenticate()
        with TasksClient(auth.app_context) as tasks_client:
            task = auth.call_with_auth_refresh(lambda: tasks_client.get(id=task_id))

        # fetch subtasks for metadata
        with DefinitionsClient(auth.app_context) as definitions_client:
            task_def = auth.call_with_auth_refresh(
                lambda: definitions_client.get(id=task.definition_id)
            )

        storage.add_task_definition(
            task_id,
            task_definition=TaskDefinition.model_validate(
                {
                    **task_def.model_dump(
                        include={"program_language", "programs", "subtasks"}
                    ),
                    "group_id": task_def.group.id,
                }
            ),
            creation_time=task.created_date,
        )

        return cls(
            task_id=task_id,
            storage=storage,
            fetch_options=fetch_options,
            result_cls=cls.result_cls,
            context_name=context_name,
        )

    def _wait_for_completion(self, timeout: float | None = None) -> TaskStatus:
        """Poll the backend until the task reaches a terminal status.

        Args:
            timeout (float | None): Maximum number of seconds to wait.
                `None` waits indefinitely. Defaults to `None`.

        Returns:
            TaskStatus: The terminal status the task reached.

        Raises:
            TimeoutError: If `timeout` elapses before a terminal status is reached.
            ValueError: If the task was cancelled or failed.
        """
        if timeout is not None:
            deadline = time.monotonic() + timeout
        else:
            deadline = None

        interval = self.fetch_options.poll_interval_initial

        while True:
            status = self.status()
            if status in self.EXIT_STATUS:
                break
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out after {timeout}s waiting for task, status is {status}"
                )
            time.sleep(interval)
            interval = min(
                interval * self.fetch_options.poll_interval_factor,
                self.fetch_options.poll_interval_max,
            )

        if status == TaskStatus.CANCELLED:
            raise ValueError("The task was cancelled.")
        if status in (TaskStatus.FAILED, TaskStatus.PAYLOAD_PROCESSING_ERROR):
            task = self.get_task()
            # TODO: check if there was a compilation error and fetch that too
            raise ValueError(f"The task failed with errors: {task.error_reasons}")

        return status

    def _fetch_subtask_page(
        self,
        client: ResultsClient,
        subtask_page: int,
    ) -> bool:
        """Fetch one subtask page and all required shot pages.

        Args:
            client (ResultsClient): Authenticated results client.
            subtask_page (int): Subtask page index to fetch.

        Returns:
            bool: True when there are no more subtask pages to fetch; otherwise
                False.
        """
        shots_page = 0
        full_shots_page = True
        found_incomplete = False
        full_subtask_page = True

        while full_shots_page:
            response = client.get(
                id=self.task_id,
                page=subtask_page,
                size=self.fetch_options.subtasks_per_fetch,
                sort="completed_date,asc",  # sort, such that completed subtasks come first
                shots_page=shots_page,
                shots_size=self.fetch_options.shots_per_fetch,
            )

            task_results = response.get("elements", [])
            full_subtask_page = len(task_results) > 0
            temp_data = []  # accumulate data and write in batch
            shots_this_page: dict[int, int] = {}  # track shot page size
            for task_result in task_results:
                subtasks = task_result.get("subtasks", [])
                full_subtask_page = (
                    full_subtask_page
                    and len(subtasks) >= self.fetch_options.subtasks_per_fetch
                )

                # completed_date is stable across shot pages, and later shot
                # pages can return a subtask with no shots (so its index can't
                # be derived). Update only on the first shot page.
                if shots_page == 0:
                    self.storage.update_subtasks_completed_date(
                        task_id=self.task_id, subtasks=subtasks
                    )

                for subtask in subtasks:
                    subtask_status = subtask.get("status", "").upper()
                    if not found_incomplete and subtask_status != "COMPLETED":
                        # NOTE: this is the page where incomplete subtasks start;
                        # we have to start again here in future fetch calls
                        found_incomplete = True  # make sure we only track the first one
                        self._first_incomplete_subtask_page = subtask_page

                    # Add rows to the result store
                    shot_results = subtask.get("shot_results", [])
                    for shot_result in shot_results:
                        shot_index = shot_result.get("shot_index")
                        subtask_shot_index = shot_result.get("subtask_shot_index")
                        subtask_index = shot_result.get("subtask_index")
                        frame_type = shot_result.get("frame_type")
                        measurement = shot_result.get("measurement", {})
                        bitstring = measurement.get("measurement_values")

                        # increment shot count
                        shots_this_page[subtask_index] = (
                            shots_this_page.get(subtask_index, 0) + 1
                        )

                        data = ShotResult(
                            task_id=self.task_id,
                            shot_index=shot_index,
                            subtask_index=subtask_index,
                            subtask_shot_index=subtask_shot_index,
                            frame_type=frame_type.upper(),
                            bitstring=np.array(bitstring, dtype=bool),
                        )
                        temp_data.append(data)

            self.storage.add_shots(temp_data)
            shots_page += 1
            full_shots_page = any(
                sr >= self.fetch_options.shots_per_fetch
                for sr in shots_this_page.values()
            )

        return not full_subtask_page


FutureType = TypeVar("FutureType", bound=Future[Any], default=Future[Result])
