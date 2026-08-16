"""Builders and fake clients for the qlam-core HTTP surface.

Two layers live here:

1. Typed builders that return real pydantic models from `qlam_core.*`. Because
   these go through pydantic validation, any drift in field names (e.g. the
   `Task.definition_id` vs. `task.definition` mismatch that production code
   used to hit) fails loudly when a test calls a builder with the wrong kwarg.

2. Raw-dict builders for `ResultsClient.get`, which returns `JsonDict` rather
   than a typed model. The shape mirrors the live sanitized result envelope
   (see `examples/results_envelope_completed.json`).

Defaults follow the live API casing (`"Completed"`, `"Detected"`) so anything
testing the production normalization paths (`.upper()` in future.py) exercises
realistic input. The bloqade local dict schema lives in `local.py`.

These builders and the `examples/` dumps they mirror were verified against
qlam-core v0.6.x (the `~=0.6.0` pin in pyproject.toml).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable
from uuid import UUID

from qlam_core.auth.user_info import UserInfo
from qlam_core.plugins.compilations.api.compilations_models import (
    ProgramFailure,
    PublicCompilation,
    PublicCompilationStatus,
)
from qlam_core.plugins.definitions.api import definitions_models as _defs
from qlam_core.plugins.definitions.api.definitions_models import (
    GroupSummary as DefinitionGroupSummary,
    TaskDefinitionResponse,
)
from qlam_core.plugins.groups.api.groups_models import GroupResponse
from qlam_core.plugins.tasks.api.tasks_models import (
    GroupSummary as TaskGroupSummary,
    Program,
    Subtask,
    Task,
    TaskCreationRequest,
    TaskDefinition,
    TaskMetadata,
    TaskStatus,
)
from qlam_core.plugins.user_tenant.api.user_tenant_models import GroupAssignment

# Anchor values shared by every builder. Using a constant UUID/timestamp keeps
# fixture equality predictable and matches `examples/task_completed.json`.
DEFAULT_TASK_ID = "799ea417-f001-41ae-b788-7cba56e8da27"
DEFAULT_DEFINITION_ID = "db25a08c-0318-4a0d-b161-2d1658d15fa9"
DEFAULT_COMPILATION_ID = "7cd3b1aa-b0d8-4839-b060-79c8d160883e"
DEFAULT_GROUP_ID = UUID("00000000-0000-0000-0000-000000000000")
DEFAULT_GROUP_NAME = "default-group"
DEFAULT_TENANT_ID = UUID("ab12cd34-ef56-4789-abcd-ef0123456789")
DEFAULT_USER_ID = UUID("acbabea1-b48d-40c4-a7f6-d05bcf75cdd0")
DEFAULT_CREATED_DATE = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Typed builders (tasks_models / definitions_models / compilations_models)    #
# --------------------------------------------------------------------------- #


def make_task_metadata(
    *,
    user_metadata: str | None = None,
    system_metadata: str | None = None,
    **extras: Any,
) -> TaskMetadata:
    return TaskMetadata(
        user_metadata=user_metadata,
        system_metadata=system_metadata,
        **extras,
    )


def make_program(
    *,
    content: str = "program",
    program_metadata: TaskMetadata | None = None,
) -> Program:
    return Program(content=content, program_metadata=program_metadata)


def make_subtask(
    *,
    program_index: int = 0,
    num_shots: int = 1,
    arguments: dict[str, float] | None = None,
    subtask_metadata: TaskMetadata | None = None,
) -> Subtask:
    return Subtask(
        program_index=program_index,
        num_shots=num_shots,
        arguments=arguments,
        subtask_metadata=subtask_metadata,
    )


def make_task_definition(
    *,
    program_language: str | None = "squin.v0.1.0",
    programs: list[Program] | None = None,
    subtasks: list[Subtask] | None = None,
    group_id: UUID | None = None,
) -> TaskDefinition:
    if programs is None:
        programs = [make_program()]
    if subtasks is None:
        subtasks = [make_subtask()]
    return TaskDefinition(
        program_language=program_language,
        programs=programs,
        subtasks=subtasks,
        group_id=group_id,
    )


def make_group_assignment(
    *,
    tenant_id: UUID = DEFAULT_TENANT_ID,
    org_name: str | None = "test-org",
    audience: str = "https://v2/tasks",
    qpu_name: str | None = "test-qpu",
    groups: list[UUID] | None = None,
) -> GroupAssignment:
    """Build a real QLAM `GroupAssignment` model."""
    return GroupAssignment(
        tenant_id=tenant_id,
        org_name=org_name,
        audience=audience,
        qpu_name=qpu_name,
        groups=[DEFAULT_GROUP_ID] if groups is None else groups,
    )


def make_user_info(*, user_id: UUID | None = DEFAULT_USER_ID) -> UserInfo:
    """Build a typed QLAM UserInfo response."""
    return UserInfo(user_id=user_id)


def make_task_creation_request(
    task_definition: TaskDefinition | None = None,
) -> TaskCreationRequest:
    if task_definition is None:
        task_definition = make_task_definition()
    return TaskCreationRequest(root=task_definition)


def make_group(
    *,
    id: UUID = DEFAULT_GROUP_ID,  # noqa: A002 — mirrors qlam field name
    name: str = DEFAULT_GROUP_NAME,
    description: str = "test group",
    is_shared: bool = False,
    created_date: datetime = DEFAULT_CREATED_DATE,
    created_by: UUID = DEFAULT_USER_ID,
    modified_date: datetime = DEFAULT_CREATED_DATE,
    modified_by: UUID = DEFAULT_USER_ID,
) -> GroupResponse:
    """Build a real QLAM `GroupResponse` model."""
    return GroupResponse(
        id=id,
        name=name,
        description=description,
        is_shared=is_shared,
        created_date=created_date,
        created_by=created_by,
        modified_date=modified_date,
        modified_by=modified_by,
    )


def make_task(
    *,
    id: str | None = DEFAULT_TASK_ID,  # noqa: A002 — mirrors qlam field name
    task_status: TaskStatus = TaskStatus.COMPLETED,
    definition_id: str | None = DEFAULT_DEFINITION_ID,
    compilation_id: str | None = DEFAULT_COMPILATION_ID,
    created_by: UUID = DEFAULT_USER_ID,
    created_date: datetime = DEFAULT_CREATED_DATE,
    modified_date: datetime | None = None,
    modified_by: UUID | None = None,
    scheduled_date: datetime | None = None,
    group: TaskGroupSummary | None = None,
    error_reasons: list[str] | None = None,
    **extras: Any,
) -> Task:
    """Build a real `Task` pydantic model.

    Passing a field name the model does not declare — e.g. the historical
    `definition=` typo — does NOT silently succeed: it becomes a
    `model_extra` entry, so production code that accesses it as a declared
    attribute will still fail.
    """
    return Task(
        id=id,
        task_status=task_status,
        definition_id=definition_id,
        compilation_id=compilation_id,
        created_by=created_by,
        created_date=created_date,
        modified_date=modified_date,
        modified_by=modified_by,
        scheduled_date=scheduled_date,
        group=(
            TaskGroupSummary(
                id=DEFAULT_GROUP_ID, name=DEFAULT_GROUP_NAME, deactivated=False
            )
            if group is None
            else group
        ),
        error_reasons=[] if error_reasons is None else error_reasons,
        **extras,
    )


def make_task_definition_response(
    *,
    id: UUID | str = DEFAULT_DEFINITION_ID,  # noqa: A002
    program_language: str = "squin.v0.1.0",
    programs: list[Program] | None = None,
    subtasks: list[Subtask] | None = None,
    created_date: datetime = DEFAULT_CREATED_DATE,
    created_by: UUID = DEFAULT_USER_ID,
    modified_date: datetime | None = None,
    modified_by: UUID | None = None,
    group: DefinitionGroupSummary | None = None,
) -> TaskDefinitionResponse:
    """Build a `TaskDefinitionResponse`.

    Note: `qlam-core` declares `Program` and `Subtask` separately in
    `tasks_models` and `definitions_models`. Pydantic checks identity, not
    structure, so the response model rejects the tasks_models classes. This
    helper accepts the tasks_models flavors for convenience and rebuilds them
    under the definitions_models classes internally.
    """
    if programs is None:
        programs = [make_program()]
    if subtasks is None:
        subtasks = [make_subtask()]
    defs_programs = [_defs.Program(**p.model_dump()) for p in programs]
    defs_subtasks = [_defs.Subtask(**s.model_dump()) for s in subtasks]
    return TaskDefinitionResponse(
        id=UUID(id) if isinstance(id, str) else id,
        program_language=program_language,
        programs=defs_programs,
        subtasks=defs_subtasks,
        created_date=created_date,
        created_by=created_by,
        modified_date=modified_date,
        modified_by=modified_by,
        group=(
            DefinitionGroupSummary(
                id=DEFAULT_GROUP_ID, name=DEFAULT_GROUP_NAME, deactivated=False
            )
            if group is None
            else group
        ),
    )


def make_public_compilation(
    *,
    id: UUID | str = DEFAULT_COMPILATION_ID,  # noqa: A002
    input_definition_id: UUID | str = DEFAULT_DEFINITION_ID,
    status: PublicCompilationStatus = PublicCompilationStatus.SUCCEEDED,
    stack_trace: str | None = None,
    program_failures: list[ProgramFailure] | None = None,
    created_date: datetime = DEFAULT_CREATED_DATE,
    created_by: UUID = DEFAULT_USER_ID,
    modified_date: datetime | None = None,
    modified_by: UUID | None = None,
) -> PublicCompilation:
    if modified_date is None:
        modified_date = created_date
    if modified_by is None:
        modified_by = created_by
    return PublicCompilation(
        id=UUID(id) if isinstance(id, str) else id,
        input_definition_id=(
            UUID(input_definition_id)
            if isinstance(input_definition_id, str)
            else input_definition_id
        ),
        status=status,
        stack_trace=stack_trace,
        program_failures=program_failures,
        created_date=created_date,
        created_by=created_by,
        modified_date=modified_date,
        modified_by=modified_by,
    )


# --------------------------------------------------------------------------- #
# Raw-dict builders for the ResultsClient envelope                            #
# --------------------------------------------------------------------------- #
# qlam-core does not model this response shape. The defaults below match the
# sanitized envelope captured from the live API; see
# examples/results_envelope_completed.json.


def make_shot_result_dict(
    *,
    shot_index: int = 0,
    subtask_shot_index: int = 0,
    subtask_index: int = 0,
    frame_type: str = "Detected",
    measurement_values: list[int] | tuple[int, ...] = (1, 0),
    shot_start_time: str | None = None,
    measurement_time: str | None = None,
    camera_id: str | None = None,
    frame_index: int | None = None,
    error_reasons: list[str] | None = None,
) -> dict:
    """One entry in `subtask["shot_results"]`.

    Only the fields bloqade actually consumes
    (`shot_index, subtask_shot_index, subtask_index, frame_type,
    measurement.measurement_values`) are required. Optional metadata fields are
    surfaced as kwargs for tests that want to mirror the live envelope.
    """
    out: dict[str, Any] = {
        "shot_index": shot_index,
        "subtask_shot_index": subtask_shot_index,
        "subtask_index": subtask_index,
        "frame_type": frame_type,
        "measurement": {"measurement_values": list(measurement_values)},
    }
    if measurement_time is not None:
        out["measurement"]["measurement_time"] = measurement_time
    if shot_start_time is not None:
        out["shot_start_time"] = shot_start_time
    if camera_id is not None:
        out["camera_id"] = camera_id
    if frame_index is not None:
        out["frame_index"] = frame_index
    if error_reasons is not None:
        out["error_reasons"] = error_reasons
    return out


def make_result_subtask(
    *,
    status: str = "Completed",
    num_shots: int | None = None,
    completed_date: datetime | str | None = DEFAULT_CREATED_DATE,
    user_metadata: str | None = "{}",
    shot_results: list[dict] | None = None,
) -> dict:
    """One entry in `element["subtasks"]`.

    Note: the live API subtask carries no `subtask_index`/`subtask_id`. The
    index appears only on each `shot_results` entry. Code that needs to derive
    the index must read it from the shot results.
    """
    if shot_results is None:
        shot_results = [make_shot_result_dict()]
    out: dict[str, Any] = {
        "status": status,
        "shot_results": shot_results,
    }
    if num_shots is not None:
        out["num_shots"] = num_shots
    if completed_date is not None:
        out["completed_date"] = completed_date
    if user_metadata is not None:
        out["subtask_metadata"] = {"user_metadata": user_metadata}
    return out


def make_result_element(
    *,
    task_id: str = DEFAULT_TASK_ID,
    status: str = "Completed",
    subtasks: list[dict] | None = None,
) -> dict:
    if subtasks is None:
        subtasks = [make_result_subtask()]
    return {
        "task_id": task_id,
        "status": status,
        "subtasks": subtasks,
    }


def make_result_envelope(
    *,
    elements: list[dict] | None = None,
    page: int = 0,
    total: int = 1,
) -> dict:
    """Top-level dict returned by `ResultsClient.get`."""
    if elements is None:
        elements = [make_result_element()]
    return {
        "elements": elements,
        "page": page,
        "total": total,
    }


# --------------------------------------------------------------------------- #
# Fake clients                                                                #
# --------------------------------------------------------------------------- #
# Every fake supports the context-manager protocol that the real qlam clients
# use, records each call, and accepts pre-built return values. Tests that need
# to fail a call set `*_raises` instead of `*_return`.


class _RecordingContextManager:
    """Mixin: context-manager protocol + a `calls` list."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def _record(self, name: str, **kwargs: Any) -> None:
        self.calls.append((name, kwargs))


class FakeGroupsClient(_RecordingContextManager):
    """Replaces `qlam_core.plugins.groups.api.client.GroupsClient` in tests."""

    def __init__(
        self,
        app_context: Any = None,
        *,
        get_returns: dict[UUID, GroupResponse] | None = None,
    ) -> None:
        _RecordingContextManager.__init__(self)
        self.app_context = app_context
        self.get_returns = {} if get_returns is None else get_returns

    def get(self, id):  # noqa: A002
        self._record("get", id=id)
        if id not in self.get_returns:
            raise AssertionError(f"FakeGroupsClient.get called with unknown id {id}")
        return self.get_returns[id]


class FakeUsersClient(_RecordingContextManager):
    """Replaces `qlam_core.plugins.users.api.client.UsersClient` in tests."""

    def __init__(
        self,
        app_context: Any = None,
        *,
        get_groups_return: list[GroupAssignment] | None = None,
    ) -> None:
        _RecordingContextManager.__init__(self)
        self.app_context = app_context
        self.get_groups_return = [] if get_groups_return is None else get_groups_return

    def get_groups(self, id):  # noqa: A002
        self._record("get_groups", id=id)
        return self.get_groups_return


class FakeTasksClient(_RecordingContextManager):
    """Replaces `qlam_core.plugins.tasks.api.client.TasksClient` in tests.

    Pass typed `Task` instances (e.g. via `make_task(...)`) for `get_return` /
    `create_return` to keep wire-shape validation honest; raise from `*_raises`
    to simulate API errors. Cancellation is no-op by default and returns None.
    """

    def __init__(
        self,
        app_context: Any = None,
        *,
        get_return: Task | Callable[[str], Task] | None = None,
        create_return: Task | Callable[[Any], Task] | None = None,
        cancel_raises: Exception | None = None,
    ) -> None:
        _RecordingContextManager.__init__(self)
        self.app_context = app_context
        self.get_return = get_return
        self.create_return = create_return
        self.cancel_raises = cancel_raises

    def get(self, id):  # noqa: A002
        self._record("get", id=id)
        ret = self.get_return
        if ret is None:
            raise AssertionError("FakeTasksClient.get called but no get_return set")
        if callable(ret):
            return ret(id)
        return ret

    def create(self, body):
        self._record("create", body=body)
        ret = self.create_return
        if ret is None:
            raise AssertionError(
                "FakeTasksClient.create called but no create_return set"
            )
        if callable(ret):
            return ret(body)
        return ret

    def cancel(self, id):  # noqa: A002
        self._record("cancel", id=id)
        if self.cancel_raises is not None:
            raise self.cancel_raises
        return None


class FakeDefinitionsClient(_RecordingContextManager):
    """Replaces `qlam_core.plugins.definitions.api.client.DefinitionsClient`.

    Production code at `future.from_task_id` consumes the result via
    `storage.add_task_definition(...)` — it accepts both `TaskDefinitionResponse`
    and a bare `TaskDefinition` because the storage backend only reads
    `program_language`, `programs`, and `subtasks`. Tests are free to pass
    either; both are supported.
    """

    def __init__(
        self,
        app_context: Any = None,
        *,
        get_return: TaskDefinitionResponse | TaskDefinition | None = None,
    ) -> None:
        _RecordingContextManager.__init__(self)
        self.app_context = app_context
        self.get_return = get_return

    def get(self, id):  # noqa: A002
        self._record("get", id=id)
        if self.get_return is None:
            raise AssertionError(
                "FakeDefinitionsClient.get called but no get_return set"
            )
        return self.get_return


class FakeCompilationsClient(_RecordingContextManager):
    def __init__(
        self,
        app_context: Any = None,
        *,
        get_return: PublicCompilation | None = None,
    ) -> None:
        _RecordingContextManager.__init__(self)
        self.app_context = app_context
        self.get_return = get_return

    def get(self, id):  # noqa: A002
        self._record("get", id=id)
        if self.get_return is None:
            raise AssertionError(
                "FakeCompilationsClient.get called but no get_return set"
            )
        return self.get_return


class FakeAuthClient(_RecordingContextManager):
    """Replaces `qlam_core.auth.client.AuthClient` in tests.

    Records each `refresh_credentials` call and returns the configured
    mapping. The default `refresh_result={"qlam": True}` simulates a
    successful refresh; pass `{"qlam": False}` (or an empty dict) to
    simulate a refresh that produced no fresh credentials.
    """

    def __init__(
        self,
        app_context: Any = None,
        *,
        refresh_result: dict[str, bool] | None = None,
        user_info: UserInfo | None = None,
    ) -> None:
        _RecordingContextManager.__init__(self)
        self.app_context = app_context
        self.refresh_result = (
            {"qlam": True} if refresh_result is None else refresh_result
        )
        self.user_info = make_user_info() if user_info is None else user_info

    def get_user_info(self, provider: str | None = None) -> UserInfo:
        self._record("get_user_info", provider=provider)
        return self.user_info

    def refresh_credentials(self, provider: str | None = None, *, force: bool = False):
        self._record("refresh_credentials", provider=provider, force=force)
        return self.refresh_result


class FakeResultsClient(_RecordingContextManager):
    """Replaces `qlam_core.plugins.results.api.client.ResultsClient`.

    Tests can either:
    * pass a single `envelope_return` (returned for every call), or
    * pass `envelope_fn` to compute the envelope from kwargs (used by tests
      that depend on `shots_page` / `page` for pagination behavior).
    """

    def __init__(
        self,
        app_context: Any = None,
        *,
        envelope_return: dict | None = None,
        envelope_fn: Callable[..., dict] | None = None,
    ) -> None:
        _RecordingContextManager.__init__(self)
        self.app_context = app_context
        self.envelope_return = envelope_return
        self.envelope_fn = envelope_fn

    def get(self, **kwargs):
        self._record("get", **kwargs)
        if self.envelope_fn is not None:
            return self.envelope_fn(**kwargs)
        if self.envelope_return is not None:
            return self.envelope_return
        raise AssertionError(
            "FakeResultsClient.get called but no envelope_return / envelope_fn set"
        )
