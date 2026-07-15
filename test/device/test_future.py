import importlib
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest
from qlam_core.errors import APIError
from qlam_core.plugins.tasks.api.tasks_models import TaskStatus

from bloqade.core.device.future import ApiFetchOptions, Future
from bloqade.core.device.local_storage import (
    DictStorage,
    ShotFilter,
)
from bloqade.core.device.result import Result

from .fixtures import local, remote

future_mod = importlib.import_module("bloqade.core.device.future")
mixins_mod = importlib.import_module("bloqade.core.device.mixins")

CREATION_TIME = local.CREATION_TIME


class CustomResult(Result):
    pass


class DefaultContextFuture(Future):
    context_name = "class-context"


def make_task_definition(content: str = "program"):
    return remote.make_task_definition(programs=[remote.make_program(content=content)])


make_shot = local.make_shot
assert_shots_equal = local.assert_shots_equal


def test_resolve_context_name_uses_explicit_or_class_default():
    assert Future._resolve_context_name("explicit") == "explicit"
    assert DefaultContextFuture._resolve_context_name(None) == "class-context"

    with pytest.raises(ValueError, match="no default context_name"):
        Future._resolve_context_name(None)


def test_results_from_storage_defaults_to_task_detected_filter_and_result_class():
    storage = DictStorage()
    future = Future(
        task_id="task-1",
        storage=storage,
        context_name="ctx",
        result_cls=CustomResult,
    )

    result = future.results_from_storage()

    assert isinstance(result, CustomResult)
    assert result.storage is storage
    assert result.shot_filter == ShotFilter(
        task_ids=("task-1",),
        frame_type="DETECTED",
    )


def test_results_from_storage_uses_explicit_filter():
    shot_filter = ShotFilter(task_ids=("task-2",), frame_type="raw")
    future = Future(task_id="task-1", storage=DictStorage(), context_name="ctx")

    result = future.results_from_storage(shot_filter=shot_filter)

    assert result.shot_filter is shot_filter


def test_from_storage_uses_single_stored_task_id_and_optional_new_storage():
    storage = DictStorage()
    new_storage = DictStorage()
    storage.add_task_definition("task-1", make_task_definition(), CREATION_TIME)
    fetch_options = ApiFetchOptions(shots_per_fetch=5)

    future = Future.from_storage(
        storage=storage,
        new_storage=new_storage,
        fetch_options=fetch_options,
        context_name="ctx",
    )

    assert future.task_id == "task-1"
    assert future.storage is new_storage
    assert future.fetch_options is fetch_options
    assert future.context_name == "ctx"


def test_from_storage_preserves_explicit_qpu_mode():
    storage = DictStorage()
    storage.add_task_definition("task-1", make_task_definition(), CREATION_TIME)

    future = Future.from_storage(
        storage=storage,
        context_name="ctx",
        qpu_mode="squin-256q",
    )

    assert future.qpu_mode == "squin-256q"


def test_from_storage_rejects_empty_storage():
    with pytest.raises(ValueError, match="Found no task IDs"):
        Future.from_storage(storage=DictStorage(), context_name="ctx")


def test_from_storage_requires_task_id_when_storage_has_multiple_tasks():
    storage = DictStorage()
    storage.add_task_definition("task-1", make_task_definition("p1"), CREATION_TIME)
    storage.add_task_definition("task-2", make_task_definition("p2"), CREATION_TIME)

    with pytest.raises(ValueError, match="More than one task ID found"):
        Future.from_storage(storage=storage, context_name="ctx")


def test_from_storage_rejects_missing_explicit_task_id():
    storage = DictStorage()
    storage.add_task_definition("task-1", make_task_definition(), CREATION_TIME)

    with pytest.raises(ValueError, match="Task with ID missing not found"):
        Future.from_storage(storage=storage, task_id="missing", context_name="ctx")


def test_from_task_id_fetches_definition_and_stores_it(monkeypatch):
    storage = DictStorage()
    task_definition_response = remote.make_task_definition_response(
        id="11111111-1111-1111-1111-111111111111",
    )
    task = remote.make_task(
        id="task-1",
        definition_id="11111111-1111-1111-1111-111111111111",
        task_status=TaskStatus.CREATED,
        created_date=CREATION_TIME,
    )
    tasks_client = remote.FakeTasksClient(get_return=task)
    defs_client = remote.FakeDefinitionsClient(get_return=task_definition_response)
    auth_context_names = []

    def authenticate(auth):
        auth_context_names.append(auth.context_name)

    monkeypatch.setattr(future_mod.AuthMixin, "authenticate", authenticate)
    monkeypatch.setattr(future_mod, "TasksClient", lambda app_context: tasks_client)
    monkeypatch.setattr(
        future_mod, "DefinitionsClient", lambda app_context: defs_client
    )

    future = Future.from_task_id(
        task_id="task-1",
        storage=storage,
        context_name="ctx",
    )

    assert auth_context_names == ["ctx"]
    assert future.task_id == "task-1"
    assert future.storage is storage
    assert tasks_client.calls == [("get", {"id": "task-1"})]
    assert defs_client.calls == [
        ("get", {"id": "11111111-1111-1111-1111-111111111111"})
    ]
    stored_def = storage.get_task_definition("task-1")
    assert stored_def.program_language == task_definition_response.program_language
    assert [p.content for p in stored_def.programs] == [
        p.content for p in task_definition_response.programs
    ]
    assert storage.get_task_creation_time("task-1") == CREATION_TIME


def test_from_task_id_passes_explicit_qpu_mode_to_backend_fetches(monkeypatch):
    task_definition_response = remote.make_task_definition_response(
        id="11111111-1111-1111-1111-111111111111",
    )
    task = remote.make_task(
        id="task-1",
        definition_id="11111111-1111-1111-1111-111111111111",
        task_status=TaskStatus.CREATED,
        created_date=CREATION_TIME,
    )
    tasks_client = remote.FakeTasksClient(get_return=task)
    defs_client = remote.FakeDefinitionsClient(get_return=task_definition_response)

    monkeypatch.setattr(future_mod.AuthMixin, "authenticate", lambda auth: None)
    monkeypatch.setattr(future_mod, "TasksClient", lambda app_context: tasks_client)
    monkeypatch.setattr(
        future_mod, "DefinitionsClient", lambda app_context: defs_client
    )

    future = Future.from_task_id(
        task_id="task-1",
        context_name="ctx",
        qpu_mode="squin-256q",
    )

    assert future.qpu_mode == "squin-256q"
    assert tasks_client.calls == [("get", {"id": "task-1", "qpu_mode": "squin-256q"})]
    assert defs_client.calls == [
        (
            "get",
            {
                "id": "11111111-1111-1111-1111-111111111111",
                "qpu_mode": "squin-256q",
            },
        )
    ]


def test_future_defaults_to_fresh_dict_storage_per_instance():
    first = Future(task_id="task-1", context_name="ctx")
    second = Future(task_id="task-2", context_name="ctx")

    assert isinstance(first.storage, DictStorage)
    assert isinstance(second.storage, DictStorage)
    assert first.storage is not second.storage


def test_from_task_id_defaults_to_fresh_dict_storage_when_storage_is_none(monkeypatch):
    task_definition_response = remote.make_task_definition_response(
        id="22222222-2222-2222-2222-222222222222",
    )
    task = remote.make_task(
        id="task-1",
        definition_id="22222222-2222-2222-2222-222222222222",
        task_status=TaskStatus.CREATED,
        created_date=CREATION_TIME,
    )
    tasks_client = remote.FakeTasksClient(get_return=task)
    defs_client = remote.FakeDefinitionsClient(get_return=task_definition_response)

    monkeypatch.setattr(future_mod.AuthMixin, "authenticate", lambda auth: None)
    monkeypatch.setattr(future_mod, "TasksClient", lambda app_context: tasks_client)
    monkeypatch.setattr(
        future_mod, "DefinitionsClient", lambda app_context: defs_client
    )

    future = Future.from_task_id(task_id="task-1", context_name="ctx")

    assert isinstance(future.storage, DictStorage)
    stored_def = future.storage.get_task_definition("task-1")
    assert stored_def.program_language == task_definition_response.program_language
    assert future.storage.get_task_creation_time("task-1") == CREATION_TIME


def test_status_done_and_cancelled_delegate_to_task_status(monkeypatch):
    future = Future(task_id="task-1", storage=DictStorage(), context_name="ctx")
    monkeypatch.setattr(
        future,
        "get_task",
        lambda: remote.make_task(task_status=TaskStatus.CANCELLED),
    )

    assert future.status() == TaskStatus.CANCELLED
    assert future.done() is True
    assert future.cancelled() is True


def test_wait_for_completion_returns_completed_status(monkeypatch):
    future = Future(task_id="task-1", storage=DictStorage(), context_name="ctx")
    monkeypatch.setattr(future, "status", lambda: TaskStatus.COMPLETED)

    assert future._wait_for_completion(timeout=0) == TaskStatus.COMPLETED


def test_wait_for_completion_rejects_cancelled_and_failed_tasks(monkeypatch):
    future = Future(task_id="task-1", storage=DictStorage(), context_name="ctx")
    monkeypatch.setattr(future, "status", lambda: TaskStatus.CANCELLED)

    with pytest.raises(ValueError, match="cancelled"):
        future._wait_for_completion(timeout=0)

    monkeypatch.setattr(future, "status", lambda: TaskStatus.FAILED)
    monkeypatch.setattr(
        future,
        "get_task",
        lambda: remote.make_task(
            task_status=TaskStatus.FAILED,
            error_reasons=["hardware unavailable"],
        ),
    )

    with pytest.raises(ValueError, match="hardware unavailable"):
        future._wait_for_completion(timeout=0)


def test_wait_for_completion_times_out_before_terminal_status(monkeypatch):
    future = Future(task_id="task-1", storage=DictStorage(), context_name="ctx")
    monkeypatch.setattr(future, "status", lambda: TaskStatus.SCHEDULED)

    with pytest.raises(TimeoutError, match="Timed out"):
        future._wait_for_completion(timeout=0)


def test_result_waits_fetches_and_returns_storage_result(monkeypatch):
    future = Future(task_id="task-1", storage=DictStorage(), context_name="ctx")
    calls = []

    monkeypatch.setattr(
        future,
        "_wait_for_completion",
        lambda timeout: calls.append(("wait", timeout)),
    )
    monkeypatch.setattr(future, "fetch", lambda: calls.append(("fetch", None)))

    result = future.result(timeout=1.25)

    assert calls == [("wait", 1.25), ("fetch", None)]
    assert result.shot_filter.task_ids == ("task-1",)


def test_partial_result_fetches_without_waiting(monkeypatch):
    future = Future(task_id="task-1", storage=DictStorage(), context_name="ctx")
    calls = []
    monkeypatch.setattr(future, "fetch", lambda: calls.append("fetch"))

    result = future.partial_result()

    assert calls == ["fetch"]
    assert result.shot_filter.task_ids == ("task-1",)


def test_export_to_copies_filtered_shots_and_task_definitions():
    source = DictStorage()
    destination = DictStorage()
    task_definition = make_task_definition()
    source.add_task_definition("task-1", task_definition, CREATION_TIME)
    source.add_task_definition("task-2", make_task_definition("other"), CREATION_TIME)
    kept_shots = [
        make_shot(task_id="task-1", shot_index=0),
        make_shot(task_id="task-1", shot_index=1),
    ]
    other_shot = make_shot(task_id="task-2", shot_index=0)
    source.add_shots([*kept_shots, other_shot])
    future = Future(task_id="task-1", storage=source, context_name="ctx")

    future.export_to(
        destination,
        chunk_size=1,
        shot_filter=ShotFilter(task_ids=("task-1",)),
    )

    assert_shots_equal(list(destination.get_shots()), kept_shots)
    assert destination.get_task_definition("task-1") == task_definition
    assert destination.task_ids() == {"task-1"}


def test_fetch_and_export_to_fetches_before_exporting(monkeypatch):
    source = DictStorage()
    destination = DictStorage()
    task_definition = make_task_definition()
    source.add_task_definition("task-1", task_definition, CREATION_TIME)
    shot = make_shot(task_id="task-1", shot_index=0)
    future = Future(task_id="task-1", storage=source, context_name="ctx")

    def fetch():
        source.add_shots([shot])

    monkeypatch.setattr(future, "fetch", fetch)

    future.fetch_and_export_to(destination)

    assert_shots_equal(list(destination.get_shots()), [shot])
    assert destination.get_task_definition("task-1") == task_definition


def test_fetch_subtask_page_parses_results_and_tracks_first_incomplete_page():
    storage = DictStorage()
    future = Future(
        task_id="task-1",
        storage=storage,
        context_name="ctx",
        fetch_options=ApiFetchOptions(
            subtasks_per_fetch=10,
            shots_per_fetch=100,
        ),
    )

    # Note: the real API returns title-case "Completed"/"Detected" — the
    # production code uppercases on ingest, so either case works. We mix one
    # title-case ("Completed") + one upper-case ("SCHEDULED") here to make the
    # case-normalization seam explicit.
    envelope = remote.make_result_envelope(
        total=1,
        elements=[
            remote.make_result_element(
                subtasks=[
                    remote.make_result_subtask(
                        status="Completed",
                        completed_date=None,
                        user_metadata=None,
                        shot_results=[
                            remote.make_shot_result_dict(
                                shot_index=0,
                                subtask_shot_index=0,
                                subtask_index=0,
                                frame_type="detected",
                                measurement_values=[1, 0, 1],
                            )
                        ],
                    ),
                    remote.make_result_subtask(
                        status="SCHEDULED",
                        completed_date=None,
                        user_metadata=None,
                        shot_results=[
                            remote.make_shot_result_dict(
                                shot_index=1,
                                subtask_shot_index=0,
                                subtask_index=1,
                                frame_type="raw",
                                measurement_values=[0, 1, 0],
                            )
                        ],
                    ),
                ]
            )
        ],
    )

    client = remote.FakeResultsClient(envelope_return=envelope)

    done = future._fetch_subtask_page(client=client, subtask_page=3)  # type: ignore

    assert done is True
    assert future._first_incomplete_subtask_page == 3
    assert client.calls == [
        (
            "get",
            {
                "id": "task-1",
                "page": 3,
                "size": 10,
                "sort": "completed_date,asc",
                "shots_page": 0,
                "shots_size": 100,
            },
        )
    ]
    shots = sorted(storage.get_shots(), key=lambda shot: shot.shot_index)
    assert [shot.frame_type for shot in shots] == ["DETECTED", "RAW"]
    np.testing.assert_array_equal(shots[0].bitstring, np.array([True, False, True]))
    np.testing.assert_array_equal(shots[1].bitstring, np.array([False, True, False]))


def test_future_passes_explicit_qpu_mode_to_backend_clients(monkeypatch):
    returned_task = remote.make_task(task_status=TaskStatus.CREATED)
    returned_compilation = remote.make_public_compilation()
    future = Future(
        task_id="task-1",
        storage=DictStorage(),
        context_name="ctx",
        qpu_mode="squin-256q",
        fetch_options=ApiFetchOptions(subtasks_per_fetch=10, shots_per_fetch=100),
    )

    tasks_client = remote.FakeTasksClient(get_return=returned_task)
    compilations_client = remote.FakeCompilationsClient(get_return=returned_compilation)
    results_client = remote.FakeResultsClient(
        envelope_return=remote.make_result_envelope(elements=[], total=0)
    )

    monkeypatch.setattr(future, "authenticate", lambda: None)
    monkeypatch.setattr(future_mod, "TasksClient", lambda app_context: tasks_client)
    monkeypatch.setattr(
        future_mod, "CompilationsClient", lambda app_context: compilations_client
    )
    monkeypatch.setattr(future_mod, "ResultsClient", lambda app_context: results_client)

    assert future.get_task() is returned_task
    assert future.cancel() is None
    assert (
        future.get_compilation(compilation_id="compilation-1") is returned_compilation
    )
    future.fetch()

    assert tasks_client.calls == [
        ("get", {"id": "task-1", "qpu_mode": "squin-256q"}),
        ("cancel", {"id": "task-1", "qpu_mode": "squin-256q"}),
    ]
    assert compilations_client.calls == [
        ("get", {"id": "compilation-1", "qpu_mode": "squin-256q"})
    ]
    assert results_client.calls == [
        (
            "get",
            {
                "id": "task-1",
                "qpu_mode": "squin-256q",
                "page": 0,
                "size": 10,
                "sort": "completed_date,asc",
                "shots_page": 0,
                "shots_size": 100,
            },
        )
    ]


def test_fetch_subtask_page_persists_completed_dates_from_api_schema(storage):
    # NOTE: the real results API subtask object carries NO subtask identifier
    # (no `subtask_index`, no `subtask_id`); the `subtask_index` is present only
    # on each `shot_results` entry. Verified by recording a live response.
    storage.add_task_definition(
        "task-1",
        remote.make_task_definition(
            programs=[remote.make_program(content="program")],
            subtasks=[
                remote.make_subtask(num_shots=1),
                remote.make_subtask(num_shots=1),
                remote.make_subtask(num_shots=1),
            ],
        ),
        CREATION_TIME,
    )
    future = Future(
        task_id="task-1",
        storage=storage,
        context_name="ctx",
        fetch_options=ApiFetchOptions(subtasks_per_fetch=10, shots_per_fetch=100),
    )

    completed_at_0 = datetime(2026, 5, 21, 10, 0, 0, tzinfo=timezone.utc)
    completed_at_1_iso = "2026-05-21T11:00:00+00:00"

    envelope = remote.make_result_envelope(
        total=1,
        elements=[
            remote.make_result_element(
                subtasks=[
                    remote.make_result_subtask(
                        completed_date=completed_at_0,
                        shot_results=[
                            remote.make_shot_result_dict(shot_index=0, subtask_index=0)
                        ],
                    ),
                    remote.make_result_subtask(
                        completed_date=completed_at_1_iso,
                        shot_results=[
                            remote.make_shot_result_dict(shot_index=1, subtask_index=1)
                        ],
                    ),
                    remote.make_result_subtask(
                        status="Scheduled",
                        completed_date=None,
                        shot_results=[],
                    ),
                ]
            )
        ],
    )

    client = remote.FakeResultsClient(envelope_return=envelope)
    future._fetch_subtask_page(client=client, subtask_page=0)  # type: ignore

    subtasks = sorted(
        storage.get_subtasks(), key=lambda subtask: subtask["subtask_index"]
    )
    assert subtasks[0]["completed_date"] == completed_at_0
    assert subtasks[1]["completed_date"] == datetime.fromisoformat(completed_at_1_iso)
    assert subtasks[2]["completed_date"] is None


def test_fetch_subtask_page_updates_completed_dates_only_on_first_shot_page():
    # completed_date never changes across shot pages, and on later shot pages a
    # subtask can come back with empty shot_results (so its index can't be
    # derived). Only run the update on the first shot page.
    dict_storage = DictStorage()
    dict_storage.add_task_definition(
        "task-1",
        remote.make_task_definition(
            programs=[remote.make_program(content="program")],
            subtasks=[remote.make_subtask(num_shots=2)],
        ),
        CREATION_TIME,
    )
    future = Future(
        task_id="task-1",
        storage=dict_storage,
        context_name="ctx",
        fetch_options=ApiFetchOptions(subtasks_per_fetch=10, shots_per_fetch=2),
    )

    completed_at = datetime(2026, 5, 21, 10, 0, 0, tzinfo=timezone.utc)

    update_calls = []
    original_update = dict_storage.update_subtasks_completed_date

    def spy(task_id, subtasks):
        update_calls.append(subtasks)
        return original_update(task_id=task_id, subtasks=subtasks)

    dict_storage.update_subtasks_completed_date = spy

    def envelope_fn(**kwargs):
        # A full first shot page (2 == shots_per_fetch) forces a second fetch;
        # that second page returns the subtask with no shots left.
        if kwargs["shots_page"] == 0:
            shot_results = [
                remote.make_shot_result_dict(shot_index=0, subtask_index=0),
                remote.make_shot_result_dict(shot_index=1, subtask_index=0),
            ]
        else:
            shot_results = []
        return remote.make_result_envelope(
            total=1,
            elements=[
                remote.make_result_element(
                    subtasks=[
                        remote.make_result_subtask(
                            completed_date=completed_at,
                            shot_results=shot_results,
                        )
                    ]
                )
            ],
        )

    client = remote.FakeResultsClient(envelope_fn=envelope_fn)
    future._fetch_subtask_page(client=client, subtask_page=0)  # type: ignore

    # Two shot pages were fetched, but the completed-date update ran only once.
    assert [c[1]["shots_page"] for c in client.calls] == [0, 1]
    assert len(update_calls) == 1
    assert dict_storage.get_subtasks()[0]["completed_date"] == completed_at


def test_cancel_warns_when_backend_cancel_raises(monkeypatch):
    future = Future(task_id="task-1", storage=DictStorage(), context_name="ctx")
    monkeypatch.setattr(future, "authenticate", lambda: None)

    client = remote.FakeTasksClient(
        cancel_raises=RuntimeError("cannot cancel task-1"),
    )
    monkeypatch.setattr(future_mod, "TasksClient", lambda app_context: client)

    with pytest.warns(UserWarning, match="cannot cancel task-1"):
        assert future.cancel() is None


def test_get_task_authenticates_and_returns_backend_task(monkeypatch):
    returned_task = remote.make_task(task_status=TaskStatus.CREATED)
    future = Future(task_id="task-1", storage=DictStorage(), context_name="ctx")
    authenticated = []

    client = remote.FakeTasksClient(get_return=returned_task)

    monkeypatch.setattr(future, "authenticate", lambda: authenticated.append(True))
    monkeypatch.setattr(future_mod, "TasksClient", lambda app_context: client)

    assert future.get_task() is returned_task
    assert authenticated == [True]


def test_get_compilation_uses_task_compilation_id_when_omitted(monkeypatch):
    returned_compilation = remote.make_public_compilation()
    future = Future(task_id="task-1", storage=DictStorage(), context_name="ctx")
    authenticated = []

    client = remote.FakeCompilationsClient(get_return=returned_compilation)

    monkeypatch.setattr(future, "authenticate", lambda: authenticated.append(True))
    monkeypatch.setattr(
        future,
        "get_task",
        lambda: SimpleNamespace(compilation_id=str(returned_compilation.id)),
    )
    monkeypatch.setattr(future_mod, "CompilationsClient", lambda app_context: client)

    assert future.get_compilation() is returned_compilation
    assert authenticated == [True]
    assert client.calls == [("get", {"id": str(returned_compilation.id)})]


def test_fetch_retries_only_the_failing_page_on_403(monkeypatch):
    future = Future(task_id="task-1", storage=DictStorage(), context_name="ctx")

    class _DummyResultsClient:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    attempts = []

    def fake_fetch_subtask_page(self, *, client, subtask_page):
        attempts.append(subtask_page)
        if subtask_page == 1 and attempts.count(1) == 1:
            raise APIError(message="permission denied", status_code=403)
        return subtask_page >= 2

    auth_client = remote.FakeAuthClient(refresh_result={"qlam": True})

    monkeypatch.setattr(future_mod.AuthMixin, "authenticate", lambda auth: None)
    monkeypatch.setattr(
        future_mod, "ResultsClient", lambda app_context: _DummyResultsClient()
    )
    monkeypatch.setattr(Future, "_fetch_subtask_page", fake_fetch_subtask_page)
    monkeypatch.setattr(mixins_mod, "AuthClient", lambda app_context: auth_client)

    future.fetch()

    assert attempts == [0, 1, 1, 2]
    assert [name for name, _ in auth_client.calls] == ["refresh_credentials"]
