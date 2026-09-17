"""Compatibility checks for Bloqade payloads and the installed qlam-core models."""

import importlib.metadata
import json
import re
from pathlib import Path
from uuid import UUID

import pytest
from kirin.prelude import basic_no_opt
from kirin.serialization import JSONSerializer
from qlam_core.plugins.compilations.api.compilations_models import PublicCompilation
from qlam_core.plugins.definitions.api.definitions_models import (
    TaskDefinitionResponse,
)
from qlam_core.plugins.task_profiles.api.task_profiles_models import (
    TaskProfileRequest,
)
from qlam_core.plugins.tasks.api.tasks_models import (
    CompilationReference,
    Task,
    TaskCreationRequest,
    TaskDefinition,
    TaskDefinitionReference,
)

from bloqade.core.device.local_storage import DictStorage, SQLiteStorage
from bloqade.core.device.task import (
    KernelBatchTask,
    ParameterScanTask,
    SingleKernelTask,
)
from bloqade.core.device.task_builder import FinalizeContext, TaskBuilder

from .fixtures import local, remote

EXAMPLES = Path(__file__).parent / "fixtures" / "examples"
PROFILE_ID = UUID("12345678-1234-5678-1234-567812345678")
GROUP_ID = UUID("11111111-1111-1111-1111-111111111111")


@basic_no_opt
def main():
    return


@basic_no_opt
def other():
    return


# --------------------------------------------------------------------------- #
# Validation runs against the *installed* qlam-core, so pin down which one
# --------------------------------------------------------------------------- #


def _pinned_qlam_core_minor() -> tuple[int, int]:
    """Read the `qlam-core~=X.Y.Z` pin from bloqade-core's own metadata."""
    requirements = importlib.metadata.requires("bloqade-core") or []
    (pin,) = [r for r in requirements if r.startswith("qlam-core")]
    match = re.search(r"~=\s*(\d+)\.(\d+)", pin)
    assert match, f"unexpected qlam-core pin format: {pin!r}"
    return int(match.group(1)), int(match.group(2))


def test_installed_qlam_core_matches_the_pin():
    """Every check below runs against the installed models; make sure they are
    the ones this branch was written against, not a stale environment."""
    installed = importlib.metadata.version("qlam-core")
    major, minor = (int(x) for x in installed.split(".")[:2])

    assert (major, minor) == _pinned_qlam_core_minor(), (
        f"installed qlam-core {installed} does not match the pyproject pin; "
        "run `uv sync` before trusting the wire-shape tests"
    )


def test_models_carry_the_0_7_0_wire_shapes():
    """Shapes introduced in qlam-core 0.7.0 that this branch depends on."""
    # profile_id on every task-creation shape and on the Task record
    for model in (TaskDefinition, TaskDefinitionReference, CompilationReference, Task):
        assert "profile_id" in model.model_fields, model
    # group became required on stored definitions and compilations
    assert TaskDefinitionResponse.model_fields["group"].is_required()
    assert PublicCompilation.model_fields["group"].is_required()
    # task profiles are a new resource: free-form content with two known keys
    assert {"name", "dry_run"} <= TaskProfileRequest.model_fields.keys()
    assert TaskProfileRequest.model_config.get("extra") == "allow"


# --------------------------------------------------------------------------- #
# Live API captures must still parse under the current qlam-core models
# --------------------------------------------------------------------------- #

_STALE = pytest.mark.xfail(
    strict=True,
    reason=(
        "captured against qlam-core 0.6.x, before `group` became a required "
        "field in 0.7.0 — re-capture this example from the live API"
    ),
)


@pytest.mark.parametrize(
    ("filename", "model"),
    [
        pytest.param("task_completed.json", Task, id="task_completed"),
        pytest.param(
            "task_execution_completed.json", Task, id="task_execution_completed"
        ),
        pytest.param("task_failed.json", Task, id="task_failed"),
        pytest.param(
            "task_definition_response_completed.json",
            TaskDefinitionResponse,
            marks=_STALE,
            id="definition_response_completed",
        ),
        pytest.param(
            "task_definition_response_failed.json",
            TaskDefinitionResponse,
            marks=_STALE,
            id="definition_response_failed",
        ),
        pytest.param(
            "public_compilation_succeeded.json",
            PublicCompilation,
            marks=_STALE,
            id="compilation_succeeded",
        ),
        pytest.param(
            "public_compilation_failed.json",
            PublicCompilation,
            marks=_STALE,
            id="compilation_failed",
        ),
    ],
)
def test_live_api_examples_match_current_models(filename, model):
    model.model_validate_json((EXAMPLES / filename).read_text(), strict=True)


# --------------------------------------------------------------------------- #
# What bloqade sends must be wire-valid
# --------------------------------------------------------------------------- #


def _request(definition: TaskDefinition) -> dict:
    return TaskCreationRequest(root=definition).model_dump(
        mode="json", exclude_none=True
    )


@pytest.mark.parametrize(
    "task",
    [
        pytest.param(
            SingleKernelTask(
                context_name="ctx", program_language="squin", kernel=main, num_shots=3
            ),
            id="single",
        ),
        pytest.param(
            SingleKernelTask(
                context_name="ctx",
                program_language="squin",
                kernel=main,
                num_shots=3,
                arguments={"theta": 0.5},
                metadata={"tag": "t"},
                group="grp",
                profile_id=str(PROFILE_ID),
            ),
            id="single-full",
        ),
        pytest.param(
            KernelBatchTask(
                context_name="ctx",
                program_language="squin",
                kernels=[main, other],
                num_shots=[1, 2],
            ),
            id="batch",
        ),
        pytest.param(
            ParameterScanTask(
                context_name="ctx",
                program_language="squin",
                kernel=main,
                arguments=[{"x": 1.0}, {"x": 2.0}],
                num_shots=5,
                metadata=[{"i": 0}, {"i": 1}],
            ),
            id="scan",
        ),
    ],
)
def test_legacy_task_shapes_emit_schema_valid_requests(task):
    TaskCreationRequest.model_validate_json(
        json.dumps(_request(task.create_task_definition())), strict=True
    )


def test_task_builder_finalize_emits_schema_valid_definition():
    builder = TaskBuilder()
    builder.add_subtask(main, 10, {"kind": "a"})
    builder.add_subtask(other, 20)
    builder.add_subtask(main, 30)  # reuses program 0
    ctx = FinalizeContext(
        program_language="squin",
        language_version="0.1.0",
        kernel_serializer=JSONSerializer(),
    )

    definition = builder._finalize(ctx)
    payload = _request(definition)

    TaskCreationRequest.model_validate_json(json.dumps(payload), strict=True)
    # The builder's display-only name must never reach the wire.
    assert all("kernel_name" not in st for st in payload["subtasks"])


@pytest.mark.parametrize("backend", ["dict", "sqlite"])
def test_storage_round_trip_preserves_wire_shape(backend, tmp_path):
    storage = (
        DictStorage()
        if backend == "dict"
        else SQLiteStorage(str(tmp_path / "s.sqlite"))
    )
    original = remote.make_task_definition(group_id=GROUP_ID, profile_id=PROFILE_ID)
    storage.add_task_definition("task-1", original, local.CREATION_TIME)

    restored = storage.get_task_definition("task-1")

    assert restored.model_dump(mode="json", exclude_none=True) == original.model_dump(
        mode="json", exclude_none=True
    )


def test_task_profile_content_stays_free_form():
    """Server stores profile content verbatim; do not check for undeclared fields."""
    payload = TaskProfileRequest.model_validate(
        {"name": "afm-sweep", "layers": 3, "annotation": None}
    ).model_dump(mode="json", exclude_none=True)

    assert payload == {"name": "afm-sweep", "layers": 3}  # exclude_none drops the null
