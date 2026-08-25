import base64
import json
from typing import cast

import pytest
from kirin import ir
from kirin.ir.exception import ValidationError
from kirin.prelude import basic_no_opt
from kirin.serialization import JSONSerializer
from kirin.validation import ValidationSuite
from kirin.validation.validationpass import ValidationResult

from bloqade.core.device.task import KernelSerializer
from bloqade.core.device.task_builder import (
    FinalizeContext,
    SubtaskValidationError,
    TaskBuilder,
    TaskFinalizeError,
    _format_validation_errors,
)


@basic_no_opt
def no_args():
    return


@basic_no_opt
def with_args(x: float, y: float):
    return


@basic_no_opt
def other():
    return


class RecordingSerializer:
    def __init__(self, payload: str | bytes = "serialized"):
        self.payload = payload
        self.versions: list[str] = []

    def encode(self, encoded_module):
        self.versions.append(encoded_module.version)
        return self.payload


def finalize_context(
    *,
    serializer: KernelSerializer | None = None,
    dialect_group: ir.DialectGroup | None = None,
    validation_suite: ValidationSuite | None = None,
) -> FinalizeContext:
    return FinalizeContext(
        program_language="squin",
        language_version="v2",
        kernel_serializer=serializer or JSONSerializer(),
        dialect_group=dialect_group,
        validation_suite=validation_suite,
    )


def test_add_subtask_deduplicates_programs_and_binds_arguments():
    builder = TaskBuilder()

    first_index = builder.add_subtask(
        with_args,
        4,
        {"label": "positional"},
        1.5,
        2.5,
    )
    reused_index = builder.add_subtask(
        with_args,
        7,
        metadata={"label": "keyword"},
        x=3.5,
        y=4.5,
    )
    other_index = builder.add_subtask(other, 2)

    assert (first_index, reused_index, other_index) == (0, 1, 2)
    assert len(builder._programs) == 2
    assert [s.qlam_subtask.arguments for s in builder._subtasks] == [
        {"x": 1.5, "y": 2.5},
        {"x": 3.5, "y": 4.5},
        {},
    ]
    assert builder._subtasks[0].qlam_subtask.subtask_metadata is not None
    assert builder._subtasks[0].qlam_subtask.subtask_metadata.user_metadata == (
        json.dumps({"label": "positional"})
    )
    assert builder._subtasks[2].qlam_subtask.subtask_metadata is None


def test_add_subtask_rejects_invalid_metadata_and_arguments():
    builder = TaskBuilder()

    with pytest.raises(ValueError, match="dictionary or None"):
        builder.add_subtask(no_args, 1, metadata="bad")  # type: ignore[arg-type]
    assert builder == TaskBuilder()

    with pytest.raises(TypeError, match="unexpected keyword argument 'unknown'"):
        builder.add_subtask(
            with_args,
            1,
            x=1.0,
            y=2.0,
            unknown=2.0,  # type: ignore[call-arg]
        )
    assert builder == TaskBuilder()


def test_add_subtask_rejects_non_json_metadata_without_mutating_builder():
    builder = TaskBuilder()

    with pytest.raises(TypeError, match="not JSON serializable"):
        builder.add_subtask(no_args, 1, metadata={"value": object()})

    assert builder == TaskBuilder()


def test_add_subtask_requires_a_python_function(monkeypatch):
    builder = TaskBuilder()
    monkeypatch.setattr(no_args, "py_func", None)

    with pytest.raises(TypeError, match="Cannot determine the argument names"):
        builder.add_subtask(no_args, 1)

    assert builder == TaskBuilder()


@pytest.mark.parametrize("num_shots", [-1, 0])
def test_nonpositive_shots_raise_subtask_validation_error(num_shots):
    builder = TaskBuilder()

    with pytest.raises(SubtaskValidationError, match="num_shots must be at least 1"):
        builder.add_subtask(no_args, num_shots)

    assert builder == TaskBuilder()
    assert builder.add_subtask(no_args, 1) == 0
    assert builder._subtasks[0].kernel_name == "no_args"


def test_kernel_names_are_unique_and_copy_preserves_the_counter(monkeypatch):
    builder = TaskBuilder()
    builder.add_subtask(no_args, 1)
    builder.add_subtask(no_args, 2)

    copied = builder.copy()
    copied.add_subtask(no_args, 3)

    assert [s.kernel_name for s in builder._subtasks] == ["no_args", "no_args_1"]
    assert [s.kernel_name for s in copied._subtasks] == [
        "no_args",
        "no_args_1",
        "no_args_2",
    ]
    assert copied._programs is not builder._programs
    assert copied._subtasks is not builder._subtasks
    assert copied._subtasks[0] is builder._subtasks[0]

    anonymous = other
    monkeypatch.setattr(anonymous, "sym_name", None)
    anonymous_builder = TaskBuilder()
    anonymous_builder.add_subtask(anonymous, 1)
    assert anonymous_builder._subtasks[0].kernel_name == "kernel"


def test_summary_and_detailed_output_include_subtask_information():
    builder = TaskBuilder()
    builder.add_subtask(with_args, 9, None, 1.0, 2.0)

    assert str(builder) == builder.summary()
    assert "0. Subtask: with_args, program 0 -> 9 shots" in builder.summary()

    detailed = builder.print_detailed()
    assert "Programs:" in detailed
    assert "Subtasks:" in detailed
    assert "Args {'x': 1.0, 'y': 2.0} -> 9 shots" in detailed


def test_finalize_builds_plain_ordered_payload_without_mutating_builder():
    serializer = RecordingSerializer()
    builder = TaskBuilder()
    builder.add_subtask(with_args, 3, {"run": 1}, 1.0, 2.0)
    builder.add_subtask(other, 5)
    builder.add_subtask(with_args, 7, None, 3.0, 4.0)
    before_finalize = builder.copy()

    definition = builder._finalize(
        finalize_context(serializer=cast(KernelSerializer, serializer))
    )

    assert definition.program_language == "squin.v2"
    assert definition.group_id is None
    assert [program.content for program in definition.programs] == [
        "serialized",
        "serialized",
    ]
    assert serializer.versions == ["v2", "v2"]
    assert [subtask.program_index for subtask in definition.subtasks] == [0, 1, 0]
    assert [subtask.num_shots for subtask in definition.subtasks] == [3, 5, 7]
    assert all(not hasattr(subtask, "kernel_name") for subtask in definition.subtasks)
    assert builder == before_finalize

    repeated_definition = builder._finalize(
        finalize_context(serializer=cast(KernelSerializer, serializer))
    )
    assert repeated_definition.model_dump() == definition.model_dump()
    assert builder == before_finalize

    assert builder.add_subtask(no_args, 11) == 3
    assert len(builder._subtasks) == 4
    assert len(definition.subtasks) == 3


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b"binary program", base64.b64encode(b"binary program").decode("ascii")),
        ("text program", "text program"),
    ],
)
def test_serialize_kernel_accepts_binary_and_text_payloads(payload, expected):
    serializer = RecordingSerializer(payload)

    actual = TaskBuilder()._serialize_kernel(
        no_args,
        cast(KernelSerializer, serializer),
        "1",
    )

    assert actual == expected


def test_serialize_kernel_rejects_other_payload_types():
    serializer = RecordingSerializer(cast(str, {"bad": "payload"}))

    with pytest.raises(TypeError, match="must return str or bytes, got dict"):
        TaskBuilder()._serialize_kernel(
            no_args,
            cast(KernelSerializer, serializer),
            "1",
        )


class ErrorWithHint(Exception):
    def hint(self):
        return "use a supported operation"


class ErrorWithoutHint(Exception):
    pass


def test_format_validation_errors_includes_pass_names_and_hints():
    result = ValidationResult(
        cast(
            dict[str, list[ValidationError]],
            {
                "device-check": [
                    ErrorWithHint("invalid operation"),
                    ErrorWithoutHint("second error"),
                ]
            },
        )
    )

    assert _format_validation_errors(result) == [
        "[device-check] invalid operation",
        "  hint: use a supported operation",
        "[device-check] second error",
    ]


class ResultSuite:
    def __init__(self, result: ValidationResult):
        self.result = result
        self.methods: list[ir.Method] = []

    def validate(self, method: ir.Method) -> ValidationResult:
        self.methods.append(method)
        return self.result


class CrashingSuite:
    def validate(self, method: ir.Method) -> ValidationResult:
        raise RuntimeError(f"could not validate {method.sym_name}")


def test_kernel_validation_accepts_supported_dialects_and_valid_result():
    builder = TaskBuilder()
    builder.add_subtask(no_args, 1)
    result = ValidationResult({"device-check": []})
    suite = ResultSuite(result)

    builder._validate_kernels(
        dialect_group=no_args.dialects,
        validation_suite=cast(ValidationSuite, suite),
    )

    assert suite.methods == [no_args]


def test_kernel_validation_aggregates_suite_and_dialect_errors():
    builder = TaskBuilder()
    builder.add_subtask(no_args, 1)
    result = ValidationResult(
        cast(
            dict[str, list[ValidationError]],
            {"device-check": [ErrorWithHint("invalid operation")]},
        )
    )
    suite = ResultSuite(result)

    with pytest.raises(TaskFinalizeError) as exc_info:
        builder._validate_kernels(
            dialect_group=ir.DialectGroup([]),
            validation_suite=cast(ValidationSuite, suite),
        )

    message = str(exc_info.value)
    assert "1 of 1 program(s) failed validation" in message
    assert "[device-check] invalid operation" in message
    assert "uses dialect(s) not supported by this device" in message
    assert exc_info.value.results == {"program 0 (no_args)": result}


def test_kernel_validation_reports_a_crashing_suite_and_anonymous_kernel(
    monkeypatch,
):
    builder = TaskBuilder()
    builder.add_subtask(other, 1)
    monkeypatch.setattr(other, "sym_name", None)

    with pytest.raises(TaskFinalizeError) as exc_info:
        builder._validate_kernels(
            validation_suite=cast(ValidationSuite, CrashingSuite()),
        )

    assert "program 0 (<anonymous>)" in str(exc_info.value)
    assert "validation suite raised RuntimeError" in str(exc_info.value)
    assert exc_info.value.results == {}


def test_validation_options_can_be_skipped_independently():
    builder = TaskBuilder()
    builder.add_subtask(no_args, 1)

    builder._validate_kernels()
    builder._validate_kernels(dialect_group=no_args.dialects)
    valid_suite = ResultSuite(ValidationResult({"device-check": []}))
    builder._validate_kernels(validation_suite=cast(ValidationSuite, valid_suite))

    assert valid_suite.methods == [no_args]


def test_finalize_error_defaults_to_empty_results():
    error = TaskFinalizeError("bad task")

    assert str(error) == "bad task"
    assert error.results == {}
