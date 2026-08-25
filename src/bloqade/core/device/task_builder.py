import base64
import inspect
import json
from dataclasses import dataclass, field
from typing import TypeVar

from kirin import ir
from kirin.validation import ValidationSuite
from kirin.validation.validationpass import ValidationResult
from qlam_core.plugins.tasks.api.tasks_models import (
    Program,
    Subtask,
    TaskDefinition,
    TaskMetadata,
)
from typing_extensions import ParamSpec

from .task import KernelSerializer

CallArgs = ParamSpec("CallArgs")
T = TypeVar("T")


@dataclass(frozen=True)
class FinalizeContext:
    program_language: str
    language_version: str
    kernel_serializer: KernelSerializer
    dialect_group: ir.DialectGroup | None = None
    validation_suite: ValidationSuite | None = None


class SubtaskValidationError(Exception):
    """Validation error if a subtask's arguments, shot count, or metadata are invalid."""


@dataclass
class _Subtask:
    kernel_name: str
    qlam_subtask: Subtask


class TaskFinalizeError(Exception):
    """Kernels failed dialect or validation checks; nothing was submitted."""

    def __init__(
        self, message: str, *, results: dict[str, ValidationResult] | None = None
    ):
        super().__init__(message)
        self.results = results or {}


def _format_validation_errors(result: ValidationResult) -> list[str]:
    """Mirror kirin's own error extraction, minus the ANSI colour codes."""
    lines: list[str] = []
    for pass_name, errors in result.errors.items():
        for err in errors:
            lines.append(f"[{pass_name}] {err.args[0] if err.args else err}")
            hint = err.hint() if hasattr(err, "hint") else None
            if hint:
                lines.append(f"  hint: {hint}")
    return lines


@dataclass
class TaskBuilder:
    """Incrementally assemble subtasks before finalizing them for a device."""

    _programs: dict[ir.Method, int] = field(default_factory=dict)
    _subtasks: list[_Subtask] = field(default_factory=list)
    _max_program_idx: int = 0
    _kernel_name_counter: dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.summary()

    def summary(self) -> str:
        """Return a human-readable summary printed on dry-run.

        Returns:
            str: Summary describing what would be submitted.
        """
        ret_str = ""
        ret_str += "Task:\n"

        for idx, subtask in enumerate(self._subtasks):
            subtask_str = f"{idx}. Subtask: {subtask.kernel_name}, program {subtask.qlam_subtask.program_index} -> {subtask.qlam_subtask.num_shots} shots\n"
            ret_str += subtask_str

        return ret_str

    def print_detailed(self) -> str:
        """Return a detailed summary containing program IR and subtask arguments."""
        ret_str = ""
        ret_str += "Task:\n"
        ret_str += "\tPrograms:\n"
        for program, program_idx in sorted(self._programs.items(), key=lambda x: x[1]):
            ret_str += f"\t\t{program_idx}. {program.print_str()}"

        ret_str += "\tSubtasks:\n"
        for subtask_idx, subtask in enumerate(self._subtasks):
            ret_str += f"\t\t{subtask_idx}. Program {subtask.qlam_subtask.program_index}, Args {subtask.qlam_subtask.arguments} -> {subtask.qlam_subtask.num_shots} shots\n"

        return ret_str

    # TODO: do we assume that the kernel is a kirin kernel? We do for hashing
    def _validate_subtask(self, subtask: _Subtask) -> bool:
        # TODO: can edit this for a more robust validation pass later
        # NOTE: kernel validation will be done in _finalize() per kernel depending on the validation pass we use

        # TODO: define a maximum number of shots we allow?
        if subtask.qlam_subtask.num_shots <= 0:
            raise ValueError("num_shots must be at least 1")

        # TODO: what other validation to do?
        return True

    def add_subtask(
        self,
        kernel: ir.Method[CallArgs, T],
        num_shots: int,
        metadata: dict | None = None,
        *args: CallArgs.args,
        **kernel_args: CallArgs.kwargs,
    ) -> int:
        # TODO: define equality? we COULD implement hashing on kirin kernels. as a first pass maybe it's OK to
        # just do exact object check
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("expecting `metadata` to be a dictionary or None. ")

        base_kernel_name = kernel.sym_name if kernel.sym_name is not None else "kernel"
        kernel_name_index = self._kernel_name_counter.get(base_kernel_name, 0)
        kernel_name = (
            base_kernel_name
            if kernel_name_index == 0
            else f"{base_kernel_name}_{kernel_name_index}"
        )

        if kernel in self._programs:
            program_idx = self._programs[kernel]
            is_new_program = False
        else:
            program_idx = self._max_program_idx
            is_new_program = True

        if metadata is not None:
            subtask_metadata = TaskMetadata(user_metadata=json.dumps(metadata))
        else:
            subtask_metadata = None

        if kernel.py_func is None:
            raise TypeError(
                f"Cannot determine the argument names for kernel {kernel.sym_name!r}"
            )

        signature = inspect.signature(kernel.py_func)
        bound = signature.bind(*args, **kernel_args)

        arguments = dict(bound.arguments)

        qlam_subtask = Subtask(
            program_index=program_idx,
            num_shots=num_shots,
            arguments=arguments,
            subtask_metadata=subtask_metadata,
        )

        new_subtask = _Subtask(kernel_name=kernel_name, qlam_subtask=qlam_subtask)
        try:
            self._validate_subtask(subtask=new_subtask)
        except ValueError as e:
            raise SubtaskValidationError(f"Error during subtask validation: {e}")

        # Commit builder state only after every operation above has succeeded.
        if is_new_program:
            self._programs[kernel] = program_idx
            self._max_program_idx += 1
        self._kernel_name_counter[base_kernel_name] = kernel_name_index + 1
        self._subtasks.append(new_subtask)
        return len(self._subtasks) - 1

        # does validation of the subtask added in terms of number of shots, metadata, ...

    # # TODO: need to think about "add_batch_subtask" more... do we need it? can add other convenience things later.
    # def add_batch_subtask(self, kernels, num_shots: list[int], *, metadata: list[dict], args: list[dict]):
    #     # ...

    def _validate_kernels(
        self,
        dialect_group: ir.DialectGroup | None = None,
        validation_suite: ValidationSuite | None = None,
    ):
        if dialect_group is None and validation_suite is None:
            return
        reports: list[str] = []
        results: dict[str, ValidationResult] = {}

        if validation_suite is None:

            def _validate(
                method: ir.Method, problems_list: list[str], kernel_label: str
            ):
                return

        else:

            def _validate(
                method: ir.Method, problems_list: list[str], kernel_label: str
            ):
                try:
                    result = validation_suite.validate(method)
                except Exception as exc:  # noqa: BLE001 — a pass itself blew up
                    problems_list.append(
                        f"validation suite raised {type(exc).__name__}: {exc}"
                    )
                else:
                    results[kernel_label] = result
                    if not result.is_valid:
                        problems_list.extend(_format_validation_errors(result))

        if dialect_group is None:

            def _check_dialect_group(
                method: ir.Method, problems_list: list[str], kernel_label: str
            ):
                return

        else:

            def _check_dialect_group(
                method: ir.Method, problems_list: list[str], kernel_label: str
            ):
                unsupported = method.dialects.data - dialect_group.data
                if unsupported:
                    problems_list.append(
                        "uses dialect(s) not supported by this device: "
                        + ", ".join(sorted(d.name for d in unsupported))
                    )

        for kernel, index in sorted(self._programs.items(), key=lambda x: x[1]):
            label = f"program {index} ({kernel.sym_name or '<anonymous>'})"
            problems: list[str] = []

            _validate(kernel, problems, label)
            _check_dialect_group(kernel, problems, label)

            if problems:
                reports.append(
                    label + ":\n" + "\n".join(f"    - {p}" for p in problems)
                )

        if reports:
            raise TaskFinalizeError(
                f"{len(reports)} of {len(self._programs)} program(s) failed validation:\n\n"
                + "\n".join(f"  {r}" for r in reports),
                results=results,
            )

    def _serialize_kernel(
        self,
        kernel: ir.Method,
        kernel_serializer: KernelSerializer,
        program_language_version: str,
    ) -> str:
        encoded_module = kernel.dialects.encode(
            kernel, version=program_language_version
        )
        payload = kernel_serializer.encode(encoded_module)

        if isinstance(payload, bytes):
            return base64.b64encode(payload).decode("ascii")

        if isinstance(payload, str):
            return payload

        raise TypeError(
            "kernel_serializer.encode must return str or bytes, "
            f"got {type(payload).__name__}"
        )

    def _get_programs(
        self, kernel_serializer: KernelSerializer, program_language_version: str
    ) -> list[Program]:
        program_list = []
        for kernel, index in sorted(self._programs.items(), key=lambda x: x[1]):
            serialized_program = self._serialize_kernel(
                kernel,
                kernel_serializer=kernel_serializer,
                program_language_version=program_language_version,
            )
            program_list.append(Program(content=serialized_program))
        return program_list

    def _finalize(self, ctx: FinalizeContext) -> TaskDefinition:
        """Build a `TaskDefinition` without mutating this builder.

        Finalization validates and serializes the current builder contents. It
        may be called repeatedly, and the builder remains editable afterward.

        Override this method directly if your use-case doesn't fit the API
        contract.

        Returns:
            TaskDefinition: Definition ready to be submitted.
        """
        # Serialize the programs and run the validation

        self._validate_kernels(
            dialect_group=ctx.dialect_group, validation_suite=ctx.validation_suite
        )

        programs = self._get_programs(
            kernel_serializer=ctx.kernel_serializer,
            program_language_version=ctx.language_version,
        )

        subtasks = []
        for subtask in self._subtasks:

            subtasks.append(
                Subtask(
                    program_index=subtask.qlam_subtask.program_index,
                    num_shots=subtask.qlam_subtask.num_shots,
                    arguments=subtask.qlam_subtask.arguments,
                    subtask_metadata=subtask.qlam_subtask.subtask_metadata,
                )
            )

        # NOTE: strictly need to removeprefix here?
        program_language_with_version = (
            f"{ctx.program_language}.v{ctx.language_version.removeprefix('v')}"
        )
        return TaskDefinition(
            program_language=program_language_with_version,
            programs=programs,
            subtasks=subtasks,
            group_id=None,
        )

    def copy(self) -> "TaskBuilder":
        """Return a shallow, independently editable copy of this builder."""
        return TaskBuilder(
            _programs=dict(self._programs),
            _subtasks=list(self._subtasks),
            _max_program_idx=self._max_program_idx,
            _kernel_name_counter=dict(self._kernel_name_counter),
        )
