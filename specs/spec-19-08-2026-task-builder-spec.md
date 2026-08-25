# Spec: `TaskBuilder`

Status: Implemented in PR #109; pending review
Owner: Phillip Weinberg (@weinbe58)
Tracking issue: QuEraComputing/bloqade-internal#398 ("TaskBuilder API")
Target package: `bloqade-core` (`bloqade.core.device`)

## 1. Motivation

The legacy device API creates one of three fixed task shapes:
`SingleKernelTask`, `KernelBatchTask`, or `ParameterScanTask`. Each task eagerly
binds its kernels, shots, arguments, and metadata at construction time.

`TaskBuilder` adds an incremental construction API. A caller can add subtasks
one at a time, reuse a kernel across subtasks, inspect or copy the assembled
work, perform a dry run, and finally submit it through a `Device`.

The builder owns task construction state. The device owns the language,
serializer, validation configuration, authentication, group selection,
submission, storage, and future type.

## 2. Scope

### In scope

- A standalone, argument-free `TaskBuilder` exported from
  `bloqade.core.device`.
- Incremental `add_subtask` calls with positional or keyword kernel arguments.
- Program deduplication by kernel object identity.
- Transactional add-time validation for metadata, kernel arguments, and shots.
- Shallow copying through `copy()`.
- Human-readable `summary()` and `print_detailed()` output.
- Non-mutating conversion to a QLAM `TaskDefinition` through `_finalize()`.
- Kernel dialect and `ValidationSuite` checks during finalization.
- Dry-run and submission through `Device.run_async()`.
- Per-submission group selection using existing QLAM configuration precedence.

### Out of scope

- `add_batch_subtask`; callers can use `add_subtask` in a loop.
- In-place editing methods such as `modify_subtask` or `modify_program`.
- Precompilation and QLAM compilation-reference workflows.
- Retrieval, result concatenation, and storage hardening changes.
- Simulator-specific implementations.
- Removing the three legacy `TaskABC` task shapes.

## 3. User-facing workflow

```python
from bloqade.core.device import TaskBuilder

device = SomeLogicalDevice()

task = TaskBuilder()
i0 = task.add_subtask(kernel_a, 1000)
i1 = task.add_subtask(kernel_b, 23, {"purpose": "calibration"})
i2 = task.add_subtask(kernel_a, 100, theta=1.5)

assert (i0, i1, i2) == (0, 1, 2)

preview = task.copy()
preview.add_subtask(kernel_c, 50)

print(task.summary())
print(task.print_detailed())

# Validate, serialize, and print a preview. Nothing is submitted, and the
# builder remains editable.
device.run_async(task, dry_run=True)
task.add_subtask(kernel_d, 10)

# Validate and serialize the current contents again, then submit them.
future = device.run_async(task, dry_run=False)
result = future.result()
```

`dry_run` is required so submission is always an explicit choice.

## 4. Builder representation

`TaskBuilder` is a mutable dataclass in
`src/bloqade/core/device/task_builder.py`:

```python
@dataclass
class TaskBuilder:
    _programs: dict[ir.Method, int] = field(default_factory=dict)
    _subtasks: list[_Subtask] = field(default_factory=list)
    _max_program_idx: int = 0
    _kernel_name_counter: dict[str, int] = field(default_factory=dict)
```

- `_programs` maps each distinct kernel object to its program index.
- `_subtasks` preserves subtask insertion order.
- `_max_program_idx` is the next program index.
- `_kernel_name_counter` creates unique display names when several subtasks
  use kernels with the same `sym_name`.

There is no finalized or sealed state. Finalizing produces a payload snapshot
without changing the builder.

### 4.1 Program identity

Programs are deduplicated using the `ir.Method` object as a dictionary key.
Kirin methods use identity hashing, so adding the same method object again
reuses its program index. A distinct method object remains a distinct program,
even when its IR or symbol name is identical.

### 4.2 Internal subtask wrapper

```python
@dataclass
class _Subtask:
    kernel_name: str
    qlam_subtask: Subtask
```

Composition keeps the display-only `kernel_name` out of the QLAM request.
`_finalize()` reconstructs plain QLAM `Subtask` instances for the payload.

## 5. `TaskBuilder` API

### 5.1 `add_subtask`

```python
def add_subtask(
    self,
    kernel: ir.Method[CallArgs, T],
    num_shots: int,
    metadata: dict | None = None,
    *args: CallArgs.args,
    **kernel_args: CallArgs.kwargs,
) -> int: ...
```

The method returns the newly appended **subtask index**, not the program index.
Reusing a program therefore still returns a new value for every call.

Kernel arguments are bound against `inspect.signature(kernel.py_func)`:

```python
signature = inspect.signature(kernel.py_func)
bound = signature.bind(*args, **kernel_args)
arguments = dict(bound.arguments)
```

This supports positional and keyword arguments and rejects missing, duplicate,
or unknown arguments before submission. `ParamSpec` exposes the kernel
signature to static type checkers. A kernel without `py_func` raises
`TypeError`, because its Python argument names cannot be recovered.

`metadata` must be a dictionary or `None`. A dictionary is serialized into
`TaskMetadata(user_metadata=json.dumps(metadata))`; non-JSON-serializable
metadata raises at add time.

The base display name is `kernel.sym_name`, falling back to `"kernel"` when it
is absent. Repeated names become `name_1`, `name_2`, and so on.

#### Transactional behavior

Adding a subtask is transactional. The method computes the prospective name
and program index, serializes metadata, binds arguments, constructs the QLAM
model, and validates the subtask before changing builder state. If any step
fails, `_programs`, `_subtasks`, `_max_program_idx`, and
`_kernel_name_counter` remain unchanged.

On success, a new kernel is inserted into `_programs`, the name counter is
advanced, and the subtask is appended together.

The named builder parameters `kernel`, `num_shots`, and `metadata` are reserved
for keyword calls. A kernel parameter with one of those names can only be
reached positionally.

### 5.2 Add-time validation

`_validate_subtask()` currently requires `num_shots >= 1`. A failure is exposed
as `SubtaskValidationError`. Kernel-level validation is deferred until
finalization because it depends on the selected device.

### 5.3 `copy()`

`copy()` returns a shallow, independently editable builder:

- `_programs`, `_subtasks`, and `_kernel_name_counter` receive new containers.
- `_max_program_idx` is preserved.
- Kernel and `_Subtask` objects are shared and treated as values.
- Later additions to one builder do not modify the other builder's containers.

### 5.4 Display

- `str(builder)` delegates to `summary()`.
- `summary()` returns one line per subtask containing its display name, program
  index, and shots.
- `print_detailed()` includes each program's Kirin IR and each subtask's
  program index, bound arguments, and shots.

Both methods return strings and do not change builder state.

## 6. Finalization

### 6.1 `FinalizeContext`

The device supplies finalization inputs through a frozen dataclass:

```python
@dataclass(frozen=True)
class FinalizeContext:
    program_language: str
    language_version: str
    kernel_serializer: KernelSerializer
    dialect_group: ir.DialectGroup | None = None
    validation_suite: ValidationSuite | None = None
```

Keeping this context off the builder allows one builder to be previewed or
submitted using device-owned configuration.

### 6.2 `_finalize(ctx) -> TaskDefinition`

`_finalize()` validates and serializes the builder's current contents into a
QLAM `TaskDefinition`. It does not mutate, seal, or cache the builder. It is
safe to call repeatedly with respect to builder state, and later additions are
allowed.

The operation performs these steps:

1. Validate every unique program using `_validate_kernels()`.
2. Serialize programs in program-index order.
3. Reconstruct plain QLAM `Subtask` objects in subtask order.
4. Format the language as
   `f"{program_language}.v{language_version.removeprefix('v')}"`.
5. Return `TaskDefinition(..., group_id=None)`.

The returned definition is a snapshot. Later builder additions do not modify a
previously returned definition.

`ValidationSuite` itself maintains analysis state, so concurrent calls sharing
one suite are not guaranteed to be safe even though the builder is not
mutated.

### 6.3 Kernel validation

`_validate_kernels(dialect_group=None, validation_suite=None)` returns
immediately when neither check is configured. Otherwise, each unique program
is checked in program-index order.

- When `validation_suite` is provided, `validation_suite.validate(kernel)` is
  called. Invalid results are formatted with pass names and hints and retained
  on the raised error.
- When `dialect_group` is provided, every dialect used by the kernel must be in
  the device group. Unsupported dialect names are sorted in the report.
- When both are configured, both checks run and their problems are aggregated.

Kernels are labelled as `program <index> (<sym_name>)`; anonymous kernels use
`<anonymous>`. Any failure raises one `TaskFinalizeError` with a per-program
report and a `results` dictionary containing available `ValidationResult`s.

### 6.4 Serialization

Program serialization matches the legacy `TaskABC` path:

1. `kernel.dialects.encode(kernel, version=language_version)`
2. `kernel_serializer.encode(encoded_module)`
3. Pass string payloads through unchanged.
4. Base64-encode byte payloads.
5. Raise `TypeError` for any other serializer result.

## 7. Device integration

The builder path adds these device fields:

```python
program_language: str = "squin"
language_version: str = "0.1.0"
kernel_serializer: KernelSerializer = field(default_factory=JSONSerializer)
validation_suite: ValidationSuite | None = None
dialect_group: ir.DialectGroup | None = None
```

`Device._finalize_context()` bundles these fields for the builder.

### 7.1 `run_async`

```python
@overload
def run_async(
    task_builder: TaskBuilder,
    *,
    dry_run: Literal[True],
    group: str | None = None,
    storage: StorageBackend | None = None,
    fetch_options: ApiFetchOptions = DEFAULT_FETCH_OPTIONS,
) -> None: ...

@overload
def run_async(
    task_builder: TaskBuilder,
    *,
    dry_run: Literal[False],
    group: str | None = None,
    storage: StorageBackend | None = None,
    fetch_options: ApiFetchOptions = DEFAULT_FETCH_OPTIONS,
) -> FutureType: ...
```

The method always calls `_finalize()` first. Validation and serialization
errors therefore appear during both previews and real submissions.

When `dry_run=True`:

- The builder summary is printed.
- No authentication or QLAM API call occurs.
- No task definition is written to storage.
- `group`, `storage`, and `fetch_options` have no effect.
- `None` is returned.
- The builder remains editable and may be dry-run or submitted again.

When `dry_run=False`, the finalized definition is passed to
`submit_task_definition()`, which returns the device's configured future type.

### 7.2 Group and submission behavior

`group` is a per-call argument rather than builder or device state. If the
definition does not already have a group ID, submission selects a group using:

1. The explicit `run_async(..., group=...)` value.
2. `plugins.tasks.group` from QLAM configuration.
3. `defaults.group` from the current QLAM context.
4. No group ID, allowing QLAM's backend default behavior.

A selected group name or UUID string is resolved through `GroupsClient` before
`TasksClient.create`. The submitted definition and creation time are stored,
and the returned future contains the task ID, fetch options, storage, and
context name.

### 7.3 Legacy API

`Device.task()`, `Device.batch_task()`, and `Device.parameter_scan()` remain
available and return the existing `TaskABC` subclasses. Those task objects
retain their own `run_async()` and submission path. This PR does not remove or
redirect them through `TaskBuilder`.

## 8. Design decisions

1. The builder is standalone and device-agnostic.
2. The device owns finalization context and builder submission.
3. Finalization materializes a payload snapshot; it does not freeze the
   builder.
4. Program deduplication uses kernel object identity.
5. Kernel arguments are bound against the original Python signature.
6. `_Subtask` uses composition so display data cannot leak onto the wire.
7. Add-time failures are transactional.
8. Kernel validation runs once per unique program at finalization time.
9. Dialect validation implements only the subset check: kernel dialects must be
   supported by the device dialect group.
10. Group selection is per submission so one device can submit to different
    groups.
11. In-place editing and batch convenience methods remain deferred.

## 9. Changes from the earlier draft

The implemented API intentionally differs from the earlier
`spec/task-builder` branch in these ways:

- The module is named `task_builder.py`, and `TaskBuilder` is exported from
  `bloqade.core.device`.
- `_programs` is an identity-keyed dictionary rather than a list.
- `_Subtask` wraps an eagerly constructed QLAM `Subtask`.
- `add_subtask` accepts positional kernel arguments and uses `ParamSpec`.
- Metadata and subtask validation happen at add time.
- Display names are uniquified.
- `add_batch_subtask` is deferred.
- Group selection is a per-`run_async` argument instead of device state.
- Submission helpers currently exist separately on `Device` and `TaskABC`;
  the proposed shared submitter mixin was not introduced.
- `Device.task_builder()` was not introduced because direct `TaskBuilder()`
  construction is available.

## 10. Test coverage

`test/device/test_task_builder.py` covers:

- Subtask indices and identity-based program reuse.
- Positional and keyword argument binding.
- Transactional failure paths.
- Metadata serialization and shot validation.
- Unique display names and copy behavior.
- Payload ordering, serialization, and non-mutating repeatable finalization.
- Dialect and validation-suite success, failure, and aggregation.
- Summary and detailed display output.

`test/device/test_device_task_builder.py` covers:

- Dry-run preview without submission and continued builder editing.
- Real submission and future construction.
- Storage behavior.
- Explicit, plugin, and context-default group selection.
- Existing group-ID preservation and missing-task-ID errors.

The legacy task tests remain unchanged as a regression suite.

## 11. Deferred follow-ups

- Decide whether `add_batch_subtask` is useful enough to add as convenience.
- Consider extracting the duplicated `Device` and `TaskABC` submission logic.
- Consider exporting `FinalizeContext`, `TaskFinalizeError`, and
  `SubtaskValidationError` from `bloqade.core.device` if downstream code needs
  them as public API.
- Improve multi-line validation-report formatting.
- Decide whether empty kernel arguments should remain `{}` or become `None`.
- Decide whether shot values should reject `bool` explicitly rather than rely
  on the annotated `int` API and QLAM model coercion.
