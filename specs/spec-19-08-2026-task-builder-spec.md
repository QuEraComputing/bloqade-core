# Spec: `TaskBuilder`

Status: Draft
Owner: Phillip Weinberg (@weinbe58)
Tracking issue: QuEraComputing/bloqade-internal#398 ("TaskBuilder API")
Target package: `bloqade-core` (`bloqade.core.device`)

## 1. Motivation

Today `bloqade-core` exposes task creation through `Device` factory methods
(`task`, `batch_task`, `parameter_scan`), each producing a distinct `TaskABC`
subclass that eagerly binds kernels, shot counts, arguments, and metadata at
construction time. This works for the fixed shapes we ship, but it does not
give users an incremental, inspectable way to assemble a multi-subtask job — the
workflow described in the tracking issue where a caller adds one subtask at a
time, possibly reusing the same kernel, and only later submits the whole thing.

`TaskBuilder` is a mutable, standalone object for **incrementally assembling** a
task (a set of unique *programs* plus a list of *subtasks* that reference them),
which is then validated and turned into a qlam `TaskDefinition` at submission
time by a `Device`.

The broader goal from the issue is that the basic Python API for *creating* and
*executing* tasks reads the same across the hardware and simulator paths.
`TaskBuilder` is the shared construction surface for that.

## 2. Scope

### In scope

- A new standalone `TaskBuilder` class in `bloqade.core.device`.
- Wiring `TaskBuilder` into the **existing** submission APIs via a new
  `Device.run_async(builder, *, dry_run, ...)` entry point.
- The builder methods: `add_subtask`, `add_batch_subtask`, `copy`, `_finalize`,
  `__str__` / `__repr__`, `print_detailed`.
- Program deduplication (reusing the same kernel object reuses one program).
- Kernel validation at finalize time: a **level-1 dialect-group check** plus an
  optional kirin `ValidationSuite`, with a detailed per-kernel failure report.

### Out of scope (explicitly deferred)

- `device.compile_task_definition(...)` / precompiled task definitions and any
  QLAM compilation-reference flow.
- The retrieval side: `Future` save/load/fetch/concatenate semantics, storage
  hardening, and result-type-by-dialect. (Tracked separately; see issue #398
  and Jon Wurtz's comments.)
- Any simulator-specific implementation. `bloqade-core` only provides the
  generic building blocks; concrete devices, dialect groups, and validation
  suites are supplied by downstream packages (e.g. bloqade-lanes).
- In-place editing of an assembled task (`modify_subtask`,
  `modify_subtask_shots`, `modify_program`). Intentionally omitted — see
  §7 "Design decisions".

## 3. Background: what already exists

(For reference — see [`task.py`](../src/bloqade/core/device/task.py) and
[`device.py`](../src/bloqade/core/device/device.py).)

- `TaskABC` (`task.py`) collects kernels + per-subtask data and builds a qlam
  `TaskDefinition` via `create_task_definition()` → `programs()` +
  `Subtask` list. It owns the submit path: `run_async(dry_run=...)` →
  `create_task_definition()` → `submit_task_definition(...)`.
- `submit_task_definition` resolves the group, authenticates, calls
  `TasksClient.create`, stores the definition, and returns a `Future`.
- `Device` (`device.py`) is a factory that injects `context_name`,
  `future_cls`, `kernel_serializer`, and `group` into the tasks it builds.
- The qlam payload types come from
  `qlam_core.plugins.tasks.api.tasks_models`: `Program(content: str)`,
  `Subtask(program_index, num_shots, arguments, subtask_metadata)`,
  `TaskDefinition(program_language, programs, subtasks, group_id)`,
  `TaskCreationRequest`, `TaskMetadata`.
- Kernels are `kirin.ir.Method`. `kernel.dialects` is the kirin
  `DialectGroup` attached to the method; `DialectGroup.data` is a
  `frozenset[Dialect]`.

### External types (from `kirin-toolchain`)

- `kirin.ir.DialectGroup` — `.data: frozenset[Dialect]`, iterable, has
  `is_structurally_equal`.
- `kirin.validation.ValidationSuite(passes=[...], fail_fast=False)` with
  `.validate(method) -> ValidationResult`. `ValidationResult` exposes
  `.is_valid`, `.errors: dict[str, list[ValidationError]]`, `.error_count()`,
  and `.raise_if_invalid()` (raises `kirin.ir.exception.ValidationErrorGroup`).

> Note: `ValidationSuite` and `DialectGroup` are **not** referenced anywhere in
> `bloqade-core` today. This feature introduces the first use of them in the
> device path. They are always provided by a downstream `Device`, never
> constructed here.

## 4. User-facing workflow

```python
# A downstream device is configured with its dialect group, validation suite,
# and program language.
device = SomeLogicalDevice()          # subclass of bloqade.core.device.Device

task = TaskBuilder()                  # standalone, argument-free
i0 = task.add_subtask(k1, 1000)       # -> 0  (subtask index)
i1 = task.add_subtask(k2, 23)         # -> 1
i2 = task.add_subtask(k3, 230)        # -> 2
i3 = task.add_subtask(k1, 100)        # -> 3  (reuses k1's program; new subtask)

# Kernel arguments are passed as keyword arguments; metadata is keyword-only.
i4 = task.add_subtask(k7, 500, metadata={"tag": "cal"}, theta=1.5)  # -> 4

# Branch without mutating the original.
new_task = task.copy()
new_task.add_subtask(k4, 50)          # does not affect `task`

# Convenience for many subtasks at once (shots only).
task.add_batch_subtask([k5, k6], [10, 20])   # -> [5, 6]

# Inspect before running.
print(task)                           # short summary (see §5.6)
print(task.print_detailed())          # programs (IR) + subtasks

# Dry-run then submit. The device supplies language + dialect group +
# validation suite to _finalize.
device.run_async(task, dry_run=True)  # prints summary, no submission
future = device.run_async(task, dry_run=False)
result = future.result()
```

## 5. API design

### 5.1 `TaskBuilder`

A standalone, mutable `@dataclass` living in a new module
`src/bloqade/core/device/builder.py`, exported from
`bloqade.core.device.__init__`.

Internal state:

- `_programs: list[ir.Method]` — the unique kernels ("programs"), in insertion
  order. Deduplicated **by object identity** (`is`).
- `_subtasks: list[_Subtask]` — ordered subtask entries. `_Subtask` is a small
  internal dataclass: `program_index: int`, `num_shots: int`,
  `arguments: dict | None`, `metadata: dict | None`.

The builder holds **no** device, dialect group, validation suite, serializer,
or language. Those are supplied by the `Device` at finalize time (§5.5).

### 5.2 `add_subtask(kernel, num_shots, *, metadata=None, **kernel_args) -> int`

Kernel arguments are supplied as **keyword arguments** — you are effectively
*calling* the kernel — and collected into the subtask's `arguments` dict.

```python
task.add_subtask(kernel, num_shots=100, theta=1.5, phi=0.2)
# -> arguments = {"theta": 1.5, "phi": 0.2}
```

- Looks up `kernel` in `_programs` by identity; appends it if absent (this is
  the only place a new program is created).
- Collects `**kernel_args` into the subtask's `arguments` (`None` /`{}` when no
  kwargs are given). `metadata` is a keyword-only dict.
- Appends a `_Subtask` referencing that program index.
- Returns the **subtask index** (the position in `_subtasks`).
- **Reserved names.** Because kernel arguments ride on `**kwargs`, the
  parameter names `kernel`, `num_shots`, and `metadata` cannot be used as
  kernel argument names via this method.

### 5.3 `add_batch_subtask(kernels, num_shots) -> list[int]`

Convenience wrapper over `add_subtask` for adding many subtasks at once.
Deliberately **shots-only** — it does not take per-kernel arguments or
metadata. When a subtask needs arguments, call `add_subtask` in a loop.

- `num_shots`: `int` (broadcast to every kernel) or `list[int]` (per kernel,
  length-checked against `kernels`).
- Returns the list of subtask indices produced, in order.
- Raises `ValueError` on a `num_shots` length mismatch, before adding anything
  (no partial mutation).

### 5.4 `copy() -> TaskBuilder`

A **shallow** copy that makes "tasks are mutated in place" explicit and lets a
caller branch:

- Creates fresh `_programs` and `_subtasks` **list containers** so that adding
  to the copy does not affect the original (and vice versa).
- Does **not** copy the kernel objects; both builders reference the same
  `ir.Method` instances (kernels are treated as immutable values).
- Does not deep-copy the `arguments` / `metadata` dicts.

### 5.5 `_finalize(...) -> TaskDefinition`

Internal (leading underscore); normally called by `Device.run_async`. **Pure /
non-mutating**: it does not change builder state and may be called repeatedly
(e.g. once for a dry-run summary, again for the real submission).

It receives a single **`FinalizeContext`** — a frozen dataclass bundling
everything the qlam payload and validation need, built by the `Device` from its
own fields:

```python
@dataclass(frozen=True)
class FinalizeContext:
    program_language: str
    language_version: str
    kernel_serializer: KernelSerializer
    dialect_group: ir.DialectGroup | None = None
    validation_suite: ValidationSuite | None = None


def _finalize(self, ctx: FinalizeContext) -> TaskDefinition: ...
```

Behavior (reading fields from `ctx`):

1. **Dialect-group check (level 1)** — for each unique program kernel, if
   `dialect_group is not None`, require
   `kernel.dialects.data <= dialect_group.data` (every statement the user wrote
   is allowed by the group). Collect offending dialects
   (`kernel.dialects.data - dialect_group.data`) per kernel.
2. **Validation suite** — for each unique program kernel, if
   `validation_suite is not None`, run `validation_suite.validate(kernel)` and
   collect any invalid result.
3. If either check produced failures, raise a single **`TaskFinalizeError`**
   (a dedicated `bloqade` exception) whose body is a **detailed per-kernel
   report** (kernel `sym_name` + the specific dialect and/or validation
   errors), aggregating both check kinds into one uniform message. It also
   holds the underlying kirin `ValidationResult`s for programmatic access. No
   submission happens.
4. On success, build the qlam payload:
   - `programs = [Program(content=encode(kernel)) for kernel in _programs]`,
     serialized with `kernel_serializer` at `language_version` (same encoding
     path as `TaskABC.serialize_kernel`).
   - `subtasks = [Subtask(program_index=..., num_shots=..., arguments=...,
     subtask_metadata=...) for each _subtask]` (metadata JSON-dumped into
     `TaskMetadata`, matching `TaskABC.create_task_definition`).
   - `program_language` string formatted as
     `f"{program_language}.v{language_version.removeprefix('v')}"`.
   - `group_id=None` (group resolution stays in the submit path).

### 5.6 Display

- `__repr__`: default dataclass repr (standard, unambiguous).
- `__str__` / `summary()`:

  ```
  Task:
     1. SubTask: <kernel name>, program <program_index> -> <num_shots> shots
     2. ...
  ```

- `print_detailed() -> str`:

  ```
  Task:
      programs:
          1. <IR dump via ir.Method.print(...)>
          2. ...
      subtasks:
          1. program <program_index> -> <num_shots> shots
          2. ...
  ```

  The IR dump is captured to a string (kirin's `Method.print` writes to a rich
  console; capture via a recording console).

## 6. Wiring into existing APIs

### 6.1 Shared submission mixin (refactor)

Extract the submit path currently on `TaskABC` — `_configured_group`,
`_resolve_group_id`, `submit_task_definition` — into a
`TaskSubmitterMixin(AuthMixin)` so both `TaskABC` and `Device` can submit a
prepared `TaskDefinition`.

- The mixin hosts the `group` and `future_cls` fields.
- Keep the mixin **in `task.py`** so the module-level `TasksClient` /
  `GroupsClient` names it uses remain patchable by the existing tests (which
  monkeypatch `bloqade.core.device.task.TasksClient`).
- `TaskABC` behavior is unchanged (it inherits the mixin); existing tests stay
  green.

### 6.2 `Device` additions

- New fields (all with defaults; downstream device subclasses set the
  dialect-specific ones):
  - `program_language: str = "squin"`
  - `language_version: str = "0.1.0"`
  - `dialect_group: ir.DialectGroup | None = None`
  - `validation_suite: ValidationSuite | None = None`
  - `group: str | None = None` (via the mixin)
- `Device` inherits `TaskSubmitterMixin`.
- `task_builder() -> TaskBuilder` — a thin factory returning `TaskBuilder()`,
  for discoverability. (Direct `TaskBuilder()` construction is also supported.)
- A `_finalize_context() -> FinalizeContext` helper that bundles the device's
  `program_language`, `language_version`, `kernel_serializer`, `dialect_group`,
  and `validation_suite`.
- `run_async(builder, *, dry_run, storage=None, fetch_options=...)` with
  `Literal[True] -> None` / `Literal[False] -> FutureType` overloads:
  1. `task_def = builder._finalize(self._finalize_context())`.
  2. If `dry_run`: print the builder summary (DRY-RUN banner + `str(builder)`),
     return `None`.
  3. Else: `return self.submit_task_definition(task_definition=task_def, ...)`.

## 7. Design decisions

Decisions locked in discussion (see issue #398 and design review):

1. **Standalone builder + `device.run_async(builder, ...)`.** The builder is a
   plain mutable object, not a `TaskABC`. The device owns the run entry point,
   matching the spec sketch `device.run_async(task, dry_run)`.
2. **`_finalize` receives a `FinalizeContext`** (bundling language,
   language version, serializer, dialect group, and validation suite), built by
   the device at run time, rather than storing them on the builder at
   construction.
3. **`_finalize` is non-mutating / re-runnable.** No finalize "lock" and no
   `_finalized` flag. A dry-run does not prevent further edits.
4. **Dialect check is hardcoded to level 1** (kernel dialects ⊆ group). The
   set-equal and same-object-identity variants from the issue are **not**
   planned — level 1 is the only check.
5. **`_finalize` failures raise a dedicated `TaskFinalizeError`** that
   aggregates dialect and validation failures into one uniform per-kernel
   report and retains the underlying kirin `ValidationResult`s.
6. **Kernel arguments are passed as `**kwargs` on `add_subtask`;
   `add_batch_subtask` is shots-only.** `add_subtask(kernel, num_shots, *,
   metadata=None, **kernel_args)` collects kernel arguments as keyword
   arguments (`kernel`/`num_shots`/`metadata` are reserved names).
   `add_batch_subtask(kernels, num_shots)` takes no per-kernel arguments; loop
   `add_subtask` when arguments are needed.
7. **`add_batch_subtask` is in scope; `modify_*` methods are not.** In-place
   editing risks silently mutating a program shared by other subtasks, and the
   team prefers friction here — a mistake means rebuilding (or `copy()`-ing and
   re-adding), which forces the user to slow down. Validation failures crash
   before any submission (unlike Aquila's submit-time validation), so there is
   no partial-submission recovery problem to solve with in-place edits.
8. **Programs are deduplicated by object identity.** Re-adding the same kernel
   reuses its program; distinct kernel objects get distinct programs.

## 8. Testing plan

New `test/device/test_builder.py`:

- `add_subtask` returns correct subtask indices; program dedup by identity;
  distinct kernels get distinct program indices.
- `add_subtask` collects `**kernel_args` into the subtask's `arguments`;
  keyword-only `metadata` is stored separately; no-kwargs case yields empty/None
  arguments.
- `add_batch_subtask` broadcast vs list shots; length-mismatch raises before
  mutation; produces no arguments/metadata.
- `copy` isolates both lists (adds on copy/original don't cross over) and shares
  kernel objects.
- `_finalize` produces the expected `programs` / `subtasks` / `program_language`
  payload; dedup reflected in `program_index`.
- `_finalize` dialect-check failure raises `TaskFinalizeError` with a per-kernel
  report; passing case and `dialect_group=None` skip.
- `_finalize` validation-suite failure raises `TaskFinalizeError` with an
  aggregated per-kernel report and retains the `ValidationResult`s;
  `validation_suite=None` skips.
- `_finalize` reports both dialect and validation failures together in one
  `TaskFinalizeError`.
- `_finalize` is non-mutating (state identical before/after; re-runnable).
- `__str__` / `print_detailed` formatting.

Extend `test/device/test_device.py`:

- `Device.run_async(builder, dry_run=True)` prints the summary and does not
  submit.
- `Device.run_async(builder, dry_run=False)` finalizes and submits (mocked
  clients), returning a future.

Regression: existing `test/device/test_task.py` must remain green after the
submission-mixin refactor.

## 9. Resolved questions

These were open during review and are now settled (folded into the sections
above):

1. **`_finalize` inputs** → bundled into a frozen `FinalizeContext` built by the
   device (§5.5, §6.2).
2. **Finalize error type** → a dedicated `TaskFinalizeError` aggregating dialect
   + validation failures, retaining the kirin `ValidationResult`s (§5.5).
3. **`add_batch_subtask` argument shape** → kernel arguments move to `**kwargs`
   on `add_subtask`; `add_batch_subtask` is shots-only (§5.2, §5.3).
4. **Retrieval / `Future` hardening** → out of scope; deferred to its own issue.
5. **Dialect-check levels 2/3** → not planned; level 1 is the only check.

## 10. Remaining decisions (implementation-time)

- Exact wording/format of the `TaskFinalizeError` per-kernel report and the
  dry-run summary banner.
- Whether `add_subtask` stores empty kwargs as `None` or `{}` in the subtask
  (payload-equivalent, but pick one for consistency).
