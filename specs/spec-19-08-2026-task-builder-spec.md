# Spec: `TaskBuilder`

Status: Draft — **revised to match the implementation** in
`src/bloqade/core/device/task_builder.py` and `device.py`
Owner: Phillip Weinberg (@weinbe58)
Tracking issue: QuEraComputing/bloqade-internal#398 ("TaskBuilder API")
Target package: `bloqade-core` (`bloqade.core.device`)

> This revision folds in the decisions actually made while implementing the
> builder. Sections 1–4 are substantively unchanged; §5–§7 are updated to
> describe the shipped design, §11 records where the implementation
> deliberately departs from the original draft, and §12 lists gaps found while
> reconciling the two. Where the code and the original draft disagree and the
> code looks unintentional, it is filed in §12 rather than promoted to spec.

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

- A standalone `TaskBuilder` class in `bloqade.core.device.task_builder`.
- Wiring it into submission via `Device.run_async(task_builder, ...)`.
- Builder methods: `add_subtask`, `copy`, `_finalize`, `summary`,
  `print_detailed`, plus the private `_validate_subtask`, `_validate_kernels`,
  `_serialize_kernel`, `_get_programs`.
- Program deduplication by object identity.
- Kernel validation at finalize time: a level-1 dialect-group check plus an
  optional kirin `ValidationSuite`, with an aggregated per-kernel report.
- Per-subtask validation at *add* time (shot count, metadata type, argument
  binding against the kernel signature).

### Out of scope (explicitly deferred)

- `add_batch_subtask`. **Moved out of scope** during implementation — see §11.7.
- `device.compile_task_definition(...)` / precompiled task definitions and the
  QLAM compilation-reference flow.
- The retrieval side: `Future` save/load/fetch/concatenate semantics, storage
  hardening, and result-type-by-dialect.
- Any simulator-specific implementation. `bloqade-core` provides the generic
  building blocks; concrete devices, dialect groups, and validation suites come
  from downstream packages (e.g. bloqade-lanes).
- In-place editing (`modify_subtask`, `modify_subtask_shots`, `modify_program`).
- Retiring `TaskABC` and the three `Device` factory methods. They remain, marked
  `# legacy`. See the companion spec,
  [device owns submission](spec-21-08-2026-device-owns-submission.md).

## 3. Background: what already exists

- `TaskABC` (`task.py`) collects kernels + per-subtask data and builds a qlam
  `TaskDefinition` via `create_task_definition()`. It still owns its own submit
  path.
- `Device` (`device.py`) is a factory for the three legacy task shapes and now
  also the submission entry point for builders.
- qlam payload types come from `qlam_core.plugins.tasks.api.tasks_models`:
  `Program(content: str)`, `Subtask(program_index, num_shots, arguments,
  subtask_metadata)`, `TaskDefinition(program_language, programs, subtasks,
  group_id)`, `TaskCreationRequest`, `TaskMetadata`.
- Kernels are `kirin.ir.Method`; `kernel.dialects` is a `DialectGroup` whose
  `.data` is a `frozenset[Dialect]`.

### External types (from `kirin-toolchain`)

- `kirin.ir.DialectGroup` — `.data: frozenset[Dialect]`; `Dialect.name: str`.
- `kirin.validation.ValidationSuite(passes=[...], fail_fast=False)` with
  `.validate(method) -> ValidationResult`. `ValidationResult` exposes
  `.is_valid`, `.errors: dict[str, list[ValidationError]]`, `.frames`,
  `.error_count()`, and `.raise_if_invalid()`.

Three properties of these types the implementation relies on, each verified:

1. **`Method.__hash__` is `id(self)`.** Identity hashing, which is what makes a
   `dict[ir.Method, int]` an identity-keyed map (§5.1).
2. **`ValidationSuite.validate` never propagates a pass failure.** It wraps each
   pass in `try/except` and records a crashing pass as a `ValidationError` under
   that pass's name, message including `traceback.format_exc()`. Only
   `pass_cls()` construction and `validator.name()` sit outside that guard.
3. **`Method.is_structurally_equal` is unusable for dedup** — it returns `False`
   for two independently-lowered but identical kernels, because
   `code.is_structurally_equal` only holds for the same object. Passing an
   explicit `context` dict does not change it. Identity is therefore the only
   viable built-in comparison.

## 4. User-facing workflow

```python
device = SomeLogicalDevice()           # supplies language, dialect group, suite

task = TaskBuilder(_programs={}, _subtasks=[])     # see §12.1
task.add_subtask(k1, 1000)
task.add_subtask(k2, 23)
task.add_subtask(k1, 100)              # reuses k1's program; new subtask

# Kernel arguments bind against the kernel's own signature — positional or
# keyword, checked by the type checker via ParamSpec.
task.add_subtask(k7, 500, {"tag": "cal"}, theta=1.5)

new_task = task.copy()                 # branch without mutating the original

print(task.summary())
print(task.print_detailed())

future = device.run_async(task)
result = future.result()
```

## 5. API design

### 5.1 `TaskBuilder`

A mutable `@dataclass` in `src/bloqade/core/device/task_builder.py`.

```python
@dataclass
class TaskBuilder:
    _programs: dict[ir.Method, int]
    _subtasks: list[_Subtask]
    _max_program_idx: int = 0
    _kernel_name_counter: dict[str, int] = field(default_factory=dict)
```

- **`_programs` is a dict keyed by the kernel object**, mapping kernel →
  program index. Because `Method.__hash__` is `id(self)`, this *is* identity
  dedup, with O(1) lookup instead of the O(n) scan a list would need. Keying on
  the object rather than on `id(kernel)` also holds a strong reference, so an
  `id` cannot be recycled onto a different kernel after garbage collection.
- **`_max_program_idx`** is the next index to assign.
- **`_kernel_name_counter`** disambiguates display names: the first `bell`
  stays `bell`, subsequent ones become `bell_1`, `bell_2`, … This exists because
  `Method.sym_name` is the *bare* name (not the qualname) and is typed
  `str | None`, so it is neither unique nor guaranteed present.

The builder holds no device, dialect group, validation suite, serializer, or
language. Those arrive in `FinalizeContext` (§5.5).

### 5.2 `_Subtask`

```python
@dataclass
class _Subtask:
    kernel_name: str
    qlam_subtask: Subtask
```

A thin wrapper pairing a display name with an **eagerly constructed qlam
`Subtask`**. The payload object is built at `add_subtask` time, not at finalize.

Deliberately *composition, not inheritance*: subclassing `Subtask` to add
`kernel_name` would make it a declared pydantic field, and since every model in
`tasks_models` sets `extra="allow"`, that field would be serialized into the
request body — where the OpenAPI schema declares `additionalProperties: false`
and would most likely drop it silently. Wrapping keeps `kernel_name` off the
wire by construction.

### 5.3 `add_subtask(kernel, num_shots, metadata=None, *args, **kernel_args) -> int`

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

**Arguments bind against the kernel's real signature.** Rather than collecting
`**kwargs` blindly, the implementation resolves them:

```python
signature = inspect.signature(kernel.py_func)
bound = signature.bind(*args, **kernel_args)
arguments = dict(bound.arguments)
```

This buys three things the original draft did not have:

- **Positional arguments work.** `add_subtask(kernel, 100, None, 1.5, 0.2)`
  binds to the kernel's parameter names.
- **Wrong argument names fail immediately**, with `signature.bind`'s error,
  rather than being sent to the backend as parameters the program does not
  declare.
- **`ParamSpec` typing.** `kernel: ir.Method[CallArgs, T]` with
  `*args: CallArgs.args` / `**kernel_args: CallArgs.kwargs` means a type checker
  validates the call against the kernel's own signature.

`kernel.py_func is None` raises `TypeError`, since argument names cannot be
recovered without it.

Remaining behavior:

- Looks up `kernel` in `_programs`; assigns and increments `_max_program_idx`
  when absent. This is the only place a program is created.
- `metadata` must be a `dict` or `None` (`ValueError` otherwise) and is
  JSON-dumped into `TaskMetadata(user_metadata=...)` immediately.
- Builds the qlam `Subtask`, wraps it in `_Subtask`, runs `_validate_subtask`,
  and appends.
- **Reserved names.** `kernel`, `num_shots`, and `metadata` are named
  parameters, so they cannot be used as keyword kernel-argument names. (They can
  still be reached positionally through `*args`.)

### 5.4 `_validate_subtask(subtask) -> bool`

Per-subtask checks at *add* time, so a failure's traceback points at the
offending call rather than at `_finalize`. Currently: `num_shots` must not be
negative. A `ValueError` from here is re-raised as `SubtaskValidationError`.

Kernel-level checks are **not** done here — they belong to `_validate_kernels`,
which needs the device's dialect group and suite.

### 5.5 `_finalize(ctx: FinalizeContext) -> TaskDefinition`

Internal; normally called by `Device.run_async`. Non-mutating with respect to
builder state and re-runnable: no finalize lock, no `_finalized` flag.

```python
@dataclass(frozen=True)
class FinalizeContext:
    program_language: str
    language_version: str
    kernel_serializer: KernelSerializer
    dialect_group: ir.DialectGroup | None = None
    validation_suite: ValidationSuite | None = None
```

Order of operations:

1. `_validate_kernels(dialect_group=ctx.dialect_group,
   validation_suite=ctx.validation_suite)` — §5.6.
2. `_get_programs(...)` — one `Program` per entry in `_programs`, ordered by
   program index, each serialized by `_serialize_kernel`.
3. Rebuild **plain `Subtask` objects** from each `_Subtask.qlam_subtask`,
   copying `program_index`, `num_shots`, `arguments`, `subtask_metadata`. This
   re-wrap is what guarantees no builder-internal field reaches the payload.
4. `program_language = f"{ctx.program_language}.v{ctx.language_version.removeprefix('v')}"`.
5. `TaskDefinition(..., group_id=None)` — group resolution stays in the submit
   path.

> **Purity caveat.** `_finalize` does not mutate the *builder*, but
> `ctx.validation_suite` is stateful: `ValidationSuite.validate` clears and
> repopulates `self._analysis_cache`. Since the suite lives on the `Device` and
> is shared across every task that device builds, `_finalize` is not safe for
> concurrent use, and shared analyses are recomputed per kernel rather than
> cached across them.

### 5.6 `_validate_kernels(dialect_group=None, validation_suite=None)`

Returns immediately when both are `None`. Otherwise, for each unique program
kernel — iterated in program-index order — it runs both checks and accumulates
problems:

- **Dialect check (level 1).** `unsupported = kernel.dialects.data -
  dialect_group.data`; non-empty means failure, reported as the sorted
  `Dialect.name`s. Using the difference rather than `<=` yields the offenders
  directly; the test is equivalent.
- **Validation suite.** `validation_suite.validate(kernel)`, with the result
  stored in `results[label]` and, when invalid, formatted by
  `_format_validation_errors`. The call is wrapped in `try/except` as a backstop
  for the two statements outside the suite's own per-pass guard.

The two checks are installed as **closures chosen once**, before the loop —
no-ops when the corresponding context field is `None` — so the per-kernel loop
carries no `None` tests.

Kernels are labelled `f"program {index} ({kernel.sym_name or '<anonymous>'})"`.
The index is included because `sym_name` is neither guaranteed present nor
unique.

On any failure, one `TaskFinalizeError` is raised carrying an aggregated
per-kernel report and the `ValidationResult`s. No submission happens.

`_format_validation_errors` flattens `result.errors` to
`[pass_name] first_line_of_message`, plus `err.hint()` when available, mirroring
kirin's own extraction without its ANSI colour codes.

### 5.7 Serialization

`_serialize_kernel(kernel, kernel_serializer, program_language_version)` lives
on the builder and reproduces `TaskABC.serialize_kernel`:
`kernel.dialects.encode(kernel, version=...)` → `kernel_serializer.encode(...)`
→ base64 when the serializer returns `bytes`, passthrough when it returns `str`,
`TypeError` otherwise.

### 5.8 Exceptions

```python
class SubtaskValidationError(Exception):
    """A subtask's arguments, shot count, or metadata are invalid."""

class TaskFinalizeError(Exception):
    """Kernels failed dialect or validation checks; nothing was submitted."""
    def __init__(self, message: str, *, results: dict[str, ValidationResult] | None = None):
        super().__init__(message)
        self.results = results or {}
```

Both take message-only `super().__init__()` so `str(exc)` stays clean, and keep
extra data keyword-only with defaults so the exceptions remain reconstructible
when pickled (`BaseException.__reduce__` calls `cls(*self.args)`).

### 5.9 `copy() -> TaskBuilder`

Shallow: fresh `_programs` dict and `_subtasks` list containers, same
`ir.Method` and `_Subtask` objects, `_max_program_idx` carried over. Kernels are
treated as immutable values.

### 5.10 Display

`summary()` renders one line per subtask (`kernel_name`, program index, shots);
`print_detailed()` renders programs (IR) then subtasks with arguments. Both
return `str`.

## 6. Wiring into `Device`

New fields on `Device`:

```python
program_language: str = ""
language_version: str = "0.1.0"
kernel_serializer: KernelSerializer = field(default_factory=JSONSerializer)
validation_suite: ValidationSuite | None = None
dialect_group: ir.DialectGroup | None = None
```

`dialect_group` and `validation_suite` default to `None`, meaning "no check
configured." An *empty* `DialectGroup` is not equivalent — it rejects every
kernel, since `kernel.dialects.data <= frozenset()` is false for any non-trivial
kernel — so `None` is the only way to express "skip." (Both types are
`@dataclass` with `eq=True`, hence unhashable, hence rejected by dataclasses as
mutable defaults; `None` avoids needing `default_factory`.)

`_finalize_context()` bundles all five into a `FinalizeContext`.

`run_async(task_builder, group=None, storage=None, fetch_options=...)`:

1. `task_definition = task_builder._finalize(self._finalize_context())`
2. `return self.submit_task_definition(task_definition=..., storage=...,
   fetch_options=..., group=group)`

`group` is a **per-call parameter** threaded into `submit_task_definition` and
`_configured_group(group)`, rather than a field. Precedence is unchanged:
explicit `group` → `plugins.tasks.group` → `defaults.group` → backend default.

The three `*_task_cls` slots and the `task` / `batch_task` / `parameter_scan`
factories remain, marked `# legacy`.

## 7. Design decisions

1. **Standalone builder + `device.run_async(builder, ...)`.** The builder is a
   plain mutable object, not a `TaskABC`; the device owns the run entry point.
2. **`_finalize` receives a `FinalizeContext`**, built by the device at run
   time, rather than storing language/serializer/validation on the builder.
3. **`_finalize` is non-mutating and re-runnable** with respect to builder
   state. No lock, no flag. (See the caveat in §5.5 about the shared suite.)
4. **Dialect check is level 1 only** (kernel dialects ⊆ group). The set-equal
   and object-identity variants are not planned.
5. **`_finalize` failures raise `TaskFinalizeError`**, aggregating dialect and
   validation failures into one per-kernel report and retaining the
   `ValidationResult`s.
6. **Programs are deduplicated by object identity**, via a `dict` keyed on the
   kernel. Structural equality is not used — `Method.is_structurally_equal`
   returns `False` for genuine twins, and `==` is field-wise over
   identity-compared fields (and inconsistent with `__hash__` for copies). A
   content fingerprint over the serialized program would work but requires
   encoding, which is unavailable at `add_subtask` time because the serializer
   and version live in the `FinalizeContext`.
7. **Kernel arguments bind against the kernel signature** via
   `inspect.signature(kernel.py_func).bind(*args, **kernel_args)`, supporting
   positional and keyword forms and rejecting unknown names at call time.
   `ParamSpec` propagates the kernel's signature to type checkers.
8. **`_Subtask` wraps a qlam `Subtask`; it does not subclass it.** Subclassing
   would put `kernel_name` on the wire (§5.2).
9. **Validation is split by stage.** Subtask-shaped checks run at `add_subtask`
   (fast, offline, traceback points at the caller); kernel-shaped checks run at
   `_finalize` (they need the device's dialect group and suite).
10. **Two exception types**, one per stage: `SubtaskValidationError` for add-time
    problems, `TaskFinalizeError` for finalize-time ones.
11. **Display names are uniquified** with a counter, because `sym_name` is
    optional and non-unique.
12. **`group` is a per-call argument to `run_async`**, not a builder or device
    field — the builder is device-agnostic, and a device is reused across
    groups.
13. **In-place editing (`modify_*`) stays out.** Editing risks silently mutating
    a program shared by other subtasks; rebuilding (or `copy()`-ing) is the
    intended friction. Validation failures raise before any submission, so
    there is no partial-submission state to repair.

## 8. Testing plan

New `test/device/test_task_builder.py`:

- `add_subtask`: program dedup by identity; distinct kernels get distinct
  program indices; the return value (see §12.2).
- Argument binding: keyword form; positional form; unknown name raises;
  `py_func is None` raises `TypeError`; no-argument case.
- `metadata`: dict is JSON-dumped into `TaskMetadata.user_metadata`; non-dict
  raises; `None` leaves `subtask_metadata` unset.
- `_validate_subtask`: negative `num_shots` raises `SubtaskValidationError`.
- Kernel-name uniquification across repeated `sym_name`s, including through
  `copy()` (§12.3).
- `copy` isolates both containers and shares kernel objects.
- `_finalize` payload: programs ordered by index, dedup reflected in
  `program_index`, fused `program_language`, `group_id=None`, plain `Subtask`
  instances with no `kernel_name` in `model_dump()`.
- `_finalize` is non-mutating and produces equal payloads on repeat calls.
- `_validate_kernels`: dialect failure; validation failure; both together in one
  `TaskFinalizeError`; `results` populated; each `None` skips its check; a
  crashing pass is reported per-pass rather than propagating.
- `summary` / `print_detailed` formatting.

Extend `test/device/test_device.py`:

- `Device.run_async(builder)` finalizes and submits against mocked clients.
- `group` precedence: explicit argument beats config.

Regression: `test/device/test_task.py` stays green — `TaskABC` is untouched.

## 9. Resolved questions

1. `_finalize` inputs → a frozen `FinalizeContext` built by the device.
2. Finalize error type → `TaskFinalizeError`, aggregating both check kinds and
   retaining `ValidationResult`s.
3. Argument shape → bound against the kernel signature, positional and keyword.
4. Program equality → object identity; structural and content-based comparison
   ruled out (§7.6).
5. Retrieval / `Future` hardening → out of scope.
6. Dialect-check levels 2/3 → not planned.

## 10. Remaining decisions (implementation-time)

- Exact wording of the `TaskFinalizeError` report (§12.7 has two concrete
  problems with the current format).
- Whether empty `kernel_args` should store `{}` or `None` in the subtask
  (payload-equivalent; currently `{}` via `dict(bound.arguments)`).
- Whether `arguments` should be coerced to `float` at add time, so what is
  stored equals what is sent. The wire type is `dict[str, float]`; note `bool`
  is a `numbers.Real` and coerces to `1.0`/`0.0`, and NaN/inf have no JSON
  representation.

## 11. Departures from the original draft

| # | original draft | implementation | why |
|---|---|---|---|
| 1 | module `builder.py`, exported from `__init__` | `task_builder.py`, **not exported** | naming; the export is still owed (§12.5) |
| 2 | `_programs: list[ir.Method]` | `dict[ir.Method, int]` + `_max_program_idx` | O(1) identity lookup; the dict holds a strong reference so `id` reuse cannot alias |
| 3 | `_Subtask` = flat internal dataclass | wrapper around an eagerly built qlam `Subtask` | payload built once at add time; wrapping (not subclassing) keeps `kernel_name` off the wire |
| 4 | `add_subtask(..., *, metadata=None, **kernel_args)` | `(..., metadata=None, *args, **kernel_args)` with signature binding + `ParamSpec` | positional args, name checking, and type-checked calls |
| 5 | no per-subtask validation | `_validate_subtask` + `SubtaskValidationError` | fail at the offending call, not at finalize |
| 6 | no name disambiguation | `_kernel_name_counter` | `sym_name` is optional and non-unique |
| 7 | `add_batch_subtask` in scope | deferred (commented out) | "need to think about it more"; `add_subtask` in a loop covers it |
| 8 | `run_async(builder, *, dry_run, ...)` with `Literal` overloads | `run_async(builder, group=None, storage=None, fetch_options=...)`, **no `dry_run`** | see §12.4 |
| 9 | `group` on the submitter mixin | per-call parameter on `run_async` / `submit_task_definition` | builder is device-agnostic; one device serves many groups |
| 10 | extract `TaskSubmitterMixin` shared by `TaskABC` and `Device` | `submit_task_definition` **duplicated** on both | see §12.6 |
| 11 | `Device.task_builder()` factory | not implemented | direct `TaskBuilder(...)` construction only |
| 12 | `program_language: str = "squin"` | `""` on `Device`; `"squin"` still on the legacy factories | see §12.8 |
| 13 | dialect check then validation suite | validation suite then dialect check | affects report ordering only |
| 14 | "same encoding path as `TaskABC.serialize_kernel`" | `_serialize_kernel` duplicated on the builder | two copies to keep in sync |

## 12. Open items found while reconciling

1. **`TaskBuilder()` does not work.** `_programs` and `_subtasks` have no
   defaults, so construction requires `TaskBuilder(_programs={}, _subtasks=[])`
   — passing private fields explicitly. §4 of the original draft promises
   argument-free construction. Fix: `field(default_factory=dict)` and
   `field(default_factory=list)`.
2. **`add_subtask` returns `program_idx`, not the subtask index.** The draft's
   worked example expects `add_subtask(k1, 100)` to return `3` (a new subtask
   reusing program 0); the implementation returns `0`. Decide which is intended
   — the subtask index is what identifies the thing just added, and program
   indices are not unique across calls.
3. **`copy()` drops `_kernel_name_counter`.** A copy restarts naming, so a
   branch can produce duplicate `kernel_name`s — the exact thing the counter
   exists to prevent.
4. **`dry_run` is unimplemented.** No dry-run parameter, no summary print, no
   `Literal` overloads. `run_async` always submits, so there is currently no way
   to preview a builder-based task — a capability `TaskABC.run_async` has.
5. **Neither exception nor `TaskBuilder`/`FinalizeContext` is exported** from
   `bloqade.core.device`, so `except TaskFinalizeError` requires a private
   submodule import. A shared `BloqadeError` base would let callers catch either
   exception with one clause.
6. **`submit_task_definition` and `_configured_group` now exist on both `Device`
   and `TaskABC`.** Group resolution, auth, the task-ID log line, and the
   storage write are duplicated. The draft's `TaskSubmitterMixin` was the fix
   for exactly this.
7. **Report formatting.** Continuation lines inherit the `- ` bullet, so a
   truncated traceback renders as `-   (full traceback...)`; and device-wide
   problems (one missing dialect, one crashing pass) repeat verbatim under every
   kernel, burying per-kernel signal. Consider hoisting problems common to all
   programs into a preamble, and reporting a crashed pass separately from a real
   violation — a broken suite is a device bug, not a user error, and counting it
   in "N of M programs failed validation" misattributes blame.
8. **`Device.program_language` defaults to `""`**, so a default `Device` emits
   `program_language=".v0.1.0"`. Either require it or default it to `"squin"` as
   the legacy factories do.
9. **`_validate_subtask` accepts `num_shots == 0`** (`< 0` is the only check) and
   accepts `True`, since `isinstance(True, int)`. The API expects a positive
   count.
10. **`SubtaskValidationError` is raised without `from e`**, discarding the
    original traceback.
11. **`summary()` / `print_detailed()` emit no newlines** between entries, so
    lines run together; and `print_detailed` interpolates `program.print()`,
    which writes to a rich console and returns `None`, rendering the literal
    `"None"`. §5.6 of the draft notes the IR dump needs a recording console.
