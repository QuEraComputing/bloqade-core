# Spec: `Device` owns submission; tasks become device-agnostic builders

Status: Draft
Target package: `bloqade-core` (`bloqade.core.device`)
Relationship to [#107](https://github.com/QuEraComputing/bloqade-core/pull/107):
sibling. #107 adds a **new** standalone `TaskBuilder` alongside the existing
`TaskABC` hierarchy and gives `Device` a `run_async(builder, …)` entry point.
This spec applies the same move to the **existing** classes: strip submission
and device identity out of `TaskABC` so the three shipped shapes become pure
builders, leaving exactly one submission path. The two specs share
`FinalizeContext`, the `TaskSubmitterMixin` refactor, and the
`Device.run_async` signature by design, and can land in either order.

## 1. Motivation

`TaskABC` currently carries three unrelated responsibilities:

1. **Shape** — which kernels, arguments, metadata, and shot counts make up which
   subtasks. Genuinely per-task.
2. **Language** — `program_language`, `language_version`, `kernel_serializer`,
   `serialize_kernel`. A property of the *device* you are submitting to, not of
   the job.
3. **Submission and identity** — `AuthMixin` (hence `context_name`), `group`,
   `future_cls`, `run_async`, `submit_task_definition`, group resolution.
   Entirely a device concern.

Bundling 2 and 3 into the task has concrete costs today:

- **`Device` needs three class-injection slots.** `single_kernel_task_cls`,
  `kernel_batch_task_cls`, and `parameter_scan_task_cls` exist only so a device
  can substitute a task class carrying different defaults, and each needs a
  `cast(...)` to type-check on Python 3.10. Overriding some but not all three
  silently produces base-class tasks — see `device_design.md` §D8a.
- **Downstream pays per-shape.** `bloqade-internal`'s Gemini logical backend
  writes a `GeminiTaskMixin` plus three near-empty task subclasses solely to set
  `program_language`, a version, `future_cls`, and `context_name`. All four are
  field defaults.
- **A task cannot be built without a context name.** `TaskABC` inherits
  `AuthMixin`, so constructing one requires naming an authentication context
  even for a dry-run or a serialization test.
- **Six `@overload`s.** The `Literal[True]/[False]` overloads on `run_async` are
  duplicated per concrete shape.
- **Two ways to submit.** After #107 lands there would be `builder` →
  `device.run_async` and `task` → `task.run_async`, with group resolution,
  logging, and the storage write reachable through both.

A previously-raised objection to putting `run_async` on `Device` was parameter
explosion: `Device.task()` already takes eight arguments and adding
`dry_run`/`storage`/`fetch_options` would make an eleven-parameter method,
tripled across shapes. **The builder is what dissolves that objection** — the
kernel/argument/metadata parameters live on the builder, so
`Device.run_async(spec, *, dry_run, storage, fetch_options)` is four
parameters and two overloads, once.

## 2. Scope

### In scope

- A `TaskSpec` Protocol: the contract `Device.run_async` accepts.
- Removing `AuthMixin`, `future_cls`, `group`, `run_async`, and
  `submit_task_definition` from `TaskABC`.
- Moving `program_language`, `language_version`, `kernel_serializer`, and
  `serialize_kernel` off `TaskABC` and onto `Device`/`FinalizeContext`.
- Renaming `create_task_definition()` → `_finalize(ctx)` on the three shapes,
  matching #107's builder.
- `TaskSubmitterMixin` extraction (identical to #107 §6.1).
- Retiring the three `*_task_cls` slots on `Device`.
- A deprecation shim so every existing `task.run_async(...)` call site keeps
  working unchanged for one release.

### Out of scope

- `TaskBuilder` itself — that is #107. This spec only guarantees `TaskBuilder`
  and the three shapes satisfy the same Protocol.
- `result_cls` forwarding, `Device.future_from_task_id` / `future_from_storage`,
  and re-parameterizing `Device` on the result type. Complementary; tracked
  separately.
- Compilation references (`definition_id` / `compilation_id` submission).
- Storage, `Future`, and `Result` changes of any kind.

## 3. API design

### 3.1 `FinalizeContext` (shared with #107)

```python
@dataclass(frozen=True)
class FinalizeContext:
    program_language: str
    language_version: str
    kernel_serializer: KernelSerializer
    dialect_group: ir.DialectGroup | None = None
    validation_suite: ValidationSuite | None = None
```

Everything a payload needs that the shape does not own. Built by the device;
frozen so it is safe as a shared value.

### 3.2 `TaskSpec` Protocol

```python
@runtime_checkable
class TaskSpec(Protocol):
    def _finalize(self, ctx: FinalizeContext) -> TaskDefinition: ...
    def summary(self) -> str: ...
```

Structural, so `TaskBuilder` (#107) and the three shapes satisfy it without a
shared base class, and a downstream package can supply its own spec object.
`Device.run_async` is typed against this Protocol and nothing else.

### 3.3 `TaskABC` after the change

Retained — the shape contract, all device-agnostic:

```python
@dataclass(kw_only=True)
class TaskABC(ABC):                      # NOTE: no AuthMixin
    @property
    @abstractmethod
    def num_subtasks(self) -> int: ...
    @abstractmethod
    def get_kernels(self) -> list[ir.Method]: ...
    @abstractmethod
    def get_arguments(self) -> list[dict] | None: ...
    @abstractmethod
    def get_metadata(self) -> list[dict] | None: ...
    @abstractmethod
    def get_num_shots(self) -> list[int]: ...

    def program_index_for_subtask(self, i: int) -> int: ...
    def validate_arguments(self) -> None: ...
    def summary(self) -> str: ...
    def _finalize(self, ctx: FinalizeContext) -> TaskDefinition: ...
```

Removed: `AuthMixin` inheritance (and therefore `context_name`),
`program_language`, `language_version`, `kernel_serializer`,
`program_language_version`, `serialize_kernel`, `future_cls`, `group`,
`run_async`, `submit_task_definition`, `_configured_group`, `_resolve_group_id`.

`_finalize(ctx)` does what `create_task_definition` did, plus what
`serialize_kernel` did, reading language/serializer from `ctx`:

1. `self.validate_arguments()`.
2. Optional dialect-group and validation-suite checks from `ctx`, raising
   `TaskFinalizeError` (#107 §5.5) — skipped when both are `None`, which is the
   default, so behavior is unchanged for existing devices.
3. `programs = [Program(content=_encode(k, ctx)) for k in self.get_kernels()]`,
   where `_encode` is the current `serialize_kernel` body as a module-level
   helper taking `ctx` (`dialects.encode(kernel, version=ctx.language_version)`
   → `ctx.kernel_serializer.encode(...)` → base64 if bytes).
4. Subtasks exactly as `create_task_definition` builds them today, including
   `program_index_for_subtask` and the JSON-dumped `TaskMetadata`.
5. `program_language=f"{ctx.program_language}.v{ctx.language_version.removeprefix('v')}"`,
   `group_id=None`.

The three concrete shapes lose their `program_language` requirement and are
otherwise untouched — same fields, same getters, same `summary()`.

`_finalize` is **pure and re-runnable**: dry-run then submit calls it twice and
must produce equal payloads.

### 3.4 `Device` after the change

```python
@dataclass(kw_only=True)
class Device(TaskSubmitterMixin, Generic[FutureType]):
    # language, hoisted off the task
    program_language: str = "squin"
    language_version: str = "0.1.0"
    kernel_serializer: KernelSerializer = field(default_factory=JSONSerializer)
    # optional validation, supplied by downstream devices
    dialect_group: ir.DialectGroup | None = None
    validation_suite: ValidationSuite | None = None
    # from TaskSubmitterMixin: context_name, group, future_cls

    def _finalize_context(self) -> FinalizeContext: ...

    def task(self, kernel, num_shots=1, arguments=None, metadata=None,
             *, program_language=None, language_version=None,
             kernel_serializer=None) -> SingleKernelTask: ...
    def batch_task(...) -> KernelBatchTask: ...
    def parameter_scan(...) -> ParameterScanTask: ...

    @overload
    def run_async(self, spec: TaskSpec, *, dry_run: Literal[True],
                  storage: StorageBackend | None = None,
                  fetch_options: ApiFetchOptions = DEFAULT_FETCH_OPTIONS) -> None: ...
    @overload
    def run_async(self, spec: TaskSpec, *, dry_run: Literal[False], ...) -> FutureType: ...
    def run_async(self, spec, *, dry_run, storage=None,
                  fetch_options=DEFAULT_FETCH_OPTIONS) -> FutureType | None:
        task_def = spec._finalize(self._finalize_context())
        if dry_run:
            print(_DRY_RUN_BANNER + spec.summary())
            return None
        return self.submit_task_definition(
            task_definition=task_def, storage=storage, fetch_options=fetch_options
        )
```

Notes:

- The factory methods keep their per-call `program_language` /
  `language_version` / `kernel_serializer` parameters, but they now default to
  `None` and are **overrides recorded on the returned builder** rather than
  required construction arguments. A builder with all three unset inherits the
  device's values at `run_async` time. (Implementation: three optional fields on
  `TaskABC`, `_finalize` preferring the builder's value over `ctx`'s — the same
  `None`-means-inherit pattern `_resolve_kernel_serializer` already uses.)
- The three `*_task_cls` slots are **deleted**. A device that needs a different
  payload shape now passes its own `TaskSpec` to `run_async`; it does not need
  the factory to construct one. The QASM2 demo's specialization becomes
  `Device(kernel_serializer=QASM2Serializer())`.
- `group` resolution, `authenticate`, `TasksClient.create`, the
  `logger.info(task_id)` line, and `storage.add_task_definition` stay exactly
  where they are, in `TaskSubmitterMixin.submit_task_definition` — one path.

### 3.5 `TaskSubmitterMixin` (identical to #107 §6.1)

`AuthMixin` subclass hosting `group` and `future_cls`, plus
`_configured_group`, `_resolve_group_id`, and `submit_task_definition`. Stays in
`task.py` so the module-level `TasksClient` / `GroupsClient` names that existing
tests monkeypatch remain patchable. `Device` inherits it; `TaskABC` no longer
does.

## 4. Migration

The one breaking change is that `task.run_async(...)` no longer exists on a
builder that has no device. Every current call site — `docs/device_workflow.md`,
`demo/qasm_single_task.py`, `test/device/test_task.py`, and downstream code
built on `device.task(...)` — uses a builder obtained *from* a device, so it can
keep working:

**Phase 1 (this spec).** Builders created by a `Device` factory method carry a
private `_device` back-reference. `TaskABC.run_async(*, dry_run, storage=None,
fetch_options=…)` is retained as a shim:

```python
def run_async(self, *, dry_run, storage=None, fetch_options=DEFAULT_FETCH_OPTIONS):
    warnings.warn(
        "TaskABC.run_async is deprecated; use device.run_async(task, ...).",
        DeprecationWarning, stacklevel=2,
    )
    if self._device is None:
        raise TypeError(
            "This task was constructed directly and has no device. "
            "Submit it with device.run_async(task, dry_run=...)."
        )
    return self._device.run_async(self, dry_run=dry_run, storage=storage,
                                  fetch_options=fetch_options)
```

Every existing call site is source-compatible and warns. Directly-constructed
builders get an actionable error rather than a missing attribute.

**Phase 2 (next minor release).** Delete the shim and `_device`. Update
`device_workflow.md`, the demo, and `device_design.md` (D6's "considered:
`Device.run_async`" note, D8a, D14, D16, and the three-classes ledger all
change).

The back-reference is the only concession to compatibility and is explicitly
temporary; a builder is otherwise device-free, which is the point of the spec.

## 5. What this buys

- **One submission path.** Group resolution, auth, the task-ID log line, and the
  durable storage write are reachable exactly one way.
- **Two overloads instead of six.**
- **Three `Device` fields and three `cast(...)` calls deleted** with the
  `*_task_cls` slots, along with the partial-override footgun.
- **Downstream devices shrink to field defaults.** Gemini logical loses
  `GeminiTaskMixin` and its three task subclasses:
  ```python
  @dataclass(kw_only=True)
  class GeminiLogicalDevice(Device):
      context_name: str = "gemini-logical"
      program_language: str = "squin"
      language_version: str = field(default_factory=lambda: version("bloqade-circuit"))
  ```
- **Builders are testable and serializable.** No `context_name`, no clients, no
  credentials — a payload test constructs a shape and calls `_finalize` with a
  hand-made context.
- **Converges with #107.** `TaskBuilder` and the three shapes become
  interchangeable arguments to one method.

## 6. Design decisions

1. **`TaskSpec` is a Protocol, not a base class.** #107's builder is not a
   `TaskABC` and should not have to become one; a downstream package should be
   able to hand `run_async` any object that can finalize itself.
2. **Language lives on the device, not the task.** A task does not choose its
   language — the machine does. Per-call overrides remain for the mixed-language
   session, as `None`-means-inherit fields on the builder.
3. **`validate_arguments` runs inside `_finalize`, not in `run_async`.** It is
   device-agnostic and must run before any network call; keeping it in
   `_finalize` means a directly-constructed builder validates identically.
4. **`summary()` stays on the spec; the DRY-RUN banner moves to `Device`.** The
   shape knows what it contains; the banner is presentation owned by the caller
   that decided not to submit.
5. **The `*_task_cls` slots are deleted rather than deprecated.** With language
   and submission off the task there is nothing left for a device to inject;
   keeping them would preserve the partial-override hazard for no benefit. The
   escape hatch is now "pass your own `TaskSpec`", which is strictly more
   capable — it can express payload shapes the three classes cannot, such as a
   non-injective subtask→program mapping.
6. **`_finalize` is private on a public Protocol.** Deliberate, following #107:
   users build and inspect; only a device finalizes. `summary()` is the public
   inspection surface.
7. **Deprecate rather than break.** A private `_device` back-reference on
   device-built tasks keeps every documented call site working for one release,
   at the cost of one temporary field.

## 7. Testing plan

- `TaskABC` subclasses construct with **no** `context_name` and no
  `program_language`; `dataclasses.fields()` confirms both are gone.
- `_finalize(ctx)` produces byte-identical payloads to today's
  `create_task_definition()` for all three shapes, including
  `program_language` string formatting, program dedup in `ParameterScanTask`,
  metadata JSON, and `group_id=None`.
- `_finalize` is pure: two calls return equal payloads and leave the builder
  unchanged.
- Builder-level overrides beat `ctx` values; unset overrides inherit them.
- `Device.run_async(spec, dry_run=True)` prints and does not authenticate
  (assert no client construction).
- `Device.run_async(spec, dry_run=False)` submits once, writes the definition to
  storage, and returns a future (mocked clients, as today).
- `Device.run_async` accepts a hand-written `TaskSpec` that is not a `TaskABC`.
- Deprecation: `device.task(...).run_async(dry_run=False)` warns and submits;
  `SingleKernelTask(...).run_async(...)` on a directly-constructed builder
  raises `TypeError` with the remediation message.
- Regression: existing `test/device/test_task.py` and `test_device.py` pass with
  only the deprecation warning added.

## 8. Open questions

1. Should `run_async` accept a bare `ir.Method` as sugar for a one-kernel spec
   (`device.run_async(kernel, num_shots=…)`)? Convenient, but re-introduces
   task-construction parameters onto `run_async` — the thing this spec removes.
   Recommendation: no.
2. Should `Device.task()/batch_task()/parameter_scan()` survive at all once
   #107's `TaskBuilder` exists, or become `TaskBuilder` classmethods
   (`TaskBuilder.scan(kernel, arguments)`)? Deferred until #107 lands; the
   Protocol makes either outcome non-breaking for `run_async`.
3. Does `validate_arguments` belong on the `TaskSpec` Protocol as a public
   method, so a caller can pre-check without finalizing? Cheap to add; unclear
   whether anyone wants it separately from `summary()`.
