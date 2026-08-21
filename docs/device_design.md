# Design decisions in `bloqade.core.device`

This page documents *why* the device subsystem is built the way it is. It is
written for contributors and for anyone specializing the API in a downstream
package (a hardware SDK, `bloqade-circuit`, a simulator backend). For a
task-oriented walkthrough of the happy path, see
[Running kernels on a device](device_workflow.md).

## How to read this

Every entry has the same three parts:

- **The decision** — what the code does, stated flatly, with the identifiers
  involved so you can find it.
- **Why** — the reason it is that way. Every numbered `Dn`/`Rn` entry has one.
  If one ever loses its `**Why.**`, that is a bug in the document: an
  undocumented decision is indistinguishable from an accident, and the next
  person will "clean it up." (Sections 9–13 are reference material, not
  decisions, and are exempt.)
- **Consequence** — what you inherit by keeping it, where there is something
  non-obvious to inherit. Omitted when the decision is self-contained.

Decisions are numbered `D1`, `D2`, … so they can be cited in reviews and
issues. Numbering is append-only: a decision inserted into an existing section
later gets a letter suffix (`D8a`) rather than renumbering everything below it,
so citations stay valid. Decisions that were made and later revised are kept in
[Revised decisions](#revised-decisions) rather than deleted — the reasoning
that lost is often the reasoning that comes back.

### Index

**[Architecture](#1-architecture)** — D1 five layers · D1a `typing_extensions`
for 3.10 · D1b explicit re-exports · D1c what is not exported · D2 storage is
the source of truth · D2a what is frozen and what is not · D3 class-injection
slots · D4 `kw_only` dataclasses · D4a mixin-first MRO · D5 synchronous, no
`asyncio`

**[Device](#2-devicepy-the-factory)** — D6 factory cannot submit · D7 three
task shapes · D8 defaults, not state · D8a three task-class slots · D9 per-call
overrides · D9a no backend introspection on `Device`

**[Auth](#3-mixinspy-authentication)** — D10 auth as a mixin · D11 clients per
call · D12 lazy idempotent login · D13 single 401/403 retry

**[Task](#4-taskpy-payload-construction)** — D14 abstract getters, one
assembler · D14a `num_subtasks` as an abstract property · D15 local validation
first · D16 `dry_run` overloads · D17 program
dedup hook · D18 injectable serializer · D19 fused language string · D20 JSON
metadata · D20a group as a name · D21 group resolved at submit · D22 durable
before return · D22a explicit `group_id=None` · D23 in-memory storage default ·
[ledger: three task classes versus one](#ledger-three-task-classes-versus-one)

**[Future](#5-futurepy-the-async-handle)** — D24 handle, not container · D24a
mutable, with private watermark state · D25 frozen fetch options · D25a the
page-size defaults · D26 capped backoff · D27 terminal statuses · D28 two-axis
pagination · D29 idempotent by construction · D30 `completed_date` handling ·
D31 blocking vs partial · D32 recovery constructors · D32a `result_cls`
forwarding · D33 boundary normalization · D34 class-level `context_name` · D35
chunked export

**[Storage](#6-local_storagepy-persistence)** — D36 storage ABC · D36a
streaming `get_shots` · D37 flat shot rows · D38 decomposed metadata · D38a
`DictStorage` mirrors SQL · D38b one mutable column · D39 two filter levels ·
D40 predicates compile to filters · D41 copies on read · D42 empty means
nothing · D43 SQLite choices · D43a idempotent bootstrap · D43b surrogate key ·
D44 schema versioning · D45 ABC evolution policy · D46 summarizing `repr`

**[Result](#7-resultpy-the-analysis-view)** — D47 a query, not data · D47a
`DETECTED` default · D47b validation memoized · D47c filter downcast · D48 merge
contract · D49 merged vs full · D50 raw bitstrings only · D50a one query per
subtask · D51 subclass-preserving narrowing · D52 dual filters in `where_shots`
· D53 documented coercion

**[Logging](#8-log_infopy-logging)** — D54 opt-in · D54a loguru namespace ·
D54b `warn` versus `logger`

**Reference, not decisions** — [extension points](#9-extension-points) ·
[the verified QLAM contract](#10-the-api-contract-verified) ·
[dependency map](#11-dependency-map) ·
[data model: QLAM primitives, local tables, `ShotResult`, frame alignment](#12-data-model-reference) ·
[frequently asked](#13-frequently-asked) ·
[revised decisions](#revised-decisions) ·
[known trade-offs](#known-trade-offs-and-open-todos)

**Related specs.** [PR #107](https://github.com/QuEraComputing/bloqade-core/pull/107)
(standalone `TaskBuilder`, kernel validation) and
[device owns submission](https://github.com/QuEraComputing/bloqade-core/blob/main/specs/spec-21-08-2026-device-owns-submission.md)
(tasks become device-agnostic builders). Decisions here that those specs would
change are marked where they appear.

This reflects `main` at `d0dcb66`. The subsystem was introduced in
[PR #65](https://github.com/QuEraComputing/bloqade-core/pull/65) and extended
in [#101](https://github.com/QuEraComputing/bloqade-core/pull/101)
(QLAM 0.6.0 integration). Claims about backend behaviour were checked against
the QLAM OpenAPI specs and the compiler pipeline; see
[The API contract, verified](#10-the-api-contract-verified).

---

## 1. Architecture

### D1 — Five layers, one direction of flow

```
Device  ──▶  Task  ──▶  Future  ──▶  StorageBackend  ──▶  Result
factory      payload     handle      durable rows        query view
```

Each layer knows only the layer it hands off to. `Device` never talks to the
network; `Task` never reads results; `Future` never interprets bitstrings;
`StorageBackend` never knows what a kernel is; `Result` never talks to the API.

**Why.** The five concerns have genuinely different lifetimes. A device is
configuration that lives for a session. A task is a payload that is valid
before any credentials exist. A future is a handle that must survive a process
restart. Storage outlives the process entirely. A result is a question asked
against storage, possibly months later. Collapsing any two of them couples a
long-lived concern to a short-lived one.

**Consequence.** Any single layer can be replaced without touching the others,
which is what makes [D3](#d3-every-seam-is-a-class-injection-slot) work.

### D1a — `typing_extensions` for `Self` and `TypeVar(default=…)`

`future.py` and `result.py` import `Self` and `TypeVar` from
`typing_extensions`, not `typing`.

**Why.** The package supports Python 3.10. `Self` (PEP 673) is 3.11+ in the
standard library and `TypeVar(default=…)` (PEP 696) is 3.13+, but both are
load-bearing here: `Self` is what makes `Future.from_storage` return the
*subclass*, and the `default=Future[Result]` on `FutureType` is what lets
`Device` be written without a mandatory type parameter.
`typing_extensions` backports both.

**Consequence.** When the floor rises to 3.13, these imports and the casts from
[D3](#d3-every-seam-is-a-class-injection-slot) can be deleted together — they
are the same workaround wearing two hats.

### D1b — Re-exports are explicit `X as X`

`__init__.py` is written `from .device import Device as Device`, never
`from .device import Device`.

**Why.** The redundant-looking alias is the PEP 484 marker for an intentional
re-export. Without it, strict type checkers treat the name as a private import
and flag downstream `from bloqade.core.device import Device` as an error. It
also documents intent: anything in this file is public API, anything reachable
only through a submodule is not.

### D1c — The exported surface is the nouns users construct

Exported: `Device`, `Future`, `Result`, `DictStorage`, `SQLiteStorage`,
`ShotFilter`, `ShotResult`, `KernelSerializer`, `set_logging`. Not exported:
`TaskABC` and the three task classes, `StorageBackend`, `StorageFilter`,
`AuthMixin`, `ApiFetchOptions`.

**Why.** Tasks are obtained from a device, never constructed directly, so
exporting them would advertise a second way to do the same thing. The
abstract bases are for implementers, who are already reading the module.

**Consequence.** Two of the omissions are arguably wrong rather than
deliberate: `StorageBackend` is what a custom backend must subclass, and
`ApiFetchOptions` is what a user must construct to change poll intervals — both
currently require importing from `bloqade.core.device.local_storage` and
`.future`. See [Known trade-offs](#known-trade-offs-and-open-todos).

### D2 — Storage is the source of truth, not `Future` and not `Result`

`Future` holds `task_id`, `storage`, `fetch_options`, `context_name` — no
shots. `Result` holds `storage` + a `ShotFilter` — no shots. Every read
re-queries the backend store.

**Why.** Shot data is the only large object in the system and the only one that
must survive a crash. Keeping it in exactly one place means there is never a
stale in-memory copy to reconcile, `Result` narrowing costs nothing, and
"close the laptop, resume tomorrow" needs no special path — it is the normal
path.

**Consequence.** Reads are not free: `Result.shot_results()` hits the store
every call. Callers that need the arrays repeatedly should hold onto the
returned list. In exchange, holding a hundred `Result` views costs nothing.

### D2a — Values are frozen, handles are not

Frozen: `ShotResult`, `StorageFilter`, `ShotFilter`, `ApiFetchOptions`. Not
frozen: `Device`, the task classes, `Future`, `Result`.

**Why.** The split follows identity. A filter, a fetch-options bundle, and a
shot row are *values* — two with the same contents are interchangeable — so
freezing them buys safe sharing as default arguments
([D25](#d25-apifetchoptions-is-frozen-and-there-is-one-module-level-default)),
`dataclasses.replace` narrowing ([D51](#d51-where_-returns-typeself-over-a-replaced-filter)),
and no aliasing bugs. A device, task, future, or result is an *entity* with a
lifecycle, and two of them hold caches that must mutate
([D24a](#d24a-future-is-mutable-and-its-watermark-is-private-initfalse-state),
[D47b](#d47b-validation-is-memoized-in-a-private-field)).

**Consequence.** `ShotResult` is frozen but holds an `np.ndarray`, so it is
unhashable and only shallowly immutable — mutating `shot.bitstring` in place is
possible and unguarded.

### D3 — Every seam is a class-injection slot

`Device.future_cls`, `Device.single_kernel_task_cls`, `.kernel_batch_task_cls`,
`.parameter_scan_task_cls`, `Future.result_cls`, and the `storage` argument are
all substitution points. Generic parameters carry the substitution through the
type checker: `FutureType = TypeVar("FutureType", bound=Future[Any],
default=Future[Result])` and `ResultType = TypeVar("ResultType",
bound="Result")`.

**Why.** `bloqade-core` deliberately knows nothing about any specific machine,
logical encoding, or program language. Downstream packages need to specialize
one layer — usually `Result` — without forking the other four. Injection slots
plus defaulted TypeVars give them that *and* keep `device.task(...).run_async(
dry_run=False).result()` correctly typed as their own `Result` subclass.

**Consequence.** The `cast(...)` calls and `# type: ignore` comments in
`device.py` and `task.py` exist only to satisfy Python 3.10, where
`type[Future[Result]]` cannot be a `TypeVar` default in a dataclass field.
The NOTE comments say so; do not "clean them up" without dropping 3.10.

### D4 — Everything is a `@dataclass(kw_only=True)`

No custom `__init__` anywhere in the subsystem. Filters are
`@dataclass(frozen=True)`.

**Why.** `kw_only=True` lets a subclass add required fields without breaking
positional-argument order in the base — the whole extension story depends on
that. Frozen filters can be safely shared as default argument values and
narrowed with `dataclasses.replace`, which is the mechanism behind every
`Result.where_*` method.

### D4a — The auth mixin comes first in every MRO

`class TaskABC(AuthMixin, ABC, Generic[FutureType])`, `class Device(AuthMixin,
Generic[FutureType])`, `class Future(AuthMixin, Generic[ResultType])`.

**Why.** Dataclass fields are collected in reverse MRO order, so the base
listed first contributes its fields *earliest*. Putting `AuthMixin` first means
`context_name` is established before any subclass field, which keeps the
generated `__init__` signatures stable across the hierarchy. With
`kw_only=True` ([D4](#d4-everything-is-a-dataclasskw_onlytrue)) this no longer
affects call sites, but consistent ordering keeps `dataclasses.fields()`
output — used by tests and `repr` — predictable.

**Consequence.** `TaskABC` originally read `Generic[FutureType], AuthMixin,
ABC`; it was reordered in #101. Keep the convention when adding a class.

### D5 — Synchronous polling, no `asyncio`, not a `concurrent.futures.Future`

`Future` spawns no threads and starts no event loop. `_wait_for_completion`
sleeps in the calling thread.

**Why.** The primary consumer is a notebook or a script. An event loop imposes
`async`/`await` on every user of the library and interacts badly with Jupyter's
own loop; a thread pool introduces hidden concurrency and makes the object
unpicklable and unreproducible. Because the object is inert data keyed by
`task_id`, it can instead be *reconstructed* — see
[D32](#d32-two-recovery-constructors-that-refuse-to-guess).

**Consequence.** A user waiting on ten tasks blocks on them in sequence. That
is accepted: the intended pattern for breadth is one task with many subtasks
([D7](#d7-three-task-shapes-not-one-generic-submit)), not many tasks in
parallel.

---

## 2. `device.py` — the factory

### D6 — A factory that deliberately cannot submit

`Device` has no `run`, `submit`, or `execute`. Its three methods return task
objects.

**Why.** Building a payload is pure; submitting has four side effects (auth,
network, backend state, storage writes). Splitting them means payload
construction is testable with no credentials, inspectable before it leaves the
machine, and reusable — you can hold a task and submit it twice.

**Consequence.** Two-step usage (`task = device.task(...)`, then
`task.run_async(...)`) is slightly more verbose than a one-liner.

**Considered: `Device.run_async`.** A combined build-and-submit method on
`Device` would be legal — it could delegate to `self.task(...).run_async(...)`
and preserve every invariant — and it is what comparable SDKs do
(`backend.run(circuit, shots=…)`). It is absent for three reasons, none of them
a hard invariant. First, parameter count: `task()` already takes eight
arguments and `run_async` adds three from an unrelated concern, so a combined
method is eleven parameters, and the `Literal` overloads
([D16](#d16-one-run_async-with-dry_run-plus-overloads-on-literal)) would go from
two to six across the three shapes. Second, coupling: `device.py` imports
nothing from `local_storage` and does not touch `ApiFetchOptions`, and
submission would pull both into the one class kept free of I/O. Third, the task
object is genuinely useful — it can be inspected, archived, or submitted twice,
which a single call cannot return alongside a future.

If it is added, it should be one sugar method on the single-kernel path that
delegates rather than reimplements, and `storage` should be *required* on it —
with the in-memory default from [D23](#d23-storage-defaults-to-a-fresh-dictstorage),
a one-liner makes it easy to submit a billable task whose ID lives only in a
discarded object.

### D7 — Three task shapes, not one generic `submit`

`task()`, `batch_task()`, `parameter_scan()` differ only in how they fan out:

| method | subtasks | programs | `program_index` |
|---|---|---|---|
| `task` | 1 | 1 | 0 |
| `batch_task` | one per kernel | one per kernel | `i` |
| `parameter_scan` | one per argument set | **1, reused** | `0` |

**Why.** All three compile to the same QLAM `TaskDefinition`, so a single
generic method was possible. The three names exist because the *shapes* have
different correctness rules (what must be length-matched against what) and
different efficiency implications (a 500-point scan must upload one program,
not 500 identical copies). Encoding that in a type rather than in documentation
means `validate_arguments` and `program_index_for_subtask` can be specialized
per shape.

**Consequence.** A fourth fan-out shape is a new `TaskABC` subclass plus a
`Device` slot, not a new flag.

### D8 — `Device` carries defaults, never per-task state

Fields: `context_name` (via `AuthMixin`), `future_cls`, `kernel_serializer`,
and the three `*_task_cls` slots (all `init=False`).

**Why.** A device is a *configuration object*, so it must be safe to reuse
across arbitrarily many submissions and safe to share. Anything per-submission
lives on the task.

**Consequence.** Concrete devices are made by subclassing and overriding the
`init=False` slots, not by passing them in — see
[Extension points](#9-extension-points).

### D8a — Three separate task-class slots, one per shape, all `init=False`

```python
single_kernel_task_cls: type[SingleKernelTask[FutureType]] = field(
    default=cast(type[SingleKernelTask[FutureType]], SingleKernelTask),
    init=False,
)
kernel_batch_task_cls: type[KernelBatchTask[FutureType]] = field(...)
parameter_scan_task_cls: type[ParameterScanTask[FutureType]] = field(...)
```

**Why three and not one.** The three task classes are not interchangeable —
they have different constructor signatures, because the shapes take different
data: `SingleKernelTask(kernel=…, arguments: dict | None)` versus
`KernelBatchTask(kernels=…, arguments: list[dict] | None)` versus
`ParameterScanTask(kernel=…, arguments: list[dict])` (required, it *is* the
axis). A single `task_cls` slot would have to be typed as the common
supertype `type[TaskABC]`, and every factory method would then be calling a
constructor the type checker cannot verify. One slot per shape keeps each
factory method's construction call and its return annotation exact, so
`device.parameter_scan(...)` is known to return a `ParameterScanTask` and IDE
completion works on the shape-specific fields.

**Why `init=False`.** Which task class a device builds is a property of the
device *type*, not of an instance: a QASM2 device always builds QASM2 tasks.
Exposing them as constructor parameters would invite mixing incompatible
classes at runtime and would clutter `Device(context_name=…)`, which is the
only argument an end user should need. Overriding is therefore done by
subclassing with `field(default=…, init=False)`.

**Why the `cast`.** `SingleKernelTask` is generic in `FutureType`, and the bare
class object does not type as `type[SingleKernelTask[FutureType]]` when the
TypeVar is unbound in a dataclass default under Python 3.10. The cast silences
that while keeping the informative annotation — same motivation as
[D3](#d3-every-seam-is-a-class-injection-slot).

**Consequence.** A specialization that cares about all three shapes must
override all three slots; overriding one leaves the other two on the base
classes. See the note in
[Known trade-offs](#known-trade-offs-and-open-todos).

### D9 — Per-call overrides fall back to device defaults

Every factory method accepts `program_language`, `language_version`,
`kernel_serializer`, and `group`, and `_resolve_kernel_serializer` falls back
to the device's value when the argument is `None`.

**Why.** Two audiences: a downstream SDK sets these once on a device subclass
and never thinks about them again; a user experimenting with a second language
in the same session needs a per-call escape hatch. `None`-means-inherit gives
both without two APIs.

### D9a — `Device` exposes no backend introspection

There is no `list_groups`, `list_tasks`, `list_definitions`, `qpu_status`, or
`current_user` on `Device`, even though the QLAM clients offer all of them.

**Why.** A device is a task factory ([D6](#d6-a-factory-that-deliberately-cannot-submit)).
Every listing method would add a network call to a class whose whole value is
that it makes none, and would duplicate what the `qsh` CLI and the QLAM clients
already do. Group *selection* is configuration, so it belongs in `~/.qsh` and in
one `group` argument ([D21](#d21-group-names-are-resolved-to-uuids-at-submission-not-construction)),
not in a discovery API.

**Consequence.** Users who need to browse groups or tasks use `qsh` or a QLAM
client directly. A `list_groups` method was in fact built during #101 and
removed before merge — see [R5](#r5-devicelist_groups-was-built-then-removed).

---

## 3. `mixins.py` — authentication

### D10 — Auth is a mixin, because three unrelated classes need it

`Device`, `TaskABC`, and `Future` all make authenticated calls and share no
meaningful supertype.

**Why.** A common base class would force an inheritance relationship that does
not exist (a future is not a kind of task). Passing an auth object as a field
would put the same plumbing in three constructors. A mixin dataclass
contributes exactly one field, `context_name`, to each.

### D11 — `app_context` is a property; API clients are created per call

`AppContext` is constructed fresh on every access, and every call site opens a
client in a `with` block.

**Why.** No live connection, session, or token cache ever lives inside a
`Future` or `Task`. That is what keeps those objects inert, copyable, and free
of teardown obligations. The cost — reconstructing a context and checking auth
per call — is negligible next to an HTTP round-trip.

**Consequence.** There is no connection reuse across calls. Accepted
deliberately; revisit only with measurements.

### D12 — `authenticate()` is lazy, idempotent, and called at every entry point

It checks `is_authenticated()` and logs in only if needed.

**Why.** Users should never manage a login lifecycle. Making it idempotent
means every network method can call it unconditionally, so there is no "did I
authenticate yet?" state to get wrong. Laziness means the interactive browser
flow fires on the first real API call, not at import.

### D13 — `call_with_auth_refresh` retries exactly once, on 401/403 only

On those statuses it attempts a non-interactive `refresh_credentials()`; if no
provider yields fresh credentials it re-raises the original error, and a second
failure is not retried.

**Why.** Long scans outlive access tokens, and the failure would otherwise
surface as a spurious mid-fetch error. One retry is the whole scope on purpose:
any other status is a real error that must not be masked, and retrying twice
turns an expired-refresh-token situation into a loop.

**Consequence.** Every QLAM call in the subsystem is wrapped in this. New API
calls should be wrapped too, or they will be the one place a long job dies.

---

## 4. `task.py` — payload construction

### D14 — Abstract *getters*, one concrete assembler

Subclasses implement `num_subtasks`, `get_kernels`, `get_arguments`,
`get_metadata`, `get_num_shots` (and optionally `program_index_for_subtask`,
`summary`). The base class owns `programs()`, `create_task_definition()`,
`validate_arguments()`, `run_async()`, `submit_task_definition()`.

**Why.** The QLAM `TaskDefinition` shape is fixed and non-obvious (programs
array + subtasks referencing `program_index`). Re-deriving it per task type
invites drift. Inverting the pattern — subclasses declare data, base class
assembles — makes a new task shape about twenty lines and keeps serialization,
validation, group resolution, and submission in exactly one place.

**Consequence.** A task whose payload genuinely does not fit overrides
`create_task_definition` directly; the docstring names this as the sanctioned
escape hatch.

### D14a — `num_subtasks` is a public abstract property, not a stored field

```python
@property
@abstractmethod
def num_subtasks(self) -> int: ...
```

**Why a property.** It is *derived*, not configured: it is `len(self.arguments)`
for a scan, `len(self.kernels)` for a batch, and the literal `1` for a single
kernel. Storing it as a field would create a second source of truth that can
disagree with the data it counts — a user could construct a task with three
argument sets and `num_subtasks=2`, and the length checks in
`validate_arguments` would then be validating one lie against another. As a
property it cannot drift.

**Why abstract.** It is precisely the piece the base class cannot know. Every
other member of the assembly contract (`programs`, `create_task_definition`,
`validate_arguments`) is written in terms of it, so making it abstract is what
turns "declare your fan-out" into a compile-time obligation for a new task
shape rather than a documentation note.

**Why not private.** It is public — no leading underscore — and deliberately so:
`validate_arguments` and `summary` read it, subclasses implement it, and it is
the natural thing for a user to check before submitting (`task.num_subtasks`
answers "how many subtasks will this be?" without building a payload). What can
look private is that it is *read-only*: there is no setter, because
[D6](#d6-a-factory-that-deliberately-cannot-submit) means a task's fan-out is
fixed by the data you constructed it with.

### D15 — All length validation happens locally, before authentication

`validate_arguments()` compares `len(arguments)`, `len(metadata)`, and
`len(num_shots)` against `num_subtasks` and raises `ValueError` with the actual
counts.

**Why.** Fail fast, offline, with a message that names the mismatch — rather
than after a login prompt and a rejected HTTP request. This is also what makes
`num_shots: int | list[int]` broadcasting safe to offer: the broadcast result is
checked like anything else.

### D16 — One `run_async` with `dry_run`, plus `@overload`s on `Literal`

Two overloads give `dry_run=True → None` and `dry_run=False → FutureType`.

**Why.** Dry-run and real submission must traverse *identical* validation and
`create_task_definition` code, or the dry-run lies about what would be sent.
That argues for one method. A `bool` parameter would then erase the return type
to `FutureType | None` for both callers; the `Literal` overloads restore
precision without duplicating the body.

### D17 — Program deduplication is a one-method hook

`program_index_for_subtask(i)` returns `i` by default; `ParameterScanTask`
returns `0`.

**Why.** It is the *only* structural difference between a batch and a scan.
Expressing it as a hook rather than a branch inside
`create_task_definition` keeps the assembler shape-agnostic and makes future
shapes (e.g. two programs alternating across subtasks) trivial.

### D18 — Serialization is an injectable `Protocol`, and bytes get base64'd

`KernelSerializer` is a structural `Protocol` with a single `encode` method.
`serialize_kernel` calls `kernel.dialects.encode(kernel, version=...)`, passes
the module to the serializer, base64-encodes `bytes` results, passes `str`
results through, and raises `TypeError` on anything else.

**Why.** Structural typing means a serializer needs no import from bloqade and
no base class — `kirin.serialization.JSONSerializer` already satisfies it
accidentally, which is the point. Base64 exists because `Program.content` is a
string field in the API; a binary serializer would otherwise be unusable.
`serialize_kernel` remains a method, so overriding it entirely (as the QASM2
demo does) is still available for languages that bypass Kirin encoding.

**Consequence.** Two override levels with different blast radii: swap
`kernel_serializer` to change the encoding of a Kirin module; override
`serialize_kernel` to change what is encoded at all.

### D19 — Language and version are fused into one wire string

`create_task_definition` emits
`f"{program_language}.v{version.removeprefix('v')}"`, e.g. `squin.v0.1.0`.

**Why.** The API models a single `program_language` string and documents the
convention by example (`"flair.v1"`). Keeping `program_language` and
`language_version` as separate user-facing fields and fusing them at the
boundary means users never hand-format the wire value, and the version stays
available as data for `kernel.dialects.encode`.

**Consequence.** `removeprefix('v')` tolerates users writing `"v0.1.0"`. The
API does not validate the fused string, so a typo shows up as a backend
rejection.

### D20 — User metadata is JSON-serialized into a pass-through string

Per-subtask `metadata` dicts become
`TaskMetadata(user_metadata=json.dumps(metadata[i]))`.

**Why.** `arguments` is `dict[str, float]` on the wire — it cannot carry a tag,
a label, or a nested structure. `user_metadata` is an uninterpreted string, so
JSON-encoding a dict into it gives users an arbitrarily shaped, queryable
annotation channel. `Result.where_metadata` decodes it back.

**Consequence.** Metadata must be JSON-serializable, and
`filter_by_metadata` raises on values it cannot decode. Documented, with
`where_subtasks` as the manual-parsing fallback.

### D20a — The user-facing group is a name; the wire wants a UUID

`TaskABC.group` is `str | None` and documented as a name, while
`TaskDefinition.group_id` is a `UUID`.

**Why.** Nobody types UUIDs. Groups have human names in `qsh` config and in the
web UI, so the field a user sets should accept the thing they already know. The
conversion is one API call and belongs in the library, not in the user's head.
`GroupsClient.resolve_id` accepts either form, so a user who *does* have a UUID
can still pass it as a string.

**Consequence.** The field name (`group`) differs from the wire name
(`group_id`) on purpose; they are different types carrying different
identifiers, and collapsing them would force one audience to convert.

### D21 — Group names are resolved to UUIDs at submission, not construction

`create_task_definition` sets `group_id=None`. At submit time, if the
definition has no group, `_configured_group()` picks the first of: task-level
`group`, `~/.qsh` `plugins.tasks.group`, `defaults.group`; a name is then
resolved to a UUID via `GroupsClient.resolve_id`, and if nothing is configured
the field is omitted so QLAM chooses.

**Why.** Users think in group names; the API wants UUIDs. Resolving a name
requires a network call, so doing it during `create_task_definition` would
break [D6](#d6-a-factory-that-deliberately-cannot-submit) — dry-runs would
need credentials. Deferring to submission keeps construction offline. The
precedence chain deliberately mirrors the `qsh` CLI so a user's existing config
means the same thing in both tools.

### D22 — The task ID is durable before the future is returned

`submit_task_definition` logs the ID and calls
`storage.add_task_definition(...)` immediately after submission, before
constructing the `Future`.

**Why.** This is the "queue time is money" failure mode. Between submission and
the first `result()` call there is a window where a crash would orphan a
running task — the backend charges for it, and the user has no ID. Writing the
ID and full payload to storage first closes that window and is precisely what
makes `Future.from_storage` a real recovery path rather than a nicety.

**Consequence.** Submission requires a storage object even if the user does not
care about persistence — hence [D23](#d23-storage-defaults-to-a-fresh-dictstorage).

### D22a — `create_task_definition` passes `group_id=None` explicitly

The field defaults to `None` in the pydantic model, yet the call site writes it
out.

**Why.** It marks the seam. `create_task_definition` is offline
([D6](#d6-a-factory-that-deliberately-cannot-submit)), so it *cannot* fill in a
group; `submit_task_definition` then checks `if task_definition.group_id is
None` and resolves one. Writing the `None` makes that handoff visible at both
ends, and makes clear that a caller who builds a definition by hand may supply
a `group_id` and have it respected rather than overwritten.

**Consequence.** The precedence rule is "an explicit `group_id` on the
definition wins over everything," which is only obvious because the field is
mentioned in both places.

### D23 — Storage defaults to a fresh `DictStorage`

`run_async(storage=None)`, `submit_task_definition(storage=None)`, and
`Future.from_task_id(storage=None)` all construct an in-memory `DictStorage`;
`Future.storage` uses `field(default_factory=DictStorage)`.

**Why.** [D22](#d22-the-task-id-is-durable-before-the-future-is-returned)
makes storage structurally mandatory, but demanding it in a first-five-minutes
snippet is bad ergonomics. An in-memory default satisfies the invariant with no
ceremony.

**Consequence.** The default gives you no crash protection — the point of D22 —
so anything that costs real queue time should pass a `SQLiteStorage`. Note what
this default is *not*: a default file path. See
[R2](#r2-a-default-storage-file-path-was-considered-and-rejected).

### Ledger — three task classes versus one

[D7](#d7-three-task-shapes-not-one-generic-submit) and
[D14](#d14-abstract-getters-one-concrete-assembler) commit to three concrete
subclasses of `TaskABC`. The honest accounting:

**In favour.**

- *Illegal states are unrepresentable.* You cannot build a task with a list of
  kernels *and* a list of scan arguments over one kernel, because no class has
  both fields. A single class with `kernels` and `arguments` would admit that
  combination and need a runtime check to reject it.
- *Types match the data.* `SingleKernelTask.arguments` is a `dict`;
  the other two take `list[dict]`. One class would have to accept
  `dict | list[dict]` and normalize, which pushes a shape question into every
  reader.
- *Per-shape behaviour has somewhere to live.* `program_index_for_subtask`
  ([D17](#d17-program-deduplication-is-a-one-method-hook)) and `summary`
  genuinely differ per shape. In one class both become `if`-ladders on a mode
  flag.
- *The API self-documents.* Three factory methods with three return types tell a
  reader what the backend supports. `device.submit(kernels=…, arguments=…,
  mode="scan")` does not.
- *Extension is additive.* A fourth shape is a new subclass; nothing existing
  changes.

**Against.**

- *Boilerplate.* Each class re-implements four getters, and the
  `int | list[int]` broadcast in `get_num_shots` is duplicated verbatim between
  `KernelBatchTask` and `ParameterScanTask`. That could live on the base.
- *Three slots, three casts, one footgun.* The classes require the `Device`
  fields in [D8a](#d8a-three-separate-task-class-slots-one-per-shape-all-initfalse),
  including the partial-override hazard.
- *Adding a base field touches every shape.* #101 added `language_version`,
  `kernel_serializer`, and `group`; each had to be threaded through three
  `Device` methods and appears in three docstrings.
- *The shapes overlap.* `SingleKernelTask` is `ParameterScanTask` with one
  argument set, and also `KernelBatchTask` with one kernel. The distinction is
  ergonomic, not structural — it exists so the common case takes a `dict`
  instead of a one-element list.
- *No hybrids.* A kernels × arguments cross-product, or a batch where subtasks
  map to programs non-injectively, fits none of the three and needs a fourth
  class — even though the wire format expresses it natively with an arbitrary
  `program_index` per subtask.

**Verdict.** The split earns its keep on the first two points; the wire format
allows more shapes than the three classes expose, and the boilerplate is a real
but small cost. The cheap improvement is not fewer classes but less duplication:
move `get_num_shots`'s broadcast onto `TaskABC` and give `Device` one hook
instead of three slots.

**The deeper answer: the abstraction is only needed while construction is
declarative.** `TaskABC`'s five abstract getters exist to defer one question —
"what shape is this?" — to a subclass, and to answer it *lazily*, at
`create_task_definition` time, from whole-task fields. An incremental builder
answers the same question *eagerly*, one `add_subtask` call at a time, and
stores the already-resolved result: a list of unique programs plus a list of
subtasks each carrying its own `program_index`, `num_shots`, `arguments`, and
`metadata`. Once that flat state exists, there is nothing left to abstract:

| `TaskABC` member | in a builder |
|---|---|
| `num_subtasks` | `len(self._subtasks)` — concrete |
| `get_kernels` / `get_arguments` / `get_metadata` / `get_num_shots` | the state itself; no accessors needed |
| `program_index_for_subtask` | stored per subtask at add time — strictly more expressive |
| `validate_arguments` | mostly unnecessary: one `add_subtask` appends exactly one subtask with its own data, so the length invariants cannot be violated. What remains (`num_shots >= 1`, float-valued arguments) is per-value validation belonging in `add_subtask` |
| `summary` / `_finalize` | one concrete implementation each |

So the honest answer to "do we need `TaskABC`" is **no** — provided the
construction surface becomes incremental. The three shapes survive as
classmethods over the same state (`TaskBuilder.single`, `.batch`, `.scan`), each
a short loop, which preserves the declarative entry point *and* the
illegal-states-unrepresentable property from the ledger above: a builder has no
`kernels` field and no scan-`arguments` field to mix up, because it has neither.

What is genuinely lost is the type-level distinction between shapes — a
function can no longer be annotated to accept only a parameter scan. The one
place that annotation exists today, `bloqade-flair`'s
`SubmissionScanner.to_task() -> ParameterScanTask`, only needs its result to be
submittable, so the loss is nominal.

What survives is a single *structural* abstraction rather than an ABC: the
`TaskSpec` Protocol (`_finalize(ctx)` + `summary()`) that `Device.run_async`
accepts. That costs nothing, requires no inheritance, and lets a downstream
package supply its own payload shape.

Sequencing matters, though: this is only safe once submission has left the task
([spec: device owns submission](https://github.com/QuEraComputing/bloqade-core/blob/main/specs/spec-21-08-2026-device-owns-submission.md))
and the builder exists ([PR #107](https://github.com/QuEraComputing/bloqade-core/pull/107)).
Deleting `TaskABC` first would strand `run_async`, `future_cls`, and group
resolution with nowhere to live.

---

## 5. `future.py` — the async handle

### D24 — `Future` is a handle, not a container

Its entire state is `task_id`, `storage`, `fetch_options`, `context_name`,
`result_cls`, and one private page watermark. No shots, no task record, no
cached status.

**Why.** Follows from [D2](#d2-storage-is-the-source-of-truth-not-future-and-not-result),
but the sharper reason is that a future's *identity* is the `task_id` — that is
the only piece the backend and the user's log agree on. Anything else held on
the object is a cache that can disagree with the backend, and status is the one
thing that must never be stale, so `status()` re-fetches every time
([D27](#d27-terminal-statuses-are-a-class-attribute-failures-raise)).

**Consequence.** Polling in a tight loop is a request per iteration. That is why
the backoff in [D26](#d26-exponential-backoff-with-a-cap-against-a-monotonic-deadline)
lives in `_wait_for_completion` rather than in `status()`.

### D24a — `Future` is mutable, and its watermark is private `init=False` state

`_first_incomplete_subtask_page: int = field(init=False, default=0)`, mutated
by `_fetch_subtask_page`.

**Why.** `init=False` keeps it out of the constructor because it is not
configuration — a caller supplying it could silently skip unfetched pages. It is
private because it is an optimization with no semantic meaning to a user
([D29](#d29-fetching-is-idempotent-by-construction-not-by-bookkeeping)): losing
it costs time, never data. And it is why `Future` cannot be frozen
([D2a](#d2a-values-are-frozen-handles-are-not)).

**Consequence.** Two futures on the same `task_id` keep independent watermarks
and will each re-fetch — correct, just duplicated work.

### D25 — `ApiFetchOptions` is frozen, and there is one module-level default

`DEFAULT_FETCH_OPTIONS = ApiFetchOptions()` is used as the default argument
throughout.

**Why.** Bundling five pagination/backoff knobs into one object keeps six
signatures from growing five parameters each. Freezing it makes a shared
instance safe as a default argument — the classic mutable-default bug is
impossible. A named module constant makes the shared default explicit rather
than relying on `= ApiFetchOptions()` being evaluated once.

### D25a — Ten subtasks and one hundred shots per request

`subtasks_per_fetch = 10`, `shots_per_fetch = 100`.

**Why.** Ten is the API's own documented default `size`, so the default request
is the one the backend is tuned for. The product — up to ~1000 shot rows per
response — is also the write batch: `_fetch_subtask_page` accumulates a page
into `temp_data` and hands it to `add_shots` in one `executemany`, so the page
size doubles as the transaction size. It is large enough that SQLite commits are
amortized and small enough that a dropped connection loses little work.

**Consequence.** The two knobs are coupled in effect even though they are
independent fields: raising `shots_per_fetch` raises peak memory *and*
transaction size together.

### D26 — Exponential backoff with a cap, against a monotonic deadline

`0.5s`, doubling, capped at `30s`; the timeout deadline uses
`time.monotonic()`.

**Why.** A simulator finishes in seconds and deserves a fast first poll; a real
queue can take hours and must not be hammered. Capping keeps the poll rate
bounded without giving up responsiveness early. `monotonic` rather than wall
clock means an NTP correction or DST change cannot corrupt a timeout.

### D27 — Terminal statuses are a class attribute; failures raise

`EXIT_STATUS` lists `CANCELLED`, `FAILED`, `PAYLOAD_PROCESSING_ERROR`,
`COMPLETED`. `_wait_for_completion` breaks on any of them, then raises
`ValueError` for cancellation and for failure — the failure message includes
`task.error_reasons`.

**Why.** Waiting and failure-detection are the same loop; separating "reached a
terminal state" from "reached a *good* terminal state" lets `done()` mean
"stop polling" while `result()` means "give me data or explain why you can't."
Errors surface as exceptions at the point the user asked for data, which is
where they can act on them. `EXIT_STATUS` is a class attribute so a subclass
targeting a backend with extra states can extend it.

### D28 — Two-axis pagination, completed-first sort, and an incremental watermark

`_fetch_subtask_page` paginates over subtask pages *and* shot pages, requests
`sort="completed_date,asc"`, and records the first page containing a
non-`COMPLETED` subtask in `_first_incomplete_subtask_page`. The next `fetch()`
resumes there.

**Why.** A large scan has two independently unbounded dimensions (subtasks,
shots per subtask), so one page cursor cannot express the position. Sorting
completed subtasks first turns "where should I resume?" into a single integer:
everything before the watermark is final and never needs re-reading. That is
what makes `partial_result()` cheap to call in a polling loop over a long job.

### D29 — Fetching is idempotent by construction, not by bookkeeping

The watermark is an optimization, not a correctness mechanism. Deduplication
lives in storage: `UNIQUE(task_id, shot_index, frame_type)` in SQLite and the
same tuple as a dict key in `DictStorage`.

**Why.** Any client-side "have I seen this shot?" ledger is one bug away from
dropping data. Making re-fetch harmless at the storage layer means the fetch
loop can be sloppy about overlap, retried freely, and interrupted at any point.

**Consequence.** The known cost is documented in a TODO: the watermark is not
persisted, so a new session re-fetches every shot. Wasteful, never wrong — the
right trade given D29.

### D30 — `completed_date` is written on the first shot page, indexed via `shot_results`

The update runs only when `shots_page == 0`, and the subtask index is read from
`shot_results[0]["subtask_index"]`; subtasks with no shot results are skipped.

**Why.** Two API facts, both recorded as comments: `completed_date` is stable
across shot pages, so writing it once is sufficient; and the API's subtask
object carries no index of its own — the index only appears on shot rows. A
later shot page can legitimately return a subtask with an empty `shot_results`
list, at which point the index is unrecoverable, so those rows are skipped
rather than guessed.

### D31 — `result()` blocks, `partial_result()` does not

Both end in `results_from_storage()`, scoped to this task ID and the
`DETECTED` frame.

**Why.** Two genuinely different intents: "I want the answer" and "show me what
exists so far." Sharing the tail means the returned view is identical in both
cases, so monitoring code and final-analysis code are the same code.

### D32 — Two recovery constructors that refuse to guess

`from_task_id` fetches the task and its definition and writes the definition to
storage before returning. `from_storage` discovers the ID in storage; with more
than one candidate it raises and lists every ID with its creation time.

**Why.** Recovery is the point of [D22](#d22-the-task-id-is-durable-before-the-future-is-returned),
and it needs both directions: from an ID a user pasted from a log, and from a
`.sql` file they still have. `from_task_id` writes the definition because
`Result` merging and validation need the subtask structure, not just shots.
`from_storage` raises rather than picking the newest task, because silently
attaching to the wrong task produces plausible wrong numbers — the worst
possible failure mode in a measurement tool. The error message doubles as the
listing the user needs to choose.

### D32a — Recovery constructors forward `result_cls` explicitly

Both classmethods pass `result_cls=cls.result_cls` into the constructor.

**Why.** `result_cls` is a dataclass field with a default, so constructing
`cls(...)` without it would take the *field default* — base `Result` — even
when `cls` is a subclass that set `result_cls = MyResult` as a class
attribute. Forwarding it explicitly is what makes
`MyFuture.from_task_id(...).result()` return a `MyResult`. Same mechanism as
[D34](#d34-_resolve_context_name-reads-a-class-level-default-via-getattr):
class-level overrides on a dataclass need to be read deliberately.

**Consequence.** Any new field a subclass is expected to set as a class
attribute must be forwarded here too, or it will be silently dropped on
recovery.

### D33 — Foreign API models are normalized to `TaskDefinition` at the boundary

`from_task_id` reshapes the Definitions endpoint response with
`TaskDefinition.model_validate({**task_def.model_dump(include={...}),
"group_id": task_def.group.id})`.

**Why.** The definitions endpoint returns a different model than the one
submission accepts. Storage speaks exactly one dialect — `TaskDefinition` — so
translation happens once, at the ingress point, rather than leaking a second
shape into every storage backend and every `Result` accessor.

### D34 — `_resolve_context_name` reads a class-level default via `getattr`

Classmethods accept `context_name: str | None`; when `None`, they fall back to
`getattr(cls, "context_name", None)` and raise a message naming the fix.

**Why.** `context_name` is a dataclass *field*, so it has no value on the class
— but a downstream SDK wants `class MyFuture(Future): context_name =
"my-context"` so its users can call `MyFuture.from_task_id(task_id=...)`
without repeating the context every time. `getattr` is what makes that
subclass-provided class attribute visible. The explicit error keeps the base
class from failing obscurely when nobody set one.

### D35 — `export_to` chunks shots and copies definitions

Writes in batches of 1000 and copies the task definitions for the exported task
IDs.

**Why.** Chunking bounds memory when promoting a large in-memory session to
SQLite. Copying definitions is not optional: shots without their subtask
structure cannot be validated or merged, so a definition-less export would
produce a store that `Result` cannot read.

---

## 6. `local_storage.py` — persistence

### D36 — A storage ABC rather than a serialization format

`StorageBackend` is an ABC with nine abstract methods; `get_task_definition`,
`get_arguments`, and the four `filter_by_*` helpers are concrete on the base.

**Why.** `Result` needs random access by `(task, subtask, shot, frame)` and
predicate filtering, and the data must outlive the process. A pickle or an HDF5
dump gives neither. Sizing the abstract surface to *rows in, rows out* and
implementing everything derivable on the base means a new backend gets
definition reconstruction and all predicate filtering for free.

### D36a — `get_shots` returns a generator in both backends

Declared `Iterable[ShotResult]`; `DictStorage` yields from its dict and
`SQLiteStorage` yields per cursor row.

**Why.** A completed scan can hold millions of shot rows, and the two consumers
want different things: `Future.export_to` streams and never needs the whole set
in memory, while `Result._shot_results_for_subtasks` immediately materializes a
numpy array per subtask. Yielding serves the streaming consumer and costs the
materializing one nothing.

**Consequence.** The result is single-pass. Code that iterates a `get_shots`
return twice sees an empty second pass, and `len()` does not work on it — both
easy mistakes, neither caught by the type annotation.

### D37 — One flat row per shot per frame, with `frame_type` in the identity

`ShotResult(task_id, shot_index, subtask_index, subtask_shot_index, frame_type,
bitstring)`, deduplicated on `(task_id, shot_index, frame_type)`.

**Why.** Denormalized rows are directly SQL-filterable — no joins in the hot
path — and give a stable natural key, which is what
[D29](#d29-fetching-is-idempotent-by-construction-not-by-bookkeeping)
requires. `frame_type` is part of the key because hardware emits several frames
per physical shot (`DETECTED`, `SORTED`, …); leaving it out would make frames
overwrite each other. Both indices are kept because `subtask_shot_index` is
what you group by within a scan point and `shot_index` is what the API's
pair-filters speak.

### D38 — Task metadata is decomposed into tables, not stored as a blob

Three logical tables: `task_definitions`, `programs`, `subtasks`.

**Why.** Users filter on `arguments`, `metadata`, and `program_index`, so those
have to be columns, not fields inside an opaque JSON document. The reverse
direction is preserved by `get_task_definition`, which reassembles a real
`TaskDefinition` from the rows — so decomposition costs nothing in fidelity.

### D38a — `DictStorage` mirrors the SQL schema instead of nesting naturally

Its `_metadata` is `{"task_definitions": {task_id: row}, "programs": {(task_id,
i): row}, "subtasks": {(task_id, i): row}}` — three flat maps keyed by the same
tuples SQLite uses as primary keys.

**Why.** The natural Python shape would be one nested dict per task. It was
rejected because the two backends must return *identically shaped* rows from
`get_subtasks` and `get_programs` — `Result` merging, the `filter_by_*` helpers,
and the whole test suite consume that shape. Mirroring the tables makes
divergence structurally unlikely: adding a column means adding a key in both
places, and a test written against one backend passes against the other.

**Consequence.** `DictStorage` is a slightly awkward in-memory database rather
than an idiomatic dict, and denormalized `task_id` is repeated in every row.
That redundancy is the price of one row shape.

### D38b — `completed_date` is the only mutable subtask column

`update_subtasks_completed_date` is the sole mutating method on stored metadata;
everything else is `INSERT OR IGNORE`.

**Why.** A task definition is immutable once submitted — that is what makes
re-adding it idempotent ([D43](#d43-sqlite-specifics-each-chosen-against-a-concrete-failure))
and what lets `Result` treat rows as facts. Completion time is the one attribute
that is genuinely unknown at submission and learned later, so it is the one
attribute allowed to change. Keeping the mutable surface to a single method also
keeps the audit story simple: nothing else can rewrite history.

**Consequence.** Anything else the backend learns later (per-subtask status, for
instance) currently has nowhere to be stored, which is why the fetch loop keeps
the incomplete-page watermark in memory instead
([D24a](#d24a-future-is-mutable-and-its-watermark-is-private-initfalse-state)).

### D39 — Two filter levels, frozen, with explicit pair fields

`StorageFilter(task_ids, subtask_indices, task_subtask_pairs)` and
`ShotFilter(… , frame_type, task_shot_pairs)`. All criteria are AND-ed.
`ShotFilter.__post_init__` upper-cases `frame_type`.

**Why.** Subtask-level questions and shot-level questions need different
vocabularies, and inheritance lets shot filters be used wherever a storage
filter is expected. The `*_pairs` fields exist because AND-ing independent
`task_ids` × `subtask_indices` cannot express "these specific (task, subtask)
combinations" — which is exactly what predicate narrowing produces. Frozen for
[D4](#d4-everything-is-a-dataclasskw_onlytrue) reasons. Case normalization is
done once at construction so no comparison site has to remember it.

### D40 — Predicates compile to filters; they never return rows

`filter_by_subtasks`, `filter_by_arguments`, `filter_by_metadata`, and
`filter_by_shots` all return a filter object populated with matching pairs.

**Why.** This is the seam between arbitrary Python and the query layer. The
predicate runs in Python — full expressiveness, no DSL — but the *result* is
declarative, so it composes: `Result.where_*` can intersect it with existing
scope and defer the actual heavy shot selection to SQL. Returning rows instead
would force every narrowing step to materialize data.

**Consequence.** Predicate evaluation pulls metadata rows into Python, so
filtering is O(subtasks), not index-accelerated. Fine at metadata scale;
`filter_by_shots` is the expensive one.

### D41 — Getters must return independent copies

Stated as a NOTE in the ABC docstring; `DictStorage` wraps returns in
`dict(...)` and `list(map(dict, ...))`.

**Why.** `Result.subtasks()` mutates the rows it receives — it pops `task_id`
and `metadata` while merging. Without copies, reading a merged view would
corrupt the in-memory store. The team chose to document the contract rather
than redesign the merge, since the copy is cheap and the alternative touches
every accessor.

**Consequence.** A real coupling between two modules that is enforced only by
prose. Any new backend that forgets it will produce a very confusing bug.

### D42 — An empty filter tuple matches nothing

Every SQL branch emits `where_clauses.append("0")` for an empty tuple;
`DictStorage` reaches the same outcome by membership test.

**Why.** `task_ids=()` means "no tasks selected." Treating an empty collection
as "no filter" — the natural outcome of a naive `if not task_ids` — would turn
a narrowed-to-nothing `Result` into a silent full-table scan. Since filters are
produced mechanically by [D40](#d40-predicates-compile-to-filters-they-never-return-rows),
a predicate matching nothing is a routine occurrence, not an edge case.

### D43 — SQLite specifics, each chosen against a concrete failure

**Why (in general).** SQLite is the persistence layer because it is in the
standard library, needs no server, produces a single file a user can copy or
attach to a bug report, and gives transactional writes — the four properties a
scientist's results file needs. Every detail below is a consequence of that
choice meeting a specific failure mode.

- `INSERT OR IGNORE` + `UNIQUE` — implements [D29](#d29-fetching-is-idempotent-by-construction-not-by-bookkeeping)
  in one keyword.
- Bitstrings stored as `"0101"` text, read back via
  `np.array(list(...), np.uint8).view(bool)` — inspectable with any SQLite
  browser and diffable in a bug report. A TODO notes optional compression;
  legibility won for now.
- `task_shot_pairs` are passed as a single JSON document and unpacked with
  `json_each`/`json_extract` — SQLite's bound-parameter limit is around 32k, and
  a shot-level filter routinely exceeds it. This is the only place a filter is
  not expressed as parameters.
- `PRIMARY KEY` on `(task_id, program_index)` and `(task_id, subtask_index)` —
  makes re-adding a definition idempotent, matching shot behaviour.
- `close()` plus `__enter__`/`__exit__` — the class docstring recommends the
  `with` form so the file lock is released promptly instead of at GC time.

### D43a — Tables are created on every open, and rows come back as `sqlite3.Row`

`__init__` runs `CREATE TABLE IF NOT EXISTS` for all four tables every time, and
sets `conn.row_factory = sqlite3.Row`.

**Why.** Idempotent bootstrap means there is no "initialize the database" step
to forget and no difference between opening a new file and an existing one —
`SQLiteStorage("new.sql")` and `SQLiteStorage("old.sql")` are the same call.
`sqlite3.Row` gives mapping access, which is what lets `get_subtasks` do
`dict(row)` and return the exact same key set as `DictStorage`
([D38a](#d38a-dictstorage-mirrors-the-sql-schema-instead-of-nesting-naturally)) —
with the default tuple factory, every accessor would hand-build dicts by column
position.

### D43b — A surrogate `row_number` alongside the natural unique key

The `results` table has `row_number INTEGER PRIMARY KEY AUTOINCREMENT` *and*
`UNIQUE(task_id, shot_index, frame_type)`.

**Why.** The unique constraint enforces identity
([D37](#d37-one-flat-row-per-shot-per-frame-with-frame_type-in-the-identity)); the
surrogate key gives insertion order, so an unsorted `SELECT * FROM results`
returns shots in the order they were fetched rather than an implementation-defined
order. It also keeps the natural key free to change — widening it (to include
`camera_id`, say) is an additive migration rather than a primary-key rewrite.

**Consequence.** One extra integer per shot row, and `rowid` is not reused after
deletes. Neither matters at the scale involved.

### D44 — Schema is versioned, migrations are forward-only, mismatches are fatal

A `bloqade_schema` table holds one version string. `0.1.0 → 0.2.0` is
performed in `__init__` (adds nullable `task_definitions.group_id`, restamps
the version, logs what it did); any *other* mismatch raises `ValueError`.

**Why.** These files hold data the user paid queue time for, so silently
reading them with the wrong schema assumptions is unacceptable — hence the hard
error. The one migration is additive and idempotent (`PRAGMA table_info` guard),
so it is safe to run on every open. It is deliberately one-way and says so in
both the log line and the comment: an upgraded file can no longer be opened by
an older bloqade. Loud and one-way beats bidirectional and clever.

### D45 — New capabilities join the ABC as concrete defaults, not abstract methods

`get_task_group_id` returns `None` on the base class and is overridden by both
built-in backends. Its docstring states the reason.

**Why.** Adding an abstract method breaks every third-party backend at import
time for a feature they may not care about. A concrete default degrades
gracefully: an old backend simply reconstructs definitions without a group.
This is now the policy for extending `StorageBackend`.

### D46 — `DictStorage.__repr__` summarizes instead of dumping

Returns `DictStorage(num_shots=…, task_ids=…)`.

**Why.** The original returned `repr(self._data)`, which printed every shot in a
notebook when the object was the last expression in a cell. A summary is the
information you actually want at a glance.

---

## 7. `result.py` — the analysis view

### D47 — A `Result` is a query, not data

Two fields: `storage` and `shot_filter`. Nothing is read until an accessor is
called.

**Why.** Follows from [D2](#d2-storage-is-the-source-of-truth-not-future-and-not-result).
It makes narrowing free, makes views trivially serializable as
(store, filter) pairs, and means a result computed today and one reconstructed
from the same store next month are the same object.

### D47a — `DETECTED` is the default frame type

`_default_shot_filter()` returns `ShotFilter(frame_type="DETECTED")`, and
`Future.results_from_storage` scopes to `DETECTED` when given no filter.

**Why.** A shot yields several frames — `LOADED`, `SORTED`, `DETECTED` — and an
unfiltered query would return all of them interleaved, so
`shot_results()` would silently mix frames into one array and produce a
meaningless average. `DETECTED` is the measurement outcome, which is what
"the result" means to a user asking for results. Defaulting to it makes the
common case correct rather than merely permissive.

**Consequence.** Other frames are reachable only by constructing a filter
explicitly (`replace(result.shot_filter, frame_type="SORTED")`), which is also
the mechanism [D52](#d52-where_shots-separates-the-predicates-scope-from-the-results-scope)
uses for post-selection.

### D47b — Validation is memoized in a private field

`_is_valid: bool = field(init=False, default=False)`; `validate()` returns
immediately once it is set, and single-task views set it without doing any work.

**Why.** `validate()` reads every selected subtask row, and it is called from
`subtasks()`, which is called from `shot_results()` and `arguments()`. Without
memoization an ordinary three-line analysis re-validates three times. The
single-task short-circuit exists because the check is meaningless with one
task — there is nothing to disagree with.

**Consequence.** The cache is per-view and never invalidated, which is safe
only because a `Result`'s filter is fixed at construction and stored definitions
are immutable ([D38b](#d38b-completed_date-is-the-only-mutable-subtask-column)).
Adding a way to mutate a filter in place would break it.

### D47c — `Result.storage_filter` downcasts to the subtask-level filter

A property that rebuilds a `StorageFilter` from the `task_ids`,
`subtask_indices`, and `task_subtask_pairs` of the view's `ShotFilter`, dropping
`frame_type` and `task_shot_pairs`.

**Why.** `ShotFilter` extends `StorageFilter`
([D39](#d39-two-filter-levels-frozen-with-explicit-pair-fields)), so it would be
*accepted* wherever a storage filter is expected — and would then be wrong:
`frame_type` and `task_shot_pairs` name columns the `subtasks` table does not
have, and a shot-level narrowing must not silently restrict which subtasks are
visible. Projecting explicitly makes "which parts of the scope apply to metadata
queries" a single visible decision instead of an accident of inheritance.

### D48 — Merging across task IDs by `subtask_index`, with an explicit compatibility contract

`validate()` requires that task IDs agreeing on a `subtask_index` also agree on
`program_index` and on `arguments`; differing `num_shots` is explicitly
allowed; `None` and `{}` are treated as equal arguments. The check caches
`_is_valid` and short-circuits for a single task ID.

**Why.** The motivating workflow is "run the same scan again tomorrow and pool
the statistics." Different shot counts are the *point*, so they cannot be an
error; a different program or different parameters at the same index means the
two runs are not the same experiment and pooling them would silently produce a
wrong number. `None` vs `{}` is an artifact of how the API represents "no
arguments", not a real difference.

**Consequence.** QLAM records no notion of scan identity
([D7](#d7-three-task-shapes-not-one-generic-submit) is client-side), so this
check *reconstructs* it from the definition rows. Every error message ends with
the `verify=False` hint, keeping the check overridable rather than a wall.

### D49 — Merged and full accessors are separate methods

`subtasks()`/`arguments()` merge (drop `task_id` and `metadata`, sum
`num_shots`, order by subtask index); `full_subtasks()`/`full_arguments()`
return raw rows.

**Why.** A merged row cannot honestly carry per-task fields, so the merged view
drops them rather than picking one arbitrarily. Provenance is still needed for
debugging and for auditing what went into an average, so it gets its own
accessor instead of a flag.

### D50 — `shot_results()` returns raw physical bitstrings and nothing else

One 2-D boolean array per merged subtask, ordered by subtask index. No
decoding, no register mapping, no counts.

**Why.** `bloqade-core` does not know your logical encoding, qubit layout, or
error-correction scheme. Guessing would be wrong; offering a half-abstraction
would be worse. The base class provides the honest primitive and leaves
interpretation to a subclass ([D3](#d3-every-seam-is-a-class-injection-slot)).

### D50a — One query per subtask, not one grouped query

`_shot_results_for_subtasks` loops over subtasks, `replace`s the filter with a
single `subtask_indices=(i,)`, and queries once per subtask.

**Why.** The output is a *list of arrays*, one per subtask, so the rows must be
grouped somewhere. Grouping in SQL would mean one query plus a Python
partitioning pass keyed on `subtask_index`, with the ordering guarantees that
implies; grouping by issuing one scoped query per subtask reuses the existing
filter machinery and makes the grouping obviously correct by construction. For
scan sizes measured in tens or hundreds of subtasks against a local file, the
per-query overhead is not the bottleneck — decoding bitstrings is.

**Consequence.** This is an N+1 query pattern and the first thing to change if
a very wide scan becomes slow. Noted as a trade-off, not a hidden cost.

### D51 — `where_*` returns `type(self)` over a `replace`d filter

Narrowing constructs `type(self)(storage=…, shot_filter=replace(…))`.

**Why.** `type(self)` means a downstream `Result` subclass survives narrowing —
without it, `MyResult(...).where_metadata(...)` would silently degrade to a base
`Result` and lose every domain method. `replace` on a frozen filter means
narrowing composes: existing scope such as `frame_type` is preserved rather
than rebuilt.

### D52 — `where_shots` separates the predicate's scope from the result's scope

`predicate_filter` selects the shots the predicate sees and defaults to
`self.shot_filter`; the returned view keeps `self`'s scope intersected with the
matching pairs.

**Why.** Real post-selection reads one frame and returns another — the docstring's
example is "keep shots where `SORTED` was all ones, then give me their
`DETECTED` data." One filter cannot express a precondition on frame A and a
payload from frame B. Defaulting to `self.shot_filter` keeps the simple case
simple.

### D53 — The API's `float` coercion is documented, not hidden

`where_arguments` carries a NOTE: `Subtask.arguments` is `dict[str, float]`, so
`True` becomes `1.0` and identity predicates (`is True`) match nothing; use
`== 1` or a non-bool discriminator.

**Why.** Wrapping arguments to restore Python types would be a lie — the stored
and submitted values really are floats, and the backend really did receive
`1.0`. Documenting the sharp edge where users will hit it is the honest option.

---

## 8. `log_info.py` — logging

### D54 — Logging is opt-in, and a bad path warns instead of raising

`set_logging(enabled=True, path="bloqade.log", level="INFO")` is the entry
point, exported from the package. Nothing is written unless it is called or
`BLOQADE_LOGGING=1` is set before import. An `OSError` while creating the sink
emits a `RuntimeWarning` and leaves logging disabled.

**Why.** A library that creates files in the user's working directory on import
is a bad citizen — it surprises people, dirties repositories, and breaks in
read-only or containerized environments. Opt-in respects the host application's
control of its own logging. Warning rather than raising follows from logging
being ancillary: an unwritable log path must never take down a submission.

**Consequence.** The safety net that motivated the original always-on default
is now something users must switch on. What logging protects — recovering a
`task_id` after a crash — is instead covered structurally by
[D22](#d22-the-task-id-is-durable-before-the-future-is-returned). See
[R1](#r1-file-logging-was-on-by-default).

---

### D54a — Loguru, with `bloqade` as a disable-able namespace

The dependency is `loguru`, not `logging`, and the off switch is
`logger.disable("bloqade")`.

**Why.** The subsystem needs exactly one thing from a logging library: attach or
detach a file sink in one call, without touching global state that belongs to the
host application. `logging` would require a named logger, a handler, a formatter,
and care not to disturb the root logger's configuration. Loguru's
`disable(name)` also scopes the off switch to this package by module name, so
disabling bloqade's logging cannot accidentally silence anything else in the
process.

**Consequence.** A dependency, and a second logging system in any application
that already uses `logging`. Records do not propagate between the two;
integration means adding a loguru sink that forwards to `logging`.

### D54b — `warnings.warn` for degraded behaviour, `logger` for the audit trail

`logger.info` records submissions and status fetches. `warnings.warn` is used
for a failed `cancel()` and for an unwritable log path.

**Why.** They have different audiences. The log is an after-the-fact record a
user consults to recover a `task_id`, so it is written to a file and is silent
by default ([D54](#d54-logging-is-opt-in-and-a-bad-path-warns-instead-of-raising)).
A warning is a *now* problem the caller may need to act on — "your cancel did
not go through" — so it goes to stderr through the mechanism Python users
already filter and capture in tests, and it fires whether or not logging is
enabled.

**Consequence.** Warnings are not written to the log file, so a
recovered-after-the-fact log will not show that a cancel failed.

---

## 9. Extension points

| To change | Do this | Wire it in |
|---|---|---|
| Kernel encoding of a Kirin module | supply a `KernelSerializer` (any object with `encode`) | `Device(kernel_serializer=…)` or per call |
| A non-Kirin language (e.g. QASM2 text) | override `TaskABC.serialize_kernel` | `Device.single_kernel_task_cls` |
| A new fan-out shape | subclass `TaskABC`, implement the five getters | a new `*_task_cls` slot |
| A payload the assembler can't express | override `create_task_definition` | as above |
| Fetch/poll behaviour, extra terminal states | subclass `Future`; override `_fetch_subtask_page` or `EXIT_STATUS` | `Device(future_cls=…)` |
| Domain analysis (counts, decoding, expectations) | subclass `Result` | `Future.result_cls` |
| A new persistence format | subclass `StorageBackend` (nine abstract methods) | pass as `storage=` |
| Non-interactive auth | subclass `AuthMixin` | inherited by device/task/future |

A concrete language specialization, from the shipped demo:

```python
@dataclass
class QASM2Task(SingleKernelTask):
    def serialize_kernel(self, kernel: Method) -> str:
        return QASM2().emit_str(kernel)

@dataclass
class QASM2Device(Device):
    single_kernel_task_cls: type[SingleKernelTask[Future[Result]]] = field(
        default=QASM2Task, init=False
    )
```

A full hardware specialization is the same pattern on three layers: a `Result`
subclass with the machine's decoding, a `Future` subclass pinning `result_cls`
and a class-level `context_name` ([D34](#d34-_resolve_context_name-reads-a-class-level-default-via-getattr)),
and a `Device` subclass pinning `future_cls` and the task classes. Users then
touch only `Device`.

**In practice, subclassing is rarely needed.** The real parameter-scan consumer,
`bloqade-flair`'s `SubmissionScanner.to_task`, uses `Device` unmodified: it
expands a scan grid, flattens each point into type-tagged float keys
([§10](#10-the-api-contract-verified)), and calls
`device.parameter_scan(kernel=…, arguments=…, program_language="flair")` — a
per-call override ([D9](#d9-per-call-overrides-fall-back-to-device-defaults)),
no `Device` subclass, no task subclass. That is the intended shape for a language
front-end: build the argument list and the language string, and let the injected
serializer do the rest.

A custom storage backend implements `add_shots`, `get_shots`, `task_ids`,
`add_task_definition`, `get_program_language`, `get_task_creation_time`,
`get_programs`, `get_subtasks`, `update_subtasks_completed_date` — and inherits
`get_task_definition`, `get_arguments`, `get_task_group_id`, and all four
`filter_by_*`. It must honour [D41](#d41-getters-must-return-independent-copies)
(return copies) and [D42](#d42-an-empty-filter-tuple-matches-nothing) (empty
tuple matches nothing).

---

## 10. The API contract, verified

Checked against `qlam-task-manager` and `qlam-task-definitions`
(`public_openapi.yml`), `qlam-result-manager` / `qlam-result-transformer`
(`TaskResult.yaml`), and `compiler-services`.

**Confirmed.** Many-subtasks-to-one-program with per-subtask
`arguments` and `num_shots` is the API's designed shape — the definitions spec's
own example submits three subtasks across two programs with `program_index` 0,
1, 0. `Subtask.arguments` really is `additionalProperties: {type: number,
format: float}`, so [D53](#d53-the-apis-float-coercion-is-documented-not-hidden)
is a wire constraint, not a pydantic artifact. `SubTaskResult` genuinely has no
index field — the index appears only on shot rows, exactly as
[D30](#d30-completed_date-is-written-on-the-first-shot-page-indexed-via-shot_results)
assumes. `shots_page` is documented as "applied uniformly to every subtask on
the current page", which is what
[D28](#d28-two-axis-pagination-completed-first-sort-and-an-incremental-watermark)'s
two-axis loop is built around, and `completed_date` is a supported sort
property. Cancellation exists only as `/v2/{qpu_mode}/tasks/{id}/cancel` —
whole-task, no per-subtask form.

**Refinement — argument binding is real, and there is a convention above it.**
`arguments` are not merely passed through: the Gemini Flair compiler plugin
calls `unflatten_dict(subtask_arguments)` and then
`assign_var_by_dict(mt=program, parameters=scan_vars)`, i.e. per-subtask
*partial compilation* of the scan variables into the program. The plugins for
plain `flair` explicitly discard them at compile time and forward them to the
compiled subtask instead. The float-only limit is worked around by a flattening
convention in `bloqade-flair` (`flatten_dict`/`unflatten_dict`): keys carry a
path and a trailing type tag, so `{'x:int': 3.0, 'v:0:float': 1.0, 'v:1:float':
2.0}` round-trips to `{'x': 3, 'v': [1.0, 2.0]}` — ints, bools, nested dicts,
and lists all survive. `bloqade-core` does not implement this convention and
should not: it is a contract between a language front-end and its compiler
plugin. Note the consequence, though — raw untagged keys such as
`{"theta": 0.5}` would fail that plugin's `unflatten_dict` assertion, so a
`parameter_scan` aimed at the Gemini Flair path must send flattened keys.
`bloqade-flair`'s `SubmissionScanner.to_task` is the reference for that: it
flattens and namespaces (`user_args`, `flags`) before calling
`device.parameter_scan(...)`.

**Unused API capability.** `TaskCreationRequest` is a `oneOf` over three
shapes: an inline `TaskDefinition`, a `TaskDefinitionReference`
(`definition_id`), and a `CompilationReference` (`compilation_id`).
`bloqade-core` only ever sends the first. Similarly, `GET
/v2/tasks/{task_id}/results` returns "the full set of results for the task as a
binary stream, bypassing pagination" under `Accept:
application/octet-stream` — `Future.fetch` always paginates JSON instead, which
is the right default for incremental fetch
([D28](#d28-two-axis-pagination-completed-first-sort-and-an-incremental-watermark))
but leaves a faster path unused for the "task is finished, give me everything"
case. Re-running an identical scan —
[D48](#d48-merging-across-task-ids-by-subtask_index-with-an-explicit-compatibility-contract)'s
motivating workflow — therefore re-uploads every program instead of referencing
the definition already stored server-side, and re-runs compilation instead of
referencing the compilation. Storage already keeps everything needed to support
the reference forms.

**The local schema mirrors the wire model, not the server's storage.** QLAM is
service-per-database and considerably more normalized. `qlam-task-definitions`
(`V0__init.sql`) has `program`, `program_content` (payload split out so listing
does not drag content), `taskdef`, and `subtask` — UUID surrogate keys,
`tenant_id`, `tce_name`, `arguments JSONB`, metadata split into
`*_user_metadata TEXT` and `*_system_metadata JSONB`, and the language version
stored as **three integer columns** (`version_major/minor/patch`) plus a
`checksum`. `qlam-result-manager` (`V8__create_v2_tables.sql`) has seven tables
— `task`, `sub_task`, `shot`, `shot_stopped`, `detection`, `execution_plan`,
`atom_site` — with no foreign keys and no unique constraints, because
"application-level idempotency handles duplicates."

| concern | QLAM | `bloqade.core.device` |
|---|---|---|
| identity | UUID surrogate keys + `tenant_id` | natural composite keys, no tenant |
| language version | three integer columns | one fused string ([D19](#d19-language-and-version-are-fused-into-one-wire-string)) |
| program payload | split into `program_content` | inline in `programs.content` |
| metadata | user `TEXT` / system `JSONB`, separately | one JSON blob in `subtasks.metadata` |
| measurement row | `detection(task_id, shot_index, camera_id, frame_index, frame_type_code, measurements INTEGER[])` | `results(task_id, shot_index, subtask_index, subtask_shot_index, frame_type, bitstring TEXT)` |
| per-shot status | `shot.status`, `shot_stopped.error_reasons[]`, `qpu_status` | not stored |
| geometry | `atom_site(geometry_id, site_index, x, y)` | not stored |
| subtask order | `sub_task.sub_task_order` | positional `subtask_index` |

Two consequences matter. The server's measurement index is
`(task_id, shot_index, camera_id, frame_index)` and carries *no* unique
constraint, which confirms that bloqade's `UNIQUE(task_id, shot_index,
frame_type)` is a narrower key than the source data — multi-camera or
multi-frame shots would collapse. And `measurements` is `INTEGER[]` server-side,
not bits, so the `dtype=bool` cast in `_fetch_subtask_page` is an assumption
about value range rather than a representation change. Local storage is
therefore a deliberately lossy projection: it holds what is needed to rebuild a
`TaskDefinition` and analyse bitstrings, and re-fetching will not fill in
per-shot errors, QPU status, or geometry because the fetch loop never reads
them.

**Also noted.** `program_language` is *not* required by the task-manager schema
(only `programs` and `subtasks` are), and the language also appears in the URL
as `qpu_mode` in `<lang>-<constraint>` form (e.g. `squin-256q`) — so
[D19](#d19-language-and-version-are-fused-into-one-wire-string)'s fused string
is a convention, unvalidated and partly redundant. The spec's own examples write
`flair.v1` and `flair.v0.1`, while bloqade emits three-component versions such
as `squin.v0.1.0`. `SubTaskStatus` and `ResultStatus` are their own enums
(`Created`, `Partial`, `Completed`, `Cancelled`, `Failed`) distinct from
`TaskStatus`. A third frame type, `LOADED`, exists alongside `SORTED` and
`DETECTED`.

---

## 11. Dependency map

Inside the package the import graph is a chain, not a web:

```
log_info ──▶ local_storage ──▶ result ──▶ future ──▶ task ──▶ device
                    └───────────────────────┴─────────┴────────┘
mixins ─────────────────────────────────────▶ future, task, device
```

Why each edge exists:

| edge | reason |
|---|---|
| `local_storage → log_info` | only to log the schema migration ([D44](#d44-schema-is-versioned-migrations-are-forward-only-mismatches-are-fatal)) |
| `result → local_storage` | a `Result` *is* a storage query — it needs `ShotFilter`, `StorageFilter`, `ShotResult` ([D47](#d47-a-result-is-a-query-not-data)) |
| `future → result` | `future.result()` constructs one; needs `Result` and `ResultType` ([D3](#d3-every-seam-is-a-class-injection-slot)) |
| `future → local_storage` | writes fetched shots; needs `StorageBackend` and the `DictStorage` default ([D23](#d23-storage-defaults-to-a-fresh-dictstorage)) |
| `task → future` | **the load-bearing edge** — `submit_task_definition` returns a future, so `task.py` imports `Future`, `FutureType`, `ApiFetchOptions`, `DEFAULT_FETCH_OPTIONS` |
| `device → task, future` | the factory constructs tasks and holds `future_cls` |
| `mixins → future, task, device` | all three authenticate, which is why all three know a `context_name` ([D10](#d10-auth-is-a-mixin-because-three-unrelated-classes-need-it)) |

**The shape is wrong, and the fix is structural.** Payload construction is the
most abstract concern in the package — it needs `kirin` and the qlam payload
models and nothing else — yet `task.py` sits second from the top and
transitively depends on storage, results, and four QLAM clients. Moving
submission onto `Device` turns the chain into a fan:

```
task ────────────┐          task depends only on kirin + qlam models
local_storage ───┼──▶ device
result ─▶ future ┘
```

`task.py` becomes a leaf importable with no network client and no storage
backend — the structural statement of "tasks are device-agnostic builders." See
the [spec](https://github.com/QuEraComputing/bloqade-core/blob/main/specs/spec-21-08-2026-device-owns-submission.md).

External dependencies and why each is present: `qlam-core` (API clients and
payload models), `kirin-toolchain` (`ir.Method`, dialect encoding, `JSONSerializer`),
`numpy` (bitstring arrays), `loguru` ([D54a](#d54a-loguru-with-bloqade-as-a-disable-able-namespace)),
`typing_extensions` ([D1a](#d1a-typing_extensions-for-self-and-typevardefault)).

---

## 12. Data model reference

### 12.1 The QLAM primitives

QLAM splits what is casually called "a task" across four resources: a
**definition** (the payload), an optional **compilation** (processed payload), a
**task** (execution record), and **results**. The payload primitive is a program
pool plus an execution list that indexes into it:

```json
{
  "program_language": "flair.v0.1",
  "programs":  [{"content": "<opaque string>", "program_metadata": {...}}],
  "subtasks":  [{"program_index": 0, "num_shots": 100,
                 "arguments": {"theta": 0.1},
                 "subtask_metadata": {"user_metadata": "..."}}],
  "group_id": "<uuid>"
}
```

`programs` and `subtasks` are both `minItems: 1`; `program_index` must be less
than `len(programs)`; `content` is an **opaque string** the API never parses;
every schema is `additionalProperties: false`, so metadata strings are the only
extension slot. The execution record — `{id, task_status, definition_id,
compilation_id, group_id, created_*, modified_*, scheduled_date,
error_reasons[]}` — holds no payload and no results. Task creation accepts a
`oneOf`: an inline definition, a `definition_id`, or a `compilation_id`, which is
the clearest statement that a task *is* a pointer to a payload plus a group.
Results invert the nesting: task → subtask → shot → frame row. Everything
task-side is addressed under `/v2/{qpu_mode}/…` where `qpu_mode` is
`<lang>-<constraint>` (`squin-256q`); results live at
`/v2/tasks/{task_id}/results`.

There is **no scan resource**. A parameter scan is one task whose subtasks all
carry `program_index=0` and differ only in `arguments`
([D17](#d17-program-deduplication-is-a-one-method-hook)) — the fan-out is native,
the *name* is client-side. What is server-side is the binding: the Gemini Flair
compiler plugin calls `unflatten_dict(subtask_arguments)` and then
`assign_var_by_dict(mt=program, parameters=scan_vars)`, i.e. per-subtask partial
compilation.

`arguments` is **optional per subtask** — only `program_index` and `num_shots`
are required — and is one flat `dict[str, float]`, never a list. One map is one
point in parameter space, which is why N scan points need N subtasks. Presence
may vary per subtask on the wire; `TaskABC.get_arguments()` is all-or-nothing at
the task level, but `[{"x": 1.0}, None]` produces a valid mixed payload, which is
why the read path is typed `list[dict | None]`.

### 12.2 The local tables

One SQLite file, five tables ([D38](#d38-task-metadata-is-decomposed-into-tables-not-stored-as-a-blob)):

```sql
task_definitions(task_id PK, program_language, creation_time, group_id)
programs(task_id, program_index, content, PRIMARY KEY(task_id, program_index))
subtasks(task_id, subtask_index, program_index, num_shots,
         arguments TEXT, metadata TEXT, completed_date,
         PRIMARY KEY(task_id, subtask_index))
results(row_number PK AUTOINCREMENT, task_id, shot_index, subtask_index,
        subtask_shot_index, frame_type, bitstring TEXT,
        UNIQUE(task_id, shot_index, frame_type))
bloqade_schema(version_number PK)          -- "0.2.0"
```

`arguments` and `metadata` are JSON text, timestamps are ISO strings, and
`bitstring` is `"0101"` text ([D43](#d43-sqlite-specifics-each-chosen-against-a-concrete-failure)).
The first three tables are a normalized `TaskDefinition`; one file holds many
`task_id`s.

There are two backends behind the ABC — `DictStorage` (in-memory, per-process)
and `SQLiteStorage` (one file per instance) — and any number of instances, with
`Future.export_to` moving rows between them
([D35](#d35-export_to-chunks-shots-and-copies-definitions)). There is
deliberately no default *path* ([R2](#r2-a-default-storage-file-path-was-considered-and-rejected)).
A third copy exists in the wild: `bloqade-internal` ships a pre-#101 fork of this
module pinned at schema `0.1.0`, so a file that core has migrated to `0.2.0` can
no longer be opened by it — one-way, by design
([D44](#d44-schema-is-versioned-migrations-are-forward-only-mismatches-are-fatal)).

### 12.3 `ShotResult`, field by field

| field | meaning |
|---|---|
| `task_id` | QLAM task UUID; the join key to the three metadata tables |
| `shot_index` | position within the **whole task**, 0-based, spanning all subtasks |
| `subtask_index` | which subtask produced it — positional index into `TaskDefinition.subtasks`, i.e. which scan point or which kernel |
| `subtask_shot_index` | position **within its own subtask**, restarting at 0 per subtask |
| `frame_type` | which camera frame: `LOADED`, `SORTED`, or `DETECTED` |
| `bitstring` | the measurement, one boolean per site, indexed by site |

Two addressing schemes coexist. For a 3-point scan with 2 shots per point,
`shot_index` runs 0–5 while `(subtask_index, subtask_shot_index)` runs
(0,0),(0,1),(1,0),(1,1),(2,0),(2,1). All three indices arrive from the API as
required fields. `shot_index` is already unique within a task, so the other two
are technically redundant — they are kept because they are what you *group by*:
`shot_results()` returns one array per `subtask_index`, and `subtask_shot_index`
is the row index within it.

A single shot passes through stages of the atom-imaging pipeline and each stage
yields its own image and bitstring — `LOADED` (which sites received an atom),
`SORTED` (occupancy after rearrangement), `DETECTED` (the measurement). So one
physical shot becomes several rows differing only in `frame_type` and
`bitstring`, which is why `frame_type` is in the dedup key
([D37](#d37-one-flat-row-per-shot-per-frame-with-frame_type-in-the-identity)) and
why every view defaults to `DETECTED`
([D47a](#d47a-detected-is-the-default-frame-type)).

`bitstring` arrives as `measurement.measurement_values`, typed `INTEGER[]` on
the wire and stored `INTEGER[]` server-side in `detection`. Position is the site
index, which aligns with the geometry the server keeps in `atom_site` and
bloqade does not store — so mapping a bit to a physical position needs the
program, not the shot row.

### 12.4 Aligning frames: `(task_id, shot_index)` is the key

Both frames of one physical shot share the same `shot_index`, so
`(task_id, shot_index)` is the correct and sufficient join key for pairing
`SORTED` with `DETECTED` across subtask boundaries — `subtask_index` is not
needed, because `shot_index` is already global across subtasks. That is exactly
what `filter_by_shots` produces: pairs of `(task_id, shot_index)`, deliberately
carrying no `frame_type`, which is what lets a filter derived from one frame
apply to a view scoped to another
([D52](#d52-where_shots-separates-the-predicates-scope-from-the-results-scope)).
Include `task_id` because `shot_index` is unique only *within* a task.

Do **not** align positionally. `get_shots` has no `ORDER BY`, so row order is
insertion order and incidental; and nothing guarantees every shot has every
frame, so a single missing `SORTED` row shifts a `zip` silently. Join on the key,
or stay in the `Result` API and let `task_shot_pairs` give inner-join semantics
for free.

---

## 13. Frequently asked

**Is a parameter scan natively supported by QLAM?** The fan-out is
([§12.1](#121-the-qlam-primitives)); the *name* is not. There is no scan
resource, no server-side sweep expansion — `ParameterScanTask` materializes
every point client-side — and no scan identity on the task record, which is why
`Result.validate()` has to reconstruct it by comparing `program_index` and
`arguments` per `subtask_index`
([D48](#d48-merging-across-task-ids-by-subtask_index-with-an-explicit-compatibility-contract)).
Argument *binding*, however, is genuinely server-side.

**Does every subtask have arguments?** No — optional, one flat float map each,
and presence may vary per subtask. See [§12.1](#121-the-qlam-primitives).

**Why is `num_subtasks` a property and not a field?** It is derived, and a stored
copy could disagree with the data it counts. It is public and read-only, not
private. See [D14a](#d14a-num_subtasks-is-a-public-abstract-property-not-a-stored-field).

**Why not put `run_async` on `Device`?** Nothing forbids it; it costs eleven
parameters and six overloads *as long as task construction shares the signature*.
Passing a builder dissolves that. See the "Considered" note in
[D6](#d6-a-factory-that-deliberately-cannot-submit) and the
[spec](https://github.com/QuEraComputing/bloqade-core/blob/main/specs/spec-21-08-2026-device-owns-submission.md).

**Do we need three task subclasses? Do we need `TaskABC` at all?** See the
[ledger](#ledger-three-task-classes-versus-one). Short version: the three are
justified while construction is declarative, and unnecessary once it is
incremental — at which point `TaskABC` can be deleted in favour of one builder
plus a `TaskSpec` Protocol.

**Why does each device need a `Future` subclass *and* a `Result` subclass?** The
`Result` subclass is real work — decoding needs the *program*, via
`storage.get_programs()`, not just the bitstrings. The `Future` subclass is pure
wiring, forced by `submit_task_definition` not forwarding `result_cls`; see
[Known trade-offs](#known-trade-offs-and-open-todos).

**Could the per-program post-processing be generic instead of a subclass?** Yes.
The device-specific part is only "deserialize this language, build a decoder from
the kernel", and storage already records `program_language` per task, so dispatch
can be data-driven — either an injected decoder object forwarded like
`result_cls`, or a language registry mirroring how `compiler-services` dispatches
compiler plugins. Two caveats: language is per *task* while merged views drop
`task_id`, so dispatch inherits [D48](#d48-merging-across-task-ids-by-subtask_index-with-an-explicit-compatibility-contract)'s
homogeneity assumption; and version matching needs a decision, since the stored
value is the fused `squin.v0.1.0` ([D19](#d19-language-and-version-are-fused-into-one-wire-string)).
A subclass remains the right home once the device grows a real vocabulary
(`logical_error_rate()`, `syndromes()`) rather than just "decode".

**Is it a problem if each kernel returns a different type?** Not at runtime — and
it cannot arise within a scan, where every subtask shares one program. The real
problems are that the current `RetType` annotation is unsound (a method-scoped
TypeVar with nothing to bind to unless functions are passed explicitly), and that
a positional list plus a silent "no decoder → raw bitstrings" fallback loses
provenance. A tagged per-subtask return would fix both.

**What validates what, and where?** Argument/metadata/shot-count lengths are
checked locally before authentication ([D15](#d15-all-length-validation-happens-locally-before-authentication));
merge compatibility is checked at read time ([D48](#d48-merging-across-task-ids-by-subtask_index-with-an-explicit-compatibility-contract));
pydantic enforces the wire schema (`additionalProperties: false`); the server
reports payload problems as `PayloadProcessingError`. Kernel-level checks — a
dialect-group subset test and an optional kirin `ValidationSuite` — are specced
in [PR #107](https://github.com/QuEraComputing/bloqade-core/pull/107) and do not
exist yet. Two gaps: nothing checks that argument *keys* match the program's
declared parameters, and nothing validates the fused language string.

---

## Revised decisions

### R1 — File logging was on by default

Shipped in #65 as an import-time `logger.add("bloqade.log", …)` unless
`BLOQADE_LOGGING=0`.

**Why it was that way.** Defensive, in the author's words: *"the 'safe' route so
people can easily keep track of their task_ids."* A submitted task costs queue
time; losing its ID costs money.

**Why it changed.** Reviewers flagged the import side effect — a library writing
files into the user's working directory. The response at the time was to ship and
see whether anyone complained, while conceding the env var was hard to discover.
They complained.

**Now:** opt-in, via `set_logging` — [D54](#d54-logging-is-opt-in-and-a-bad-path-warns-instead-of-raising).
The recovery need it served is met by
[D22](#d22-the-task-id-is-durable-before-the-future-is-returned) instead.

### R2 — A default storage file path was considered and rejected

Proposed in review: default to something like `.bloqade/tasks.sql` so users need
not think about storage.

**Why it was rejected.** Storage *appends*. A shared default file accumulates
tasks with unrelated subtask structures, and `Result`'s merge validation
([D48](#d48-merging-across-task-ids-by-subtask_index-with-an-explicit-compatibility-contract))
then forces the user to retroactively pick task IDs — long after the context is
gone. Explicit storage means explicit aggregation scope.

**Still in force.** [D23](#d23-storage-defaults-to-a-fresh-dictstorage) added
an in-memory default, which lowers first-use friction without creating a shared
file. A default *path* remains rejected.

### R3 — Serialization moved from subclass to injected object

#65 made `serialize_kernel` the only extension point, so changing encodings
meant a task subclass and a `Device` subclass.

**Why it changed.** Subclassing is a type-level commitment for what turned out
to be a per-call parameter — and it forced the three-slot override problem in
[D8a](#d8a-three-separate-task-class-slots-one-per-shape-all-initfalse). A
`Protocol`-typed field carries the same information as data.

**Now:** `kernel_serializer` with a per-call override
([D18](#d18-serialization-is-an-injectable-protocol-and-bytes-get-base64d)).
Subclassing is reserved for changing *what* is encoded, not how.

### R4 — `program_language_version` went from `""` to a real default

Originally an empty string with "override this to set versions".

**Why it changed.** The empty default produced a meaningless wire string and
passed `version=""` into `kernel.dialects.encode`, so the encoded module carried
no version either. A default that is wrong everywhere is worse than a default
that is merely arbitrary.

**Now:** a `language_version` field defaulting to `"0.1.0"`, fused into the wire
string ([D19](#d19-language-and-version-are-fused-into-one-wire-string)).

### R5 — `Device.list_groups` was built, then removed

A `list_groups` method and a `current_user_id` helper were added on `Device`
during the #101 work and removed again before the branch merged, replaced by
submission with a group *name*.

**Why it changed.** Listing groups put a network call on the one class whose
value is making none ([D6](#d6-a-factory-that-deliberately-cannot-submit)),
duplicated `qsh`, and answered a question users mostly ask once. Accepting a
name and resolving it at submission
([D20a](#d20a-the-user-facing-group-is-a-name-the-wire-wants-a-uuid),
[D21](#d21-group-names-are-resolved-to-uuids-at-submission-not-construction))
covers the actual need with no new surface.

**Now:** `Device` has no introspection methods at all
([D9a](#d9a-device-exposes-no-backend-introspection)). Worth knowing this was
tried, because "just add a `list_*` helper" is a recurring request.

---

## Known trade-offs and open TODOs

Recorded in the code; none are hidden, all are cheap to fix if they start to
hurt.

- **The fetch watermark is not persisted** (`future.py` TODO). A new session
  re-fetches every shot. Safe by [D29](#d29-fetching-is-idempotent-by-construction-not-by-bookkeeping),
  merely wasteful.
- **`frame_type` is a bare string** with a TODO to make it an enum. Normalized
  to upper case in two places instead. The API's set is `LOADED`, `SORTED`,
  `DETECTED`.
- **The shot dedup key may be too narrow.** `(task_id, shot_index, frame_type)`
  ([D37](#d37-one-flat-row-per-shot-per-frame-with-frame_type-in-the-identity))
  omits `camera_id` and `frame_index`, both of which the API's `ShotResult`
  carries — `frame_index` is documented as "the frame index within the shot."
  With a single camera and one frame per type this is unique; with more than one
  camera, rows would silently overwrite each other rather than coexist. Worth
  confirming against hardware before it matters.
- **Per-shot `error_reasons` are discarded.** `_fetch_subtask_page` reads only
  `measurement_values`, so a shot the backend flagged as erroneous is stored
  indistinguishably from a clean one. `measurement_values` are also typed as
  integers in the spec and cast with `dtype=bool`, which would silently map any
  value above 1 to `True`.
- **A permanently failed subtask pins the fetch watermark.**
  `_fetch_subtask_page` treats any status other than `COMPLETED` as "incomplete",
  but `SubTaskStatus` includes the terminal states `Failed` and `Cancelled`. A
  task with one failed subtask therefore re-reads from that page on every
  subsequent `fetch()`. Harmless by
  [D29](#d29-fetching-is-idempotent-by-construction-not-by-bookkeeping), but it
  defeats the optimization for exactly the tasks most likely to be polled a lot.
- **Bitstrings are uncompressed text** (`local_storage.py` TODO). Legibility
  over size; revisit at large shot counts.
- **`cancel()` swallows the exception** into a `warn`, which the author noted
  "may lose error details."
- **`KernelBatchTask.summary()` prints argument names without values**, unlike
  `SingleKernelTask.summary()`, which formats `key=value`.
- **Compilation errors are not fetched on failure** (`_wait_for_completion`
  TODO); `get_compilation` exists but must be called manually.
- **Storage getters must return copies** ([D41](#d41-getters-must-return-independent-copies)),
  a cross-module contract enforced only by documentation.
- **Partial task-class overrides fail silently.** A `Device` subclass that sets
  only `single_kernel_task_cls` ([D8a](#d8a-three-separate-task-class-slots-one-per-shape-all-initfalse))
  still builds base-class tasks from `batch_task()` and `parameter_scan()` — so
  a language-specialized device would serialize batches with the wrong encoder
  and give no warning. The demo's `QASM2Device` has exactly this shape. Since
  [R3](#r3-serialization-moved-from-subclass-to-injected-object) made
  serializers injectable, most specializations no longer need task subclasses at
  all; for those that do, a single hook (a `task_cls_for(shape)` method, or a
  mixin applied to all three defaults) would remove the footgun.
- **`StorageBackend` and `ApiFetchOptions` are not re-exported**
  ([D1c](#d1c-the-exported-surface-is-the-nouns-users-construct)), so writing a
  custom backend or changing a poll interval requires a submodule import. The
  first is defensible; the second looks like an oversight, since a user tuning
  `poll_interval_max` is not an implementer.
- **One query per subtask** in `Result.shot_results()`
  ([D50a](#d50a-one-query-per-subtask-not-one-grouped-query)) — an N+1 pattern,
  fine at current scan widths, first thing to change if it stops being fine.
- **`get_shots` has no `ORDER BY`**, while `get_programs` and `get_subtasks`
  both do (`ORDER BY task_id, program_index` / `task_id, subtask_index`). Shot
  rows therefore come back in `rowid` order — insertion order, which is fetch
  order ([D43b](#d43b-a-surrogate-row_number-alongside-the-natural-unique-key))
  — and that ordering is incidental, not specified. Anything that aligns two
  frame arrays positionally (`SORTED` against `DETECTED`, say) is relying on it.
  Adding `ORDER BY task_id, shot_index` would make the guarantee explicit and
  match the other two accessors.
- **`get_shots` results are single-pass**
  ([D36a](#d36a-get_shots-returns-a-generator-in-both-backends)) but typed as
  `Iterable`, so double iteration silently yields nothing the second time.
- **A `Result` subclass forces a `Future` subclass.**
  `submit_task_definition` constructs `self.future_cls(task_id=…,
  fetch_options=…, storage=…, context_name=…)` and never forwards `result_cls`,
  so the only way to make submission return a specialized result is to bake
  `result_cls` into a `Future` subclass. The Gemini logical backend's future is
  17 lines of exactly that wiring and no behaviour. Adding a `result_cls` field
  to `Device`/`TaskABC` and forwarding it would remove the requirement; the
  argument against is that two independent slots (`future_cls`, `result_cls`)
  can be set to disagree, whereas `Future[GeminiLogicalResult]` carries its own
  guarantee.
- **A `ShotResult` is only shallowly frozen**
  ([D2a](#d2a-values-are-frozen-handles-are-not)); its `bitstring` array is
  mutable in place.
