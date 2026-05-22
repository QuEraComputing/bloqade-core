# Running kernels on a device

`bloqade.core.device` provides the pieces needed to submit a kernel to a
backend, wait for it to finish, and read out its shot results. The four
concepts you interact with are `Device`, the task objects it builds, the
`Future` returned on submission, and the `Result` view fetched from that
future.

```
Device ──task()──▶ Task ──run_async()──▶ Future ──result()──▶ Result
```

## Concepts

**Device.** A factory for tasks. It captures the qlam context name to
authenticate against and the task class to use. The device does not submit
anything itself; it builds task objects via [`Device.task`](reference/bloqade/core/device/device.md#bloqade.core.device.device.Device.task),
[`Device.batch_task`](reference/bloqade/core/device/device.md#bloqade.core.device.device.Device.batch_task),
and [`Device.parameter_scan`](reference/bloqade/core/device/device.md#bloqade.core.device.device.Device.parameter_scan).

**Task.** Bundles one or more kernels with per-subtask metadata, arguments,
and shot counts. A task is what actually gets sent to the backend. Calling
[`run_async(dry_run=True)`](reference/bloqade/core/device/task.md#bloqade.core.device.task.TaskABC.run_async)
validates the task and prints a summary without submitting;
`run_async(dry_run=False)` serializes the kernels, submits them, and returns
a `Future`. The concrete task shapes are
[`SingleKernelTask`](reference/bloqade/core/device/task.md#bloqade.core.device.task.SingleKernelTask),
[`KernelBatchTask`](reference/bloqade/core/device/task.md#bloqade.core.device.task.KernelBatchTask),
and [`ParameterScanTask`](reference/bloqade/core/device/task.md#bloqade.core.device.task.ParameterScanTask),
all built on the shared [`TaskABC`](reference/bloqade/core/device/task.md#bloqade.core.device.task.TaskABC)
base class.

**Future.** A handle to a submitted task. It polls task status against the
backend, fetches available shot results into storage, and constructs a
`Result` view. [`future.result(timeout=...)`](reference/bloqade/core/device/future.md)
blocks until the task reaches a terminal status, then fetches all shots and
returns a `Result`. A future can also be reattached to an existing task ID
via [`Future.from_task_id`](reference/bloqade/core/device/future.md) or
[`Future.from_storage`](reference/bloqade/core/device/future.md), which is
how you resume from a previous session.

**Result.** A view over the shots in storage scoped to one or more task IDs
and a frame type. [`result.shot_results()`](reference/bloqade/core/device/result.md)
returns one boolean `np.ndarray` per subtask. The `where_subtasks`,
`where_arguments`, `where_metadata`, and `where_shots` methods narrow the
view without copying data.

## The intended flow

In the example below, you can see how the intended flow looks when submitting
a simple squin kernel to a machine.
The `context_name` sets the context as specified in your `~/.qsh/config.json`.

```python
from bloqade import squin
from bloqade.core.device import Device

# 1. Define a kernel.
@squin.kernel
def bell():
    q = squin.qalloc(2)
    squin.h(q[0])
    squin.cx(q[0], q[1])

# 2. Build a task from the device.
device = Device(context_name="my-context")
task = device.task(
    kernel=bell,
    num_shots=2,
    metadata={"tag": "bell"},      # metadata is arbitrary user JSON
    program_language="squin",
)

# 3a. Dry-run first to see what would be submitted.
task.run_async(dry_run=True)

# 3b. Submit. The first call triggers browser-based authentication.
future = task.run_async(dry_run=False)

# 4. Wait for completion and read shot bitstrings.
result = future.result(timeout=80.0)
print(result.shot_results())
```

`device.task(...)` returns a
[`SingleKernelTask`](reference/bloqade/core/device/task.md#bloqade.core.device.task.SingleKernelTask).
Use `device.batch_task(...)` for a
[`KernelBatchTask`](reference/bloqade/core/device/task.md#bloqade.core.device.task.KernelBatchTask)
that runs multiple independent kernels in one task, or
`device.parameter_scan(...)` for a
[`ParameterScanTask`](reference/bloqade/core/device/task.md#bloqade.core.device.task.ParameterScanTask)
that runs one kernel against several argument sets. The downstream steps
(`run_async`, `future.result`) are identical.

## Persisting results

`run_async` accepts a `storage` argument that controls where the task
definition and fetched shots are kept. If you omit it, a fresh
[`DictStorage`](reference/bloqade/core/device/local_storage.md) is used —
that is, the data lives in memory only and is lost when the process exits.
Pass a [`SQLiteStorage`](reference/bloqade/core/device/local_storage.md)
instead to keep everything on disk and resume work in a later session:

```python
from bloqade.core.device import SQLiteStorage

storage = SQLiteStorage("results.sql")
future = task.run_async(dry_run=False, storage=storage)
result = future.result(timeout=80.0)

storage.close()  # optional; GC will also close it
```

In a later session, reattach to the same task without resubmitting:

```python
from bloqade.core.device import Future
from bloqade.core.device import SQLiteStorage

storage = SQLiteStorage("results.sql")
future = Future.from_storage(storage=storage, context_name="my-context")
result = future.result(timeout=80.0)
```

When more than one task ID is present in the storage, pass `task_id=...` to
disambiguate. To fetch a task that you do not have stored yet, use
`Future.from_task_id(task_id=..., storage=storage, context_name=...)` —
this fetches the task definition from the backend into `storage` and
returns a future you can drive normally.

## Filtering results

A `Result` is a view over storage rather than a materialized list of shots.
Each `where_*` method returns a new, narrower view by combining the current
[`ShotFilter`](reference/bloqade/core/device/local_storage.md) with the
matches from a predicate. Nothing is copied, so the views are cheap and can
be chained. Each method takes a `Callable[[X], bool]` predicate whose input
type `X` matches the row kind being filtered:

- [`where_subtasks`](reference/bloqade/core/device/result.md#bloqade.core.device.result.Result.where_subtasks)
  — `Callable[[dict], bool]`. Predicate receives the full subtask record
  (program index, num_shots, arguments, metadata, …).
- [`where_arguments`](reference/bloqade/core/device/result.md#bloqade.core.device.result.Result.where_arguments)
  — `Callable[[dict | None], bool]`. Predicate receives the per-subtask
  arguments dict only (or `None`). Convenient for slicing a parameter scan.
- [`where_metadata`](reference/bloqade/core/device/result.md#bloqade.core.device.result.Result.where_metadata)
  — `Callable[[dict | None], bool]`. Predicate receives the JSON-decoded
  user metadata you attached when building the task (or `None`).
- [`where_shots`](reference/bloqade/core/device/result.md#bloqade.core.device.result.Result.where_shots)
  — `Callable[[ShotResult], bool]`. Predicate receives individual
  [`ShotResult`](reference/bloqade/core/device/local_storage.md) rows;
  the optional `predicate_filter: ShotFilter | None` argument controls
  which shots the predicate is evaluated against (see frame types below).

```python
# Suppose `result` came from a parameter scan whose arguments look like
#   {"theta": float, "phi": float, "lam": float}
# and whose per-subtask metadata looks like
#   {"recheck": bool, ...}

def lam_is_zero(arguments: dict | None) -> bool:
    return arguments is not None and arguments.get("lam") == 0.0

def needs_recheck(metadata: dict | None) -> bool:
    return metadata is not None and metadata.get("recheck", False)

# Slice the scan to lam == 0, then keep only subtasks tagged for re-analysis.
narrowed = result.where_arguments(lam_is_zero).where_metadata(needs_recheck)
print(narrowed.shot_results())
```

### Frame types

Every shot is recorded under one of two frame types:

- **`DETECTED`** — the final measurement at the end of the program. This is
  the default scope of a `Result` and what `shot_results()` returns.
- **`SORTED`** — the readout taken right after atom sorting. Useful as a
  precondition to check whether sorting was successful, so you can discard
  the corresponding `DETECTED` shot when it was not.

`where_shots` makes this precondition explicit by accepting a
`predicate_filter`. The predicate is evaluated against the shots selected
by that filter, but the returned view keeps the original frame scope:

```python
from bloqade.core.device import ShotFilter, ShotResult

def atom_sorting_was_successful(shot: ShotResult) -> bool:
    return bool(shot.bitstring.all())

clean = result.where_shots(
    atom_sorting_was_successful,
    predicate_filter=ShotFilter(frame_type="SORTED"),
)
print(clean.shot_results())  # DETECTED shots whose SORTED frame passed
```

The [`ShotFilter`](reference/bloqade/core/device/local_storage.md) and
[`StorageFilter`](reference/bloqade/core/device/local_storage.md) classes
in `local_storage.py` describe every field you can scope on directly if
you need finer control than the `where_*` helpers provide.

## Specializing a task

The default task serializes kernels with `kirin`'s JSON encoder, which fits
most use cases. When the backend expects a different program format, or when
you want to attach extra fields to the submitted task, subclass the relevant
task type and point the device at it.

As an example, consider a backend that expects QASM2 source. The default
[`SingleKernelTask`](reference/bloqade/core/device/task.md#bloqade.core.device.task.SingleKernelTask)
would encode the kernel as kirin JSON, which the backend cannot parse.
Overriding [`serialize_kernel`](reference/bloqade/core/device/task.md#bloqade.core.device.task.TaskABC.serialize_kernel)
on a subclass and pointing a `Device` subclass at it is enough to switch
the wire format without changing anything downstream:

```python
from dataclasses import dataclass, field
from bloqade.qasm2.emit import QASM2
from kirin.ir.method import Method
from bloqade.core.device import Device, Future, Result
from bloqade.core.device.task import SingleKernelTask

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

A `Device` holds a class reference for each task shape it can build
(`single_kernel_task_cls`, `kernel_batch_task_cls`, `parameter_scan_task_cls`).
Override the ones you need; the device's `task`, `batch_task`, and
`parameter_scan` methods will instantiate your subclass. The
`program_language` argument is just a string recorded on the task
definition, so set it to whatever the backend expects (here, `"qasm"`):

```python
from bloqade import qasm2

@qasm2.main
def bell():
    q = qasm2.qreg(2)
    qasm2.h(q[0])
    qasm2.cx(q[0], q[1])

device = QASM2Device(context_name="gemini-qasm")
task = device.task(            # returns a QASM2Task instance
    kernel=bell,
    num_shots=2,
    metadata={"tag": "bell"},
    program_language="qasm",
)
future = task.run_async(dry_run=False)
result = future.result(timeout=80.0)
```

Other common extension points on the task itself are
[`program_language_version`](reference/bloqade/core/device/task.md#bloqade.core.device.task.TaskABC.program_language_version)
(recorded alongside the program),
[`summary`](reference/bloqade/core/device/task.md#bloqade.core.device.task.TaskABC.summary)
(text printed on dry-run), and
[`create_task_definition`](reference/bloqade/core/device/task.md#bloqade.core.device.task.TaskABC.create_task_definition)
(full control over the submitted payload).

The same pattern extends to the `Future` class. `Device.future_cls` is the
class the task instantiates after submission, so setting it to a `Future`
subclass on your device lets you attach backend-specific helpers — extra
polling logic, logical-result decoding, custom export hooks — without
touching the task or device flow. `Future.result_cls` plays the same role
for `Result`, so a specialized `Future` can surface a specialized result
view through `future.result(...)`.

Runnable versions of both flows live in the repository:

- [`demo/qasm_single_task.py`](https://github.com/QuEraComputing/bloqade-core/blob/main/demo/qasm_single_task.py)
  — the snippet above, using the default in-memory storage.
- [`demo/qasm_single_task_persistent.py`](https://github.com/QuEraComputing/bloqade-core/blob/main/demo/qasm_single_task_persistent.py)
  — the same task pinned to a `SQLiteStorage` file so results survive
  across sessions.

## Logging

Task submissions and status fetches are logged at `INFO` level via
[`loguru`](https://github.com/Delgan/loguru) to a file called `bloqade.log`
in the current working directory. To opt out, set the `BLOQADE_LOGGING`
environment variable to anything other than `"1"` before importing
`bloqade.core.device`:

```bash
export BLOQADE_LOGGING=0
```
