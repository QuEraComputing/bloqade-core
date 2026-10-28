# Building tasks with `TaskBuilder`

`TaskBuilder` is the canonical interface for constructing and submitting
kernel workloads through `bloqade-core`. It separates two responsibilities:

- the builder records which kernels to run, in what order, with which
  arguments, metadata, and shot counts;
- the device owns the language, serializer, supported dialects, and validation
  rules required by its backend.

The device combines those pieces when you call `device.run_async(...)`.

!!! warning "Deprecated task factories"

    `device.task()`, `device.batch_task()`, and `device.parameter_scan()` are
    deprecated. They remain available for backwards compatibility, but new
    code should use `TaskBuilder` and `device.run_async(builder, ...)`.

```text
kernels ──add_subtask()──▶ TaskBuilder ──Device.run_async()──▶ Future ──result()──▶ Result
```

## Basic usage

Create a builder, append each kernel invocation, and submit the builder through
the device:

```python
from bloqade import squin
from bloqade.core.device import Device, TaskBuilder


@squin.kernel
def bell():
    q = squin.qalloc(2)
    squin.h(q[0])
    squin.cx(q[0], q[1])


device = Device(
    context_name="my-context",
    program_language="squin",
    language_version="0.1.0",
)

builder = TaskBuilder()
subtask_index = builder.add_subtask(
    bell,
    num_shots=100,
    metadata={"experiment": "bell", "basis": "Z"},
)
assert subtask_index == 0

# Validate, serialize, and print the task without contacting QLAM.
device.run_async(builder, dry_run=True)

# Finalize and submit the same builder.
future = device.run_async(builder, dry_run=False)
result = future.result(timeout=300)
print(result.shot_results())
```

Use a concrete subclass of `Device` to specify parameters used for validating the supplied kernels.

## Add multiple subtasks

Every `add_subtask()` call adds one subtask at the end of the task. The return
value is its zero-based index:

```python
builder = TaskBuilder()

x_index = builder.add_subtask(x_kernel, num_shots=100)
y_index = builder.add_subtask(y_kernel, num_shots=200)
z_index = builder.add_subtask(z_kernel, num_shots=300)

assert (x_index, y_index, z_index) == (0, 1, 2)
```

A builder is mutable and append-only. Subtask order is submission order; there
is currently no operation for removing or reordering existing entries.

Adding the same kernel object again creates another subtask but reuses its
program entry:

```python
builder.add_subtask(x_kernel, num_shots=25, metadata={"cycle": 0})
builder.add_subtask(x_kernel, num_shots=25, metadata={"cycle": 1})
```

The kernel is serialized once, while the two subtasks retain independent shot
counts and metadata. Currently, program deduplication is based on kernel object identity;
separately constructed kernel objects are treated as separate programs.

## Kernel arguments and metadata

Pass kernel arguments after `metadata`, either positionally or by keyword.
Keyword arguments are usually clearer:

```python
builder.add_subtask(
    parameterized_kernel,
    100,
    metadata={"sample": "A", "basis": "X"},
    theta=0.25,
    repetitions=3,
)
```

Arguments are validated against the kernel's Python signature and stored by
parameter name. Metadata must be a JSON-serializable dictionary and is attached
to the individual subtask. It can later be used to group or filter results.

## Inspect and copy a builder

`summary()` shows program references, arguments, metadata, and shot counts.
`print_detailed()` additionally includes each program's Kirin IR:

```python
print(builder.summary())
print(builder.print_detailed())
```

Use `copy()` when several tasks share a common prefix:

```python
baseline = TaskBuilder()
baseline.add_subtask(calibration_kernel, 100)

experiment = baseline.copy()
experiment.add_subtask(experiment_kernel, 1_000)
```

The mutable containers are copied, so appending to `experiment` does not change
`baseline`. Kernel objects and existing subtask records are shared rather than
deep-copied.

## Finalization belongs to the device

When you pass in the `TaskBuilder` object to the device via `Device.run_async(TaskBuilder, dry_run=...)`, the kernels will be validated, serialized, and a QLAM `TaskDefinition` will be created. If `dry_run=True`, a summary of the programs and subtasks will be printed.

The builder remains editable after either finalization or a dry run.

## Migrating from the deprecated task factories

The deprecated factories all map to repeated `add_subtask()` calls. Replace a
single-kernel `device.task(...)` call with one builder entry:

```python
builder = TaskBuilder()
builder.add_subtask(
    bell,
    num_shots=100,
    metadata={"experiment": "bell"},
)

device.run_async(builder, dry_run=True)
future = device.run_async(builder, dry_run=False)
result = future.result(timeout=300)
```

Existing callers can continue using `device.task()`, `device.batch_task()`, and
`device.parameter_scan()` while migrating, but they should not be used in new
code.

A batch is a builder containing different kernels:

```python
builder = TaskBuilder()
for kernel, shots in zip(kernels, shot_counts):
    builder.add_subtask(kernel, num_shots=shots)

future = device.run_async(builder, dry_run=False)
```

A parameter scan reuses the same kernel with different argument sets:

```python
builder = TaskBuilder()
for parameter_set in parameter_sets:
    builder.add_subtask(
        parameterized_kernel,
        num_shots=100,
        **parameter_set,
    )

future = device.run_async(builder, dry_run=False)
```

Unlike the three deprecated factories, the builder also supports mixing these
patterns in one workload.

See [Running kernels on a device](device_workflow.md) for result filtering,
frame selection, storage, groups, and future reconstruction.
