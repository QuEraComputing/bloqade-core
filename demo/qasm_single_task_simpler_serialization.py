"""Simple demo to submit a single task (using QASM for testing purposes) using
in-memory storage only (results are not persisted across processes).

For a persistence-enabled version of this demo, see
qasm_single_task_persistent.py.

NOTE: requires bloqade-circuit[qasm2] to be installed.
"""

from dataclasses import dataclass

from bloqade.qasm2.emit import QASM2
from kirin.ir.method import Method

from bloqade import qasm2
from bloqade.core.device import Device, TaskBuilder, set_logging

set_logging()


@dataclass(frozen=True)
class QASM2Serializer:
    kernel: Method

    def encode(self, _encoded_module: object) -> str:
        return QASM2().emit_str(self.kernel)


# 1. Create a simple kernel (simulator supports up to 10 qubits)
@qasm2.main
def bell():
    q = qasm2.qreg(2)
    qasm2.h(q[0])
    qasm2.cx(q[0], q[1])


# 2. Configure the device and build the task -- optionally set some metadata.
# NOTE: context_name and program_language are set to qasm for testing.
device = Device(
    context_name="gemini-qasm",
    qpu_mode="qasm-10q",
    program_language="qasm",
    language_version="2.0.0",
    kernel_serializer=QASM2Serializer(bell),
)
builder = TaskBuilder()
builder.add_subtask(
    bell, num_shots=2, metadata={"tag": "bell"}
)  # metadata is completely customizable

# 3a. Dry run
device.run_async(builder, dry_run=True)

# 3b. Actually submit
# NOTE: at this point, a browser window should open to authenticate
future = device.run_async(builder, dry_run=False)

# NOTE: if you want to resume fetching results, just comment out the line above
# and use the one below
# from bloqade.core.device import Future
# future = Future.from_task_id(task_id="<task-id>", context_name="gemini-qasm")

# 4. Wait for completion and get all results
result = future.result(timeout=80.0)

# 5. Print results -- no logical post-processing can be done here
print(result.shot_results())
