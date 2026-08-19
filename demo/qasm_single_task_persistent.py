"""Simple demo to submit a single task (using QASM for testing purposes)

NOTE: requires bloqade-circuit[qasm2] to be installed.
"""

from dataclasses import dataclass, field

from bloqade.qasm2.emit import QASM2
from kirin.ir.method import Method as Method

from bloqade import qasm2
from bloqade.core.device import Device, Future, Result, SQLiteStorage
from bloqade.core.device.task import SingleKernelTask


# NOTE: custom task that overrides serialization with QASM2 string emit
@dataclass
class QASM2Task(SingleKernelTask):
    @property
    def program_language_version(self) -> str:
        return "2.0.0"

    def serialize_kernel(self, kernel: Method) -> str:
        return QASM2().emit_str(kernel)


@dataclass
class QASM2Device(Device):
    single_kernel_task_cls: type[SingleKernelTask[Future[Result]]] = field(
        default=QASM2Task, init=False
    )


# 1. Create a simple kernel (simulator supports up to 10 qubits)
@qasm2.main
def bell():
    q = qasm2.qreg(2)
    qasm2.h(q[0])
    qasm2.cx(q[0], q[1])


# 2. Create the task using the device -- optionally set some metadata
# NOTE: context_name and program_language are set to qasm for testing
device = QASM2Device(context_name="testbed")
task = device.task(
    kernel=bell, num_shots=2, metadata={"tag": "bell"}, program_language="qasm"
)  # metadata is completely customizable

# 3. Submit task -- requires specifying storage
persistent_storage = SQLiteStorage("qasm_single_task.sql")

# 3a. Dry run
task.run_async(dry_run=True, storage=persistent_storage)

# 3b. Actually submit
# NOTE: at this point, a browser window should open to authenticate
future = task.run_async(dry_run=False, storage=persistent_storage)

# NOTE: if you want to resume fetching results, just comment out the line above
# and use the one below
# future = Future.from_storage(storage=persistent_storage, context_name="testbed")

# 4. Wait for completion and get all results
result = future.result(timeout=80.0)

# 5. Print results -- no logical post-processing can be done here
print(result.shot_results())

# 6. Finally, close the SQL connection (not strictly necessary, GC will handle it too)
persistent_storage.close()
