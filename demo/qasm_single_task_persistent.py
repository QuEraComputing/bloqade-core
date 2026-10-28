"""Simple demo to submit a single task (using QASM for testing purposes)

NOTE: requires bloqade-circuit[qasm2] to be installed.
"""

from bloqade.qasm2.emit import QASM2
from kirin.serialization.core.serializationmodule import SerializationModule

from bloqade import qasm2
from bloqade.core.device import Device, SQLiteStorage, TaskBuilder


# Device serializers receive an encoded Kirin module. Decode it before QASM emit.
class QASM2Serializer:
    def encode(self, encoded_module: SerializationModule) -> str:
        kernel = qasm2.main.decode(encoded_module)
        return QASM2().emit_str(kernel)


# 1. Create a simple kernel (simulator supports up to 10 qubits)
@qasm2.main
def bell():
    q = qasm2.qreg(2)
    qasm2.h(q[0])
    qasm2.cx(q[0], q[1])


# 2. Configure the device and build the task -- optionally set some metadata
# NOTE: context_name and program_language are set to qasm for testing
device = Device(
    context_name="testbed",
    program_language="qasm",
    language_version="2.0.0",
    kernel_serializer=QASM2Serializer(),
)
builder = TaskBuilder()
builder.add_subtask(
    bell, num_shots=2, metadata={"tag": "bell"}
)  # metadata is completely customizable

# 3. Submit task -- requires specifying storage
persistent_storage = SQLiteStorage("qasm_single_task.sql")

# 3a. Dry run
device.run_async(builder, dry_run=True)

# 3b. Actually submit
# NOTE: at this point, a browser window should open to authenticate
future = device.run_async(builder, dry_run=False, storage=persistent_storage)

# NOTE: if you want to resume fetching results, just comment out the line above
# and use the one below
# from bloqade.core.device import Future
# future = Future.from_storage(storage=persistent_storage, context_name="testbed")

# 4. Wait for completion and get all results
result = future.result(timeout=80.0)

# 5. Print results -- no logical post-processing can be done here
print(result.shot_results())

# 6. Finally, close the SQL connection (not strictly necessary, GC will handle it too)
persistent_storage.close()
