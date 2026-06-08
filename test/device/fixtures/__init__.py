"""Shared test fixtures for the device/ test suite.

Two strictly separated namespaces:

* `remote` — builders and fake clients that mirror the qlam-core HTTP surface
  (typed pydantic models for Task/TaskDefinition/Compilation; raw dicts for the
  ResultsClient envelope, which qlam-core does not model).
* `local`  — helpers for the bloqade-side StorageBackend dict schema and the
  `ShotResult` dataclass. These shapes are invented by bloqade and unrelated to
  the qlam wire format.

A new test should pick one namespace per call site and stick with it; mixing
remote pydantic types and local dicts in the same fixture is the drift this
module exists to prevent. Sample JSON dumps captured from the live API live
under `examples/` for reference.

The remote builders and the `examples/` dumps were verified against
qlam-core v0.2.0. If you bump that pin, re-capture the examples and re-check
the builders against the new wire shapes.
"""

from . import local, remote

__all__ = ["remote", "local"]
