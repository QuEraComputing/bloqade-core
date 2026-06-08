"""Shared pytest fixtures for the device/ test suite.

Re-exports `storage` from the local fixtures module so tests can use it as a
plain parameter (`def test_x(storage): ...`) without importing it explicitly.
"""

from .fixtures.local import storage  # noqa: F401
