"""Shared pytest fixtures for the device/ test suite.

Re-exports `storage` from the local fixtures module so tests can use it as a
plain parameter (`def test_x(storage): ...`) without importing it explicitly.
"""

import json

import pytest

from .fixtures.local import storage  # noqa: F401


@pytest.fixture(autouse=True)
def qsh_config_dir(tmp_path, monkeypatch):
    """Point qlam-core at a per-test config dir so tests never read `~/.qsh`.

    Returns the config directory so tests can write a `config.json` into it
    (see `write_qsh_config`).
    """
    config_dir = tmp_path / "qsh"
    monkeypatch.setenv("QSH_CONFIG_DIR", str(config_dir))
    return config_dir


@pytest.fixture
def write_qsh_config(qsh_config_dir):
    """Write a minimal qsh `config.json` with one context named "ctx"."""

    def _write(
        *,
        defaults_group: str | None = None,
        tasks_plugin_group: str | None = None,
    ) -> None:
        context: dict = {"name": "ctx", "qpu": "test-qpu"}
        if defaults_group is not None:
            context["defaults"] = {"group": defaults_group}
        if tasks_plugin_group is not None:
            context["plugins"] = {"tasks": {"group": tasks_plugin_group}}
        qsh_config_dir.mkdir(parents=True, exist_ok=True)
        (qsh_config_dir / "config.json").write_text(
            json.dumps({"current_context": "ctx", "contexts": [context]})
        )

    return _write
