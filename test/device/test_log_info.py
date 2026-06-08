import importlib
import types

import pytest
from loguru import logger

log_info = importlib.import_module("bloqade.core.device.log_info")


@pytest.fixture(autouse=True)
def _reset_logging():
    """Remove any sinks a test installs and restore the opt-in default."""
    before = set(logger._core.handlers)
    yield
    for handler_id in set(logger._core.handlers) - before:
        logger.remove(handler_id)
    logger.disable("bloqade")


def _bloqade_emitter():
    """A logger.info caller whose frame reports a ``bloqade.*`` module name.

    Real library logging happens from ``bloqade.core.device.*`` modules, so it
    is gated by ``logger.enable/disable("bloqade")``. Logging straight from this
    test module would not be, so we emit through a frame that loguru attributes
    to the bloqade namespace.
    """
    fake = types.ModuleType("bloqade.core.device._probe")
    fake.logger = logger
    exec("def emit(msg):\n    logger.info(msg)", fake.__dict__)
    return fake.emit


def test_logging_disabled_by_default_at_import(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BLOQADE_LOGGING", raising=False)

    importlib.reload(log_info)

    assert not (tmp_path / "bloqade.log").exists()


def test_env_var_enables_logging_at_import(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BLOQADE_LOGGING", "1")

    importlib.reload(log_info)

    assert (tmp_path / "bloqade.log").exists()


def test_set_logging_enabled_writes_to_file(tmp_path):
    path = tmp_path / "out.log"

    log_info.set_logging(enabled=True, path=str(path))
    logger.info("hello-from-set-logging")

    assert path.exists()
    assert "hello-from-set-logging" in path.read_text()


def test_set_logging_disabled_suppresses_bloqade_logs(tmp_path):
    path = tmp_path / "out.log"
    emit = _bloqade_emitter()

    log_info.set_logging(enabled=True, path=str(path))
    emit("before-disable")
    log_info.set_logging(enabled=False)
    emit("after-disable")

    contents = path.read_text()
    assert "before-disable" in contents
    assert "after-disable" not in contents


def test_set_logging_respects_level(tmp_path):
    path = tmp_path / "out.log"

    log_info.set_logging(enabled=True, path=str(path), level="ERROR")
    logger.info("info-message")
    logger.error("error-message")

    contents = path.read_text()
    assert "info-message" not in contents
    assert "error-message" in contents


def test_unwritable_path_warns_without_raising(tmp_path):
    # A directory cannot be opened as a log file -> loguru raises OSError, which
    # set_logging turns into a warning rather than letting it propagate.
    with pytest.warns(RuntimeWarning, match="could not create log file"):
        log_info.set_logging(enabled=True, path=str(tmp_path))


def test_set_logging_export_is_public():
    from bloqade.core.device import set_logging

    assert callable(set_logging)
    assert set_logging.__module__ == "bloqade.core.device.log_info"
    assert set_logging.__name__ == "set_logging"
