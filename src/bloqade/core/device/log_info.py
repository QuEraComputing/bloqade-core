import os
import warnings

from loguru import logger

_DEFAULT_PATH = "bloqade.log"
_DEFAULT_LEVEL = "INFO"
_LOG_FORMAT = "{time} | {level} | {message}"


def set_logging(
    enabled: bool = True,
    path: str = _DEFAULT_PATH,
    level: str = _DEFAULT_LEVEL,
) -> None:
    """Enable or disable bloqade file logging.

    Logging is opt-in: nothing is written until this is called, or the
    ``BLOQADE_LOGGING`` environment variable is set to ``"1"`` before
    ``bloqade.core.device`` is imported. Call this once to opt in; it is not
    meant for repeated reconfiguration.

    Args:
        enabled: Turn logging on (``True``) or off (``False``).
        path: File the logs are written to when enabled.
        level: Minimum level captured by the file sink.

    When enabled, task submissions and status fetches are written to ``path``.
    If the log file cannot be created, a :class:`RuntimeWarning` is emitted and
    logging is left disabled instead of raising.
    """
    if not enabled:
        logger.disable("bloqade")
        return

    try:
        logger.add(path, level=level, format=_LOG_FORMAT)
    except OSError as exc:
        warnings.warn(
            f"bloqade: could not create log file {path!r} ({exc}); logging disabled.",
            RuntimeWarning,
            stacklevel=2,
        )
        logger.disable("bloqade")
        return

    logger.enable("bloqade")


# Opt-in: logging stays off unless explicitly enabled.
if os.getenv("BLOQADE_LOGGING", "0") == "1":
    set_logging(enabled=True)
else:
    logger.disable("bloqade")
