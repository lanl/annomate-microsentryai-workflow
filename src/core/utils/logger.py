import sys
import logging


def setup_logging() -> None:
    """Configure the root logger for the application.

    Selects log level based on execution context: ``CRITICAL`` for compiled
    (frozen) executables and ``DEBUG`` for development runs. All messages
    are emitted to ``stdout`` with a timestamp, level, module, and
    line-number prefix.
    """
    is_compiled = getattr(sys, "frozen", False)

    log_level = logging.CRITICAL if is_compiled else logging.DEBUG
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    logger = logging.getLogger("Annomate")
    if is_compiled:
        logger.debug("Application started in PRODUCTION mode.")
    else:
        logger.debug("Application started in DEVELOPER mode.")
