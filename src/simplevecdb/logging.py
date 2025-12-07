"""
Structured logging for SimpleVecDB.

Provides consistent logging across all modules with contextual information
for debugging and operational monitoring.

Example:
    >>> from simplevecdb.logging import get_logger, log_operation
    >>>
    >>> logger = get_logger(__name__)
    >>> logger.info("Collection created", extra={"collection": "docs"})
    >>>
    >>> with log_operation("add_texts", collection="docs", count=100):
    ...     # perform operation
    ...     pass
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Iterator

# Package-level logger name
LOGGER_NAME = "simplevecdb"

# Default format includes timestamp, level, logger name, and message
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a logger instance for the specified module.

    Creates a child logger under the 'simplevecdb' namespace for consistent
    hierarchical logging control.

    Args:
        name: Module name (typically __name__). If None, returns the root
            simplevecdb logger.

    Returns:
        Logger instance configured under simplevecdb namespace.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Operation completed")
    """
    if name is None:
        return logging.getLogger(LOGGER_NAME)

    # Strip package prefix if present for cleaner names
    if name.startswith("simplevecdb."):
        name = name[len("simplevecdb.") :]

    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def configure_logging(
    level: int | str = logging.INFO,
    format_string: str | None = None,
    handler: logging.Handler | None = None,
) -> None:
    """
    Configure the simplevecdb logging system.

    Sets up logging for the entire simplevecdb package. Call this once at
    application startup to enable logging output.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Can be int or string name.
        format_string: Custom format string. If None, uses DEFAULT_FORMAT.
        handler: Custom handler. If None, creates a StreamHandler.

    Example:
        >>> import logging
        >>> from simplevecdb.logging import configure_logging
        >>>
        >>> # Basic setup
        >>> configure_logging(level=logging.DEBUG)
        >>>
        >>> # Custom format
        >>> configure_logging(
        ...     level="INFO",
        ...     format_string="%(levelname)s: %(message)s"
        ... )
    """
    logger = logging.getLogger(LOGGER_NAME)

    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create handler
    if handler is None:
        handler = logging.StreamHandler()

    # Set formatter
    formatter = logging.Formatter(format_string or DEFAULT_FORMAT)
    handler.setFormatter(formatter)
    handler.setLevel(level)

    logger.addHandler(handler)


@contextmanager
def log_operation(
    operation: str,
    logger: logging.Logger | None = None,
    level: int = logging.DEBUG,
    **context: Any,
) -> Iterator[dict[str, Any]]:
    """
    Context manager for logging operation start, success, and failure.

    Automatically logs operation start and completion with timing information.
    On exception, logs the error with full context before re-raising.

    Args:
        operation: Name of the operation being performed.
        logger: Logger to use. If None, uses the root simplevecdb logger.
        level: Log level for start/success messages (errors always use ERROR).
        **context: Additional context to include in all log messages.

    Yields:
        Dict that can be updated with additional context during operation.

    Example:
        >>> with log_operation("add_texts", collection="docs", count=100) as ctx:
        ...     result = perform_add()
        ...     ctx["inserted_ids"] = len(result)
    """
    if logger is None:
        logger = get_logger()

    ctx: dict[str, Any] = dict(context)
    start_time = time.perf_counter()

    logger.log(level, "%s started", operation, extra={"operation": operation, **ctx})

    try:
        yield ctx
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        ctx["elapsed_ms"] = round(elapsed_ms, 2)
        logger.log(
            level,
            "%s completed in %.2fms",
            operation,
            elapsed_ms,
            extra={"operation": operation, **ctx},
        )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        ctx["elapsed_ms"] = round(elapsed_ms, 2)
        ctx["error"] = str(e)
        ctx["error_type"] = type(e).__name__
        logger.error(
            "%s failed after %.2fms: %s",
            operation,
            elapsed_ms,
            e,
            extra={"operation": operation, **ctx},
            exc_info=True,
        )
        raise


def log_error(
    operation: str,
    error: Exception,
    logger: logging.Logger | None = None,
    **context: Any,
) -> None:
    """
    Log an error with operation context.

    Convenience function for logging errors with consistent formatting
    and context capture.

    Args:
        operation: Name of the operation that failed.
        error: The exception that was raised.
        logger: Logger to use. If None, uses the root simplevecdb logger.
        **context: Additional context to include in the log message.

    Example:
        >>> try:
        ...     risky_operation()
        ... except sqlite3.OperationalError as e:
        ...     log_error("database_write", e, table="vectors", row_count=100)
        ...     raise
    """
    if logger is None:
        logger = get_logger()

    logger.error(
        "%s failed: %s",
        operation,
        error,
        extra={
            "operation": operation,
            "error": str(error),
            "error_type": type(error).__name__,
            **context,
        },
        exc_info=True,
    )
