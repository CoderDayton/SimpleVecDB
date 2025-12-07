from __future__ import annotations

import importlib
import logging
import sqlite3
import sys
import time
from functools import wraps
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# Use standard logging to avoid circular import with logging module
_logger = logging.getLogger("simplevecdb.utils")


def _import_optional(name: str) -> Any:
    """Attempt to import a module while honoring tests that stub sys.modules."""
    sentinel = object()
    existing = sys.modules.get(name, sentinel)
    if existing is None:
        return None
    if existing is not sentinel:
        return existing
    try:
        return importlib.import_module(name)
    except Exception:
        return None


class DatabaseLockedError(Exception):
    """Raised when database remains locked after all retry attempts."""

    def __init__(self, message: str, attempts: int, total_wait: float):
        super().__init__(message)
        self.attempts = attempts
        self.total_wait = total_wait


def retry_on_lock(
    max_retries: int = 5,
    base_delay: float = 0.1,
    max_delay: float = 2.0,
    jitter: bool = True,
) -> Callable[[F], F]:
    """
    Decorator that retries database operations on SQLite lock errors.

    Uses exponential backoff with optional jitter to handle concurrent write
    contention gracefully. Only retries on "database is locked" errors;
    other SQLite errors are raised immediately.

    Args:
        max_retries: Maximum number of retry attempts (default: 5).
        base_delay: Initial delay in seconds before first retry (default: 0.1).
        max_delay: Maximum delay between retries in seconds (default: 2.0).
        jitter: Add randomness to delay to avoid thundering herd (default: True).

    Returns:
        Decorated function with retry behavior.

    Raises:
        DatabaseLockedError: If all retry attempts fail due to lock contention.
        sqlite3.OperationalError: For non-lock SQLite errors.

    Example:
        >>> @retry_on_lock(max_retries=3, base_delay=0.05)
        ... def insert_data(conn, data):
        ...     conn.execute("INSERT INTO items VALUES (?)", (data,))
        ...     conn.commit()
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: sqlite3.OperationalError | None = None
            total_wait = 0.0

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    error_msg = str(e).lower()
                    if "database is locked" not in error_msg:
                        # Not a lock error - raise immediately
                        raise

                    last_exception = e

                    if attempt < max_retries:
                        # Calculate delay with exponential backoff
                        delay = min(base_delay * (2**attempt), max_delay)

                        # Add jitter (±25%) to avoid thundering herd
                        if jitter:
                            import random

                            delay *= 0.75 + random.random() * 0.5

                        total_wait += delay

                        _logger.warning(
                            "Database locked, retrying in %.3fs (attempt %d/%d)",
                            delay,
                            attempt + 1,
                            max_retries,
                            extra={
                                "operation": func.__name__,
                                "attempt": attempt + 1,
                                "max_retries": max_retries,
                                "delay_seconds": round(delay, 3),
                            },
                        )
                        time.sleep(delay)

            # All retries exhausted
            _logger.error(
                "Database locked after %d attempts (%.2fs total wait)",
                max_retries + 1,
                total_wait,
                extra={
                    "operation": func.__name__,
                    "attempts": max_retries + 1,
                    "total_wait_seconds": round(total_wait, 2),
                },
            )
            raise DatabaseLockedError(
                f"Database remained locked after {max_retries + 1} attempts "
                f"(waited {total_wait:.2f}s total)",
                attempts=max_retries + 1,
                total_wait=total_wait,
            ) from last_exception

        return wrapper  # type: ignore[return-value]

    return decorator


def validate_filter(filter_dict: dict[str, Any] | None) -> None:
    """
    Validate metadata filter structure before SQL generation.

    Ensures filter keys are strings and values are supported types.
    Call this before building SQL WHERE clauses to provide clear error
    messages for invalid filters.

    Args:
        filter_dict: Metadata filter dictionary to validate.

    Raises:
        ValueError: If filter keys are not strings or values are unsupported types.

    Example:
        >>> validate_filter({"category": "tech", "score": 0.95})  # OK
        >>> validate_filter({123: "value"})  # Raises ValueError
    """
    if filter_dict is None:
        return

    for key, value in filter_dict.items():
        if not isinstance(key, str):
            raise ValueError(
                f"Filter keys must be strings, got {type(key).__name__}: {key!r}"
            )
        if not isinstance(value, (int, float, str, list)):
            raise ValueError(
                f"Filter value for '{key}' must be int, float, str, or list, "
                f"got {type(value).__name__}: {value!r}"
            )
        if isinstance(value, list):
            for i, item in enumerate(value):
                if not isinstance(item, (int, float, str)):
                    raise ValueError(
                        f"Filter list items for '{key}' must be int, float, or str, "
                        f"got {type(item).__name__} at index {i}: {item!r}"
                    )
