from __future__ import annotations

import importlib
import itertools
import logging
import os
import random
import sqlite3
import sys
import time
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Generator, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# Use standard logging to avoid circular import with logging module
_logger = logging.getLogger("simplevecdb.utils")


def _batched(iterable: Iterable[Any], n: int) -> Iterable[Sequence[Any]]:
    """Batch data into lists of length n. The last batch may be shorter."""
    if isinstance(iterable, Sequence):
        for i in range(0, len(iterable), n):
            yield iterable[i : i + n]
    else:
        it = iter(iterable)
        while True:
            batch = list(itertools.islice(it, n))
            if not batch:
                return
            yield batch


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
    total_timeout: float = 10.0,
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
        total_timeout: Absolute wall-clock budget in seconds (default: 10.0).
            Gives up early if cumulative sleep would exceed this, even if
            max_retries has not been reached.

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
                            delay *= 0.75 + random.random() * 0.5

                        # Enforce total_timeout budget
                        if total_wait + delay > total_timeout:
                            _logger.warning(
                                "Database lock retry would exceed total_timeout "
                                "(%.2fs spent, %.2fs budget) — giving up",
                                total_wait,
                                total_timeout,
                                extra={"operation": func.__name__},
                            )
                            break

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


def async_retry_on_lock(
    max_retries: int = 5,
    base_delay: float = 0.1,
    max_delay: float = 2.0,
    jitter: bool = True,
    total_timeout: float = 10.0,
) -> Callable[[F], F]:
    """
    Async decorator that retries database operations on SQLite lock errors.

    Uses asyncio.sleep instead of time.sleep, avoiding executor thread blocking.
    Same backoff logic as retry_on_lock.

    Args:
        max_retries: Maximum number of retry attempts (default: 5).
        base_delay: Initial delay in seconds before first retry (default: 0.1).
        max_delay: Maximum delay between retries in seconds (default: 2.0).
        jitter: Add randomness to delay to avoid thundering herd (default: True).
        total_timeout: Absolute wall-clock budget in seconds (default: 10.0).

    Returns:
        Decorated async function with retry behavior.

    Raises:
        DatabaseLockedError: If all retry attempts fail due to lock contention.
        sqlite3.OperationalError: For non-lock SQLite errors.
    """
    import asyncio

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: sqlite3.OperationalError | None = None
            total_wait = 0.0

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    error_msg = str(e).lower()
                    if "database is locked" not in error_msg:
                        raise

                    last_exception = e

                    if attempt < max_retries:
                        delay = min(base_delay * (2**attempt), max_delay)

                        if jitter:
                            delay *= 0.75 + random.random() * 0.5

                        if total_wait + delay > total_timeout:
                            _logger.warning(
                                "Database lock retry would exceed total_timeout "
                                "(%.2fs spent, %.2fs budget) — giving up",
                                total_wait,
                                total_timeout,
                                extra={"operation": func.__name__},
                            )
                            break

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
                        await asyncio.sleep(delay)

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


# Range/numeric filter operators (gap 5). Mongo-style operator dicts:
#   {"score": {"$gt": 0.5, "$lte": 0.9}}
#   {"tag":   {"$in": ["a", "b"]}}
#   {"flag":  {"$exists": True}}
# Plus tuple shorthand normalized to the same operator dicts:
#   {"score": (">", 0.5)}                         -> {"$gt": 0.5}
#   {"score": ("range", 0.5, 0.9)}                -> {"$between": [0.5, 0.9]}
_FILTER_OPERATORS: frozenset[str] = frozenset({
    "$eq", "$ne", "$gt", "$gte", "$lt", "$lte",
    "$in", "$nin", "$exists", "$between",
})

_TUPLE_OP_MAP: dict[str, str] = {
    "==": "$eq", "eq": "$eq",
    "!=": "$ne", "ne": "$ne",
    ">":  "$gt", "gt": "$gt",
    ">=": "$gte", "gte": "$gte",
    "<":  "$lt", "lt": "$lt",
    "<=": "$lte", "lte": "$lte",
    "in": "$in", "nin": "$nin",
    "exists": "$exists", "range": "$between", "between": "$between",
}


def _is_finite_number(x: Any) -> bool:
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        return False
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return False
    return True


def _normalize_filter_value(key: str, value: Any) -> Any:
    """Normalize tuple shorthand into operator dicts; pass through others."""
    if isinstance(value, tuple):
        if not value:
            raise ValueError(f"Filter tuple for '{key}' must not be empty")
        op_raw = value[0]
        if not isinstance(op_raw, str):
            raise ValueError(
                f"Filter tuple operator for '{key}' must be a string, "
                f"got {type(op_raw).__name__}: {op_raw!r}"
            )
        op = _TUPLE_OP_MAP.get(op_raw)
        if op is None:
            raise ValueError(
                f"Unknown tuple operator '{op_raw}' for '{key}'. "
                f"Valid: {sorted(set(_TUPLE_OP_MAP))}"
            )
        rest = list(value[1:])
        if op == "$between":
            if len(rest) != 2:
                raise ValueError(
                    f"'{key}' range/between expects exactly 2 args, got {len(rest)}"
                )
            return {op: rest}
        if op in ("$in", "$nin"):
            arg = rest[0] if len(rest) == 1 else rest
            if not isinstance(arg, list):
                arg = list(arg) if isinstance(arg, (tuple, set)) else [arg]
            return {op: arg}
        if len(rest) != 1:
            raise ValueError(
                f"'{key}' operator '{op_raw}' expects exactly 1 arg, got {len(rest)}"
            )
        return {op: rest[0]}
    return value


def normalize_filter(
    filter_dict: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Convert tuple shorthand to operator dicts; otherwise return as-is.

    Pure: callers can rely on the result not aliasing the input for keys
    that needed conversion.
    """
    if not filter_dict:
        return filter_dict
    return {k: _normalize_filter_value(k, v) for k, v in filter_dict.items()}


def validate_filter(filter_dict: dict[str, Any] | None) -> None:
    """
    Validate metadata filter structure before SQL generation.

    Accepts:
      - scalar equality: {"category": "tech"}
      - list IN: {"tag": ["a", "b"]}
      - operator dicts: {"score": {"$gt": 0.5, "$lte": 0.9}}
      - tuple shorthand: {"score": (">", 0.5)} (normalized internally)

    Args:
        filter_dict: Metadata filter dictionary to validate.

    Raises:
        ValueError: If keys are not strings, operators unknown, or values
            are unsupported types/non-finite.

    Example:
        >>> validate_filter({"category": "tech", "score": 0.95})  # OK
        >>> validate_filter({"score": {"$gt": 0.5}})              # OK
        >>> validate_filter({"score": (">", 0.5)})                # OK
        >>> validate_filter({123: "value"})                       # ValueError
    """
    if filter_dict is None:
        return

    for key, raw_value in filter_dict.items():
        if not isinstance(key, str):
            raise ValueError(
                f"Filter keys must be strings, got {type(key).__name__}: {key!r}"
            )
        # Normalize tuple shorthand for validation; the actual SQL builder
        # also normalizes, so this is just for the error path here.
        value = _normalize_filter_value(key, raw_value)

        if isinstance(value, dict):
            _validate_operator_dict(key, value)
            continue

        if isinstance(value, bool):
            # bool is a subclass of int; allow as exact equality.
            continue
        if not isinstance(value, (int, float, str, list)):
            raise ValueError(
                f"Filter value for '{key}' must be int, float, str, list, "
                f"or operator dict, got {type(value).__name__}: {value!r}"
            )
        if isinstance(value, float) and not _is_finite_number(value):
            raise ValueError(
                f"Filter value for '{key}' must be finite, got {value!r}"
            )
        if isinstance(value, list):
            _validate_filter_list(key, value)


def _validate_filter_list(key: str, value: list[Any]) -> None:
    if not value:
        raise ValueError(f"Filter list for '{key}' must not be empty")
    for i, item in enumerate(value):
        if isinstance(item, bool):
            continue
        if not isinstance(item, (int, float, str)):
            raise ValueError(
                f"Filter list items for '{key}' must be int, float, or str, "
                f"got {type(item).__name__} at index {i}: {item!r}"
            )
        if isinstance(item, float) and not _is_finite_number(item):
            raise ValueError(
                f"Filter list item for '{key}' at index {i} must be finite, "
                f"got {item!r}"
            )


def _validate_operator_dict(key: str, op_dict: dict[str, Any]) -> None:
    if not op_dict:
        raise ValueError(f"Operator dict for '{key}' must not be empty")
    for op, arg in op_dict.items():
        if op not in _FILTER_OPERATORS:
            raise ValueError(
                f"Unknown operator '{op}' for '{key}'. "
                f"Valid: {sorted(_FILTER_OPERATORS)}"
            )
        if op in ("$gt", "$gte", "$lt", "$lte"):
            if not _is_finite_number(arg):
                raise ValueError(
                    f"'{key}' {op} expects a finite number, got {arg!r}"
                )
        elif op in ("$eq", "$ne"):
            if isinstance(arg, bool):
                continue
            if arg is None or isinstance(arg, str):
                continue
            if isinstance(arg, (int, float)) and _is_finite_number(arg):
                continue
            raise ValueError(
                f"'{key}' {op} expects scalar (str/number/bool/None), "
                f"got {type(arg).__name__}: {arg!r}"
            )
        elif op in ("$in", "$nin"):
            if not isinstance(arg, list):
                raise ValueError(
                    f"'{key}' {op} expects a list, got {type(arg).__name__}"
                )
            _validate_filter_list(key, arg)
        elif op == "$exists":
            if not isinstance(arg, bool):
                raise ValueError(
                    f"'{key}' $exists expects a bool, got {type(arg).__name__}"
                )
        elif op == "$between":
            if not isinstance(arg, (list, tuple)) or len(arg) != 2:
                raise ValueError(
                    f"'{key}' $between expects [lo, hi], got {arg!r}"
                )
            lo, hi = arg
            if not (_is_finite_number(lo) and _is_finite_number(hi)):
                raise ValueError(
                    f"'{key}' $between bounds must be finite numbers, got {arg!r}"
                )



@contextmanager
def file_lock(path: Path) -> Generator[None, None, None]:
    """Advisory file lock for cross-process safety.

    Uses fcntl.flock on Unix and msvcrt.locking on Windows.
    Creates a .lock file alongside the target path.

    Args:
        path: Path to the file to lock (a .lock sibling is created).

    Yields:
        None — the lock is held for the duration of the context.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    # Open with O_CREAT|O_RDWR — no truncation. A stale lock file from a
    # crashed prior run is reused as-is (its contents are irrelevant; the
    # lock is on the FD via fcntl/msvcrt). Permissions are restricted so
    # other users on the host cannot tamper with the lock target.
    fd_int = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    fd = os.fdopen(fd_int, "r+b")  # noqa: SIM115
    try:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            try:
                if sys.platform == "win32":
                    import msvcrt

                    msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
            except OSError:
                # Even if unlock fails (rare), the close below still runs.
                pass
    finally:
        # Always close the fd so we don't leak file handles when an unlock
        # call raises. The lock file itself is intentionally NOT unlinked:
        # flock/LK_LOCK are inode-bound, so removing the path while
        # another process is still queued on flock(fd) on the old inode
        # would let a third process create a new path → new inode →
        # acquire a different lock concurrently. A surviving zero-byte
        # ``.lock`` sidecar is far cheaper than a torn save.
        try:
            fd.close()
        except OSError:
            pass
