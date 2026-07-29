"""Per-thread SQLite connections for SimpleVecDB.

SQLite gives no isolation between operations on a single connection: a reader
sharing one connection with a writer sees that writer's uncommitted rows.
Separate connections are isolated from each other, so each thread gets its own
connection to the same database. In WAL mode readers also run concurrently
with a writer.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import uuid
from typing import Any, Protocol

from .. import constants

_logger = logging.getLogger("simplevecdb.engine.connection")


class ConnectionSource(Protocol):
    """What the catalog and collection layers need from a connection owner."""

    @property
    def conn(self) -> sqlite3.Connection:
        """The connection this thread should use."""
        ...

    @property
    def shared(self) -> bool:
        """Whether threads share one connection, and so one transaction context."""
        ...

    def close_all(self) -> None:
        """Close every connection handed out."""
        ...


class ConnectionLock:
    """RLock that engages only when threads share one connection.

    A shared connection has one transaction context for every thread, so the
    lock is real for in-memory databases and injected connections. With a
    connection per thread there is no shared context to guard and SQLite
    serializes writers itself, so acquire and release become no-ops.
    """

    __slots__ = ("_lock", "engaged")

    def __init__(self, engaged: bool) -> None:
        self._lock = threading.RLock()
        self.engaged = engaged

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if not self.engaged:
            return True
        return self._lock.acquire(blocking, timeout)

    def release(self) -> None:
        if self.engaged:
            self._lock.release()

    def __enter__(self) -> "ConnectionLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()

    def __repr__(self) -> str:
        return f"ConnectionLock(engaged={self.engaged})"


def is_in_memory(path: str) -> bool:
    """Whether `path` names an in-memory database."""
    return path == ":memory:"


def shared_memory_dsn() -> str:
    """A named shared-cache DSN for an in-memory database.

    Two connections opened on `":memory:"` get two independent databases, so
    pooling one requires this form. Unused by default: shared cache takes
    table-level write locks, and a reader on another connection then fails
    with ``SQLITE_LOCKED``, which `busy_timeout` does not wait out.
    """
    return f"file:simplevecdb_{uuid.uuid4().hex}?mode=memory&cache=shared"


def apply_pragmas(conn: sqlite3.Connection) -> None:
    """Configure a freshly opened connection.

    These are connection-scoped rather than database-scoped, so a pool must
    apply them per connection; `foreign_keys` silently defaults back to off
    otherwise. `journal_mode=WAL` is the exception and persists in the file.
    """
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    # Native lock-wait window so SQLite blocks the caller in C rather than
    # surfacing 'database is locked' immediately under multi-writer load.
    conn.execute(f"PRAGMA busy_timeout={constants.SQLITE_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys=ON")


class SingleConnection:
    """Adapter presenting one already-open connection as a connection source.

    Used when a caller supplies its own `sqlite3.Connection` instead of a
    path. It provides no cross-thread isolation.
    """

    __slots__ = ("_conn",)

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    @property
    def shared(self) -> bool:
        """One connection for every thread, so the lock must engage."""
        return True

    def close_all(self) -> None:
        self._conn.close()


class ConnectionPool:
    """One SQLite connection per thread, all against the same database.

    The constructing thread's connection is opened eagerly, so an unusable
    path or key fails at construction rather than on first use, and is held
    for the pool's lifetime.
    """

    __slots__ = (
        "_dsn",
        "_uri",
        "_encryption_key",
        "_timeout",
        "_local",
        "_all",
        "_all_lock",
        "_closed",
        "_anchor",
    )

    def __init__(
        self,
        path: str,
        *,
        encryption_key: str | bytes | None = None,
        timeout: float = 30.0,
    ) -> None:
        if is_in_memory(path):
            raise ValueError(
                "ConnectionPool cannot pool an in-memory database; "
                "use open_source(), which gives it a shared connection."
            )
        self._dsn, self._uri = path, False
        self._encryption_key = encryption_key
        self._timeout = timeout
        self._local = threading.local()
        self._all: list[sqlite3.Connection] = []
        self._all_lock = threading.Lock()
        self._closed = False
        self._anchor = self._open()
        self._local.conn = self._anchor

    def _open(self) -> sqlite3.Connection:
        """Open and configure one connection."""
        if self._encryption_key is not None:
            from ..encryption import create_encrypted_connection

            conn = create_encrypted_connection(
                self._dsn,
                self._encryption_key,
                check_same_thread=False,
                timeout=self._timeout,
            )
        else:
            conn = sqlite3.connect(
                self._dsn,
                uri=self._uri,
                check_same_thread=False,
                timeout=self._timeout,
            )
        apply_pragmas(conn)
        with self._all_lock:
            self._all.append(conn)
        return conn

    @property
    def shared(self) -> bool:
        """Each thread has its own connection, so the lock can stand down."""
        return False

    @property
    def conn(self) -> sqlite3.Connection:
        """This thread's connection, opened on first use."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            if self._closed:
                raise sqlite3.ProgrammingError("Cannot operate on a closed database.")
            conn = self._open()
            self._local.conn = conn
        return conn

    def close_all(self) -> None:
        """Close every connection this pool opened, from any thread.

        Other threads' connections are closed here too; they were opened with
        `check_same_thread=False`, and a worker thread may never run again to
        close its own.
        """
        self._closed = True
        with self._all_lock:
            connections, self._all = self._all, []
        for conn in connections:
            try:
                conn.close()
            except Exception:  # pragma: no cover - close is best-effort
                _logger.debug("Failed to close a pooled connection", exc_info=True)
        self._local = threading.local()

    def __repr__(self) -> str:
        return f"ConnectionPool(dsn={self._dsn!r}, open={len(self._all)})"


def open_source(
    path: str,
    *,
    encryption_key: str | bytes | None = None,
    timeout: float = 30.0,
) -> ConnectionSource:
    """Open the right kind of connection source for `path`.

    File-backed databases get one connection per thread, so a transaction on
    one thread is invisible to reads on another. In-memory databases get a
    single shared connection and so keep cross-thread dirty reads: pooling one
    requires shared cache, whose table-level locks turn concurrent readers
    into ``SQLITE_LOCKED`` errors (see `shared_memory_dsn`).
    """
    if is_in_memory(path):
        conn = sqlite3.connect(path, check_same_thread=False, timeout=timeout)
        apply_pragmas(conn)
        return SingleConnection(conn)
    return ConnectionPool(path, encryption_key=encryption_key, timeout=timeout)


def as_source(conn_or_source: Any) -> ConnectionSource:
    """Accept either a raw connection or something already pool-shaped.

    Anything that is not already a source is treated as a raw connection.
    Checked this way round because sqlcipher connections and test doubles are
    connection-like without being `sqlite3.Connection` instances.
    """
    if isinstance(conn_or_source, (ConnectionPool, SingleConnection)):
        return conn_or_source
    return SingleConnection(conn_or_source)
