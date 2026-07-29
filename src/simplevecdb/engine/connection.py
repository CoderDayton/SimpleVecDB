"""Per-thread SQLite connections for SimpleVecDB.

SQLite gives no isolation between operations on a *single* connection: a
reader sharing one connection with a writer sees that writer's uncommitted
rows, and can act on data that later rolls back. Isolation is a property of
having separate connections — with them, "the reader is only able to see
complete committed transactions from the writer... regardless of whether the
two database connections are in the same thread, in different threads of the
same process, or in different processes."

So each thread gets its own connection to the same database. In WAL mode that
also buys snapshot isolation and lets readers run while a writer commits.
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

    def close_all(self) -> None:
        """Close every connection handed out."""
        ...


def is_in_memory(path: str) -> bool:
    """Whether `path` names an in-memory database."""
    return path == ":memory:"


def shared_memory_dsn() -> str:
    """A named shared-cache DSN for an in-memory database.

    Not used by default, and the reason is worth recording. Pooling an
    in-memory database requires this form — *"opening two database
    connections each with the filename ':memory:' will create two
    independent in-memory databases"* — but shared cache takes **table-level**
    write locks, and a reader on another connection then fails with
    ``SQLITE_LOCKED`` ("database table is locked"), which `busy_timeout` does
    not wait out. That trades a harmless stale read for a hard error, so
    in-memory databases keep a single shared connection instead.
    """
    return f"file:simplevecdb_{uuid.uuid4().hex}?mode=memory&cache=shared"


def apply_pragmas(conn: sqlite3.Connection) -> None:
    """Configure a freshly opened connection.

    Every one of these is connection-scoped, not database-scoped, so a pool
    must apply them per connection — `foreign_keys` silently defaults back to
    off otherwise. (`journal_mode=WAL` is the exception, persisting in the
    file, but setting it again is harmless and keeps this in one place.)
    """
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    # Native lock-wait window so SQLite blocks the caller in C rather than
    # surfacing 'database is locked' immediately under multi-writer load.
    conn.execute(f"PRAGMA busy_timeout={constants.SQLITE_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys=ON")


class SingleConnection:
    """Adapter presenting one already-open connection as a connection source.

    Used when a caller supplies its own `sqlite3.Connection` directly instead
    of a path, so those call sites keep working unchanged. It cannot provide
    cross-thread isolation — that is the point of `ConnectionPool`.
    """

    __slots__ = ("_conn",)

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    def close_all(self) -> None:
        self._conn.close()


class ConnectionPool:
    """One SQLite connection per thread, all against the same database.

    The first connection is opened eagerly and kept for the pool's lifetime:
    a shared-cache in-memory database is reclaimed when its *last* connection
    closes, so without an anchor the data would vanish whenever a worker
    thread's connection happened to be the last one out.
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

        Connections belonging to other threads are closed here too. That is
        safe because they were opened with `check_same_thread=False`, and it
        is necessary because a worker thread may never run again to close its
        own.
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

    File-backed databases get one connection per thread, which is what makes
    a transaction on one thread invisible to reads on another. In-memory
    databases get a single shared connection: pooling one requires shared
    cache, whose table-level locks turn concurrent readers into
    ``SQLITE_LOCKED`` errors (see `shared_memory_dsn`). They therefore keep
    the old behaviour, cross-thread dirty reads included — acceptable for a
    database that cannot outlive the process.
    """
    if is_in_memory(path):
        conn = sqlite3.connect(path, check_same_thread=False, timeout=timeout)
        apply_pragmas(conn)
        return SingleConnection(conn)
    return ConnectionPool(path, encryption_key=encryption_key, timeout=timeout)


def as_source(conn_or_source: Any) -> ConnectionSource:
    """Accept either a raw connection or something already pool-shaped.

    Tested by identity rather than by `isinstance(..., sqlite3.Connection)`:
    sqlcipher connections and test doubles are connection-like without being
    instances of it, and wrapping those is exactly the intent.
    """
    if isinstance(conn_or_source, (ConnectionPool, SingleConnection)):
        return conn_or_source
    return SingleConnection(conn_or_source)
