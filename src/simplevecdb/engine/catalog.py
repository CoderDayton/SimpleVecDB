"""
CatalogManager: SQLite metadata and FTS operations for SimpleVecDB.

This module handles all SQLite operations for document metadata, text content,
and full-text search (FTS5). Vector operations are handled by UsearchIndex.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, TYPE_CHECKING, Callable
from collections.abc import Iterable, Sequence

from ..utils import _batched

from ..utils import validate_filter, retry_on_lock, normalize_filter

if TYPE_CHECKING:
    import sqlite3

_logger = logging.getLogger("simplevecdb.engine.catalog")

# Regex for safe table names (defense-in-depth)
_SAFE_TABLE_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_table_name(name: str) -> None:
    """Validate table name to prevent SQL injection (defense-in-depth)."""
    if not _SAFE_TABLE_NAME_RE.match(name):
        raise ValueError(
            f"Invalid table name '{name}'. Must be alphanumeric + underscores, "
            "starting with a letter or underscore."
        )


def _coerce_scalar(arg: Any) -> Any:
    """Coerce a Python scalar to the value SQLite stores via json_extract.

    json_extract returns 0/1 for JSON booleans, so an `$eq True` filter
    must compare against 1 (not Python True, which sqlite3 would bind as
    integer 1 anyway, but we make it explicit).
    """
    if isinstance(arg, bool):
        return 1 if arg else 0
    return arg


# Identifier safety for keys interpolated into SQL (atomic counters, etc.).
_SAFE_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(name: str, what: str = "identifier") -> None:
    """Reject names that would be unsafe to inline into SQL string literals."""
    if not isinstance(name, str) or not _SAFE_IDENT_RE.match(name):
        raise ValueError(
            f"Invalid {what} '{name}'. Must match {_SAFE_IDENT_RE.pattern}."
        )


class _TxState:
    """Shared per-VectorDB transaction depth counter (gap 2).

    Used by VectorDB.transaction() to mark all collections/catalogs as
    operating inside an outer SAVEPOINT. Catalog write helpers consult
    this to decide whether to commit on exit.
    """

    __slots__ = ("depth",)

    def __init__(self) -> None:
        self.depth: int = 0


class _CatalogWritable:
    """Context manager replacing `with self._lock, self.conn:`.

    Behaviour:
      * Always takes the catalog write lock.
      * Outside a transaction, also enters the connection's implicit-tx
        context; on exit it commits (or rolls back on error).
      * Inside a transaction (`tx_state.depth > 0`) it skips the conn
        context — the outer SAVEPOINT handles atomicity, so we must NOT
        commit at every catalog call.
    """

    __slots__ = ("_lock", "_conn", "_tx", "_owns_conn")

    def __init__(
        self,
        lock: threading.RLock,
        conn: "sqlite3.Connection",
        tx_state: _TxState,
    ) -> None:
        self._lock = lock
        self._conn = conn
        self._tx = tx_state
        self._owns_conn = False

    def __enter__(self):
        self._lock.acquire()
        if self._tx.depth == 0:
            self._conn.__enter__()
            self._owns_conn = True
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._owns_conn:
                # Defer to the connection's own commit/rollback semantics.
                self._conn.__exit__(exc_type, exc, tb)
        finally:
            self._lock.release()
        return False


# Edge column names that participate in column-direct filtering. Anything
# else falls through to JSON metadata filtering.
_EDGE_NUMERIC_COLUMNS: frozenset[str] = frozenset({
    "weight", "bonus", "hits", "last_touch"
})


def _split_edge_filter(
    filter_dict: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Partition a filter into (edge-column filter, metadata filter).

    Filter keys that name a numeric edge column compile to direct
    column comparisons; everything else routes through json_extract on
    the edge's metadata column.
    """
    if not filter_dict:
        return {}, {}
    edge_part: dict[str, Any] = {}
    meta_part: dict[str, Any] = {}
    for k, v in filter_dict.items():
        if k in _EDGE_NUMERIC_COLUMNS:
            edge_part[k] = v
        else:
            meta_part[k] = v
    return edge_part, meta_part


def _compile_edge_column_filter(
    edge_filter: dict[str, Any], params: list[Any]
) -> str:
    """Compile filters against literal edge columns (no json_extract).

    Mirrors the operator grammar from utils.normalize_filter / validate_filter
    but emits SQL referencing the column directly.
    """
    from ..utils import normalize_filter, validate_filter

    validate_filter(edge_filter)
    normalized = normalize_filter(edge_filter) or {}
    pieces: list[str] = []
    for col, value in normalized.items():
        # col is one of _EDGE_NUMERIC_COLUMNS — safe to inline.
        if isinstance(value, dict):
            for op, arg in value.items():
                if op == "$eq":
                    pieces.append(f"{col} = ?")
                    params.append(arg)
                elif op == "$ne":
                    pieces.append(f"{col} != ?")
                    params.append(arg)
                elif op == "$gt":
                    pieces.append(f"{col} > ?")
                    params.append(arg)
                elif op == "$gte":
                    pieces.append(f"{col} >= ?")
                    params.append(arg)
                elif op == "$lt":
                    pieces.append(f"{col} < ?")
                    params.append(arg)
                elif op == "$lte":
                    pieces.append(f"{col} <= ?")
                    params.append(arg)
                elif op == "$in":
                    placeholders = ",".join("?" for _ in arg)
                    pieces.append(f"{col} IN ({placeholders})")
                    params.extend(arg)
                elif op == "$nin":
                    placeholders = ",".join("?" for _ in arg)
                    pieces.append(f"{col} NOT IN ({placeholders})")
                    params.extend(arg)
                elif op == "$between":
                    lo, hi = arg
                    pieces.append(f"{col} BETWEEN ? AND ?")
                    params.extend([lo, hi])
                elif op == "$exists":
                    pieces.append(
                        f"{col} IS NOT NULL" if arg else f"{col} IS NULL"
                    )
                else:
                    raise ValueError(f"Unsupported edge operator '{op}'")
        elif isinstance(value, list):
            placeholders = ",".join("?" for _ in value)
            pieces.append(f"{col} IN ({placeholders})")
            params.extend(value)
        else:
            pieces.append(f"{col} = ?")
            params.append(value)
    return " AND ".join(pieces)


class CatalogManager:
    """
    Handles SQLite metadata and FTS operations.

    This manager is responsible for:
    - Creating and managing SQLite tables (metadata and FTS)
    - Adding, deleting, and removing document metadata
    - Building filter clauses for metadata queries
    - FTS5 full-text search indexing

    Note: Vector operations are handled by UsearchIndex, not CatalogManager.

    Args:
        conn: SQLite database connection
        table_name: Name of the metadata table
        fts_table_name: Name of the full-text search table
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        fts_table_name: str,
        lock: threading.RLock | None = None,
        tx_state: "_TxState | None" = None,
    ):
        # Defense-in-depth: validate table names. After this point every
        # f-string SQL site that interpolates self._table_name (or its
        # _pending_vectors / _edges / _events / _ttl / _clusters
        # suffixes) is safe by construction; do not reintroduce arbitrary
        # name interpolation downstream.
        _validate_table_name(table_name)
        _validate_table_name(fts_table_name)

        self.conn = conn
        self._table_name = table_name
        self._fts_table_name = fts_table_name
        self._fts_enabled = False
        self._cluster_table_name = f"{table_name}_clusters"
        self._cluster_table_ready = False
        # Serializes Python-level access to the shared sqlite3.Connection. The
        # connection is opened with check_same_thread=False; SQLite itself is
        # safe under WAL, but Python's `with conn:` transaction context is not
        # — two threads entering it simultaneously interleave their writes
        # under one implicit transaction. The lock prevents that.
        self._lock: threading.RLock = lock if lock is not None else threading.RLock()
        # Optional shared cross-collection transaction state. When the
        # state's depth > 0, _writable() suppresses inner conn commits so
        # the outer SAVEPOINT controls atomicity.
        self._tx_state: _TxState = tx_state if tx_state is not None else _TxState()

    def _writable(self):
        """Acquire the write lock and (when no outer txn) the conn context.

        Outside a transaction, this is equivalent to `with self._lock,
        self.conn:` — every catalog write commits when the block exits.

        Inside a transaction (caller has called VectorDB.transaction()),
        the connection's commit-on-exit is suppressed so the SAVEPOINT
        opened by the transaction owns atomicity.
        """
        return _CatalogWritable(self._lock, self.conn, self._tx_state)

    def create_tables(self) -> None:
        """Create metadata and FTS tables if they don't exist."""
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                metadata TEXT,
                embedding BLOB,
                parent_id INTEGER REFERENCES {self._table_name}(id) ON DELETE SET NULL
            )
            """
        )
        # Create index for parent_id lookups
        self.conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table_name}_parent
            ON {self._table_name}(parent_id)
            WHERE parent_id IS NOT NULL
            """
        )
        # Index on text for find_ids_by_texts / remove_texts which previously
        # full-scanned. Costs disk proportional to total text size; payback
        # is large on collections that frequently look up by text content.
        self.conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table_name}_text
            ON {self._table_name}(text)
            """
        )
        # Migrate existing tables that lack columns
        self._ensure_embedding_column()
        self._ensure_parent_id_column()
        self._ensure_fts_table()
        # 2.6.1 auxiliary tables (pending vectors, edges, events, TTL).
        # Each is idempotent (CREATE TABLE IF NOT EXISTS), so existing 2.6.0
        # databases gain them transparently on first open.
        self._ensure_pending_vectors_table()
        self._ensure_edges_table()
        self._ensure_events_table()
        self._ensure_ttl_table()

    def _ensure_pending_vectors_table(self) -> None:
        """Buffer of vector updates flushed to usearch in batches (gap 1)."""
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table_name}_pending_vectors (
                doc_id      INTEGER PRIMARY KEY
                            REFERENCES {self._table_name}(id) ON DELETE CASCADE,
                embedding   BLOB NOT NULL,
                source      TEXT,
                enqueued_at REAL NOT NULL
            )
            """
        )

    def _ensure_edges_table(self) -> None:
        """Weighted directed edges between documents (gap 3)."""
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table_name}_edges (
                src_id     INTEGER NOT NULL
                           REFERENCES {self._table_name}(id) ON DELETE CASCADE,
                dst_id     INTEGER NOT NULL
                           REFERENCES {self._table_name}(id) ON DELETE CASCADE,
                kind       TEXT NOT NULL DEFAULT '',
                weight     REAL NOT NULL DEFAULT 0.0,
                hits       INTEGER NOT NULL DEFAULT 0,
                bonus      REAL NOT NULL DEFAULT 0.0,
                last_touch REAL NOT NULL,
                metadata   TEXT,
                PRIMARY KEY (src_id, dst_id, kind)
            )
            """
        )
        self.conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table_name}_edges_dst
            ON {self._table_name}_edges(dst_id, kind)
            """
        )
        self.conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table_name}_edges_weight
            ON {self._table_name}_edges(kind, weight)
            """
        )
        self.conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table_name}_edges_last_touch
            ON {self._table_name}_edges(kind, last_touch)
            """
        )

    def _ensure_events_table(self) -> None:
        """Append-only change feed (gap 7). Subscribers poll WHERE seq > ?."""
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table_name}_events (
                seq     INTEGER PRIMARY KEY AUTOINCREMENT,
                ts      REAL NOT NULL,
                kind    TEXT NOT NULL,
                doc_id  INTEGER,
                payload TEXT
            )
            """
        )
        self.conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table_name}_events_kind_seq
            ON {self._table_name}_events(kind, seq)
            """
        )

    def _ensure_ttl_table(self) -> None:
        """TTL/expiry entries (gap 8). Sweep deletes rows whose expires_at <= now()."""
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table_name}_ttl (
                doc_id     INTEGER PRIMARY KEY
                           REFERENCES {self._table_name}(id) ON DELETE CASCADE,
                expires_at REAL NOT NULL,
                on_expire  TEXT NOT NULL DEFAULT 'delete'
            )
            """
        )
        self.conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table_name}_ttl_expires
            ON {self._table_name}_ttl(expires_at)
            """
        )

    def _ensure_embedding_column(self) -> None:
        """Add embedding column if missing (migration for v2.0.0)."""
        try:
            cursor = self.conn.execute(f"PRAGMA table_info({self._table_name})")
            columns = {row[1] for row in cursor.fetchall()}
            if "embedding" not in columns:
                self.conn.execute(
                    f"ALTER TABLE {self._table_name} ADD COLUMN embedding BLOB"
                )
                _logger.info(
                    "Migrated table %s: added embedding column", self._table_name
                )
        except Exception as e:
            _logger.warning("Could not check/add embedding column: %s", e)

    def _ensure_parent_id_column(self) -> None:
        """Add parent_id column if missing (migration for v2.1.0)."""
        try:
            cursor = self.conn.execute(f"PRAGMA table_info({self._table_name})")
            columns = {row[1] for row in cursor.fetchall()}
            if "parent_id" not in columns:
                self.conn.execute(
                    f"ALTER TABLE {self._table_name} ADD COLUMN parent_id INTEGER "
                    f"REFERENCES {self._table_name}(id) ON DELETE SET NULL"
                )
                # Create index for efficient parent lookups
                self.conn.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._table_name}_parent
                    ON {self._table_name}(parent_id)
                    WHERE parent_id IS NOT NULL
                    """
                )
                _logger.info(
                    "Migrated table %s: added parent_id column", self._table_name
                )
        except Exception as e:
            _logger.warning("Could not check/add parent_id column: %s", e)

    def _ensure_fts_table(self) -> None:
        """Create FTS5 virtual table for full-text search.

        Retries on transient lock errors but permanently disables FTS
        if the module is unavailable.
        """
        import sqlite3

        for attempt in range(3):
            try:
                self.conn.execute(
                    f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS {self._fts_table_name}
                    USING fts5(text)
                    """
                )
                self._fts_enabled = True
                return
            except sqlite3.OperationalError as e:
                msg = str(e).lower()
                if "database is locked" in msg and attempt < 2:
                    import time
                    time.sleep(0.1 * (attempt + 1))
                    continue
                _logger.warning("FTS5 not available - keyword search disabled: %s", e)
                self._fts_enabled = False
                return

    @property
    def fts_enabled(self) -> bool:
        """Whether FTS5 is available for keyword search."""
        return self._fts_enabled

    def _upsert_fts_rows(self, ids: Sequence[int], texts: Sequence[str]) -> None:
        """Update FTS index for given document IDs.

        Internal helper. Must be called inside an active transaction
        (``with self._writable()``) so the FTS shadow table stays in
        sync with the main table on crash.

        Args:
            ids: Document IDs to update
            texts: Corresponding text content
        """
        if not self._fts_enabled or not ids:
            return
        placeholders = ",".join(["?"] * len(ids))
        self.conn.execute(
            f"DELETE FROM {self._fts_table_name} WHERE rowid IN ({placeholders})",
            tuple(ids),
        )
        rows = list(zip(ids, texts))
        self.conn.executemany(
            f"INSERT INTO {self._fts_table_name}(rowid, text) VALUES (?, ?)", rows
        )

    def _delete_fts_rows(self, ids: Sequence[int]) -> None:
        """Remove documents from FTS index.

        Internal helper. Must be called inside an active transaction so
        the FTS shadow table stays in sync with the main table on crash.

        Args:
            ids: Document IDs to remove
        """
        if not self._fts_enabled or not ids:
            return
        placeholders = ",".join(["?"] * len(ids))
        self.conn.execute(
            f"DELETE FROM {self._fts_table_name} WHERE rowid IN ({placeholders})",
            tuple(ids),
        )

    @retry_on_lock(max_retries=5, base_delay=0.1)
    def add_documents(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict],
        ids: Sequence[int | None] | None = None,
        embeddings: Sequence[Sequence[float]] | None = None,
        parent_ids: Sequence[int | None] | None = None,
    ) -> list[int]:
        """
        Insert or update document metadata.

        Args:
            texts: Document text content
            metadatas: Metadata dicts for each document
            ids: Optional document IDs for upsert behavior
            embeddings: Optional embedding vectors to store
            parent_ids: Optional parent document IDs for hierarchical relationships

        Returns:
            List of document IDs (rowids)
        """
        if not texts:
            return []

        _logger.debug(
            "Adding %d documents to metadata table",
            len(texts),
            extra={"table": self._table_name},
        )

        import numpy as np

        ids_list = list(ids) if ids else [None] * len(texts)
        parent_ids_list = list(parent_ids) if parent_ids else [None] * len(texts)

        # Convert embeddings to bytes if provided
        embedding_blobs: list[bytes | None] = []
        if embeddings is not None:
            # Batch conversion: single np.array call instead of per-item np.asarray
            emb_matrix = np.asarray(embeddings, dtype=np.float32)
            row_bytes = emb_matrix.tobytes()
            stride = emb_matrix.shape[1] * 4  # float32 = 4 bytes
            embedding_blobs = [
                row_bytes[i * stride : (i + 1) * stride]
                for i in range(emb_matrix.shape[0])
            ]
        else:
            embedding_blobs = [None] * len(texts)

        # Pre-serialize metadata (compact separators saves allocation overhead)
        _dumps = json.dumps
        meta_strs = [_dumps(m, separators=(",", ":")) for m in metadatas]

        # Split into auto-ID and explicit-ID groups so each can use the
        # correct INSERT path:
        #   - Explicit IDs: upsert (ON CONFLICT DO UPDATE) so existing rows
        #     are updated in place. last_insert_rowid is unsafe here because
        #     UPSERTs that hit the UPDATE branch do not advance it, breaking
        #     the prior arithmetic.
        #   - Auto IDs (None): plain INSERT, then RETURNING id to recover the
        #     auto-assigned values exactly. Held under self._lock so the
        #     RETURNING result is uncorrupted by concurrent writers.
        explicit_rows = []
        auto_rows = []
        auto_positions = []
        for idx, (uid, txt, meta_str, emb_blob, pid) in enumerate(
            zip(ids_list, texts, meta_strs, embedding_blobs, parent_ids_list)
        ):
            if uid is None:
                auto_rows.append((txt, meta_str, emb_blob, pid))
                auto_positions.append(idx)
            else:
                explicit_rows.append((uid, txt, meta_str, emb_blob, pid))

        real_ids: list[int] = [-1] * len(ids_list)

        with self._writable():
            if explicit_rows:
                self.conn.executemany(
                    f"""
                    INSERT INTO {self._table_name}(id, text, metadata, embedding, parent_id)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        text=excluded.text,
                        metadata=excluded.metadata,
                        embedding=excluded.embedding,
                        parent_id=excluded.parent_id
                    """,
                    explicit_rows,
                )

            if auto_rows:
                # Use a single multi-VALUES INSERT ... RETURNING id so we
                # recover the auto-assigned IDs in the exact insertion order.
                placeholders = ",".join(["(?, ?, ?, ?)"] * len(auto_rows))
                flat_params = [v for r in auto_rows for v in r]
                cursor = self.conn.execute(
                    f"INSERT INTO {self._table_name}"
                    f"(text, metadata, embedding, parent_id) "
                    f"VALUES {placeholders} RETURNING id",
                    flat_params,
                )
                returned = cursor.fetchall()
                if len(returned) != len(auto_rows):
                    raise RuntimeError(
                        f"INSERT RETURNING id returned {len(returned)} rows, "
                        f"expected {len(auto_rows)}"
                    )
                for pos, row in zip(auto_positions, returned):
                    real_ids[pos] = int(row[0])

            # Fill in explicit IDs by their original position
            explicit_iter = iter(explicit_rows)
            for idx, uid in enumerate(ids_list):
                if uid is not None:
                    real_ids[idx] = int(next(explicit_iter)[0])

            # Defense-in-depth: any leftover -1 sentinel here means an
            # INSERT path partially succeeded — never feed that to FTS as
            # a rowid. This catches both retry-loop interaction with the
            # @retry_on_lock decorator and any future code path that
            # forgets to populate real_ids before the FTS upsert.
            if any(rid < 0 for rid in real_ids):
                raise RuntimeError(
                    "Internal error: add_documents produced an unfilled "
                    "rowid sentinel; refusing to update FTS with -1."
                )

            # Update FTS index
            self._upsert_fts_rows(real_ids, texts)

        _logger.debug("Added %d documents, ids=%s", len(real_ids), real_ids[:5])
        return real_ids

    @retry_on_lock(max_retries=5, base_delay=0.1)
    def delete_by_ids(self, ids: Iterable[int]) -> list[int]:
        """
        Delete documents by their IDs.

        Args:
            ids: Document IDs to delete

        Returns:
            List of IDs that were actually deleted
        """
        ids = list(ids)
        if not ids:
            return []

        _logger.debug("Deleting %d documents", len(ids))

        placeholders = ",".join("?" for _ in ids)
        params = tuple(ids)

        with self._writable():
            # Check which IDs actually exist
            existing = self.conn.execute(
                f"SELECT id FROM {self._table_name} WHERE id IN ({placeholders})",
                params,
            ).fetchall()
            existing_ids = [r[0] for r in existing]

            if existing_ids:
                placeholders = ",".join("?" for _ in existing_ids)
                # 2.6.1 aux tables. FK enforcement may be off, so cascade
                # explicitly to avoid orphan rows.
                self.conn.execute(
                    f"DELETE FROM {self._table_name}_pending_vectors "
                    f"WHERE doc_id IN ({placeholders})",
                    tuple(existing_ids),
                )
                self.conn.execute(
                    f"DELETE FROM {self._table_name}_edges "
                    f"WHERE src_id IN ({placeholders}) "
                    f"OR dst_id IN ({placeholders})",
                    tuple(existing_ids) * 2,
                )
                self.conn.execute(
                    f"DELETE FROM {self._table_name}_ttl "
                    f"WHERE doc_id IN ({placeholders})",
                    tuple(existing_ids),
                )
                self.conn.execute(
                    f"DELETE FROM {self._table_name} WHERE id IN ({placeholders})",
                    tuple(existing_ids),
                )
                self._delete_fts_rows(existing_ids)
                for doc_id in existing_ids:
                    self.append_event_in_tx("delete", doc_id=int(doc_id))

        _logger.debug("Deleted %d documents", len(existing_ids))
        return existing_ids

    def get_documents_by_ids(
        self, ids: Sequence[int]
    ) -> dict[int, tuple[str, dict[str, Any]]]:
        """
        Fetch document text and metadata by IDs.

        Args:
            ids: Document IDs to fetch

        Returns:
            Dict mapping id -> (text, metadata)
        """
        if not ids:
            return {}

        placeholders = ",".join(["?"] * len(ids))
        with self._lock:
            rows = self.conn.execute(
                f"SELECT id, text, metadata FROM {self._table_name} WHERE id IN ({placeholders})",
                tuple(ids),
            ).fetchall()

        result = {}
        for row_id, text, meta_json in rows:
            meta = json.loads(meta_json) if meta_json else {}
            result[row_id] = (text, meta)
        return result

    def list_all_ids(self) -> list[int]:
        """Return every doc id in the table, serialized through ``self._lock``.

        Used by the rebuild-index path so the SELECT runs under the same
        re-entrant lock as concurrent writers, eliminating the bare
        ``self.conn.execute(...)`` that previously relied on caller
        discipline alone.
        """
        with self._lock:
            rows = self.conn.execute(
                f"SELECT id FROM {self._table_name}"
            ).fetchall()
        return [row[0] for row in rows]

    def get_embeddings_by_ids(self, ids: Sequence[int]) -> dict[int, Any]:
        """
        Fetch embeddings by document IDs.

        Args:
            ids: Document IDs to fetch

        Returns:
            Dict mapping id -> numpy array (or None if no embedding stored)
        """
        import numpy as np

        if not ids:
            return {}

        placeholders = ",".join(["?"] * len(ids))
        with self._lock:
            rows = self.conn.execute(
                f"SELECT id, embedding FROM {self._table_name} WHERE id IN ({placeholders})",
                tuple(ids),
            ).fetchall()

        result: dict[int, np.ndarray | None] = {}
        for row_id, emb_blob in rows:
            if emb_blob is not None:
                result[row_id] = np.frombuffer(emb_blob, dtype=np.float32)
            else:
                result[row_id] = None
        return result

    def get_documents_and_embeddings_by_ids(
        self, ids: Sequence[int]
    ) -> dict[int, tuple[str, dict[str, Any], Any]]:
        """Fetch documents with their embeddings in a single query.

        Args:
            ids: Document IDs to fetch

        Returns:
            Dict mapping id -> (text, metadata, embedding_array_or_None)
        """
        import numpy as np

        if not ids:
            return {}

        placeholders = ",".join(["?"] * len(ids))
        with self._lock:
            rows = self.conn.execute(
                f"SELECT id, text, metadata, embedding FROM {self._table_name} WHERE id IN ({placeholders})",
                tuple(ids),
            ).fetchall()

        result: dict[int, tuple[str, dict[str, Any], np.ndarray | None]] = {}
        for row_id, text, meta_json, emb_blob in rows:
            meta = json.loads(meta_json) if meta_json else {}
            emb = np.frombuffer(emb_blob, dtype=np.float32) if emb_blob is not None else None
            result[row_id] = (text, meta, emb)
        return result

    def find_ids_by_texts(
        self,
        texts: Sequence[str],
        *,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[int]:
        """Find document IDs matching exact text content.

        Args:
            texts: Text strings to search for
            limit: Maximum number of IDs to return (None = all)
            offset: Number of IDs to skip (None = 0)
        """
        if not texts:
            return []
        placeholders = ",".join(["?"] * len(texts))
        sql = f"SELECT id FROM {self._table_name} WHERE text IN ({placeholders})"
        params: list[Any] = list(texts)

        if offset is not None and limit is None:
            raise ValueError("offset requires limit")
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
            if offset is not None:
                sql += " OFFSET ?"
                params.append(offset)

        with self._lock:
            rows = self.conn.execute(sql, tuple(params)).fetchall()
        return [r[0] for r in rows]

    def find_ids_by_filter(
        self,
        filter_dict: dict[str, Any],
        filter_builder: Callable[[dict[str, Any], str], tuple[str, list[Any]]],
        *,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[int]:
        """Find document IDs matching metadata filter.

        Args:
            filter_dict: Metadata key-value pairs to filter by
            filter_builder: Function to build filter clause
            limit: Maximum number of IDs to return (None = all)
            offset: Number of IDs to skip (None = 0)
        """
        if not filter_dict:
            return []

        filter_clause, filter_params = filter_builder(filter_dict, "metadata")
        # Remove leading "AND " from clause
        filter_clause = filter_clause.replace("AND ", "", 1)
        where_clause = f"WHERE {filter_clause}" if filter_clause else ""

        sql = f"SELECT id FROM {self._table_name} {where_clause}"
        params: list[Any] = list(filter_params)

        if offset is not None and limit is None:
            raise ValueError("offset requires limit")
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
            if offset is not None:
                sql += " OFFSET ?"
                params.append(offset)

        with self._lock:
            rows = self.conn.execute(sql, tuple(params)).fetchall()
        return [r[0] for r in rows]

    def keyword_search(
        self,
        query: str,
        k: int,
        filter_dict: dict[str, Any] | None = None,
        filter_builder: Callable | None = None,
    ) -> list[tuple[int, float]]:
        """
        Perform BM25 keyword search using FTS5.

        Args:
            query: Search query (FTS5 syntax supported)
            k: Maximum results
            filter_dict: Optional metadata filter
            filter_builder: Function to build filter clause

        Returns:
            List of (id, bm25_score) tuples, sorted by relevance
        """
        if not self._fts_enabled:
            raise RuntimeError("FTS5 not available - cannot perform keyword search")
        if not query.strip():
            return []

        filter_clause = ""
        filter_params: list[Any] = []
        if filter_dict and filter_builder:
            filter_clause, filter_params = filter_builder(filter_dict, "ti.metadata")

        sql = f"""
            SELECT ti.id, bm25({self._fts_table_name}) as score
            FROM {self._fts_table_name} f
            JOIN {self._table_name} ti ON ti.id = f.rowid
            WHERE {self._fts_table_name} MATCH ?
            {filter_clause}
            ORDER BY score ASC
            LIMIT ?
        """
        params = (query,) + tuple(filter_params) + (k,)
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        return [(int(row[0]), float(row[1])) for row in rows]

    def build_filter_clause(
        self,
        filter_dict: dict[str, Any] | None,
        metadata_column: str = "metadata",
    ) -> tuple[str, list[Any]]:
        """
        Build SQL WHERE clause from metadata filter dictionary.

        Accepts the full grammar from utils.validate_filter:
          - scalar equality: {"k": v}
          - list IN: {"k": [v1, v2]}
          - Mongo-style operator dicts:
              {"k": {"$gt": x, "$lte": y}}, {"k": {"$between": [lo, hi]}},
              {"k": {"$in": [...]}}, {"k": {"$exists": True}}
          - tuple shorthand: {"k": (">", x)} (normalized internally)

        Numeric operators wrap the JSON value in CAST(... AS REAL) so
        SQLite uses index-friendly numeric comparisons even though the
        column itself is JSON text.

        Args:
            filter_dict: Filter dictionary in the grammar above.
            metadata_column: Column expression holding the JSON document.

        Returns:
            Tuple of (where_clause, parameters). where_clause is empty
            string or starts with "AND (" for direct interpolation.
        """
        if not filter_dict:
            return "", []

        validate_filter(filter_dict)
        normalized = normalize_filter(filter_dict) or {}

        clauses: list[str] = []
        params: list[Any] = []
        for key, value in normalized.items():
            json_path = f"$.{key}"
            text_extract = f"json_extract({metadata_column}, ?)"
            num_extract = f"CAST({text_extract} AS REAL)"

            if isinstance(value, dict):
                self._build_operator_clauses(
                    json_path, text_extract, num_extract, value, clauses, params
                )
                continue

            if isinstance(value, bool):
                # JSON encodes bool as 0/1 via json_extract; normalize.
                clauses.append(f"{text_extract} = ?")
                params.extend([json_path, 1 if value else 0])
            elif isinstance(value, (int, float, str)):
                clauses.append(f"{text_extract} = ?")
                params.extend([json_path, value])
            elif isinstance(value, list):
                placeholders = ",".join("?" for _ in value)
                clauses.append(f"{text_extract} IN ({placeholders})")
                params.append(json_path)
                params.extend(value)
            else:
                raise ValueError(f"Unsupported filter value type for {key}")

        where = " AND ".join(clauses)
        return f"AND ({where})" if where else "", params

    def _build_operator_clauses(
        self,
        json_path: str,
        text_extract: str,
        num_extract: str,
        op_dict: dict[str, Any],
        clauses: list[str],
        params: list[Any],
    ) -> None:
        """Compile a single key's operator dict into WHERE clause fragments."""
        for op, arg in op_dict.items():
            if op == "$eq":
                clauses.append(f"{text_extract} = ?")
                params.extend([json_path, _coerce_scalar(arg)])
            elif op == "$ne":
                # IS NOT for null-safety + difference for present values.
                clauses.append(
                    f"({text_extract} IS NULL OR {text_extract} != ?)"
                )
                params.extend([json_path, json_path, _coerce_scalar(arg)])
            elif op == "$gt":
                clauses.append(f"{num_extract} > ?")
                params.extend([json_path, arg])
            elif op == "$gte":
                clauses.append(f"{num_extract} >= ?")
                params.extend([json_path, arg])
            elif op == "$lt":
                clauses.append(f"{num_extract} < ?")
                params.extend([json_path, arg])
            elif op == "$lte":
                clauses.append(f"{num_extract} <= ?")
                params.extend([json_path, arg])
            elif op == "$in":
                placeholders = ",".join("?" for _ in arg)
                clauses.append(f"{text_extract} IN ({placeholders})")
                params.append(json_path)
                params.extend(arg)
            elif op == "$nin":
                placeholders = ",".join("?" for _ in arg)
                clauses.append(
                    f"({text_extract} IS NULL OR "
                    f"{text_extract} NOT IN ({placeholders}))"
                )
                params.append(json_path)
                params.append(json_path)
                params.extend(arg)
            elif op == "$exists":
                if arg:
                    clauses.append(f"{text_extract} IS NOT NULL")
                else:
                    clauses.append(f"{text_extract} IS NULL")
                params.append(json_path)
            elif op == "$between":
                lo, hi = arg
                clauses.append(f"{num_extract} BETWEEN ? AND ?")
                params.extend([json_path, lo, hi])
            else:
                raise ValueError(f"Unsupported operator '{op}'")

    def count(self) -> int:
        """Return total number of documents."""
        with self._lock:
            row = self.conn.execute(
                f"SELECT COUNT(*) FROM {self._table_name}"
            ).fetchone()
        return row[0] if row else 0

    def get_all_docs_with_text(
        self,
        filter_dict: dict[str, Any] | None = None,
        filter_builder: Callable[[dict[str, Any], str], tuple[str, list[Any]]]
        | None = None,
        *,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[tuple[int, str, dict[str, Any]]]:
        """
        Get documents with their text content, with optional pagination.

        Args:
            filter_dict: Optional metadata filter
            filter_builder: Function to build filter clause
            limit: Maximum number of documents to return (None = all)
            offset: Number of documents to skip (None = 0)

        Returns:
            List of (doc_id, text, metadata) tuples
        """
        filter_clause = ""
        filter_params: list[Any] = []
        if filter_dict and filter_builder:
            filter_clause, filter_params = filter_builder(filter_dict, "metadata")

        sql = f"""
            SELECT id, text, metadata FROM {self._table_name}
            WHERE 1=1 {filter_clause}
            ORDER BY id
        """
        params: list[Any] = list(filter_params)

        if offset is not None and limit is None:
            raise ValueError("offset requires limit")
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
            if offset is not None:
                sql += " OFFSET ?"
                params.append(offset)

        with self._lock:
            rows = self.conn.execute(sql, tuple(params)).fetchall()
        result = []
        for row_id, text, meta_json in rows:
            meta = json.loads(meta_json) if meta_json else {}
            result.append((int(row_id), text, meta))
        return result

    def update_metadata_batch(self, updates: list[tuple[int, dict[str, Any]]]) -> int:
        """
        Update metadata for multiple documents in a single transaction.

        Merges new metadata with existing metadata (shallow merge).

        Args:
            updates: List of (doc_id, metadata_updates) tuples

        Returns:
            Number of documents updated
        """
        if not updates:
            return 0

        with self._writable():
            updated = 0
            # Batch into chunks of 500 for performance
            for batch in _batched(updates, 500):
                ids = [u[0] for u in batch]

                # Fetch all existing metadata in one query
                placeholders = ",".join(["?"] * len(ids))
                rows = self.conn.execute(
                    f"SELECT id, metadata FROM {self._table_name} WHERE id IN ({placeholders})",
                    ids,
                ).fetchall()

                current_meta_map = {r[0]: (json.loads(r[1]) if r[1] else {}) for r in rows}

                # Prepare updates
                update_data = []
                for doc_id, meta_updates in batch:
                    if doc_id in current_meta_map:
                        meta = current_meta_map[doc_id]
                        meta.update(meta_updates)
                        update_data.append((json.dumps(meta), doc_id))
                        updated += 1

                if update_data:
                    self.conn.executemany(
                        f"UPDATE {self._table_name} SET metadata = ? WHERE id = ?",
                        update_data,
                    )

            return updated

    def increment_metadata(
        self, doc_id: int, deltas: dict[str, float | int]
    ) -> int:
        """Atomically increment numeric metadata counters (gap 4).

        Single UPDATE statement applies every delta via chained json_set,
        so multi-writer races are resolved by SQLite under WAL — no
        read-modify-write window in Python. Numeric values only; missing
        keys treat the prior value as 0.

        Args:
            doc_id: Target document id.
            deltas: Mapping of counter name -> numeric delta. Keys must
                match `^[A-Za-z_][A-Za-z0-9_]*$` (validated). Values must
                be int or float (bool/strings/None rejected).

        Returns:
            1 if the row exists and was updated, 0 otherwise.
        """
        return self.increment_metadata_many([(doc_id, deltas)])

    def increment_metadata_many(
        self, updates: list[tuple[int, dict[str, float | int]]]
    ) -> int:
        """Batch variant of increment_metadata.

        Each (doc_id, deltas) tuple becomes one UPDATE; all run inside
        the same transaction so partial application on crash is
        impossible.

        Args:
            updates: List of (doc_id, deltas) pairs.

        Returns:
            Total number of rows updated (sum of UPDATE rowcounts).
        """
        if not updates:
            return 0

        # Pre-validate all keys/values up front so we can fail before
        # opening a transaction. Keeps the SQL build loop branch-free.
        for doc_id, deltas in updates:
            if not isinstance(doc_id, int) or isinstance(doc_id, bool):
                raise TypeError(
                    f"increment_metadata: doc_id must be int, got "
                    f"{type(doc_id).__name__}"
                )
            if not deltas:
                raise ValueError(
                    "increment_metadata: deltas must be a non-empty dict"
                )
            for key, val in deltas.items():
                _validate_identifier(key, "metadata counter key")
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    raise TypeError(
                        f"increment_metadata: delta for '{key}' must be "
                        f"int or float, got {type(val).__name__}"
                    )
                if isinstance(val, float) and (val != val or val in (
                    float("inf"), float("-inf")
                )):
                    raise ValueError(
                        f"increment_metadata: delta for '{key}' must be finite"
                    )

        total = 0
        with self._writable():
            for doc_id, deltas in updates:
                sql, params = self._build_increment_sql(doc_id, deltas)
                cursor = self.conn.execute(sql, params)
                if cursor.rowcount:
                    total += cursor.rowcount
                    self.append_event_in_tx(
                        "counter",
                        doc_id=doc_id,
                        payload={"deltas": dict(deltas)},
                    )
        return total

    def _build_increment_sql(
        self, doc_id: int, deltas: dict[str, float | int]
    ) -> tuple[str, tuple[Any, ...]]:
        """Compile a chained json_set UPDATE that adds all deltas atomically.

        Builds `json_set(json_set(... base ..., '$.k1', cur1+?), '$.k2', cur2+?)`
        from the inside out. Key names are inlined after _validate_identifier
        rejects anything outside the safe alphabet; deltas are bound params.
        """
        expr = f"COALESCE({self._table_name}.metadata, '{{}}')"
        params: list[Any] = []
        for key, val in deltas.items():
            cur = (
                f"COALESCE(CAST(json_extract({self._table_name}.metadata, "
                f"'$.{key}') AS REAL), 0)"
            )
            expr = f"json_set({expr}, '$.{key}', {cur} + ?)"
            params.append(val)
        params.append(doc_id)
        sql = f"UPDATE {self._table_name} SET metadata = {expr} WHERE id = ?"
        return sql, tuple(params)

    def get_metadata_counter(
        self, doc_id: int, key: str, default: float | int = 0
    ) -> float | int | None:
        """Read a single numeric counter value from metadata.

        Returns the stored number, or `default` if the key is missing
        or non-numeric. Returns None if the row itself doesn't exist
        (so callers can distinguish missing-row from missing-key).
        """
        _validate_identifier(key, "metadata counter key")
        with self._lock:
            row = self.conn.execute(
                f"SELECT json_extract(metadata, '$.{key}') "
                f"FROM {self._table_name} WHERE id = ?",
                (doc_id,),
            ).fetchone()
        if row is None:
            return None
        value = row[0]
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return value
        return default

    # --- Pending vector buffer (gap 1 / gap 6) -----------------------------

    def upsert_pending_vector(
        self, doc_id: int, embedding_bytes: bytes, source: str | None = None
    ) -> int:
        """Buffer a vector update; flush() promotes it to the HNSW index.

        Replaces any existing pending entry for the same doc_id (last-write-
        wins per id) so duplicate calls before flush coalesce. Returns the
        rowcount (1 if a row was written, 0 if the doc_id row is missing
        and FK enforcement is on).
        """
        return self.upsert_pending_vectors_many([(doc_id, embedding_bytes, source)])

    def upsert_pending_vectors_many(
        self,
        rows: Sequence[tuple[int, bytes, str | None]],
    ) -> int:
        """Bulk variant of upsert_pending_vector; one transaction."""
        if not rows:
            return 0
        with self._writable():
            cur = self.conn.executemany(
                f"""
                INSERT INTO {self._table_name}_pending_vectors
                    (doc_id, embedding, source, enqueued_at)
                VALUES (?, ?, ?, unixepoch('subsec'))
                ON CONFLICT(doc_id) DO UPDATE SET
                    embedding   = excluded.embedding,
                    source      = excluded.source,
                    enqueued_at = excluded.enqueued_at
                """,
                rows,
            )
            self.append_event_in_tx(
                "pending_enqueue",
                payload={"count": len(rows),
                         "ids": [int(r[0]) for r in rows[:50]]},
            )
        return cur.rowcount or 0

    def list_pending_vectors(
        self, *, limit: int | None = None
    ) -> list[tuple[int, bytes, str | None]]:
        """Read pending vector rows in enqueue order.

        Returns list of (doc_id, embedding_bytes, source). Rows whose
        doc_id no longer exists in the main table are skipped; this is
        defense-in-depth in case FK enforcement is off.
        """
        sql = (
            f"SELECT p.doc_id, p.embedding, p.source FROM "
            f"{self._table_name}_pending_vectors p "
            f"JOIN {self._table_name} t ON t.id = p.doc_id "
            f"ORDER BY p.enqueued_at ASC, p.doc_id ASC"
        )
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        return [(int(r[0]), bytes(r[1]), r[2]) for r in rows]

    def delete_pending_vectors(self, doc_ids: Sequence[int]) -> int:
        """Drop pending rows for the given doc_ids (post-flush cleanup)."""
        if not doc_ids:
            return 0
        placeholders = ",".join("?" for _ in doc_ids)
        sql = (
            f"DELETE FROM {self._table_name}_pending_vectors "
            f"WHERE doc_id IN ({placeholders})"
        )
        with self._writable():
            cur = self.conn.execute(sql, tuple(doc_ids))
            if cur.rowcount:
                self.append_event_in_tx(
                    "pending_flush",
                    payload={"count": int(cur.rowcount),
                             "ids": [int(i) for i in list(doc_ids)[:50]]},
                )
        return cur.rowcount or 0

    def count_pending_vectors(self) -> int:
        """Number of buffered vector updates."""
        with self._lock:
            row = self.conn.execute(
                f"SELECT COUNT(*) FROM {self._table_name}_pending_vectors"
            ).fetchone()
        return int(row[0]) if row else 0

    def get_pending_vector(self, doc_id: int) -> bytes | None:
        """Return buffered vector bytes for a doc, or None if not pending."""
        with self._lock:
            row = self.conn.execute(
                f"SELECT embedding FROM {self._table_name}_pending_vectors "
                f"WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
        return bytes(row[0]) if row is not None else None

    # --- Edges (gap 3) -----------------------------------------------------

    def add_edge(
        self,
        src_id: int,
        dst_id: int,
        *,
        kind: str = "",
        weight: float = 0.0,
        bonus: float = 0.0,
        hits: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Insert an edge; raises if (src, dst, kind) already exists."""
        if not isinstance(kind, str):
            raise TypeError(f"edge kind must be str, got {type(kind).__name__}")
        with self._writable():
            cur = self.conn.execute(
                f"""
                INSERT INTO {self._table_name}_edges
                    (src_id, dst_id, kind, weight, hits, bonus,
                     last_touch, metadata)
                VALUES (?, ?, ?, ?, ?, ?, unixepoch('subsec'), ?)
                """,
                (
                    int(src_id), int(dst_id), kind, float(weight), int(hits),
                    float(bonus),
                    json.dumps(metadata) if metadata is not None else None,
                ),
            )
            self.append_event_in_tx(
                "edge_add",
                payload={"src": int(src_id), "dst": int(dst_id), "kind": kind,
                         "weight": float(weight)},
            )
        return cur.rowcount or 0

    def upsert_edge(
        self,
        src_id: int,
        dst_id: int,
        *,
        kind: str = "",
        weight: float | None = None,
        bonus: float | None = None,
        hits: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Insert or replace fields on an existing edge.

        Fields left as None preserve their existing value (or use the
        column default on insert).
        """
        if not isinstance(kind, str):
            raise TypeError(f"edge kind must be str, got {type(kind).__name__}")
        meta_json = json.dumps(metadata) if metadata is not None else None
        with self._writable():
            cur = self.conn.execute(
                f"""
                INSERT INTO {self._table_name}_edges
                    (src_id, dst_id, kind, weight, hits, bonus,
                     last_touch, metadata)
                VALUES (?, ?, ?,
                        COALESCE(?, 0.0), COALESCE(?, 0), COALESCE(?, 0.0),
                        unixepoch('subsec'), ?)
                ON CONFLICT(src_id, dst_id, kind) DO UPDATE SET
                    weight     = COALESCE(?, weight),
                    hits       = COALESCE(?, hits),
                    bonus      = COALESCE(?, bonus),
                    metadata   = COALESCE(?, metadata),
                    last_touch = unixepoch('subsec')
                """,
                (
                    int(src_id), int(dst_id), kind,
                    weight, hits, bonus, meta_json,
                    weight, hits, bonus, meta_json,
                ),
            )
            self.append_event_in_tx(
                "edge_upsert",
                payload={"src": int(src_id), "dst": int(dst_id), "kind": kind},
            )
        return cur.rowcount or 0

    def update_edge(
        self,
        src_id: int,
        dst_id: int,
        *,
        kind: str = "",
        weight: float | None = None,
        bonus: float | None = None,
        hits: int | None = None,
        metadata: dict[str, Any] | None = None,
        dweight: float = 0.0,
        dbonus: float = 0.0,
        dhits: int = 0,
    ) -> int:
        """Modify an edge.

        Absolute values (weight/bonus/hits/metadata) replace the column
        when not None. Deltas (dweight/dbonus/dhits) are applied
        atomically via SQL `col = col + ?`. Both can be combined: an
        absolute set runs *before* the delta on the same statement.
        Non-existent edges are not auto-created (use upsert_edge for that).
        """
        if not isinstance(kind, str):
            raise TypeError(f"edge kind must be str, got {type(kind).__name__}")
        meta_json = json.dumps(metadata) if metadata is not None else None
        with self._writable():
            cur = self.conn.execute(
                f"""
                UPDATE {self._table_name}_edges SET
                    weight     = COALESCE(?, weight) + ?,
                    bonus      = COALESCE(?, bonus)  + ?,
                    hits       = COALESCE(?, hits)   + ?,
                    metadata   = COALESCE(?, metadata),
                    last_touch = unixepoch('subsec')
                WHERE src_id = ? AND dst_id = ? AND kind = ?
                """,
                (
                    weight, float(dweight),
                    bonus, float(dbonus),
                    hits, int(dhits),
                    meta_json,
                    int(src_id), int(dst_id), kind,
                ),
            )
            if cur.rowcount:
                self.append_event_in_tx(
                    "edge_update",
                    payload={"src": int(src_id), "dst": int(dst_id),
                             "kind": kind, "dweight": float(dweight),
                             "dhits": int(dhits)},
                )
        return cur.rowcount or 0

    def delete_edge(
        self, src_id: int, dst_id: int, *, kind: str = ""
    ) -> int:
        """Delete a single edge. Returns 1 if removed, else 0."""
        with self._writable():
            cur = self.conn.execute(
                f"""
                DELETE FROM {self._table_name}_edges
                WHERE src_id = ? AND dst_id = ? AND kind = ?
                """,
                (int(src_id), int(dst_id), kind),
            )
            if cur.rowcount:
                self.append_event_in_tx(
                    "edge_delete",
                    payload={"src": int(src_id), "dst": int(dst_id),
                             "kind": kind},
                )
        return cur.rowcount or 0

    def get_edges(
        self,
        *,
        src_id: int | None = None,
        dst_id: int | None = None,
        kind: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[tuple[int, int, str, float, int, float, float, dict | None]]:
        """Read edges. Filter applies via build_filter_clause to numeric
        columns (weight/bonus/hits/last_touch) or to the JSON metadata."""
        clauses: list[str] = []
        params: list[Any] = []
        if src_id is not None:
            clauses.append("src_id = ?")
            params.append(int(src_id))
        if dst_id is not None:
            clauses.append("dst_id = ?")
            params.append(int(dst_id))
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)

        edge_filter, meta_filter = _split_edge_filter(filter)
        if edge_filter:
            clauses.append(_compile_edge_column_filter(edge_filter, params))
        if meta_filter:
            extra_clause, extra_params = self.build_filter_clause(
                meta_filter, metadata_column="metadata"
            )
            if extra_clause:
                # build_filter_clause returns "AND (...)"; strip the "AND".
                clauses.append(extra_clause[4:].strip())
                params.extend(extra_params)

        where = " AND ".join(clauses) if clauses else "1=1"
        sql = (
            f"SELECT src_id, dst_id, kind, weight, hits, bonus, "
            f"last_touch, metadata "
            f"FROM {self._table_name}_edges WHERE {where} "
            f"ORDER BY last_touch DESC, src_id, dst_id"
        )
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._lock:
            rows = self.conn.execute(sql, tuple(params)).fetchall()
        result = []
        for r in rows:
            meta = json.loads(r[7]) if r[7] else None
            result.append((
                int(r[0]), int(r[1]), str(r[2]), float(r[3]), int(r[4]),
                float(r[5]), float(r[6]), meta,
            ))
        return result

    # --- Change feed (gap 7) -----------------------------------------------

    def append_event(
        self,
        kind: str,
        *,
        doc_id: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> int:
        """Append a single event to the change feed.

        Returns the assigned `seq`. Caller is responsible for placing
        this inside an outer transaction if multiple writes must atomic.
        """
        if not isinstance(kind, str) or not kind:
            raise ValueError("event kind must be a non-empty string")
        with self._writable():
            cur = self.conn.execute(
                f"""
                INSERT INTO {self._table_name}_events
                    (ts, kind, doc_id, payload)
                VALUES (unixepoch('subsec'), ?, ?, ?)
                """,
                (
                    kind,
                    int(doc_id) if doc_id is not None else None,
                    json.dumps(payload) if payload is not None else None,
                ),
            )
        return int(cur.lastrowid or 0)

    def append_event_in_tx(
        self,
        kind: str,
        *,
        doc_id: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Append without opening a transaction (caller already has one)."""
        if not isinstance(kind, str) or not kind:
            raise ValueError("event kind must be a non-empty string")
        self.conn.execute(
            f"""
            INSERT INTO {self._table_name}_events
                (ts, kind, doc_id, payload)
            VALUES (unixepoch('subsec'), ?, ?, ?)
            """,
            (
                kind,
                int(doc_id) if doc_id is not None else None,
                json.dumps(payload) if payload is not None else None,
            ),
        )

    def last_event_seq(self) -> int:
        """Return the highest assigned sequence number, or 0 if none."""
        with self._lock:
            row = self.conn.execute(
                f"SELECT MAX(seq) FROM {self._table_name}_events"
            ).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def read_events(
        self,
        *,
        since: int = 0,
        kind: str | None = None,
        limit: int | None = None,
    ) -> list[tuple[int, float, str, int | None, dict | None]]:
        """Return events with seq > `since`, optionally filtered by kind."""
        clauses = ["seq > ?"]
        params: list[Any] = [int(since)]
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        sql = (
            f"SELECT seq, ts, kind, doc_id, payload "
            f"FROM {self._table_name}_events "
            f"WHERE {' AND '.join(clauses)} ORDER BY seq ASC"
        )
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._lock:
            rows = self.conn.execute(sql, tuple(params)).fetchall()
        result = []
        for r in rows:
            payload = json.loads(r[4]) if r[4] else None
            doc_id = int(r[3]) if r[3] is not None else None
            result.append((int(r[0]), float(r[1]), str(r[2]), doc_id, payload))
        return result

    def prune_events(self, *, before_seq: int) -> int:
        """Delete events with seq < `before_seq`. Returns count deleted."""
        with self._writable():
            cur = self.conn.execute(
                f"DELETE FROM {self._table_name}_events WHERE seq < ?",
                (int(before_seq),),
            )
        return cur.rowcount or 0

    # --- TTL / expiry (gap 8) ---------------------------------------------

    def set_ttl(
        self,
        doc_id: int,
        expires_at: float,
        *,
        on_expire: str = "delete",
    ) -> int:
        """Set or replace the TTL entry for a document.

        `expires_at` is a unix timestamp (seconds). `on_expire` is either
        "delete" (sweep removes the row) or "callback" (sweep returns the
        id but does not delete; caller acts on it).
        """
        if on_expire not in ("delete", "callback"):
            raise ValueError(
                f"on_expire must be 'delete' or 'callback', got {on_expire!r}"
            )
        with self._writable():
            self.conn.execute(
                f"""
                INSERT INTO {self._table_name}_ttl
                    (doc_id, expires_at, on_expire)
                VALUES (?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    expires_at = excluded.expires_at,
                    on_expire  = excluded.on_expire
                """,
                (int(doc_id), float(expires_at), on_expire),
            )
            self.append_event_in_tx(
                "ttl_set",
                doc_id=doc_id,
                payload={"expires_at": float(expires_at),
                         "on_expire": on_expire},
            )
        return 1

    def clear_ttl(self, doc_id: int) -> int:
        """Remove a TTL entry. Returns 1 if removed, 0 if missing."""
        with self._writable():
            cur = self.conn.execute(
                f"DELETE FROM {self._table_name}_ttl WHERE doc_id = ?",
                (int(doc_id),),
            )
        return cur.rowcount or 0

    def list_expired_ttl(
        self, *, now: float | None = None, limit: int | None = None
    ) -> list[tuple[int, float, str]]:
        """Return TTL entries whose expires_at <= now."""
        cutoff = float(now) if now is not None else None
        if cutoff is None:
            sql = (
                f"SELECT doc_id, expires_at, on_expire "
                f"FROM {self._table_name}_ttl "
                f"WHERE expires_at <= unixepoch('subsec') "
                f"ORDER BY expires_at ASC"
            )
            params: tuple[Any, ...] = ()
        else:
            sql = (
                f"SELECT doc_id, expires_at, on_expire "
                f"FROM {self._table_name}_ttl "
                f"WHERE expires_at <= ? "
                f"ORDER BY expires_at ASC"
            )
            params = (cutoff,)
        if limit is not None:
            sql += " LIMIT ?"
            params = params + (int(limit),)
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        return [(int(r[0]), float(r[1]), str(r[2])) for r in rows]

    def sweep_ttl(
        self,
        *,
        now: float | None = None,
        limit: int = 1000,
    ) -> tuple[list[int], list[int]]:
        """Apply due TTL entries.

        For each expired entry:
          - on_expire == "delete": deletes from the main table (and the
            new 2.6.1 aux tables explicitly, since FK enforcement may be
            off), then drops the TTL row.
          - on_expire == "callback": leaves the doc in place and just
            drops the TTL row.

        Returns (deleted_ids, callback_ids). Both lists are empty when
        there's nothing to do.
        """
        rows = self.list_expired_ttl(now=now, limit=limit)
        if not rows:
            return [], []
        delete_ids = [r[0] for r in rows if r[2] == "delete"]
        callback_ids = [r[0] for r in rows if r[2] == "callback"]
        all_ids = [r[0] for r in rows]
        with self._writable():
            if delete_ids:
                placeholders = ",".join("?" for _ in delete_ids)
                # Children first (FK pragma may be off).
                for child in (
                    f"{self._table_name}_pending_vectors",
                    f"{self._table_name}_edges",
                    f"{self._table_name}_ttl",
                ):
                    if child.endswith("_edges"):
                        self.conn.execute(
                            f"DELETE FROM {child} WHERE src_id IN "
                            f"({placeholders}) OR dst_id IN ({placeholders})",
                            tuple(delete_ids) * 2,
                        )
                    else:
                        self.conn.execute(
                            f"DELETE FROM {child} WHERE doc_id IN "
                            f"({placeholders})",
                            tuple(delete_ids),
                        )
                # Main row.
                self.conn.execute(
                    f"DELETE FROM {self._table_name} WHERE id IN "
                    f"({placeholders})",
                    tuple(delete_ids),
                )
                if self._fts_enabled:
                    self.conn.execute(
                        f"DELETE FROM {self._fts_table_name} "
                        f"WHERE rowid IN ({placeholders})",
                        tuple(delete_ids),
                    )
            if callback_ids:
                placeholders = ",".join("?" for _ in callback_ids)
                self.conn.execute(
                    f"DELETE FROM {self._table_name}_ttl "
                    f"WHERE doc_id IN ({placeholders})",
                    tuple(callback_ids),
                )
            for doc_id in all_ids:
                self.append_event_in_tx("ttl_expire", doc_id=doc_id)
        return delete_ids, callback_ids

    def prune_edges(
        self,
        *,
        kind: str | None = None,
        max_weight: float | None = None,
        idle_before: float | None = None,
    ) -> int:
        """Bulk-delete edges by threshold. Returns number deleted."""
        clauses: list[str] = []
        params: list[Any] = []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if max_weight is not None:
            clauses.append("weight <= ?")
            params.append(float(max_weight))
        if idle_before is not None:
            clauses.append("last_touch <= ?")
            params.append(float(idle_before))
        if not clauses:
            raise ValueError(
                "prune_edges: at least one of max_weight/idle_before/kind required"
            )
        sql = (
            f"DELETE FROM {self._table_name}_edges WHERE "
            + " AND ".join(clauses)
        )
        with self._writable():
            cur = self.conn.execute(sql, tuple(params))
        return cur.rowcount or 0


    # ------------------------------------------------------------------ #
    # Hierarchical Relationships
    # ------------------------------------------------------------------ #

    def get_children(self, parent_id: int) -> list[tuple[int, str, dict[str, Any]]]:
        """
        Get all direct children of a document.

        Args:
            parent_id: ID of the parent document

        Returns:
            List of (id, text, metadata) tuples for child documents
        """
        with self._lock:
            rows = self.conn.execute(
                f"SELECT id, text, metadata FROM {self._table_name} WHERE parent_id = ?",
                (parent_id,),
            ).fetchall()

        return [(int(r[0]), r[1], json.loads(r[2]) if r[2] else {}) for r in rows]

    def get_parent(self, doc_id: int) -> tuple[int, str, dict[str, Any]] | None:
        """
        Get the parent document of a given document.

        Args:
            doc_id: ID of the child document

        Returns:
            Tuple of (id, text, metadata) for parent, or None if no parent
        """
        # Single self-join instead of two sequential queries
        with self._lock:
            row = self.conn.execute(
                f"""SELECT p.id, p.text, p.metadata
                FROM {self._table_name} c
                JOIN {self._table_name} p ON p.id = c.parent_id
                WHERE c.id = ?""",
                (doc_id,),
            ).fetchone()

        if not row:
            return None

        return (
            int(row[0]),
            row[1],
            json.loads(row[2]) if row[2] else {},
        )

    def get_descendants(
        self, root_id: int, max_depth: int | None = None
    ) -> list[tuple[int, str, dict[str, Any], int]]:
        """
        Get all descendants of a document (recursive).

        Uses a recursive CTE for efficient traversal.

        Args:
            root_id: ID of the root document
            max_depth: Maximum depth to traverse (None uses safety cap
                from constants.MAX_HIERARCHY_DEPTH to prevent infinite recursion)

        Returns:
            List of (id, text, metadata, depth) tuples
        """
        from .. import constants

        # Apply safety cap to prevent infinite recursion from cycles. The
        # depth is bound as a parameter rather than f-string interpolated;
        # int() coercion makes the previous f-string safe today, but the
        # parameter form is one less line away from injection on a future
        # refactor.
        effective_depth = int(max_depth) if max_depth is not None else constants.MAX_HIERARCHY_DEPTH

        sql = f"""
            WITH RECURSIVE descendants(id, text, metadata, depth) AS (
                SELECT id, text, metadata, 1 as depth
                FROM {self._table_name}
                WHERE parent_id = ?

                UNION ALL

                SELECT t.id, t.text, t.metadata, d.depth + 1
                FROM {self._table_name} t
                JOIN descendants d ON t.parent_id = d.id
                WHERE depth < ?
            )
            SELECT id, text, metadata, depth FROM descendants
            ORDER BY depth, id
        """

        with self._lock:
            rows = self.conn.execute(sql, (root_id, effective_depth)).fetchall()

        return [
            (int(r[0]), r[1], json.loads(r[2]) if r[2] else {}, int(r[3])) for r in rows
        ]

    def get_ancestors(
        self, doc_id: int, max_depth: int | None = None
    ) -> list[tuple[int, str, dict[str, Any], int]]:
        """
        Get all ancestors of a document (path to root).

        Args:
            doc_id: ID of the document
            max_depth: Maximum depth to traverse (None uses safety cap
                from constants.MAX_HIERARCHY_DEPTH to prevent infinite recursion)

        Returns:
            List of (id, text, metadata, depth) tuples, from immediate parent to root
        """
        from .. import constants

        # Apply safety cap to prevent infinite recursion from cycles. Bind
        # the depth as a parameter (see get_descendants for rationale).
        effective_depth = int(max_depth) if max_depth is not None else constants.MAX_HIERARCHY_DEPTH

        sql = f"""
            WITH RECURSIVE ancestors(id, text, metadata, parent_id, depth) AS (
                SELECT id, text, metadata, parent_id, 1 as depth
                FROM {self._table_name}
                WHERE id = (SELECT parent_id FROM {self._table_name} WHERE id = ?)

                UNION ALL

                SELECT t.id, t.text, t.metadata, t.parent_id, a.depth + 1
                FROM {self._table_name} t
                JOIN ancestors a ON t.id = a.parent_id
                WHERE a.parent_id IS NOT NULL AND a.depth < ?
            )
            SELECT id, text, metadata, depth FROM ancestors
            ORDER BY depth
        """

        with self._lock:
            rows = self.conn.execute(sql, (doc_id, effective_depth)).fetchall()

        return [
            (int(r[0]), r[1], json.loads(r[2]) if r[2] else {}, int(r[3])) for r in rows
        ]

    def set_parent(self, doc_id: int, parent_id: int | None) -> bool:
        """
        Set or update the parent of a document.

        Args:
            doc_id: ID of the document to update
            parent_id: New parent ID (None to remove parent)

        Returns:
            True if document was updated, False if not found

        Raises:
            ValueError: If setting parent would create a cycle
        """
        # Cycle check + UPDATE inside one critical section so a concurrent
        # writer cannot create a cycle-forming edge between the check and the
        # UPDATE. The lock serializes; `with self.conn:` wraps the UPDATE in
        # an implicit transaction that commits on success.
        with self._writable():
            if parent_id is not None:
                if parent_id == doc_id:
                    raise ValueError("A document cannot be its own parent")
                descendants = self.get_descendants(doc_id)
                descendant_ids = {d[0] for d in descendants}
                if parent_id in descendant_ids:
                    raise ValueError(
                        f"Cannot set parent: document {parent_id} is a descendant of {doc_id}"
                    )

            cursor = self.conn.execute(
                f"UPDATE {self._table_name} SET parent_id = ? WHERE id = ?",
                (parent_id, doc_id),
            )
            return cursor.rowcount > 0

    # ------------------------------------------------------------------ #
    # Cluster State Persistence
    # ------------------------------------------------------------------ #

    def _ensure_cluster_table(self) -> None:
        """Create cluster state table if it doesn't exist."""
        if self._cluster_table_ready:
            return
        cluster_table = self._cluster_table_name
        with self._writable():
            # Re-check inside the lock so concurrent first-callers don't
            # both run the DDL. The CREATE TABLE IF NOT EXISTS is itself
            # idempotent, but doing the work twice defeats the early-exit.
            if self._cluster_table_ready:
                return
            self.conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {cluster_table} (
                    name TEXT PRIMARY KEY,
                    algorithm TEXT NOT NULL,
                    n_clusters INTEGER NOT NULL,
                    centroids BLOB,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
                """
            )
            self._cluster_table_ready = True

    def save_cluster_state(
        self,
        name: str,
        algorithm: str,
        n_clusters: int,
        centroids: bytes | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Save cluster state for later reuse.

        Args:
            name: Unique name for this cluster configuration
            algorithm: Algorithm used (kmeans, minibatch_kmeans, hdbscan)
            n_clusters: Number of clusters
            centroids: Serialized centroid array (numpy bytes)
            metadata: Additional metadata (inertia, silhouette, etc.)
        """
        self._ensure_cluster_table()
        cluster_table = self._cluster_table_name

        meta_json = json.dumps(metadata) if metadata else None

        with self._writable():
            self.conn.execute(
                f"""
                INSERT OR REPLACE INTO {cluster_table}
                (name, algorithm, n_clusters, centroids, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (name, algorithm, n_clusters, centroids, meta_json),
            )

    def load_cluster_state(
        self, name: str
    ) -> tuple[str, int, bytes | None, dict[str, Any]] | None:
        """
        Load saved cluster state.

        Args:
            name: Name of the cluster configuration

        Returns:
            Tuple of (algorithm, n_clusters, centroids_bytes, metadata) or None
        """
        self._ensure_cluster_table()
        cluster_table = self._cluster_table_name

        # Serialize on the connection-level lock — sqlite3.Connection is not
        # safe for concurrent statement execution from multiple threads.
        with self._lock:
            row = self.conn.execute(
                f"SELECT algorithm, n_clusters, centroids, metadata FROM {cluster_table} WHERE name = ?",
                (name,),
            ).fetchone()

        if not row:
            return None

        algorithm, n_clusters, centroids, meta_json = row
        metadata = json.loads(meta_json) if meta_json else {}
        return (algorithm, n_clusters, centroids, metadata)

    def list_cluster_states(self) -> list[dict[str, Any]]:
        """List all saved cluster configurations."""
        self._ensure_cluster_table()
        cluster_table = self._cluster_table_name

        with self._lock:
            rows = self.conn.execute(
                f"SELECT name, algorithm, n_clusters, created_at, metadata FROM {cluster_table}"
            ).fetchall()

        result = []
        for name, algorithm, n_clusters, created_at, meta_json in rows:
            result.append(
                {
                    "name": name,
                    "algorithm": algorithm,
                    "n_clusters": n_clusters,
                    "created_at": created_at,
                    "metadata": json.loads(meta_json) if meta_json else {},
                }
            )
        return result

    def delete_cluster_state(self, name: str) -> bool:
        """Delete a saved cluster configuration."""
        self._ensure_cluster_table()
        cluster_table = self._cluster_table_name

        with self._writable():
            cursor = self.conn.execute(
                f"DELETE FROM {cluster_table} WHERE name = ?", (name,)
            )
        return cursor.rowcount > 0
