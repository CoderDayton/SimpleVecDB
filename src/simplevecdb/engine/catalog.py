"""
CatalogManager: SQLite metadata and FTS operations for SimpleVecDB.

This module handles all SQLite operations for document metadata, text content,
and full-text search (FTS5). Vector operations are handled by UsearchIndex.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TYPE_CHECKING, Callable
from collections.abc import Iterable, Sequence

from ..utils import validate_filter, retry_on_lock

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
    ):
        # Defense-in-depth: validate table names
        _validate_table_name(table_name)
        _validate_table_name(fts_table_name)

        self.conn = conn
        self._table_name = table_name
        self._fts_table_name = fts_table_name
        self._fts_enabled = False

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
        # Migrate existing tables that lack columns
        self._ensure_embedding_column()
        self._ensure_parent_id_column()
        self._ensure_fts_table()

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
        """Create FTS5 virtual table for full-text search."""
        import sqlite3

        try:
            self.conn.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {self._fts_table_name}
                USING fts5(text)
                """
            )
            self._fts_enabled = True
        except sqlite3.OperationalError:
            _logger.warning("FTS5 not available - keyword search disabled")
            self._fts_enabled = False

    @property
    def fts_enabled(self) -> bool:
        """Whether FTS5 is available for keyword search."""
        return self._fts_enabled

    def upsert_fts_rows(self, ids: Sequence[int], texts: Sequence[str]) -> None:
        """Update FTS index for given document IDs.

        Args:
            ids: Document IDs to update
            texts: Corresponding text content
        """
        if not self._fts_enabled or not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        self.conn.execute(
            f"DELETE FROM {self._fts_table_name} WHERE rowid IN ({placeholders})",
            tuple(ids),
        )
        rows = list(zip(ids, texts))
        self.conn.executemany(
            f"INSERT INTO {self._fts_table_name}(rowid, text) VALUES (?, ?)", rows
        )

    def delete_fts_rows(self, ids: Sequence[int]) -> None:
        """Remove documents from FTS index.

        Args:
            ids: Document IDs to remove
        """
        if not self._fts_enabled or not ids:
            return
        placeholders = ",".join("?" for _ in ids)
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
            for emb in embeddings:
                arr = np.asarray(emb, dtype=np.float32)
                embedding_blobs.append(arr.tobytes())
        else:
            embedding_blobs = [None] * len(texts)

        rows = [
            (uid, txt, json.dumps(meta), emb_blob, pid)
            for uid, txt, meta, emb_blob, pid in zip(
                ids_list, texts, metadatas, embedding_blobs, parent_ids_list
            )
        ]

        with self.conn:
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
                rows,
            )

            # Get the actual rowids (handles both insert and upsert)
            real_ids = [
                r[0]
                for r in self.conn.execute(
                    f"SELECT id FROM {self._table_name} ORDER BY id DESC LIMIT ?",
                    (len(texts),),
                )
            ]
            real_ids.reverse()

            # Update FTS index
            self.upsert_fts_rows(real_ids, list(texts))

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

        with self.conn:
            # Check which IDs actually exist
            existing = self.conn.execute(
                f"SELECT id FROM {self._table_name} WHERE id IN ({placeholders})",
                params,
            ).fetchall()
            existing_ids = [r[0] for r in existing]

            if existing_ids:
                placeholders = ",".join("?" for _ in existing_ids)
                self.conn.execute(
                    f"DELETE FROM {self._table_name} WHERE id IN ({placeholders})",
                    tuple(existing_ids),
                )
                self.delete_fts_rows(existing_ids)

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

        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"SELECT id, text, metadata FROM {self._table_name} WHERE id IN ({placeholders})",
            tuple(ids),
        ).fetchall()

        result = {}
        for row_id, text, meta_json in rows:
            meta = json.loads(meta_json) if meta_json else {}
            result[row_id] = (text, meta)
        return result

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

        placeholders = ",".join("?" for _ in ids)
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

    def find_ids_by_texts(self, texts: Sequence[str]) -> list[int]:
        """Find document IDs matching exact text content."""
        if not texts:
            return []
        placeholders = ",".join("?" for _ in texts)
        rows = self.conn.execute(
            f"SELECT id FROM {self._table_name} WHERE text IN ({placeholders})",
            tuple(texts),
        ).fetchall()
        return [r[0] for r in rows]

    def find_ids_by_filter(
        self,
        filter_dict: dict[str, Any],
        filter_builder: Callable[[dict[str, Any], str], tuple[str, list[Any]]],
    ) -> list[int]:
        """Find document IDs matching metadata filter."""
        if not filter_dict:
            return []

        filter_clause, filter_params = filter_builder(filter_dict, "metadata")
        # Remove leading "AND " from clause
        filter_clause = filter_clause.replace("AND ", "", 1)
        where_clause = f"WHERE {filter_clause}" if filter_clause else ""

        rows = self.conn.execute(
            f"SELECT id FROM {self._table_name} {where_clause}",
            tuple(filter_params),
        ).fetchall()
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
        rows = self.conn.execute(sql, params).fetchall()
        return [(int(row[0]), float(row[1])) for row in rows]

    def build_filter_clause(
        self, filter_dict: dict[str, Any] | None, metadata_column: str = "metadata"
    ) -> tuple[str, list[Any]]:
        """
        Build SQL WHERE clause from metadata filter dictionary.

        Args:
            filter_dict: Metadata key-value pairs to filter by
            metadata_column: Name of JSON metadata column

        Returns:
            Tuple of (where_clause, parameters) for SQL query

        Raises:
            ValueError: If filter keys are not strings or values are unsupported types
        """
        if not filter_dict:
            return "", []

        # Validate filter structure before processing
        validate_filter(filter_dict)

        clauses = []
        params: list[Any] = []
        for key, value in filter_dict.items():
            json_path = f"$.{key}"
            if isinstance(value, (int, float)):
                clauses.append(f"json_extract({metadata_column}, ?) = ?")
                params.extend([json_path, value])
            elif isinstance(value, str):
                # Escape LIKE special characters to prevent injection
                escaped_value = (
                    value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                )
                clauses.append(f"json_extract({metadata_column}, ?) LIKE ? ESCAPE '\\'")
                params.extend([json_path, f"%{escaped_value}%"])
            elif isinstance(value, list):
                placeholders = ",".join("?" for _ in value)
                clauses.append(
                    f"json_extract({metadata_column}, ?) IN ({placeholders})"
                )
                params.extend([json_path] + value)
            else:
                raise ValueError(f"Unsupported filter value type for {key}")
        where = " AND ".join(clauses)
        return f"AND ({where})" if where else "", params

    def count(self) -> int:
        """Return total number of documents."""
        row = self.conn.execute(f"SELECT COUNT(*) FROM {self._table_name}").fetchone()
        return row[0] if row else 0

    def check_legacy_sqlite_vec(self, vec_table_name: str) -> bool:
        """
        Check if legacy sqlite-vec tables exist (for migration).

        Args:
            vec_table_name: Expected name of the old vec0 virtual table

        Returns:
            True if legacy sqlite-vec data exists
        """
        try:
            row = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (vec_table_name,),
            ).fetchone()
            return row is not None
        except Exception:
            return False

    def get_legacy_vectors(self, vec_table_name: str) -> list[tuple[int, bytes]]:
        """
        Extract vectors from legacy sqlite-vec table for migration.

        Args:
            vec_table_name: Name of the old vec0 virtual table

        Returns:
            List of (rowid, embedding_blob) tuples
        """
        try:
            rows = self.conn.execute(
                f"SELECT rowid, embedding FROM {vec_table_name}"
            ).fetchall()
            return [(int(r[0]), r[1]) for r in rows]
        except Exception as e:
            _logger.warning("Failed to read legacy vectors: %s", e)
            return []

    def drop_legacy_vec_table(self, vec_table_name: str) -> None:
        """Drop legacy sqlite-vec table after migration."""
        try:
            self.conn.execute(f"DROP TABLE IF EXISTS {vec_table_name}")
            self.conn.commit()
            _logger.info("Dropped legacy sqlite-vec table: %s", vec_table_name)
        except Exception as e:
            _logger.warning("Failed to drop legacy table %s: %s", vec_table_name, e)

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
        # First get the parent_id
        row = self.conn.execute(
            f"SELECT parent_id FROM {self._table_name} WHERE id = ?",
            (doc_id,),
        ).fetchone()

        if not row or row[0] is None:
            return None

        parent_id = row[0]

        # Then fetch the parent document
        parent_row = self.conn.execute(
            f"SELECT id, text, metadata FROM {self._table_name} WHERE id = ?",
            (parent_id,),
        ).fetchone()

        if not parent_row:
            return None

        return (
            int(parent_row[0]),
            parent_row[1],
            json.loads(parent_row[2]) if parent_row[2] else {},
        )

    def get_descendants(
        self, root_id: int, max_depth: int | None = None
    ) -> list[tuple[int, str, dict[str, Any], int]]:
        """
        Get all descendants of a document (recursive).

        Uses a recursive CTE for efficient traversal.

        Args:
            root_id: ID of the root document
            max_depth: Maximum depth to traverse (None for unlimited)

        Returns:
            List of (id, text, metadata, depth) tuples
        """
        # Validate max_depth parameter to prevent SQL injection
        if max_depth is not None:
            max_depth = int(max_depth)

        if max_depth is not None:
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
            rows = self.conn.execute(sql, (root_id, max_depth)).fetchall()
        else:
            sql = f"""
                WITH RECURSIVE descendants(id, text, metadata, depth) AS (
                    SELECT id, text, metadata, 1 as depth
                    FROM {self._table_name}
                    WHERE parent_id = ?

                    UNION ALL

                    SELECT t.id, t.text, t.metadata, d.depth + 1
                    FROM {self._table_name} t
                    JOIN descendants d ON t.parent_id = d.id
                )
                SELECT id, text, metadata, depth FROM descendants
                ORDER BY depth, id
            """
            rows = self.conn.execute(sql, (root_id,)).fetchall()

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
            max_depth: Maximum depth to traverse (None for unlimited)

        Returns:
            List of (id, text, metadata, depth) tuples, from immediate parent to root
        """
        # Validate max_depth parameter to prevent SQL injection
        if max_depth is not None:
            max_depth = int(max_depth)

        if max_depth is not None:
            sql = f"""
                WITH RECURSIVE ancestors(id, text, metadata, parent_id, depth) AS (
                    SELECT id, text, metadata, parent_id, 1 as depth
                    FROM {self._table_name}
                    WHERE id = (SELECT parent_id FROM {self._table_name} WHERE id = ?)

                    UNION ALL

                    SELECT t.id, t.text, t.metadata, t.parent_id, a.depth + 1
                    FROM {self._table_name} t
                    JOIN ancestors a ON t.id = a.parent_id
                    WHERE a.parent_id IS NOT NULL AND depth < ?
                )
                SELECT id, text, metadata, depth FROM ancestors
                ORDER BY depth
            """
            rows = self.conn.execute(sql, (doc_id, max_depth)).fetchall()
        else:
            sql = f"""
                WITH RECURSIVE ancestors(id, text, metadata, parent_id, depth) AS (
                    SELECT id, text, metadata, parent_id, 1 as depth
                    FROM {self._table_name}
                    WHERE id = (SELECT parent_id FROM {self._table_name} WHERE id = ?)

                    UNION ALL

                    SELECT t.id, t.text, t.metadata, t.parent_id, a.depth + 1
                    FROM {self._table_name} t
                    JOIN ancestors a ON t.id = a.parent_id
                    WHERE a.parent_id IS NOT NULL
                )
                SELECT id, text, metadata, depth FROM ancestors
                ORDER BY depth
            """
            rows = self.conn.execute(sql, (doc_id,)).fetchall()

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
        """
        with self.conn:
            cursor = self.conn.execute(
                f"UPDATE {self._table_name} SET parent_id = ? WHERE id = ?",
                (parent_id, doc_id),
            )
            return cursor.rowcount > 0
