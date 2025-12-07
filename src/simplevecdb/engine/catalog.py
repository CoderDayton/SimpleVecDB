from __future__ import annotations

import json
import logging
from typing import Any, TYPE_CHECKING, Callable
from collections.abc import Iterable, Sequence

from ..utils import validate_filter, retry_on_lock

if TYPE_CHECKING:
    import sqlite3
    from ..types import Quantization, DistanceStrategy

_logger = logging.getLogger("simplevecdb.engine.catalog")


class CatalogManager:
    """
    Handles collection schema and CRUD operations.

    This manager is responsible for:
    - Creating and managing SQLite tables (metadata and FTS)
    - Creating and managing sqlite-vec virtual tables
    - Adding, deleting, and removing documents
    - Building filter clauses for metadata queries

    Args:
        conn: SQLite database connection
        table_name: Name of the metadata table
        vec_table_name: Name of the vector index table
        fts_table_name: Name of the full-text search table
        quantization: Vector quantization strategy
        distance_strategy: Distance metric for similarity
        quantizer: QuantizationStrategy instance
        dim_getter: Callable to get current dimension
        dim_setter: Callable to set dimension
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        vec_table_name: str,
        fts_table_name: str,
        quantization: Quantization,
        distance_strategy: DistanceStrategy,
        quantizer: Any,
        dim_getter: Callable[[], int | None],
        dim_setter: Callable[[int], None],
    ):
        self.conn = conn
        self._table_name = table_name
        self._vec_table_name = vec_table_name
        self._fts_table_name = fts_table_name
        self.quantization = quantization
        self.distance_strategy = distance_strategy
        self._quantizer = quantizer
        self._get_dim = dim_getter
        self._set_dim = dim_setter
        self._fts_enabled = False

    def create_tables(self) -> None:
        """Create metadata and FTS tables if they don't exist."""
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                metadata TEXT
            )
            """
        )
        self._ensure_fts_table()

    def _ensure_fts_table(self) -> None:
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
            self._fts_enabled = False

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

    def ensure_virtual_table(self, dim: int) -> None:
        """Create or verify sqlite-vec virtual table with given dimension.

        Args:
            dim: Vector dimension

        Raises:
            ValueError: If dimension doesn't match existing table
        """
        from ..types import Quantization, DistanceStrategy

        current_dim = self._get_dim()
        if current_dim is not None and current_dim != dim:
            raise ValueError(f"Dimension mismatch: existing {current_dim}, got {dim}")
        if current_dim is None:
            self._set_dim(dim)
            self.conn.execute(f"DROP TABLE IF EXISTS {self._vec_table_name}")

            storage_dim = dim
            if self.quantization == Quantization.BIT:
                storage_dim = ((dim + 7) // 8) * 8

            vec_type = {
                Quantization.FLOAT: f"float[{storage_dim}]",
                Quantization.INT8: f"int8[{storage_dim}]",
                Quantization.BIT: f"bit[{storage_dim}]",
            }[self.quantization]

            sql = f"CREATE VIRTUAL TABLE {self._vec_table_name} USING vec0(embedding {vec_type}"
            if (
                self.distance_strategy
                and not vec_type.startswith("bit")
                and self.distance_strategy != DistanceStrategy.COSINE
            ):
                sql += f" distance_metric={self.distance_strategy.value}"
            sql += ")"
            self.conn.execute(sql)

    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict] | None,
        embeddings: Sequence[Sequence[float]] | None,
        ids: Sequence[int | None] | None,
        batch_processor: Callable,
    ) -> list[int]:
        """
        Insert or update documents in the collection.

        Handles batched insertion into metadata, vector, and FTS tables.
        Supports upsert behavior when IDs are provided. Automatically retries
        on database lock errors with exponential backoff.

        Args:
            texts: Document text content
            metadatas: Optional metadata dicts
            embeddings: Optional pre-computed embeddings
            ids: Optional document IDs for upsert
            batch_processor: Generator yielding (texts, metadatas, ids, serialized_vectors)

        Returns:
            List of inserted/updated document IDs
        """
        if not texts:
            return []

        _logger.debug(
            "Adding %d texts to collection",
            len(texts),
            extra={"table": self._table_name, "count": len(texts)},
        )

        all_ids = []
        for batch_data in batch_processor(texts, metadatas, embeddings, ids):
            batch_texts, batch_metadatas, batch_ids, serialized = batch_data
            batch_real_ids = self._insert_batch(
                batch_texts, batch_metadatas, batch_ids, serialized
            )
            all_ids.extend(batch_real_ids)

        _logger.debug(
            "Added %d documents successfully",
            len(all_ids),
            extra={"table": self._table_name, "ids": all_ids[:10]},
        )
        return all_ids

    @retry_on_lock(max_retries=5, base_delay=0.1)
    def _insert_batch(
        self,
        batch_texts: Sequence[str],
        batch_metadatas: Sequence[dict],
        batch_ids: Sequence[int | None],
        serialized: Sequence[bytes],
    ) -> list[int]:
        """
        Insert a single batch of documents with retry on lock.

        Internal method that handles the actual database writes for a batch.
        Decorated with @retry_on_lock for automatic retry on lock contention.

        Args:
            batch_texts: Text content for the batch
            batch_metadatas: Metadata dicts for the batch
            batch_ids: Document IDs (may be None for auto-increment)
            serialized: Serialized vector data

        Returns:
            List of document IDs for the inserted batch
        """
        rows = []
        for txt, meta, uid in zip(batch_texts, batch_metadatas, batch_ids):
            rows.append((uid, txt, json.dumps(meta)))

        with self.conn:
            self.conn.executemany(
                f"""
                INSERT INTO {self._table_name}(id, text, metadata)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    text=excluded.text,
                    metadata=excluded.metadata
                """,
                rows,
            )
            batch_real_ids = [
                r[0]
                for r in self.conn.execute(
                    f"SELECT id FROM {self._table_name} ORDER BY id DESC LIMIT ?",
                    (len(batch_texts),),
                )
            ]
            batch_real_ids.reverse()

            real_vec_rows = [
                (real_id, ser) for real_id, ser in zip(batch_real_ids, serialized)
            ]

            insert_placeholder = "?"
            if self.quantization.value == "int8":
                insert_placeholder = "vec_int8(?)"
            elif self.quantization.value == "bit":
                insert_placeholder = "vec_bit(?)"

            placeholders = ",".join("?" for _ in batch_real_ids)
            self.conn.execute(
                f"DELETE FROM {self._vec_table_name} WHERE rowid IN ({placeholders})",
                tuple(batch_real_ids),
            )

            self.conn.executemany(
                f"INSERT INTO {self._vec_table_name}(rowid, embedding) VALUES (?, {insert_placeholder})",
                real_vec_rows,
            )

            self.upsert_fts_rows(batch_real_ids, batch_texts)

        return batch_real_ids

    @retry_on_lock(max_retries=5, base_delay=0.1)
    def delete_by_ids(self, ids: Iterable[int]) -> None:
        """
        Delete documents by their IDs.

        Removes documents from metadata, vector index, and FTS tables.
        Automatically runs VACUUM to reclaim disk space. Retries on
        database lock errors with exponential backoff.

        Args:
            ids: Document IDs to delete
        """
        ids = list(ids)
        if not ids:
            return

        _logger.debug(
            "Deleting %d documents",
            len(ids),
            extra={"table": self._table_name, "ids": ids[:10]},
        )

        placeholders = ",".join("?" for _ in ids)
        params = tuple(ids)
        with self.conn:
            self.conn.execute(
                f"DELETE FROM {self._table_name} WHERE id IN ({placeholders})",
                params,
            )
            self.conn.execute(
                f"DELETE FROM {self._vec_table_name} WHERE rowid IN ({placeholders})",
                params,
            )
            self.delete_fts_rows(ids)
        self.conn.execute("VACUUM")

        _logger.debug(
            "Deleted %d documents successfully",
            len(ids),
            extra={"table": self._table_name},
        )

    def remove_texts(
        self,
        texts: Sequence[str] | None,
        filter: dict[str, Any] | None,
        filter_builder: Callable,
    ) -> int:
        """
        Remove documents by text content or metadata filter.

        Args:
            texts: Optional list of exact text strings to remove
            filter: Optional metadata filter dict
            filter_builder: Function to build SQL WHERE clause from filter

        Returns:
            Number of documents deleted

        Raises:
            ValueError: If neither texts nor filter provided
        """
        if texts is None and filter is None:
            raise ValueError("Must provide either texts or filter to remove")

        ids_to_delete: list[int] = []

        if texts:
            placeholders = ",".join("?" for _ in texts)
            rows = self.conn.execute(
                f"SELECT id FROM {self._table_name} WHERE text IN ({placeholders})",
                tuple(texts),
            ).fetchall()
            ids_to_delete.extend(r[0] for r in rows)

        if filter:
            filter_clause, filter_params = filter_builder(filter)
            filter_clause = filter_clause.replace("AND ", "", 1)
            where_clause = f"WHERE {filter_clause}" if filter_clause else ""
            rows = self.conn.execute(
                f"SELECT id FROM {self._table_name} {where_clause}",
                tuple(filter_params),
            ).fetchall()
            ids_to_delete.extend(r[0] for r in rows)

        unique_ids = list(set(ids_to_delete))
        if unique_ids:
            self.delete_by_ids(unique_ids)

        return len(unique_ids)

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
        params = []
        for key, value in filter_dict.items():
            json_path = f"$.{key}"
            if isinstance(value, (int, float)):
                clauses.append(f"json_extract({metadata_column}, ?) = ?")
                params.extend([json_path, value])
            elif isinstance(value, str):
                clauses.append(f"json_extract({metadata_column}, ?) LIKE ?")
                params.extend([json_path, f"%{value}%"])
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
