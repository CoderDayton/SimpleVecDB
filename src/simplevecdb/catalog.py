from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING, Callable
from collections.abc import Iterable, Sequence

if TYPE_CHECKING:
    import sqlite3
    from .types import Quantization, DistanceStrategy


class CatalogManager:
    """Handles collection schema and CRUD operations."""

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
        if not self._fts_enabled or not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        self.conn.execute(
            f"DELETE FROM {self._fts_table_name} WHERE rowid IN ({placeholders})",
            tuple(ids),
        )

    def ensure_virtual_table(self, dim: int) -> None:
        from .types import Quantization, DistanceStrategy

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
        if not texts:
            return []

        all_ids = []
        for batch_data in batch_processor(texts, metadatas, embeddings, ids):
            batch_texts, batch_metadatas, batch_ids, serialized = batch_data

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
                all_ids.extend(batch_real_ids)

        return all_ids

    def delete_by_ids(self, ids: Iterable[int]) -> None:
        ids = list(ids)
        if not ids:
            return
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

    def remove_texts(
        self,
        texts: Sequence[str] | None,
        filter: dict[str, Any] | None,
        filter_builder: Callable,
    ) -> int:
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
        if not filter_dict:
            return "", []
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
