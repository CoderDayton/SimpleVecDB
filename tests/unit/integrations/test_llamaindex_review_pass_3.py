"""Regression tests for the LlamaIndex integration in 2.6.0 review pass 3.

Pins invariants the prior suite missed:

- A round-trip through ``add()`` → ``query()`` returns nodes whose
  ``node_id`` equals the inserted node's ``id_``.
- ``delete()`` against a v2.5-shaped row (no ``_simplevecdb_node_id``
  metadata) followed by ``migrate_node_id_metadata()`` then
  ``delete(str(doc_id))`` actually removes the row.
- A v2.5-shaped collection triggers a ``DeprecationWarning`` at
  ``__init__`` time.
- ``delete()`` does not silently swallow ``sqlite3.DatabaseError`` from
  the metadata-fallback path.
"""

from __future__ import annotations

import warnings

import pytest

llama_index = pytest.importorskip("llama_index.core")

from llama_index.core.schema import TextNode  # noqa: E402
from llama_index.core.vector_stores import VectorStoreQuery  # noqa: E402

from simplevecdb.integrations.llamaindex import SimpleVecDBLlamaStore  # noqa: E402


def _make_node(node_id: str, text: str, embedding: list[float]) -> TextNode:
    n = TextNode(text=text, id_=node_id)
    n.embedding = embedding
    return n


class TestRoundTripQuery:
    def test_inserted_node_id_preserved_through_query(self, tmp_path):
        store = SimpleVecDBLlamaStore(
            db_path=str(tmp_path / "rtq.db"),
            collection_name="default",
        )
        try:
            n1 = _make_node("uuid-aaa", "alpha doc", [1.0, 0.0, 0.0, 0.0])
            n2 = _make_node("uuid-bbb", "beta doc", [0.0, 1.0, 0.0, 0.0])
            ids = store.add([n1, n2])
            assert set(ids) == {"uuid-aaa", "uuid-bbb"}

            result = store.query(
                VectorStoreQuery(
                    query_embedding=[1.0, 0.0, 0.0, 0.0],
                    similarity_top_k=2,
                )
            )
            returned_ids = set(result.ids or [])
            assert returned_ids == {"uuid-aaa", "uuid-bbb"}, (
                f"query() returned ids {returned_ids}, expected the "
                "original LlamaIndex node ids"
            )
        finally:
            store._db.close()


class TestMigrationThenDeleteEndToEnd:
    def test_legacy_row_migrated_then_deleted(self, tmp_path):
        """A v2.5-shaped row (no ``_simplevecdb_node_id``) gets
        migrated, then ``delete(str(doc_id))`` removes it."""
        store = SimpleVecDBLlamaStore(
            db_path=str(tmp_path / "mig.db"),
            collection_name="default",
        )
        try:
            # Insert directly into the underlying collection without the
            # 2.6 metadata stamp, simulating a v2.5 row.
            ids = store._collection.add_texts(
                ["legacy doc"],
                metadatas=[{"source": "legacy"}],
                embeddings=[[1.0, 0.0, 0.0, 0.0]],
            )
            legacy_id = ids[0]

            # Migrate.
            updated = store.migrate_node_id_metadata()
            assert updated == 1

            # delete() against str(doc_id) must now find and remove it.
            store.delete(str(legacy_id))

            remaining = store._collection.get_documents()
            assert remaining == [], (
                f"Migrated row was not deleted; remaining={remaining}"
            )
        finally:
            store._db.close()


class TestLegacyCollectionWarning:
    def test_deprecation_warning_on_v25_shaped_collection(self, tmp_path):
        # Pre-populate a v2.5-shaped collection by writing through the
        # core API directly, bypassing the LlamaIndex metadata stamp.
        from simplevecdb import VectorDB

        db_path = tmp_path / "legacy.db"
        seed = VectorDB(str(db_path))
        seed.collection("default").add_texts(
            ["legacy"],
            metadatas=[{"foo": "bar"}],
            embeddings=[[1.0, 0.0, 0.0, 0.0]],
        )
        seed.close()

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            store = SimpleVecDBLlamaStore(
                db_path=str(db_path),
                collection_name="default",
            )
            try:
                deprecations = [
                    w
                    for w in record
                    if issubclass(w.category, DeprecationWarning)
                    and "_simplevecdb_node_id" in str(w.message)
                ]
                assert deprecations, (
                    "Expected DeprecationWarning naming "
                    "'_simplevecdb_node_id' on legacy collection open"
                )
            finally:
                store._db.close()


class TestDeleteSurfacesDatabaseError:
    def test_database_error_in_fallback_propagates(self, tmp_path):
        """The metadata-fallback path in ``delete()`` previously swallowed
        every Exception. A real sqlite3.DatabaseError must now
        propagate."""
        import sqlite3
        from unittest import mock

        store = SimpleVecDBLlamaStore(
            db_path=str(tmp_path / "err.db"),
            collection_name="default",
        )
        try:
            with mock.patch.object(
                store._collection,
                "get_documents",
                side_effect=sqlite3.DatabaseError("simulated locked DB"),
            ):
                with pytest.raises(sqlite3.DatabaseError, match="simulated"):
                    # _id_map is empty, forcing the fallback path.
                    store.delete("does-not-matter")
        finally:
            store._db.close()
