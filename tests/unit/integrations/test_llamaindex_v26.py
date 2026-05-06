"""LlamaIndex integration tests for 2.6.0 changes.

Covers:
- ``migrate_node_id_metadata()`` backfill helper for legacy data
- ``delete_nodes(filters=...)`` raising NotImplementedError
- ``delete()`` falling back to a metadata query when ``_id_map`` is cold
  (typically after a process restart)
- node_id persisted into ``_simplevecdb_node_id`` metadata at insert
"""

from __future__ import annotations

import pytest

try:
    import llama_index  # noqa: F401
except ImportError:
    pytest.skip("llama-index not installed", allow_module_level=True)

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    ExactMatchFilter,
    MetadataFilters,
)

from simplevecdb.integrations.llamaindex import SimpleVecDBLlamaStore


def _make_store(tmp_path, name: str) -> SimpleVecDBLlamaStore:
    return SimpleVecDBLlamaStore(db_path=str(tmp_path / f"{name}.db"))


class TestNodeIdPersisted:
    """Inserts must stamp ``_simplevecdb_node_id`` into metadata."""

    def test_node_id_written_to_metadata_on_add(self, tmp_path):
        store = _make_store(tmp_path, "persist")
        node = TextNode(
            id_="node-abc",
            text="hello",
            embedding=[0.1] * 384,
            metadata={"source": "doc1"},
        )
        store.add([node])

        docs = store._collection.get_documents()
        assert len(docs) == 1
        _doc_id, _text, metadata = docs[0]
        assert metadata.get("_simplevecdb_node_id") == "node-abc"
        # Original metadata is preserved alongside the new key.
        assert metadata.get("source") == "doc1"


class TestDeleteFallback:
    """delete() must work even when _id_map is empty (post-restart case)."""

    def test_delete_uses_metadata_when_id_map_is_cold(self, tmp_path):
        store = _make_store(tmp_path, "cold")
        node = TextNode(
            id_="cold-node",
            text="restart-me",
            embedding=[0.3] * 384,
        )
        store.add([node])

        # Simulate a process restart by clearing the in-memory map. The
        # delete() path must still locate the row via the
        # _simplevecdb_node_id metadata.
        store._id_map.clear()

        store.delete("cold-node")

        # The collection must now be empty.
        docs = store._collection.get_documents()
        assert docs == [] or len(docs) == 0


class TestMigrateNodeIdMetadata:
    """migrate_node_id_metadata() backfills legacy rows idempotently."""

    def test_migrate_backfills_missing_metadata(self, tmp_path):
        store = _make_store(tmp_path, "migrate")

        # Bypass the normal LlamaIndex add path and inject "legacy" rows
        # directly via the underlying collection — this simulates data
        # written by simplevecdb < 2.6.0 (no _simplevecdb_node_id key).
        ids = store._collection.add_texts(
            ["legacy-1", "legacy-2"],
            [{"src": "old"}, {"src": "old"}],
            [[0.1] * 384, [0.2] * 384],
        )
        assert len(ids) == 2

        updated = store.migrate_node_id_metadata()
        assert updated == 2

        for doc_id, _text, metadata in store._collection.get_documents():
            assert metadata.get("_simplevecdb_node_id") == str(doc_id)

    def test_migrate_is_idempotent(self, tmp_path):
        store = _make_store(tmp_path, "idem")
        store._collection.add_texts(
            ["a", "b"],
            [{"x": 1}, {"x": 2}],
            [[0.1] * 384, [0.2] * 384],
        )

        first_pass = store.migrate_node_id_metadata()
        second_pass = store.migrate_node_id_metadata()

        assert first_pass == 2
        # Second pass should find nothing left to update.
        assert second_pass == 0

    def test_migrate_skips_already_stamped_rows(self, tmp_path):
        store = _make_store(tmp_path, "skip")

        # Mix one already-stamped row with one legacy row.
        store._collection.add_texts(
            ["new", "legacy"],
            [{"_simplevecdb_node_id": "preserved-id"}, {"src": "old"}],
            [[0.1] * 384, [0.2] * 384],
        )

        updated = store.migrate_node_id_metadata()
        assert updated == 1

        for _doc_id, text, metadata in store._collection.get_documents():
            if text == "new":
                # Original node_id must be preserved.
                assert metadata.get("_simplevecdb_node_id") == "preserved-id"


class TestDeleteNodesFilters:
    """delete_nodes(filters=...) must raise instead of silently no-op'ing."""

    def test_delete_nodes_with_filters_raises(self, tmp_path):
        store = _make_store(tmp_path, "filters")

        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="source", value="x")]
        )
        with pytest.raises(NotImplementedError, match="filters"):
            store.delete_nodes(filters=filters)

    def test_delete_nodes_with_empty_filter_object_raises(self, tmp_path):
        # Even an empty filters object should raise — caller is asking for
        # filter-based deletion, which the store does not support yet.
        store = _make_store(tmp_path, "emptyfilter")
        empty_filters = MetadataFilters(filters=[])
        with pytest.raises(NotImplementedError):
            store.delete_nodes(filters=empty_filters)

    def test_delete_nodes_with_node_ids_only_works(self, tmp_path):
        store = _make_store(tmp_path, "nodeids")
        node = TextNode(id_="kept", text="data", embedding=[0.1] * 384)
        store.add([node])

        # Should not raise.
        store.delete_nodes(node_ids=["kept"])
        assert len(store._collection.get_documents()) == 0
