"""Tests for hierarchical document relationships."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from simplevecdb import VectorDB, Document


class TestHierarchicalRelationships:
    """Tests for parent/child document relationships."""

    @pytest.fixture
    def db_path(self, tmp_path: Path) -> Path:
        return tmp_path / "test_hierarchy.db"

    @pytest.fixture
    def dim(self) -> int:
        return 32

    def make_embedding(self, dim: int) -> list[float]:
        return np.random.randn(dim).astype(np.float32).tolist()

    def test_add_with_parent_id(self, db_path: Path, dim: int):
        """Documents can be added with parent_id."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        # Add parent
        parent_ids = collection.add_texts(
            ["Parent document"],
            embeddings=[self.make_embedding(dim)],
        )
        parent_id = parent_ids[0]

        # Add children with parent_id
        child_ids = collection.add_texts(
            ["Child 1", "Child 2"],
            embeddings=[self.make_embedding(dim), self.make_embedding(dim)],
            parent_ids=[parent_id, parent_id],
        )

        assert len(child_ids) == 2
        assert collection.count() == 3

        db.close()

    def test_get_children(self, db_path: Path, dim: int):
        """get_children() returns direct children."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        # Create parent
        parent_id = collection.add_texts(
            ["Parent"],
            embeddings=[self.make_embedding(dim)],
        )[0]

        # Create children
        collection.add_texts(
            ["Child A", "Child B"],
            embeddings=[self.make_embedding(dim), self.make_embedding(dim)],
            parent_ids=[parent_id, parent_id],
        )

        # Get children
        children = collection.get_children(parent_id)

        assert len(children) == 2
        assert all(isinstance(c, Document) for c in children)
        child_texts = {c.page_content for c in children}
        assert child_texts == {"Child A", "Child B"}

        db.close()

    def test_get_parent(self, db_path: Path, dim: int):
        """get_parent() returns the parent document."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        # Create parent
        parent_id = collection.add_texts(
            ["Parent doc"],
            metadatas=[{"type": "parent"}],
            embeddings=[self.make_embedding(dim)],
        )[0]

        # Create child
        child_id = collection.add_texts(
            ["Child doc"],
            embeddings=[self.make_embedding(dim)],
            parent_ids=[parent_id],
        )[0]

        # Get parent
        parent = collection.get_parent(child_id)

        assert parent is not None
        assert parent.page_content == "Parent doc"
        assert parent.metadata.get("type") == "parent"

        db.close()

    def test_get_parent_returns_none_for_root(self, db_path: Path, dim: int):
        """get_parent() returns None for documents without parent."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        doc_id = collection.add_texts(
            ["Root document"],
            embeddings=[self.make_embedding(dim)],
        )[0]

        parent = collection.get_parent(doc_id)
        assert parent is None

        db.close()

    def test_get_children_empty(self, db_path: Path, dim: int):
        """get_children() returns empty list for leaf nodes."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        doc_id = collection.add_texts(
            ["Leaf document"],
            embeddings=[self.make_embedding(dim)],
        )[0]

        children = collection.get_children(doc_id)
        assert children == []

        db.close()

    def test_get_descendants(self, db_path: Path, dim: int):
        """get_descendants() returns all nested children."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        # Create tree: root -> child -> grandchild
        root_id = collection.add_texts(
            ["Root"],
            embeddings=[self.make_embedding(dim)],
        )[0]

        child_id = collection.add_texts(
            ["Child"],
            embeddings=[self.make_embedding(dim)],
            parent_ids=[root_id],
        )[0]

        collection.add_texts(
            ["Grandchild"],
            embeddings=[self.make_embedding(dim)],
            parent_ids=[child_id],
        )

        # Get all descendants
        descendants = collection.get_descendants(root_id)

        assert len(descendants) == 2
        texts = [doc.page_content for doc, _ in descendants]
        assert "Child" in texts
        assert "Grandchild" in texts

        # Check depths
        depths = {doc.page_content: depth for doc, depth in descendants}
        assert depths["Child"] == 1
        assert depths["Grandchild"] == 2

        db.close()

    def test_get_descendants_with_max_depth(self, db_path: Path, dim: int):
        """get_descendants() respects max_depth."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        # Create 3-level tree
        root_id = collection.add_texts(
            ["Root"],
            embeddings=[self.make_embedding(dim)],
        )[0]

        child_id = collection.add_texts(
            ["Child"],
            embeddings=[self.make_embedding(dim)],
            parent_ids=[root_id],
        )[0]

        collection.add_texts(
            ["Grandchild"],
            embeddings=[self.make_embedding(dim)],
            parent_ids=[child_id],
        )

        # Get only depth 1
        descendants = collection.get_descendants(root_id, max_depth=1)

        assert len(descendants) == 1
        assert descendants[0][0].page_content == "Child"

        db.close()

    def test_get_ancestors(self, db_path: Path, dim: int):
        """get_ancestors() returns path to root."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        # Create tree: root -> child -> grandchild
        root_id = collection.add_texts(
            ["Root"],
            embeddings=[self.make_embedding(dim)],
        )[0]

        child_id = collection.add_texts(
            ["Child"],
            embeddings=[self.make_embedding(dim)],
            parent_ids=[root_id],
        )[0]

        grandchild_id = collection.add_texts(
            ["Grandchild"],
            embeddings=[self.make_embedding(dim)],
            parent_ids=[child_id],
        )[0]

        # Get ancestors of grandchild
        ancestors = collection.get_ancestors(grandchild_id)

        assert len(ancestors) == 2
        texts = [doc.page_content for doc, _ in ancestors]
        assert texts == ["Child", "Root"]  # Ordered by depth

        db.close()

    def test_set_parent(self, db_path: Path, dim: int):
        """set_parent() updates document relationships."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        # Create two separate documents
        doc1_id = collection.add_texts(
            ["Doc 1"],
            embeddings=[self.make_embedding(dim)],
        )[0]

        doc2_id = collection.add_texts(
            ["Doc 2"],
            embeddings=[self.make_embedding(dim)],
        )[0]

        # Initially no parent
        assert collection.get_parent(doc2_id) is None

        # Set parent
        result = collection.set_parent(doc2_id, doc1_id)
        assert result is True

        # Now has parent
        parent = collection.get_parent(doc2_id)
        assert parent is not None
        assert parent.page_content == "Doc 1"

        db.close()

    def test_set_parent_remove(self, db_path: Path, dim: int):
        """set_parent(None) removes parent relationship."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        # Create parent-child
        parent_id = collection.add_texts(
            ["Parent"],
            embeddings=[self.make_embedding(dim)],
        )[0]

        child_id = collection.add_texts(
            ["Child"],
            embeddings=[self.make_embedding(dim)],
            parent_ids=[parent_id],
        )[0]

        # Remove parent
        collection.set_parent(child_id, None)

        # No longer has parent
        assert collection.get_parent(child_id) is None
        assert collection.get_children(parent_id) == []

        db.close()

    def test_hierarchical_search(self, db_path: Path, dim: int):
        """Hierarchical docs are searchable like normal docs."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        # Create hierarchy with known embeddings
        target_vec = np.ones(dim, dtype=np.float32)
        target_vec = (target_vec / np.linalg.norm(target_vec)).tolist()

        parent_id = collection.add_texts(
            ["Parent"],
            embeddings=[self.make_embedding(dim)],
        )[0]

        collection.add_texts(
            ["Target child"],
            metadatas=[{"is_target": True}],
            embeddings=[target_vec],
            parent_ids=[parent_id],
        )

        # Search should find child
        results = collection.similarity_search(target_vec, k=1)
        assert len(results) == 1
        assert results[0][0].page_content == "Target child"

        db.close()

    def test_delete_parent_orphans_children(self, db_path: Path, dim: int):
        """Deleting parent sets children's parent_id to NULL (ON DELETE SET NULL)."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        parent_id = collection.add_texts(
            ["Parent"],
            embeddings=[self.make_embedding(dim)],
        )[0]

        child_id = collection.add_texts(
            ["Child"],
            embeddings=[self.make_embedding(dim)],
            parent_ids=[parent_id],
        )[0]

        # Delete parent
        collection.delete_by_ids([parent_id])

        # Child still exists but has no parent
        assert collection.count() == 1
        assert collection.get_parent(child_id) is None

        db.close()

    def test_set_parent_self_reference_raises(self, db_path: Path, dim: int):
        """set_parent() rejects self-reference (doc cannot be its own parent)."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        doc_id = collection.add_texts(
            ["Document"],
            embeddings=[self.make_embedding(dim)],
        )[0]

        with pytest.raises(ValueError, match="cannot be its own parent"):
            collection.set_parent(doc_id, doc_id)

        db.close()

    def test_set_parent_cycle_detection(self, db_path: Path, dim: int):
        """set_parent() rejects cycles (descendant cannot become ancestor)."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        # Create chain: grandparent -> parent -> child
        grandparent_id = collection.add_texts(
            ["Grandparent"],
            embeddings=[self.make_embedding(dim)],
        )[0]

        parent_id = collection.add_texts(
            ["Parent"],
            embeddings=[self.make_embedding(dim)],
            parent_ids=[grandparent_id],
        )[0]

        child_id = collection.add_texts(
            ["Child"],
            embeddings=[self.make_embedding(dim)],
            parent_ids=[parent_id],
        )[0]

        # Try to make child the parent of grandparent -> cycle!
        with pytest.raises(ValueError, match="descendant"):
            collection.set_parent(grandparent_id, child_id)

        # Try to make child the parent of its own parent -> cycle!
        with pytest.raises(ValueError, match="descendant"):
            collection.set_parent(parent_id, child_id)

        db.close()


class TestHierarchyMigration:
    """Test that existing databases get parent_id column added."""

    def test_migration_adds_parent_id(self, tmp_path: Path):
        """Opening existing DB adds parent_id column."""
        db_path = tmp_path / "migrate.db"
        dim = 16

        # Create DB and add document (v2.0 schema)
        db = VectorDB(db_path)
        collection = db.collection("test")
        collection.add_texts(
            ["Existing doc"],
            embeddings=[np.random.randn(dim).tolist()],
        )
        db.close()

        # Reopen - should auto-migrate
        db2 = VectorDB(db_path)
        collection2 = db2.collection("test")

        # Should be able to use parent_ids now
        parent_id = collection2.add_texts(
            ["New parent"],
            embeddings=[np.random.randn(dim).tolist()],
        )[0]

        collection2.add_texts(
            ["New child"],
            embeddings=[np.random.randn(dim).tolist()],
            parent_ids=[parent_id],
        )

        children = collection2.get_children(parent_id)
        assert len(children) == 1

        db2.close()
