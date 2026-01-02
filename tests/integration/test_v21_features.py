# tests/integration/test_v21_features.py
"""
Integration test verifying v2.1 features work together:
- SQLCipher encryption
- Streaming insert API
- Hierarchical document relationships
"""

import os
import tempfile
from typing import Iterator

import pytest

from simplevecdb import VectorDB

# Check if encryption is available
try:
    import importlib.util
    sqlcipher_spec = importlib.util.find_spec("sqlcipher3")
    HAS_ENCRYPTION = sqlcipher_spec is not None
except ImportError:
    HAS_ENCRYPTION = False

# Test-only encryption keys - NOT for production use
# nosec: B105 - hardcoded passwords are intentional for testing
TEST_ENCRYPTION_KEY = "test-only-key-do-not-use-in-production"  # noqa: S105
TEST_ENCRYPTION_KEY_WRONG = "wrong-key-for-testing-failure"  # noqa: S105


def generate_chunked_documents(
    num_parents: int = 3, chunks_per_parent: int = 4, dim: int = 8
) -> Iterator[tuple[str, dict, list[float], int | None]]:
    """
    Generate hierarchical documents for streaming insert.

    Yields: (text, metadata, embedding, parent_id)
    """
    parent_id_offset = 1  # SQLite IDs start at 1

    for p in range(num_parents):
        parent_id = parent_id_offset + p * (chunks_per_parent + 1)
        # Parent document
        yield (
            f"Parent document {p}: Main content about topic {p}",
            {"type": "parent", "topic": p},
            [0.1 * (p + 1)] * dim,
            None,  # Parents have no parent
        )
        # Child chunks
        for c in range(chunks_per_parent):
            yield (
                f"Chunk {c} of parent {p}: Detailed content section {c}",
                {"type": "chunk", "topic": p, "chunk_idx": c},
                [0.1 * (p + 1) + 0.01 * c] * dim,
                parent_id,
            )


@pytest.mark.integration
@pytest.mark.skipif(not HAS_ENCRYPTION, reason="Encryption dependencies not installed")
def test_encrypted_streaming_hierarchy():
    """
    Test that encryption, streaming, and hierarchy work together.

    This test:
    1. Creates an encrypted database
    2. Streams hierarchical documents (parents + children)
    3. Verifies hierarchy traversal works
    4. Verifies search within subtrees works
    5. Verifies data persists after reopen with correct key
    6. Verifies wrong key fails to open
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "encrypted_hierarchy.db")

        # Phase 1: Create encrypted DB and stream hierarchical data
        db = VectorDB(db_path, encryption_key=TEST_ENCRYPTION_KEY)
        collection = db.collection("docs")  # dimension auto-detected

        # Prepare data with hierarchy
        texts: list[str] = []
        metadatas: list[dict] = []
        embeddings: list[list[float]] = []
        parent_ids: list[int | None] = []

        for text, meta, emb, pid in generate_chunked_documents(
            num_parents=3, chunks_per_parent=4
        ):
            texts.append(text)
            metadatas.append(meta)
            embeddings.append(emb)
            parent_ids.append(pid)

        # Use streaming insert with progress tracking
        progress_updates = []

        def track_progress(p):
            progress_updates.append(p)

        # Stream in batches of 5
        ids = []
        for progress in collection.add_texts_streaming(
            zip(texts, metadatas, embeddings),
            batch_size=5,
            on_progress=track_progress,
        ):
            ids.extend(progress.get("batch_ids", []))

        # Verify streaming worked
        assert len(progress_updates) > 0, "Should have progress updates"
        assert progress_updates[-1]["docs_processed"] == 15  # 3 parents + 12 chunks

        # Now set parent relationships (streaming doesn't support parent_ids directly)
        # We need to manually set them after insert
        # Parents are at indices 0, 5, 10 (every 5th starting from 0)
        # Children follow each parent
        for i, pid in enumerate(parent_ids):
            if pid is not None:
                # Map the expected parent_id to actual inserted id
                # Parent 1 -> id 1, Parent 2 -> id 6, Parent 3 -> id 11
                parent_idx = (pid - 1) // 5  # 0, 1, or 2
                actual_parent_id = ids[parent_idx * 5]  # First id in each batch of 5
                collection.set_parent(ids[i], actual_parent_id)

        db.close()

        # Phase 2: Reopen with correct key and verify hierarchy
        db2 = VectorDB(db_path, encryption_key=TEST_ENCRYPTION_KEY)
        collection2 = db2.collection("docs")

        # Verify we can retrieve data
        assert collection2.count() == 15

        # Get first parent and verify children
        parent_id = ids[0]
        children = collection2.get_children(parent_id)
        assert len(children) == 4, f"Expected 4 children, got {len(children)}"

        # Verify descendants
        descendants = collection2.get_descendants(parent_id)
        assert len(descendants) == 4, f"Expected 4 descendants, got {len(descendants)}"

        # Verify ancestors from a child
        child_id = ids[1]  # First child of first parent
        ancestors = collection2.get_ancestors(child_id)
        assert len(ancestors) == 1, f"Expected 1 ancestor, got {len(ancestors)}"
        assert ancestors[0][0].metadata.get("type") == "parent"

        # Search within subtree (children of first parent)
        query_emb = [0.1] * 8  # Close to first parent's embedding
        results = collection2.similarity_search(
            query_emb, k=10, filter={"type": "chunk", "topic": 0}
        )
        assert len(results) == 4, "Should find all 4 chunks of topic 0"

        db2.close()

        # Phase 3: Verify wrong key fails
        with pytest.raises(Exception):  # Could be various encryption errors
            bad_db = VectorDB(db_path, encryption_key=TEST_ENCRYPTION_KEY_WRONG)
            bad_db.collection("docs").count()  # Force access


@pytest.mark.integration
def test_streaming_hierarchy_unencrypted():
    """
    Test streaming + hierarchy without encryption (simpler case).
    Ensures features work independently of encryption.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "hierarchy.db")

        db = VectorDB(db_path)
        collection = db.collection("docs")  # dimension auto-detected

        # Add parent documents
        parent_ids_result = collection.add_texts(
            ["Parent A", "Parent B"],
            embeddings=[[0.1] * 8, [0.2] * 8],
            metadatas=[{"type": "parent"}, {"type": "parent"}],
        )

        # Stream children for each parent
        def child_generator():
            for i in range(6):
                parent = parent_ids_result[i % 2]  # Alternate parents
                yield (
                    f"Child {i}",
                    {"type": "child", "parent_ref": parent},
                    [0.1 + 0.01 * i] * 8,
                )

        child_ids = []
        for progress in collection.add_texts_streaming(child_generator(), batch_size=3):
            child_ids.extend(progress.get("batch_ids", []))

        # Set parent relationships
        for i, cid in enumerate(child_ids):
            collection.set_parent(cid, parent_ids_result[i % 2])

        # Verify hierarchy
        children_a = collection.get_children(parent_ids_result[0])
        children_b = collection.get_children(parent_ids_result[1])

        assert len(children_a) == 3, (
            f"Parent A should have 3 children, got {len(children_a)}"
        )
        assert len(children_b) == 3, (
            f"Parent B should have 3 children, got {len(children_b)}"
        )

        # Verify search with hierarchy filter
        results = collection.similarity_search(
            [0.1] * 8, k=10, filter={"type": "child"}
        )
        assert len(results) == 6, "Should find all 6 children"

        db.close()


@pytest.mark.integration
def test_hierarchy_depth_traversal():
    """Test multi-level hierarchy (grandparent -> parent -> child)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "deep_hierarchy.db")

        db = VectorDB(db_path)
        collection = db.collection("docs")  # dimension auto-detected

        # Create 3-level hierarchy
        # Level 0: Root
        root_ids = collection.add_texts(
            ["Root document"],
            embeddings=[[1.0, 0.0, 0.0, 0.0]],
            metadatas=[{"level": 0}],
        )

        # Level 1: Chapters (children of root)
        chapter_ids = collection.add_texts(
            ["Chapter 1", "Chapter 2"],
            embeddings=[[0.9, 0.1, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0]],
            metadatas=[{"level": 1}, {"level": 1}],
            parent_ids=[root_ids[0], root_ids[0]],
        )

        # Level 2: Sections (children of chapters)
        section_ids = collection.add_texts(
            ["Section 1.1", "Section 1.2", "Section 2.1"],
            embeddings=[
                [0.7, 0.2, 0.1, 0.0],
                [0.6, 0.3, 0.1, 0.0],
                [0.5, 0.4, 0.1, 0.0],
            ],
            metadatas=[{"level": 2}, {"level": 2}, {"level": 2}],
            parent_ids=[chapter_ids[0], chapter_ids[0], chapter_ids[1]],
        )

        # Test get_descendants from root (should get all 5 descendants)
        all_descendants = collection.get_descendants(root_ids[0])
        assert len(all_descendants) == 5, (
            f"Root should have 5 descendants, got {len(all_descendants)}"
        )

        # Test get_descendants with max_depth=1 (only chapters)
        direct_children = collection.get_descendants(root_ids[0], max_depth=1)
        assert len(direct_children) == 2, (
            f"Root should have 2 direct children, got {len(direct_children)}"
        )

        # Test get_ancestors from deepest node
        ancestors = collection.get_ancestors(section_ids[0])
        assert len(ancestors) == 2, (
            f"Section should have 2 ancestors, got {len(ancestors)}"
        )
        levels = [doc.metadata.get("level") for doc, depth in ancestors]
        assert levels == [1, 0], (
            f"Ancestors should be [chapter, root], got levels {levels}"
        )

        db.close()
