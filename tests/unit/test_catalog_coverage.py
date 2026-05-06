"""Additional coverage tests for CatalogManager.

Targets missing lines: 32, 105-112, 120-136, 150-152, 167, 185,
215, 240, 394, 460, 499, 525, 580, 630-631, 643-650, 654-659, 768.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock

import pytest

from simplevecdb.engine.catalog import CatalogManager, _validate_table_name


class TestValidateTableName:
    """Cover line 32."""

    def test_invalid_table_name_raises(self):
        with pytest.raises(ValueError, match="Invalid table name"):
            _validate_table_name("DROP TABLE; --")

    def test_invalid_starts_with_digit(self):
        with pytest.raises(ValueError, match="Invalid table name"):
            _validate_table_name("1invalid")

    def test_valid_table_name_passes(self):
        _validate_table_name("valid_table_123")  # no error

    def test_invalid_with_spaces(self):
        with pytest.raises(ValueError, match="Invalid table name"):
            _validate_table_name("has spaces")


@pytest.fixture
def conn():
    """In-memory SQLite connection for catalog tests."""
    c = sqlite3.connect(":memory:")
    yield c
    c.close()


@pytest.fixture
def catalog(conn):
    """CatalogManager with tables created."""
    cm = CatalogManager(conn, "docs", "docs_fts")
    cm.create_tables()
    return cm


class TestMigrations:
    """Cover lines 105-112, 120-136."""

    def test_ensure_embedding_column_adds_missing_column(self, conn):
        """Lines 105-112: migrate table without embedding column."""
        conn.execute(
            "CREATE TABLE legacy_docs (id INTEGER PRIMARY KEY, text TEXT, metadata TEXT)"
        )
        cm = CatalogManager(conn, "legacy_docs", "legacy_docs_fts")
        cm._ensure_embedding_column()

        # Verify column was added
        cursor = conn.execute("PRAGMA table_info(legacy_docs)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "embedding" in columns

    def test_ensure_parent_id_column_adds_missing_column(self, conn):
        """Lines 120-136: migrate table without parent_id column."""
        conn.execute(
            "CREATE TABLE legacy2 (id INTEGER PRIMARY KEY, text TEXT, metadata TEXT, embedding BLOB)"
        )
        cm = CatalogManager(conn, "legacy2", "legacy2_fts")
        cm._ensure_parent_id_column()

        cursor = conn.execute("PRAGMA table_info(legacy2)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "parent_id" in columns

    def test_ensure_embedding_column_error_logged(self, conn):
        """Line 112: warning logged on failure."""
        cm = CatalogManager(conn, "nonexistent_table_xyz", "nonexistent_fts")
        # Should not raise, just log warning
        cm._ensure_embedding_column()

    def test_ensure_parent_id_column_error_logged(self, conn):
        """Line 136: warning logged on failure."""
        cm = CatalogManager(conn, "nonexistent_table_abc", "nonexistent_fts")
        cm._ensure_parent_id_column()


class TestFTS:
    """Cover lines 150-152, 167, 185."""

    def test_fts_not_available(self, conn):
        """Lines 150-152: FTS5 not available."""
        cm = CatalogManager(conn, "docs_nofts", "docs_nofts_fts")
        # Force FTS5 to fail by replacing conn with a mock that errors on VIRTUAL TABLE
        original_execute = conn.execute
        mock_conn = MagicMock(wraps=conn)

        def fail_on_fts(*args, **kwargs):
            if "VIRTUAL TABLE" in str(args[0]):
                raise sqlite3.OperationalError("fts5 not available")
            return original_execute(*args, **kwargs)

        mock_conn.execute = fail_on_fts
        cm.conn = mock_conn
        cm._ensure_fts_table()

        assert cm.fts_enabled is False

    def test_upsert_fts_rows_empty_ids(self, catalog):
        """Line 167: _upsert_fts_rows returns early on empty ids."""
        catalog._upsert_fts_rows([], [])  # no error

    def test_upsert_fts_rows_fts_disabled(self, conn):
        """Line 167: _upsert_fts_rows returns early when FTS disabled."""
        cm = CatalogManager(conn, "docs_nofts2", "docs_nofts2_fts")
        cm.conn.execute(
            "CREATE TABLE docs_nofts2 (id INTEGER PRIMARY KEY, text TEXT, metadata TEXT, embedding BLOB, parent_id INTEGER)"
        )
        cm._fts_enabled = False
        cm._upsert_fts_rows([1], ["text"])  # no error, early return

    def test_delete_fts_rows_empty_ids(self, catalog):
        """Line 185: _delete_fts_rows returns early on empty ids."""
        catalog._delete_fts_rows([])  # no error

    def test_delete_fts_rows_fts_disabled(self, conn):
        """Line 185: _delete_fts_rows returns early when FTS disabled."""
        cm = CatalogManager(conn, "docs_nofts3", "docs_nofts3_fts")
        cm.conn.execute(
            "CREATE TABLE docs_nofts3 (id INTEGER PRIMARY KEY, text TEXT, metadata TEXT, embedding BLOB, parent_id INTEGER)"
        )
        cm._fts_enabled = False
        cm._delete_fts_rows([1, 2])  # no error, early return


class TestAddDocuments:
    """Cover lines 215, 240."""

    def test_add_documents_without_embeddings(self, catalog):
        """Line 240: embeddings=None -> embedding_blobs all None."""
        ids = catalog.add_documents(
            texts=["hello", "world"],
            metadatas=[{"k": "v"}, {"k": "v2"}],
            embeddings=None,
        )
        assert len(ids) == 2

        # Verify embeddings are None
        for doc_id in ids:
            row = catalog.conn.execute(
                "SELECT embedding FROM docs WHERE id = ?", (doc_id,)
            ).fetchone()
            assert row[0] is None

    def test_add_documents_with_debug_logging(self, catalog):
        """Line 215: debug log with extra table info."""
        # Just ensure it runs without error (logging with extra dict)
        ids = catalog.add_documents(
            texts=["test"],
            metadatas=[{}],
        )
        assert len(ids) == 1


class TestGetDocumentsAndEmbeddings:
    """Cover line 394."""

    def test_empty_ids_returns_empty(self, catalog):
        """Line 394: empty ids -> empty dict."""
        result = catalog.get_documents_and_embeddings_by_ids([])
        assert result == {}


class TestKeywordSearch:
    """Cover line 460."""

    def test_keyword_search_fts_disabled_raises(self, conn):
        """Line 460: RuntimeError when FTS is not enabled."""
        cm = CatalogManager(conn, "docs_noks", "docs_noks_fts")
        cm.conn.execute(
            "CREATE TABLE docs_noks (id INTEGER PRIMARY KEY, text TEXT, metadata TEXT, embedding BLOB, parent_id INTEGER)"
        )
        cm._fts_enabled = False

        with pytest.raises(RuntimeError, match="FTS5 not available"):
            cm.keyword_search("query", k=5)


class TestBuildFilterClause:
    """Cover lines 499, 525."""

    def test_empty_filter_returns_empty(self, catalog):
        """Line 499: empty filter_dict -> empty string."""
        clause, params = catalog.build_filter_clause(None)
        assert clause == ""
        assert params == []

        clause, params = catalog.build_filter_clause({})
        assert clause == ""
        assert params == []

    def test_unsupported_filter_type_raises(self, catalog):
        """Line 525: unsupported value type raises ValueError."""
        with pytest.raises(ValueError, match="must be int, float, str, or list"):
            catalog.build_filter_clause({"key": object()})


class TestGetAllDocsWithFilter:
    """Cover line 580."""

    def test_get_all_docs_with_filter(self, catalog):
        """Line 580: get_all_docs_with_text with filter."""
        catalog.add_documents(
            texts=["doc1", "doc2", "doc3"],
            metadatas=[
                {"category": "a"},
                {"category": "b"},
                {"category": "a"},
            ],
        )

        result = catalog.get_all_docs_with_text(
            filter_dict={"category": "a"},
            filter_builder=catalog.build_filter_clause,
        )

        assert len(result) == 2
        for _, text, meta in result:
            assert meta["category"] == "a"


class TestLegacyVec:
    """Cover lines 630-631, 643-650, 654-659."""

    def test_check_legacy_returns_false_on_exception(self, catalog):
        """Lines 630-631: exception during check -> False."""
        # Replace conn with a mock that raises on execute
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = sqlite3.OperationalError("boom")
        catalog.conn = mock_conn
        result = catalog.check_legacy_sqlite_vec("old_vec_table")
        assert result is False

    def test_check_legacy_returns_false_no_table(self, catalog):
        """Line 629: table doesn't exist -> False."""
        result = catalog.check_legacy_sqlite_vec("nonexistent_vec")
        assert result is False

    def test_get_legacy_vectors_failure(self, catalog):
        """Lines 648-650: get_legacy_vectors returns empty on error."""
        result = catalog.get_legacy_vectors("nonexistent_table")
        assert result == []

    def test_get_legacy_vectors_success(self, catalog):
        """Lines 643-647: get_legacy_vectors reads from table."""
        catalog.conn.execute(
            "CREATE TABLE old_vec (embedding BLOB)"
        )
        catalog.conn.execute(
            "INSERT INTO old_vec (rowid, embedding) VALUES (1, ?)",
            (b"\x00\x01\x02\x03",),
        )
        catalog.conn.commit()

        result = catalog.get_legacy_vectors("old_vec")
        assert len(result) == 1
        assert result[0][0] == 1
        assert result[0][1] == b"\x00\x01\x02\x03"

    def test_drop_legacy_vec_table(self, catalog):
        """Lines 654-659: drop legacy table."""
        catalog.conn.execute("CREATE TABLE old_vec2 (embedding BLOB)")
        catalog.conn.commit()

        catalog.drop_legacy_vec_table("old_vec2")

        # Table should be gone
        row = catalog.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='old_vec2'"
        ).fetchone()
        assert row is None

    def test_drop_legacy_vec_table_failure_logged(self, catalog):
        """Line 659: failure during drop is logged, not raised."""
        # Replace conn with a mock that raises on execute
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = sqlite3.OperationalError("locked")
        catalog.conn = mock_conn
        catalog.drop_legacy_vec_table("some_table")  # should not raise


class TestClusterStateOperations:
    """Cover line 768 (list_cluster_states)."""

    def test_list_cluster_states_empty(self, catalog):
        """Returns empty list when no cluster states exist."""
        result = catalog.list_cluster_states()
        assert result == []

    def test_list_cluster_states_with_data(self, catalog):
        """Returns saved cluster states."""
        catalog.save_cluster_state(
            name="test_cluster",
            algorithm="kmeans",
            n_clusters=5,
            centroids=b"\x00\x01",
            metadata={"inertia": 42.0},
        )
        catalog.save_cluster_state(
            name="another",
            algorithm="hdbscan",
            n_clusters=3,
            centroids=None,
        )

        result = catalog.list_cluster_states()
        assert len(result) == 2
        names = {r["name"] for r in result}
        assert names == {"test_cluster", "another"}

    def test_delete_cluster_state(self, catalog):
        """Delete returns True when found, False otherwise."""
        catalog.save_cluster_state("to_delete", "kmeans", 2, None)
        assert catalog.delete_cluster_state("to_delete") is True
        assert catalog.delete_cluster_state("nonexistent") is False
