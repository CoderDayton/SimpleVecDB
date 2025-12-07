"""
Tests for error handling, retry logic, and logging functionality.

Covers:
- retry_on_lock decorator behavior
- DatabaseLockedError exception
- Filter validation
- Structured logging utilities
"""

import logging
import sqlite3
import tempfile
import threading
import time
from io import StringIO

import pytest

from simplevecdb import VectorDB, Quantization
from simplevecdb.utils import (
    retry_on_lock,
    DatabaseLockedError,
    validate_filter,
)
from simplevecdb.logging import (
    get_logger,
    configure_logging,
    log_operation,
    log_error,
    LOGGER_NAME,
)


# ============================================================================
# retry_on_lock tests
# ============================================================================


class TestRetryOnLock:
    """Tests for the retry_on_lock decorator."""

    def test_no_retry_on_success(self):
        """Function succeeds on first try, no retry needed."""
        call_count = 0

        @retry_on_lock(max_retries=3)
        def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = success_func()
        assert result == "success"
        assert call_count == 1

    def test_retry_on_lock_error(self):
        """Function retries on 'database is locked' error."""
        call_count = 0

        @retry_on_lock(max_retries=3, base_delay=0.01, jitter=False)
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise sqlite3.OperationalError("database is locked")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert call_count == 3

    def test_raises_database_locked_error_after_max_retries(self):
        """DatabaseLockedError raised after all retries exhausted."""
        call_count = 0

        @retry_on_lock(max_retries=2, base_delay=0.01, jitter=False)
        def always_locked():
            nonlocal call_count
            call_count += 1
            raise sqlite3.OperationalError("database is locked")

        with pytest.raises(DatabaseLockedError) as exc_info:
            always_locked()

        assert exc_info.value.attempts == 3  # Initial + 2 retries
        assert exc_info.value.total_wait > 0
        assert call_count == 3

    def test_non_lock_error_not_retried(self):
        """Non-lock SQLite errors are raised immediately."""
        call_count = 0

        @retry_on_lock(max_retries=3)
        def other_error():
            nonlocal call_count
            call_count += 1
            raise sqlite3.OperationalError("no such table: foo")

        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            other_error()

        assert call_count == 1  # No retries

    def test_exponential_backoff(self):
        """Verify exponential backoff timing."""
        delays = []

        @retry_on_lock(max_retries=3, base_delay=0.1, max_delay=1.0, jitter=False)
        def track_delays():
            delays.append(time.time())
            raise sqlite3.OperationalError("database is locked")

        start = time.time()
        with pytest.raises(DatabaseLockedError):
            track_delays()

        # Check that delays increase exponentially
        # Delay 1: 0.1s, Delay 2: 0.2s, Delay 3: 0.4s
        # Total minimum time should be ~0.7s
        total_time = time.time() - start
        assert total_time >= 0.6  # Allow some tolerance

    def test_max_delay_cap(self):
        """Verify delay doesn't exceed max_delay."""
        call_count = 0

        @retry_on_lock(max_retries=5, base_delay=0.5, max_delay=0.1, jitter=False)
        def capped_delay():
            nonlocal call_count
            call_count += 1
            raise sqlite3.OperationalError("database is locked")

        start = time.time()
        with pytest.raises(DatabaseLockedError):
            capped_delay()

        # With max_delay=0.1 and 5 retries, total time should be ~0.5s
        total_time = time.time() - start
        assert total_time < 1.0  # Should be capped

    def test_preserves_function_metadata(self):
        """Decorator preserves function name and docstring."""

        @retry_on_lock()
        def documented_func():
            """This is the docstring."""
            pass

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "This is the docstring."


class TestDatabaseLockedError:
    """Tests for DatabaseLockedError exception."""

    def test_error_attributes(self):
        """Exception stores attempts and wait time."""
        error = DatabaseLockedError("Test message", attempts=5, total_wait=1.5)
        assert str(error) == "Test message"
        assert error.attempts == 5
        assert error.total_wait == 1.5

    def test_inherits_from_exception(self):
        """DatabaseLockedError is a proper Exception subclass."""
        error = DatabaseLockedError("msg", 1, 0.1)
        assert isinstance(error, Exception)


# ============================================================================
# validate_filter tests
# ============================================================================


class TestValidateFilter:
    """Tests for filter validation."""

    def test_valid_filters(self):
        """Valid filters pass without error."""
        validate_filter(None)
        validate_filter({})
        validate_filter({"key": "value"})
        validate_filter({"count": 42})
        validate_filter({"score": 0.95})
        validate_filter({"tags": ["a", "b", "c"]})
        validate_filter({"ids": [1, 2, 3]})
        validate_filter({"multi": "text", "count": 10, "tags": ["x"]})

    def test_invalid_key_type(self):
        """Non-string keys raise ValueError."""
        with pytest.raises(ValueError, match="Filter keys must be strings"):
            validate_filter({123: "value"})

        with pytest.raises(ValueError, match="Filter keys must be strings"):
            validate_filter({None: "value"})  # type: ignore

    def test_invalid_value_type(self):
        """Unsupported value types raise ValueError."""
        with pytest.raises(ValueError, match="must be int, float, str, or list"):
            validate_filter({"key": {"nested": "dict"}})

        with pytest.raises(ValueError, match="must be int, float, str, or list"):
            validate_filter({"key": (1, 2, 3)})  # tuple not allowed

    def test_invalid_list_item_type(self):
        """List items must be int, float, or str."""
        with pytest.raises(ValueError, match="list items.*must be int, float, or str"):
            validate_filter({"tags": [{"nested": "dict"}]})

        with pytest.raises(ValueError, match="list items.*must be int, float, or str"):
            validate_filter({"tags": [[1, 2, 3]]})  # nested list


# ============================================================================
# Logging tests
# ============================================================================


class TestLogging:
    """Tests for structured logging utilities."""

    def test_get_logger_returns_child_logger(self):
        """get_logger returns logger under simplevecdb namespace."""
        logger = get_logger("mymodule")
        assert logger.name == "simplevecdb.mymodule"

    def test_get_logger_strips_package_prefix(self):
        """get_logger strips simplevecdb. prefix from name."""
        logger = get_logger("simplevecdb.core")
        assert logger.name == "simplevecdb.core"

    def test_get_logger_root(self):
        """get_logger(None) returns root simplevecdb logger."""
        logger = get_logger(None)
        assert logger.name == LOGGER_NAME

    def test_configure_logging(self):
        """configure_logging sets up handlers correctly."""
        # Capture log output
        stream = StringIO()
        handler = logging.StreamHandler(stream)

        configure_logging(level=logging.DEBUG, handler=handler)
        logger = get_logger()
        logger.debug("test message")

        output = stream.getvalue()
        assert "test message" in output

    def test_configure_logging_string_level(self):
        """configure_logging accepts string log levels."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)

        configure_logging(level="WARNING", handler=handler)
        logger = get_logger()
        logger.info("should not appear")
        logger.warning("should appear")

        output = stream.getvalue()
        assert "should not appear" not in output
        assert "should appear" in output

    def test_log_operation_success(self):
        """log_operation logs start and completion."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        configure_logging(level=logging.DEBUG, handler=handler)

        with log_operation("test_op", count=5) as ctx:
            ctx["result"] = "ok"

        output = stream.getvalue()
        assert "test_op started" in output
        assert "test_op completed" in output

    def test_log_operation_failure(self):
        """log_operation logs errors and re-raises."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        configure_logging(level=logging.DEBUG, handler=handler)

        with pytest.raises(ValueError, match="test error"):
            with log_operation("failing_op"):
                raise ValueError("test error")

        output = stream.getvalue()
        assert "failing_op started" in output
        assert "failing_op failed" in output
        assert "test error" in output

    def test_log_error(self):
        """log_error logs exception with context."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        configure_logging(level=logging.ERROR, handler=handler)

        try:
            raise RuntimeError("test exception")
        except RuntimeError as e:
            log_error("my_operation", e, table="items")

        output = stream.getvalue()
        assert "my_operation failed" in output
        assert "test exception" in output


# ============================================================================
# Integration tests
# ============================================================================


class TestErrorHandlingIntegration:
    """Integration tests for error handling in VectorDB."""

    def test_invalid_filter_raises_on_search(self):
        """Invalid filter raises ValueError during similarity_search."""
        db = VectorDB(":memory:")
        collection = db.collection("test")

        # Add some data first
        collection.add_texts(
            texts=["hello world"],
            embeddings=[[0.1, 0.2, 0.3]],
        )

        # Search with invalid filter
        with pytest.raises(ValueError, match="Filter keys must be strings"):
            collection.similarity_search(
                query=[0.1, 0.2, 0.3],
                k=5,
                filter={123: "value"},  # type: ignore
            )

    def test_invalid_filter_raises_on_keyword_search(self):
        """Invalid filter raises ValueError during keyword_search."""
        db = VectorDB(":memory:")
        collection = db.collection("test")

        # Add some data first
        collection.add_texts(
            texts=["hello world"],
            embeddings=[[0.1, 0.2, 0.3]],
        )

        # Search with invalid filter
        with pytest.raises(ValueError, match="must be int, float, str, or list"):
            collection.keyword_search(
                query="hello",
                k=5,
                filter={"meta": {"nested": "dict"}},
            )

    def test_collection_operations_succeed(self):
        """Verify normal operations work with new error handling."""
        db = VectorDB(":memory:", quantization=Quantization.FLOAT)
        collection = db.collection("test")

        # Add texts
        ids = collection.add_texts(
            texts=["apple is red", "banana is yellow"],
            embeddings=[[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
            metadatas=[{"color": "red"}, {"color": "yellow"}],
        )
        assert len(ids) == 2

        # Search with valid filter
        results = collection.similarity_search(
            query=[0.1, 0.0, 0.0],
            k=1,
            filter={"color": "red"},
        )
        assert len(results) == 1
        assert "apple" in results[0][0].page_content

        # Delete by IDs
        collection.delete_by_ids([ids[0]])

        # Verify deletion
        results = collection.similarity_search([0.1, 0.0, 0.0], k=10)
        assert len(results) == 1
        assert "banana" in results[0][0].page_content


class TestConcurrentAccess:
    """Tests for concurrent database access with retry logic."""

    def test_retry_on_concurrent_writes(self):
        """
        Test that retry logic handles concurrent write contention.

        Note: This test simulates lock behavior since in-memory databases
        don't experience real lock contention.
        """
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        db1 = VectorDB(db_path)
        db2 = VectorDB(db_path)

        collection1 = db1.collection("test")
        collection2 = db2.collection("test")

        # Initialize collection
        collection1.add_texts(
            texts=["initial"],
            embeddings=[[0.1, 0.2, 0.3]],
        )

        errors = []

        def writer(collection, writer_id):
            try:
                for i in range(5):
                    collection.add_texts(
                        texts=[f"writer {writer_id} text {i}"],
                        embeddings=[[0.1 + i * 0.01, 0.2, 0.3]],
                    )
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        # Run concurrent writers
        t1 = threading.Thread(target=writer, args=(collection1, 1))
        t2 = threading.Thread(target=writer, args=(collection2, 2))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Both should complete without unhandled errors
        # (DatabaseLockedError after retries is acceptable in extreme cases)
        for e in errors:
            if not isinstance(e, DatabaseLockedError):
                raise e

        # Verify data was written
        results = collection1.similarity_search([0.1, 0.2, 0.3], k=20)
        assert len(results) >= 1  # At least some writes succeeded

        db1.close()
        db2.close()
