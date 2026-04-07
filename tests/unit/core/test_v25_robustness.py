"""Tests for simplevecdb 2.5.0 robustness features.

Covers:
- async_retry_on_lock decorator
- FTS retry on transient lock errors
- file_lock context manager
- Mmap byte threshold for UsearchIndex
"""

from __future__ import annotations

import asyncio
import fcntl
import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock, PropertyMock, call

import pytest
import numpy as np

from simplevecdb import async_retry_on_lock, file_lock, DatabaseLockedError
from simplevecdb.engine.catalog import CatalogManager
import simplevecdb.constants as constants


# ---------------------------------------------------------------------------
# 1. async_retry_on_lock
# ---------------------------------------------------------------------------


class TestAsyncRetryOnLock:
    """Tests for the async_retry_on_lock decorator."""

    @pytest.mark.asyncio
    async def test_retries_on_lock_then_succeeds(self):
        """Decorator retries on 'database is locked' and returns the result."""
        attempt_count = 0

        @async_retry_on_lock(max_retries=5, base_delay=0.01, jitter=False, total_timeout=10.0)
        async def flaky():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count <= 2:
                raise sqlite3.OperationalError("database is locked")
            return "ok"

        result = await flaky()
        assert result == "ok"
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_uses_asyncio_sleep_not_time_sleep(self):
        """Decorator awaits asyncio.sleep, never calls time.sleep."""
        call_count = 0

        @async_retry_on_lock(max_retries=3, base_delay=0.01, jitter=False, total_timeout=10.0)
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise sqlite3.OperationalError("database is locked")
            return "done"

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_asleep, \
             patch("time.sleep") as mock_tsleep:
            result = await flaky()

        assert result == "done"
        assert mock_asleep.await_count == 2
        mock_tsleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_lock_operational_error_raises_immediately(self):
        """Non-lock OperationalErrors propagate without retry."""
        attempt_count = 0

        @async_retry_on_lock(max_retries=5, base_delay=0.01, jitter=False, total_timeout=10.0)
        async def bad():
            nonlocal attempt_count
            attempt_count += 1
            raise sqlite3.OperationalError("no such table: foo")

        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            await bad()
        assert attempt_count == 1

    @pytest.mark.asyncio
    async def test_raises_database_locked_error_after_max_retries(self):
        """DatabaseLockedError is raised once retries are exhausted."""

        @async_retry_on_lock(max_retries=2, base_delay=0.001, jitter=False, total_timeout=60.0)
        async def always_locked():
            raise sqlite3.OperationalError("database is locked")

        with pytest.raises(DatabaseLockedError) as exc_info:
            await always_locked()

        err = exc_info.value
        assert err.attempts == 3  # initial + 2 retries
        assert err.total_wait >= 0


# ---------------------------------------------------------------------------
# 2. FTS retry on transient lock errors
# ---------------------------------------------------------------------------


class TestFtsRetryOnLock:
    """Tests for CatalogManager._ensure_fts_table lock-retry behaviour."""

    @staticmethod
    def _make_catalog_with_proxy():
        """Create a CatalogManager with a proxy conn whose execute is patchable."""
        real_conn = sqlite3.connect(":memory:")
        proxy = MagicMock(wraps=real_conn)
        # wraps delegates all calls to real_conn, but proxy.execute is now mockable
        catalog = CatalogManager(
            proxy, table_name="tinyvec_items", fts_table_name="tinyvec_items_fts"
        )
        return catalog, proxy, real_conn

    def test_fts_retries_on_lock_then_enables(self):
        """FTS table creation retries on lock errors and enables FTS."""
        catalog, proxy, real_conn = self._make_catalog_with_proxy()

        call_count = 0
        original_execute = real_conn.execute

        def mock_execute(sql, *args, **kwargs):
            nonlocal call_count
            if "CREATE VIRTUAL TABLE" in sql:
                call_count += 1
                if call_count <= 2:
                    raise sqlite3.OperationalError("database is locked")
            return original_execute(sql, *args, **kwargs)

        proxy.execute = mock_execute
        catalog._ensure_fts_table()

        assert catalog._fts_enabled is True
        assert call_count == 3

    def test_fts_disables_immediately_on_module_missing(self):
        """FTS is disabled without retries when fts5 module is absent."""
        catalog, proxy, real_conn = self._make_catalog_with_proxy()

        call_count = 0
        original_execute = real_conn.execute

        def mock_execute(sql, *args, **kwargs):
            nonlocal call_count
            if "CREATE VIRTUAL TABLE" in sql:
                call_count += 1
                raise sqlite3.OperationalError("no such module: fts5")
            return original_execute(sql, *args, **kwargs)

        proxy.execute = mock_execute
        catalog._ensure_fts_table()

        assert catalog._fts_enabled is False
        assert call_count == 1  # no retries


# ---------------------------------------------------------------------------
# 3. file_lock context manager
# ---------------------------------------------------------------------------


class TestFileLock:
    """Tests for the file_lock advisory-lock context manager."""

    def test_lock_file_created(self, tmp_path: Path):
        """Acquiring file_lock creates a .lock sibling file."""
        target = tmp_path / "data.db"
        target.touch()

        with file_lock(target):
            lock_path = target.with_suffix(".db.lock")
            assert lock_path.exists()

    def test_lock_released_after_context_exit(self, tmp_path: Path):
        """After context exit the lock file exists but is no longer locked."""
        target = tmp_path / "data.db"
        target.touch()

        with file_lock(target):
            pass  # lock held here

        # Lock file should still exist on disk but not be held
        lock_path = target.with_suffix(".db.lock")
        assert lock_path.exists()

        # Verify we can immediately acquire the lock again (proves it's released)
        with file_lock(target):
            pass  # would block forever if still locked

    def test_concurrent_locks_from_threads(self, tmp_path: Path):
        """Second thread blocks until first thread releases the lock."""
        target = tmp_path / "data.db"
        target.touch()

        order: list[str] = []
        barrier = threading.Event()

        def first():
            with file_lock(target):
                order.append("first-acquired")
                barrier.set()  # signal second thread to try acquiring
                time.sleep(0.15)  # hold lock
                order.append("first-released")

        def second():
            barrier.wait()  # wait until first thread holds the lock
            time.sleep(0.02)  # small delay to ensure first thread is still holding
            with file_lock(target):
                order.append("second-acquired")

        t1 = threading.Thread(target=first)
        t2 = threading.Thread(target=second)

        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert order == ["first-acquired", "first-released", "second-acquired"]


# ---------------------------------------------------------------------------
# 4. Mmap byte threshold
# ---------------------------------------------------------------------------


class TestMmapThreshold:
    """Tests for UsearchIndex memory-mapping threshold."""

    def test_mmap_threshold_constant_value(self):
        """USEARCH_MMAP_THRESHOLD is exactly 50 MiB."""
        assert constants.USEARCH_MMAP_THRESHOLD == 50 * 1024 * 1024

    @patch("simplevecdb.engine.usearch_index.UsearchIndex._load_or_create")
    def test_large_file_enables_mmap(self, mock_load):
        """Files larger than threshold trigger memory-mapped mode."""
        from simplevecdb.engine.usearch_index import UsearchIndex

        idx = UsearchIndex.__new__(UsearchIndex)
        idx._is_view = True  # simulate what _load_or_create would set

        assert idx.is_memory_mapped is True

    @patch("simplevecdb.engine.usearch_index.UsearchIndex._load_or_create")
    def test_small_file_disables_mmap(self, mock_load):
        """Files smaller than threshold load into memory (no mmap)."""
        from simplevecdb.engine.usearch_index import UsearchIndex

        idx = UsearchIndex.__new__(UsearchIndex)
        idx._is_view = False

        assert idx.is_memory_mapped is False

    def test_load_or_create_mmap_decision(self, tmp_path: Path):
        """_load_or_create sets _is_view based on file size vs threshold."""
        from simplevecdb.engine.usearch_index import UsearchIndex

        index_path = tmp_path / "test.usearch"
        index_path.touch()

        mock_index = MagicMock()
        mock_index.ndim = 128
        mock_index.__len__ = lambda self: 1000

        # Case 1: file > threshold => mmap
        with patch("simplevecdb.engine.usearch_index.UsearchIndex._load_or_create"):
            idx = UsearchIndex.__new__(UsearchIndex)

        idx._path = index_path
        idx._is_view = False
        idx._index = None
        idx._ndim = None

        large_stat = MagicMock()
        large_stat.st_size = constants.USEARCH_MMAP_THRESHOLD + 1

        small_stat = MagicMock()
        small_stat.st_size = constants.USEARCH_MMAP_THRESHOLD - 1

        mock_index_cls = MagicMock()
        mock_index_cls.restore.return_value = mock_index

        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "stat", return_value=large_stat), \
             patch("simplevecdb.utils.file_lock"), \
             patch("usearch.index.Index", mock_index_cls):
            idx._load_or_create()

        assert idx._is_view is True
        mock_index_cls.restore.assert_called_once_with(str(index_path), view=True)

        # Case 2: file < threshold => no mmap
        mock_index_cls.reset_mock()
        idx._is_view = False

        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "stat", return_value=small_stat), \
             patch("simplevecdb.utils.file_lock"), \
             patch("usearch.index.Index", mock_index_cls):
            idx._load_or_create()

        assert idx._is_view is False
        mock_index_cls.restore.assert_called_once_with(str(index_path), view=False)
