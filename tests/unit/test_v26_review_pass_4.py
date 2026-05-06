"""Regression tests for the fourth 2.6.0 review pass (codex P1 findings).

Pins two invariants that had been broken by earlier passes:

- Pre-2.6.0 SQLCipher passphrase-mode databases (no ``.salt`` sidecar)
  must keep opening with the same passphrase under 2.6.0+. Earlier
  passes routed every key through PBKDF2 → ``x'hex'`` raw-key form,
  which yielded a different SQLCipher internal key than the
  passphrase-derived one originally written, leaving such databases
  unopenable.
- ``file_lock`` must NOT unlink the ``.lock`` sidecar after releasing
  the flock, because flock is inode-bound — removing the path while
  another process is still queued on flock(fd) on the old inode lets
  a third process create a new path → new inode → acquire a different
  lock concurrently, defeating cross-process mutual exclusion for
  index save/rebuild.
"""

from __future__ import annotations

import os
import threading

import pytest

from simplevecdb.utils import file_lock


class TestLockFileSurvivesContextExit:
    def test_lock_path_persists_after_release(self, tmp_path):
        target = tmp_path / "idx.usearch"
        target.write_bytes(b"")
        lock_path = target.with_name(target.name + ".lock")
        with file_lock(target):
            assert lock_path.exists()
        # The lock file must remain on disk after the context exits, so
        # any process still queued on flock against its inode cannot
        # have a third process race past it via a fresh inode.
        assert lock_path.exists(), (
            "file_lock must not unlink the .lock sidecar at context exit; "
            "doing so breaks inode-bound cross-process mutual exclusion."
        )

    def test_inode_stable_across_two_acquisitions(self, tmp_path):
        """Two sequential file_lock acquisitions on the same target must
        operate on the same lock-file inode, not a fresh one."""
        target = tmp_path / "idx.usearch"
        target.write_bytes(b"")
        lock_path = target.with_name(target.name + ".lock")

        with file_lock(target):
            inode_first = os.stat(lock_path).st_ino
        with file_lock(target):
            inode_second = os.stat(lock_path).st_ino

        assert inode_first == inode_second, (
            "Lock file inode changed between acquisitions — "
            "lock_path was unlinked between them, which breaks "
            "cross-process mutual exclusion."
        )

    def test_concurrent_acquisitions_serialize(self, tmp_path):
        """Two threads acquiring the same lock must run serially."""
        target = tmp_path / "idx.usearch"
        target.write_bytes(b"")
        in_critical = threading.Event()
        seen_concurrent = threading.Event()
        result_lock = threading.Lock()
        sequence: list[str] = []

        def worker(tag: str):
            with file_lock(target):
                with result_lock:
                    sequence.append(f"enter:{tag}")
                if not in_critical.is_set():
                    in_critical.set()
                else:
                    seen_concurrent.set()
                # Tiny pause to give the other thread a chance to race in.
                threading.Event().wait(0.05)
                with result_lock:
                    sequence.append(f"exit:{tag}")
                in_critical.clear()

        t1 = threading.Thread(target=worker, args=("a",))
        t2 = threading.Thread(target=worker, args=("b",))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert not seen_concurrent.is_set(), (
            "Both threads observed each other inside the critical "
            "section — file_lock failed to serialize them."
        )
        assert sequence[0].startswith("enter:")
        assert sequence[1].startswith("exit:")
        assert sequence[2].startswith("enter:")
        assert sequence[3].startswith("exit:")


class TestLegacyPassphraseDBStillOpens:
    """A pre-2.6 encrypted SQLCipher database (no .salt sidecar) must
    keep opening with the same passphrase after upgrade."""

    @pytest.fixture
    def sqlcipher_module(self):
        return pytest.importorskip("sqlcipher3")

    def test_pre_2_6_passphrase_db_opens_under_2_6(
        self, tmp_path, sqlcipher_module
    ):
        # Step 1: create a SQLCipher DB the pre-2.6 way — passphrase
        # PRAGMA, SQLCipher does its own internal KDF, no sidecar.
        db_path = tmp_path / "legacy.db"
        passphrase = "correct horse battery staple"

        sqlcipher = sqlcipher_module.dbapi2  # type: ignore[attr-defined]
        legacy = sqlcipher.connect(str(db_path))
        legacy.execute(f"PRAGMA key = '{passphrase}'")
        legacy.execute("CREATE TABLE secrets (id INTEGER PRIMARY KEY, val TEXT)")
        legacy.execute("INSERT INTO secrets(val) VALUES ('hello')")
        legacy.commit()
        legacy.close()

        # Sanity: no sidecar was created — this is genuinely a v2.5-shaped DB.
        sidecar = db_path.with_name(db_path.name + ".salt")
        assert not sidecar.exists()

        # Step 2: open it via the 2.6 factory using the same passphrase.
        from simplevecdb.encryption import create_encrypted_connection

        conn = create_encrypted_connection(db_path, passphrase)
        try:
            row = conn.execute(
                "SELECT val FROM secrets WHERE id=1"
            ).fetchone()
            assert row is not None
            assert row[0] == "hello"
        finally:
            conn.close()

        # Step 3: confirm we did NOT silently create a sidecar (which
        # would imply we mistook this DB for a fresh one and would have
        # written a wrong-key derivation under the new path).
        assert not sidecar.exists(), (
            "Legacy passphrase DB must keep using SQLCipher's internal "
            "KDF; a sidecar was incorrectly created."
        )

    def test_post_2_6_db_still_uses_raw_key_path(
        self, tmp_path, sqlcipher_module
    ):
        """Brand-new DBs written under 2.6+ must use the new raw-key
        path with a sidecar; reopening must still work."""
        from simplevecdb.encryption import create_encrypted_connection

        db_path = tmp_path / "fresh.db"
        passphrase = "another secret"

        # Fresh DB — uses the new path and writes a sidecar.
        conn = create_encrypted_connection(db_path, passphrase)
        try:
            conn.execute("CREATE TABLE t (x INTEGER)")
            conn.execute("INSERT INTO t VALUES (1)")
            conn.commit()
        finally:
            conn.close()

        sidecar = db_path.with_name(db_path.name + ".salt")
        assert sidecar.exists(), "Brand-new DB must get a .salt sidecar"

        # Reopen — must still use the new path and decrypt successfully.
        conn = create_encrypted_connection(db_path, passphrase)
        try:
            row = conn.execute("SELECT x FROM t").fetchone()
            assert row[0] == 1
        finally:
            conn.close()
