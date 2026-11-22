"""Exercise keyed embedding persistence, migration, and SQLite access costs."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from struct import pack
from threading import Barrier
from unittest.mock import Mock

import pytest

from codebase_rag.embedding_cache import KeyedEmbeddingCache


def test_pending_committed_and_precise_vectors(tmp_path: Path) -> None:
    """Observe pending edits locally and committed doubles across cache instances."""
    path = tmp_path / "cache.sqlite"
    writer = KeyedEmbeddingCache(path)
    reader = KeyedEmbeddingCache(path)
    precise = [1.0000000000000002, -0.0, float.fromhex("0x0.0000000000001p-1022")]
    writer.put("key", precise)
    assert writer.get("key") == precise
    assert not path.exists()
    assert reader.get("key") is None
    writer.save()
    result = reader.get("key")
    assert result is not None
    assert pack("<3d", *result) == pack("<3d", *precise)

    writer.put("key", [3.0])
    writer.clear()
    writer.save()
    writer.close()
    assert writer.get("key") == precise
    writer.close()
    reader.close()


def test_get_many_chunks_by_bind_limit_preserving_positions(tmp_path: Path) -> None:
    """Read only unique missing keys from SQLite and preserve duplicate positions."""
    cache = KeyedEmbeddingCache(tmp_path / "cache.sqlite")
    cache.put_many(["d", "a", "b"], [[4.0], [1.0], [2.0]])
    cache.save()
    cache.put("a", [10.0])
    connection = cache._connection
    assert connection is not None
    connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 2)
    statements: list[str] = []
    connection.set_trace_callback(statements.append)
    assert cache.get_many(["d", "missing", "b", "d", "a", "b"]) == {
        0: [4.0],
        2: [2.0],
        3: [4.0],
        4: [10.0],
        5: [2.0],
    }
    assert len(statements) == 2
    assert all("WHERE key IN" in statement for statement in statements)
    assert sum(statement.count("'d'") for statement in statements) == 1
    assert all("'a'" not in statement for statement in statements)
    cache.close()


def test_migration_imports_once_and_preserves_newer_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Commit legacy rows once and leave a previously saved vector intact."""
    path = tmp_path / "cache.sqlite"
    legacy = tmp_path / "cache.json"
    updated = KeyedEmbeddingCache(path)
    updated.put("already-saved", [8.0])
    updated.save()
    updated.close()

    legacy.write_text(json.dumps({"already-saved": [1.0], "legacy": [2.0]}))
    migrated = KeyedEmbeddingCache(path, legacy)
    assert migrated.get_many(["already-saved", "legacy"]) == {0: [8.0], 1: [2.0]}
    migrated.close()

    legacy.write_text(json.dumps({"legacy": [3.0], "added-later": [2.0]}))
    read_text = Mock(side_effect=AssertionError("steady-state open read legacy JSON"))
    monkeypatch.setattr(Path, "read_text", read_text)
    migrated.load()
    assert migrated.get_many(["legacy", "added-later"]) == {0: [2.0]}
    read_text.assert_not_called()
    migrated.close()


def test_legacy_import_revision_is_bound_to_first_import(tmp_path: Path) -> None:
    """Keep aliases eligible only for the revision that imported them."""
    path = tmp_path / "cache.sqlite"
    legacy = tmp_path / "cache.json"
    legacy.write_text('{"alias": [1.0]}')
    first = KeyedEmbeddingCache(path, legacy, legacy_revision="revision-a")
    assert first.legacy_matches("revision-a")
    first.close()

    legacy.write_text('{"alias": [2.0]}')
    second = KeyedEmbeddingCache(path, legacy, legacy_revision="revision-b")
    assert second.get("alias") == [1.0]
    assert not second.legacy_matches("revision-b")
    assert second.legacy_matches("revision-a")
    second.put("revision-b-key", [2.0])
    second.save()
    assert second.get("revision-b-key") == [2.0]
    second.close()


def test_existing_import_without_revision_cannot_be_claimed(tmp_path: Path) -> None:
    """Keep an old migration marker usable without granting revision ownership."""
    path = tmp_path / "cache.sqlite"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA user_version = 1")
    legacy = tmp_path / "cache.json"
    legacy.write_text('{"alias": [1.0]}')
    cache = KeyedEmbeddingCache(path, legacy, legacy_revision="revision-a")
    assert not cache.legacy_matches("revision-a")
    assert cache.get("alias") is None
    cache.close()


@pytest.mark.parametrize(
    "bad_json", ['{"partial": [2.0]', '{"partial": [2.0], "invalid": ["no"]}']
)
def test_malformed_legacy_is_atomic_and_can_be_repaired(
    tmp_path: Path, bad_json: str
) -> None:
    """Roll back a truncated or invalid import and retry after a file repair."""
    legacy = tmp_path / "cache.json"
    legacy.write_text(bad_json)
    cache = KeyedEmbeddingCache(
        tmp_path / "cache.sqlite", legacy, legacy_revision="revision-a"
    )
    cache.load()
    assert cache.get("partial") is None
    assert not cache.legacy_matches("revision-a")
    cache.put("fresh", [7.0])
    cache.save()
    cache.close()

    legacy.write_text('{"partial": [2.0]}')
    assert cache.get_many(["fresh", "partial"]) == {0: [7.0], 1: [2.0]}
    assert cache.legacy_matches("revision-a")
    cache.close()


def test_concurrent_cache_instances_merge_writes(tmp_path: Path) -> None:
    """Allow multiple first openers to import and save different keys concurrently."""
    path = tmp_path / "cache.sqlite"
    legacy = tmp_path / "cache.json"
    legacy.write_text('{"from-json": [12.0]}')
    ready = Barrier(6)

    def save_from_worker(index: int) -> None:
        """Write one distinct key from a separately opened cache instance."""
        cache = KeyedEmbeddingCache(path, legacy)
        cache.put(str(index), [float(index)])
        ready.wait(timeout=10)
        cache.save()
        cache.close()

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(save_from_worker, range(6)))

    cache = KeyedEmbeddingCache(path, legacy)
    assert cache.get_many([*(str(index) for index in range(6)), "from-json"]) == {
        index: [value]
        for index, value in enumerate([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 12.0])
    }
    cache.close()


def test_failed_save_rolls_back_and_can_retry(tmp_path: Path) -> None:
    """Keep unsaved values for retry following a failure partway through a save."""
    path = tmp_path / "cache.sqlite"
    writer = KeyedEmbeddingCache(path)
    writer.load()
    with closing(sqlite3.connect(path)) as db:
        db.execute("""
            CREATE TRIGGER block_write BEFORE INSERT ON embeddings
            WHEN NEW.key = 'blocked'
            BEGIN SELECT RAISE(ABORT, 'blocked for test'); END
            """)
        writer.put_many(["first", "blocked"], [[1.0], [2.0]])
        writer.save()
        assert db.execute("SELECT key FROM embeddings").fetchall() == []
        assert writer.get_many(["first", "blocked"]) == {0: [1.0], 1: [2.0]}
        db.execute("DROP TRIGGER block_write")
        writer.save()
        assert db.execute("SELECT key FROM embeddings ORDER BY key").fetchall() == [
            ("blocked",),
            ("first",),
        ]
    writer.close()


def test_noop_save_and_open_failure_recovery(tmp_path: Path) -> None:
    """Keep an unused cache absent and retry a blocked save after repair."""
    parent = tmp_path / "blocked"
    path = parent / "cache.sqlite"
    writer = KeyedEmbeddingCache(path)
    writer.save()
    assert writer.get_many([]) == {}
    assert not parent.exists()

    parent.write_text("prevent directory creation")
    assert writer.get("key") is None
    writer.put("key", [1.0])
    writer.save()
    assert writer.get("key") == [1.0]
    parent.unlink()
    writer.save()
    writer.close()
    assert writer.get("key") == [1.0]
    writer.close()


def test_lookup_and_save_work_stay_local_as_unrelated_rows_grow(tmp_path: Path) -> None:
    """Count SQLite VM steps and writes with a fixed request and growing cache."""
    path = tmp_path / "cache.sqlite"
    writer = KeyedEmbeddingCache(path)
    writer.put("target", [1.0, 2.0])
    writer.save()
    connection = writer._connection
    assert connection is not None

    def measure_lookup() -> int:
        """Measure an actual lookup and assert SQLite seeks via the primary key."""
        steps = 0
        statements: list[str] = []

        def count_step() -> int:
            """Count one instruction executed by the SQLite VM."""
            nonlocal steps
            steps += 1
            return 0

        connection.set_progress_handler(count_step, 1)
        connection.set_trace_callback(statements.append)
        assert writer.get_many(["target", "absent", "target"]) == {
            0: [1.0, 2.0],
            2: [1.0, 2.0],
        }
        connection.set_progress_handler(None, 0)
        connection.set_trace_callback(None)
        assert len(statements) == 1
        plan = connection.execute(f"EXPLAIN QUERY PLAN {statements[0]}").fetchall()
        assert any("USING PRIMARY KEY" in detail for _, _, _, detail in plan)
        return steps

    before = measure_lookup()
    writer.put_many([f"other:{i}" for i in range(20_000)], [[0.0]] * 20_000)
    writer.save()
    # A missing key may take a different branch in the SQLite seek loop.
    assert abs(measure_lookup() - before) <= 2

    writes_before = connection.total_changes
    writer.put("target", [1.0, 2.0])
    writer.save()
    writer.put("another", [3.0])
    writer.save()
    writer.save()
    assert connection.total_changes - writes_before == 1
    writer.close()
