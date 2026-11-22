"""Persist embedding vectors under keys derived by the embedding provider."""

from __future__ import annotations

import json
import sqlite3
import struct
from itertools import batched
from threading import Lock
from typing import TYPE_CHECKING

from loguru import logger

from . import logs as ls

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


_WRITE_VECTORS = """
    INSERT INTO embeddings (key, vector) VALUES (?, ?)
    ON CONFLICT(key) DO UPDATE SET vector = excluded.vector
    WHERE embeddings.vector != excluded.vector
"""
_IMPORT_VECTORS = """
    INSERT INTO embeddings (key, vector) VALUES (?, ?)
    ON CONFLICT(key) DO NOTHING
"""
_LOOKUP_BATCH_SIZE = 512


def _encode(vector: Sequence[float]) -> bytes:
    """Store IEEE 754 doubles in a platform-independent byte order."""
    return struct.pack(f"<{len(vector)}d", *vector)


class KeyedEmbeddingCache:
    """Keep pending writes in memory and read saved vectors from SQLite by key.

    Separate cache instances share committed changes through SQLite. An instance
    serializes access to its connection and pending writes across its callers.
    """

    def __init__(
        self,
        path: Path,
        legacy_path: Path | None = None,
        *,
        legacy_revision: str | None = None,
    ) -> None:
        """Configure the database and optional JSON source with its known revision."""
        self._path = path
        self._legacy_path = legacy_path
        self._legacy_revision = legacy_revision
        self._imported_legacy_revision: str | None = None
        self._connection: sqlite3.Connection | None = None
        self._pending: dict[str, bytes] = {}
        self._lock = Lock()

    def _open(self) -> sqlite3.Connection:
        """Open and initialize the database while the caller holds the cache lock."""
        if self._connection is not None:
            return self._connection
        self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._path, check_same_thread=False)
        try:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    key TEXT PRIMARY KEY,
                    vector BLOB NOT NULL
                ) WITHOUT ROWID
                """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS legacy_import (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    revision TEXT NOT NULL
                )
                """)
            if self._legacy_path is not None:
                self._import_legacy(connection, self._legacy_path)
            row = connection.execute(
                "SELECT revision FROM legacy_import WHERE id = 1"
            ).fetchone()
            self._imported_legacy_revision = row[0] if row is not None else None
        except sqlite3.Error:
            connection.close()
            raise
        self._connection = connection
        return connection

    def _import_legacy(self, connection: sqlite3.Connection, path: Path) -> None:
        """Import valid legacy vectors once, giving existing rows precedence.

        An invalid or unreadable JSON file leaves the import eligible for retry
        when the cache is reopened. A missing file completes the import.
        """
        if connection.execute("PRAGMA user_version").fetchone()[0]:
            return
        try:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                # A concurrent opener may have finished the import while we waited.
                if connection.execute("PRAGMA user_version").fetchone()[0]:
                    return
                try:
                    legacy = json.loads(path.read_text(encoding="utf-8"))
                except FileNotFoundError:
                    legacy = {}
                if not isinstance(legacy, dict):
                    raise ValueError("Expected a JSON object keyed by embedding hash")
                for key, vector in legacy.items():
                    if not isinstance(vector, list) or not all(
                        type(value) in (int, float) for value in vector
                    ):
                        raise ValueError("Expected an array of embedding numbers")
                    connection.execute(_IMPORT_VECTORS, (key, _encode(vector)))
                if legacy and self._legacy_revision:
                    connection.execute(
                        "INSERT INTO legacy_import (id, revision) VALUES (1, ?)",
                        (self._legacy_revision,),
                    )
                connection.execute("PRAGMA user_version = 1")
        except (
            OSError,
            ValueError,
            OverflowError,
            RecursionError,
            struct.error,
        ) as error:
            logger.warning(ls.EMBEDDING_CACHE_LOAD_FAILED, path=path, error=error)

    def load(self) -> None:
        """Open the cache and attempt migration without loading the saved vectors."""
        with self._lock:
            try:
                self._open()
            except (OSError, sqlite3.Error) as error:
                logger.warning(
                    ls.EMBEDDING_CACHE_LOAD_FAILED, path=self._path, error=error
                )

    def legacy_matches(self, revision: str) -> bool:
        """Authorize legacy aliases only for the exact revision recorded at import."""
        with self._lock:
            try:
                self._open()
            except (OSError, sqlite3.Error) as error:
                logger.warning(
                    ls.EMBEDDING_CACHE_LOAD_FAILED, path=self._path, error=error
                )
                return False
            return bool(revision) and self._imported_legacy_revision == revision

    def get(self, key: str) -> list[float] | None:
        """Return the pending or saved vector for a key, or a cache miss."""
        return self.get_many((key,)).get(0)

    def get_many(self, keys: Sequence[str]) -> dict[int, list[float]]:
        """Return pending and saved vectors by input position, including duplicates."""
        with self._lock:
            unique_keys = set(keys)
            hits = {
                key: self._pending[key] for key in unique_keys if key in self._pending
            }
            remaining = unique_keys - hits.keys()
            if remaining:
                try:
                    connection = self._open()
                    limit = connection.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
                    for batch in batched(remaining, min(_LOOKUP_BATCH_SIZE, limit)):
                        placeholders = ",".join("?" for _ in batch)
                        # Only the placeholder count is interpolated. Keys are bound.
                        rows = connection.execute(
                            f"SELECT key, vector FROM embeddings WHERE key IN ({placeholders})",  # noqa: S608
                            batch,
                        )
                        hits.update(rows)
                except (OSError, sqlite3.Error) as error:
                    logger.warning(
                        ls.EMBEDDING_CACHE_LOAD_FAILED, path=self._path, error=error
                    )
            vectors = {
                key: [value for (value,) in struct.iter_unpack("<d", vector)]
                for key, vector in hits.items()
            }
            return {i: vectors[key] for i, key in enumerate(keys) if key in vectors}

    def put(self, key: str, vector: Sequence[float]) -> None:
        """Stage one vector for the next save and make it visible to this instance."""
        with self._lock:
            self._pending[key] = _encode(vector)

    def put_many(self, keys: Sequence[str], vectors: Sequence[Sequence[float]]) -> None:
        """Stage matching key/vector pairs, taking the last value for duplicate keys."""
        with self._lock:
            self._pending.update(
                {
                    key: _encode(vector)
                    for key, vector in zip(keys, vectors, strict=False)
                }
            )

    def save(self) -> None:
        """Commit pending vectors together, retaining them if the write fails."""
        with self._lock:
            if not self._pending:
                return
            try:
                connection = self._open()
                with connection:
                    connection.executemany(_WRITE_VECTORS, self._pending.items())
            except (OSError, sqlite3.Error) as error:
                logger.warning(
                    ls.EMBEDDING_CACHE_SAVE_FAILED, path=self._path, error=error
                )
            else:
                self._pending.clear()

    def clear(self) -> None:
        """Discard unsaved writes while leaving committed vectors available."""
        with self._lock:
            self._pending.clear()

    def close(self) -> None:
        """Release the SQLite connection without committing pending vectors."""
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
