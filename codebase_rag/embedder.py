from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from loguru import logger

from . import constants as cs
from . import exceptions as ex
from . import logs as ls
from .config import API_KEY_INFO, normalised_credential, settings
from .embedding_cache import KeyedEmbeddingCache
from .embedding_identity import (
    effective_max_length,
    embedding_namespace,
    legacy_embedding_revision,
)
from .utils.dependencies import has_torch, has_transformers

if TYPE_CHECKING:
    from . import unixcoder_runtime

    get_model = unixcoder_runtime.get_model


_LEGACY_UNIXCODER_MAX_LENGTH = 512


def _legacy_cache_namespace() -> str:
    # Embeddings from different providers/models live in different vector
    # spaces; namespacing cache keys prevents a provider or model switch
    # from replaying stale vectors of the wrong space.
    if settings.EMBEDDING_PROVIDER == cs.EmbeddingProvider.OPENAI:
        namespace = f"{cs.EmbeddingProvider.OPENAI}:{settings.OPENAI_EMBEDDING_MODEL}"
        if settings.OPENAI_EMBEDDING_DIMENSIONS is not None:
            namespace = f"{namespace}:{settings.OPENAI_EMBEDDING_DIMENSIONS}"
        return namespace
    return f"{cs.EmbeddingProvider.UNIXCODER}:{cs.UNIXCODER_MODEL}"


class EmbeddingCache:
    """Keep the standalone JSON cache API for callers with an explicit path."""

    __slots__ = ("_cache", "_path")

    def __init__(self, path: Path | None = None) -> None:
        self._cache: dict[str, list[float]] = {}
        self._path = path

    @staticmethod
    def _content_hash(content: str) -> str:
        return hashlib.sha256(
            f"{_legacy_cache_namespace()}\x00{content}".encode()
        ).hexdigest()

    def get(self, content: str) -> list[float] | None:
        return self._cache.get(self._content_hash(content))

    def put(self, content: str, embedding: list[float]) -> None:
        self._cache[self._content_hash(content)] = embedding

    def get_many(self, snippets: list[str]) -> dict[int, list[float]]:
        results: dict[int, list[float]] = {}
        for i, snippet in enumerate(snippets):
            if (cached := self.get(snippet)) is not None:
                results[i] = cached
        return results

    def put_many(self, snippets: list[str], embeddings: list[list[float]]) -> None:
        for snippet, embedding in zip(snippets, embeddings):
            self.put(snippet, embedding)

    def save(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("w", encoding="utf-8") as f:
                json.dump(self._cache, f)
        except Exception as e:
            logger.warning(ls.EMBEDDING_CACHE_SAVE_FAILED, path=self._path, error=e)

    def load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            with self._path.open("r", encoding="utf-8") as f:
                self._cache = json.load(f)
            logger.debug(
                ls.EMBEDDING_CACHE_LOADED, count=len(self._cache), path=self._path
            )
        except Exception as e:
            logger.warning(ls.EMBEDDING_CACHE_LOAD_FAILED, path=self._path, error=e)
            self._cache = {}

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


class _SnippetEmbeddingCache:
    """Map snippets into the configured embedding space before keyed storage."""

    def __init__(
        self, path: Path, legacy_path: Path | None, legacy_revision: str | None
    ) -> None:
        """Keep the cache location and open the keyed store on demand."""
        self._path = path
        self._store = KeyedEmbeddingCache(
            path, legacy_path, legacy_revision=legacy_revision
        )

    @staticmethod
    def _keys(snippets: list[str], max_length: int | None) -> list[str]:
        """Hash snippets under the current embedding namespace."""
        namespace = embedding_namespace(max_length)
        return [
            hashlib.sha256(f"{namespace}\x00{snippet}".encode()).hexdigest()
            for snippet in snippets
        ]

    def get(self, content: str, max_length: int | None = None) -> list[float] | None:
        """Return the vector cached for this snippet and embedding configuration."""
        return self.get_many([content], max_length).get(0)

    def put(
        self, content: str, embedding: list[float], max_length: int | None = None
    ) -> None:
        """Queue a vector for persistence under the active namespace."""
        self.put_many([content], [embedding], max_length)

    def get_many(
        self, snippets: list[str], max_length: int | None = None
    ) -> dict[int, list[float]]:
        """Read only requested keys and promote eligible legacy hits."""
        keys = self._keys(snippets, max_length)
        hits = self._store.get_many(keys)
        if (
            len(hits) == len(keys)
            or effective_max_length(max_length) != _LEGACY_UNIXCODER_MAX_LENGTH
            or (revision := legacy_embedding_revision()) is None
            or not self._store.legacy_matches(revision)
        ):
            return hits
        missing = [index for index in range(len(keys)) if index not in hits]
        aliases = [EmbeddingCache._content_hash(snippets[index]) for index in missing]
        for alias_index, embedding in self._store.get_many(aliases).items():
            index = missing[alias_index]
            hits[index] = embedding
            self._store.put(keys[index], embedding)
        return hits

    def put_many(
        self,
        snippets: list[str],
        embeddings: list[list[float]],
        max_length: int | None = None,
    ) -> None:
        """Queue vectors in snippet order under the active namespace."""
        self._store.put_many(self._keys(snippets, max_length), embeddings)

    def save(self) -> None:
        """Persist queued vectors."""
        self._store.save()

    def close(self) -> None:
        """Discard pending vectors and release the underlying connection."""
        self._store.clear()
        self._store.close()


_embedding_cache: _SnippetEmbeddingCache | None = None


def get_embedding_cache() -> _SnippetEmbeddingCache:
    """Return the shared snippet cache for the active vector-store directory."""
    global _embedding_cache
    if _embedding_cache is None:
        from .vector_store import embedding_cache_dir

        directory = embedding_cache_dir()
        cache_path = directory / cs.KEYED_EMBEDDING_CACHE_FILENAME
        legacy_revision = legacy_embedding_revision()
        legacy_path = (
            directory / cs.EMBEDDING_CACHE_FILENAME
            if legacy_revision is not None
            else None
        )
        _embedding_cache = _SnippetEmbeddingCache(
            cache_path, legacy_path, legacy_revision
        )
    return _embedding_cache


def clear_embedding_cache() -> None:
    """Drop the process cache reference and release its connection."""
    global _embedding_cache
    if _embedding_cache is not None:
        _embedding_cache.close()
        _embedding_cache = None


def _openai_client() -> httpx.Client:
    headers: dict[str, str] = {}
    api_key = normalised_credential(
        settings.OPENAI_EMBEDDING_API_KEY
    ) or normalised_credential(
        os.environ.get(API_KEY_INFO[cs.Provider.OPENAI]["env_var"])
    )
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return httpx.Client(
        base_url=settings.OPENAI_EMBEDDING_BASE_URL,
        headers=headers,
        timeout=settings.OPENAI_EMBEDDING_TIMEOUT,
    )


def _openai_embed_batch(snippets: list[str]) -> list[list[float]]:
    embeddings: list[list[float]] = []
    batch_size = settings.OPENAI_EMBEDDING_BATCH_SIZE
    with _openai_client() as client:
        for start in range(0, len(snippets), batch_size):
            batch = snippets[start : start + batch_size]
            payload: dict[str, object] = {
                "model": settings.OPENAI_EMBEDDING_MODEL,
                "input": batch,
            }
            if settings.OPENAI_EMBEDDING_DIMENSIONS is not None:
                payload["dimensions"] = settings.OPENAI_EMBEDDING_DIMENSIONS
            response = client.post(cs.OPENAI_EMBEDDINGS_PATH, json=payload)
            if response.status_code != cs.HTTP_OK:
                raise RuntimeError(
                    ex.OPENAI_EMBEDDING_HTTP_ERROR.format(
                        status=response.status_code, body=response.text[:500]
                    )
                )
            embeddings.extend(_parse_embedding_rows(response, len(batch)))
    return embeddings


def _parse_embedding_rows(response: httpx.Response, expected: int) -> list[list[float]]:
    try:
        rows = response.json()["data"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise RuntimeError(
            ex.OPENAI_EMBEDDING_MALFORMED_RESPONSE.format(error=e)
        ) from e
    if not isinstance(rows, list):
        raise RuntimeError(
            ex.OPENAI_EMBEDDING_MALFORMED_RESPONSE.format(error="'data' is not a list")
        )
    if len(rows) != expected:
        raise RuntimeError(
            ex.OPENAI_EMBEDDING_COUNT_MISMATCH.format(got=len(rows), expected=expected)
        )
    # Place each row at its declared index instead of sorting: a buggy
    # server emitting duplicate or out-of-range indices must fail loudly
    # rather than pair snippets with the wrong vectors.
    placed: list[list[float] | None] = [None] * expected
    for row in rows:
        try:
            index, embedding = row["index"], row["embedding"]
        except (KeyError, TypeError) as e:
            raise RuntimeError(
                ex.OPENAI_EMBEDDING_MALFORMED_RESPONSE.format(error=e)
            ) from e
        if (
            not isinstance(index, int)
            or not 0 <= index < expected
            or placed[index] is not None
        ):
            raise RuntimeError(ex.OPENAI_EMBEDDING_BAD_INDEX.format(index=index))
        placed[index] = embedding
    return [row for row in placed if row is not None]


def __getattr__(name: str) -> object:
    """Expose the existing get_model entry point and its cache controls lazily."""
    if name == "get_model":
        from .unixcoder_runtime import get_model

        return get_model
    raise AttributeError(name)


def _require_local_dependencies() -> None:
    """Raise the existing semantic-extra error before importing the local runtime."""
    if not (has_torch() and has_transformers()):
        raise RuntimeError(ex.SEMANTIC_EXTRA)


def _unixcoder_embed_code(code: str, max_length: int | None) -> list[float]:
    """Route a cache miss to the configured worker or local inference."""
    if os.environ.get("CGR_UNIXCODER_WORKER_DIR"):
        from . import embedding_worker

        return embedding_worker.embed_code(code, effective_max_length(max_length))
    _require_local_dependencies()
    from . import unixcoder_runtime

    return unixcoder_runtime._unixcoder_embed_code(code, max_length)


def _unixcoder_embed_batch(
    snippets: list[str], max_length: int | None, batch_size: int
) -> list[list[float]]:
    """Route uncached snippets to the configured worker or local inference."""
    if os.environ.get("CGR_UNIXCODER_WORKER_DIR"):
        from . import embedding_worker

        return embedding_worker.embed_batch(
            snippets, effective_max_length(max_length), batch_size
        )
    _require_local_dependencies()
    from . import unixcoder_runtime

    return unixcoder_runtime._unixcoder_embed_batch(snippets, max_length, batch_size)


def embed_code(code: str, max_length: int | None = None) -> list[float]:
    """Return one snippet vector and cache a successful provider call."""
    cache = get_embedding_cache()
    if (cached := cache.get(code, max_length)) is not None:
        return cached
    try:
        if settings.EMBEDDING_PROVIDER == cs.EmbeddingProvider.OPENAI:
            result = _openai_embed_batch([code])[0]
        else:
            result = _unixcoder_embed_code(code, max_length)
    except Exception:
        logger.exception(ls.EMBEDDING_SNIPPET_FAILED, length=len(code))
        raise
    cache.put(code, result, max_length)
    return result


def embed_code_batch(
    snippets: list[str],
    max_length: int | None = None,
    batch_size: int = cs.EMBEDDING_DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    """Return vectors in snippet order, embedding only cache misses."""
    if not snippets:
        return []

    cache = get_embedding_cache()
    cached_results = cache.get_many(snippets, max_length)

    if len(cached_results) == len(snippets):
        logger.debug(ls.EMBEDDING_CACHE_HIT, count=len(snippets))
        return [cached_results[i] for i in range(len(snippets))]

    uncached_indices = [i for i in range(len(snippets)) if i not in cached_results]
    uncached_snippets = [snippets[i] for i in uncached_indices]

    if settings.EMBEDDING_PROVIDER == cs.EmbeddingProvider.OPENAI:
        all_new_embeddings = _openai_embed_batch(uncached_snippets)
    else:
        all_new_embeddings = _unixcoder_embed_batch(
            uncached_snippets, max_length, batch_size
        )

    cache.put_many(uncached_snippets, all_new_embeddings, max_length)

    results: list[list[float]] = [[] for _ in snippets]
    for i, emb in cached_results.items():
        results[i] = emb
    for idx, orig_i in enumerate(uncached_indices):
        results[orig_i] = all_new_embeddings[idx]

    return results
