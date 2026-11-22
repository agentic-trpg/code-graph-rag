"""Exercise configured inference and revision-aware cache reuse."""

import sys
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import codebase_rag
from codebase_rag import constants as cs
from codebase_rag import embedder, embedding_worker, vector_store
from codebase_rag.config import settings
from codebase_rag.embedding_identity import embedding_revision


@pytest.fixture(autouse=True)
def isolated_embedder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[None]:
    """Keep the cache and model settings isolated from any user installation."""
    embedder.clear_embedding_cache()
    monkeypatch.setattr(vector_store, "embedding_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", cs.EmbeddingProvider.UNIXCODER)
    monkeypatch.setattr(settings, "EMBEDDING_MAX_LENGTH", 512)
    for key in (
        "CGR_EMBEDDING_REVISION",
        "CGR_EMBEDDING_LEGACY_REVISION",
        "CGR_UNIXCODER_MODEL_PATH",
        "CGR_UNIXCODER_WORKER_DIR",
    ):
        monkeypatch.delenv(key, raising=False)
    yield
    embedder.clear_embedding_cache()


def test_worker_misses_and_persisted_hits_keep_local_runtime_unloaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Send only misses to the worker and retain the result across cache instances."""
    monkeypatch.setenv("CGR_UNIXCODER_WORKER_DIR", "configured-worker")
    monkeypatch.delattr(codebase_rag, "unixcoder_runtime", raising=False)
    monkeypatch.setitem(sys.modules, "codebase_rag.unixcoder_runtime", None)
    monkeypatch.setattr(embedder, "has_torch", lambda: False)
    single = MagicMock(return_value=[1.0])
    batch = MagicMock(return_value=[[2.0]])
    monkeypatch.setattr(embedding_worker, "embed_code", single)
    monkeypatch.setattr(embedding_worker, "embed_batch", batch)

    assert embedder.embed_code("a") == [1.0]
    assert embedder.embed_code_batch(["a", "b"], batch_size=2) == [[1.0], [2.0]]
    embedder.get_embedding_cache().save()
    embedder.clear_embedding_cache()
    assert embedder.embed_code_batch(["b", "a"]) == [[2.0], [1.0]]
    single.assert_called_once_with("a", 512)
    batch.assert_called_once_with(["b"], 512, 2)


def test_distinct_revisions_and_token_limits_never_share_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing the effective local limit or revision selects a new cache key."""
    inference = MagicMock(side_effect=[[1.0], [2.0], [3.0]])
    monkeypatch.setattr(embedder, "_unixcoder_embed_code", inference)
    monkeypatch.setenv("CGR_EMBEDDING_REVISION", "a" * 40)

    assert embedder.embed_code("snippet") == [1.0]
    assert embedder.embed_code("snippet", max_length=512) == [1.0]
    assert embedder.embed_code("snippet", max_length=256) == [2.0]
    monkeypatch.setenv("CGR_EMBEDDING_REVISION", "b" * 40)
    assert embedder.embed_code("snippet") == [3.0]
    assert inference.call_count == 3


def test_openai_endpoints_with_the_same_model_have_distinct_cache_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A different server can expose the same model name for different weights."""
    inference = MagicMock(side_effect=[[[1.0]], [[2.0]]])
    monkeypatch.setattr(embedder, "_openai_embed_batch", inference)
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", cs.EmbeddingProvider.OPENAI)
    monkeypatch.setattr(settings, "OPENAI_EMBEDDING_BASE_URL", "http://first.example")
    assert embedder.embed_code("snippet") == [1.0]
    monkeypatch.setattr(settings, "OPENAI_EMBEDDING_BASE_URL", "http://second.example")
    assert embedder.embed_code("snippet") == [2.0]
    assert embedder.embed_code("snippet") == [2.0]
    assert inference.call_count == 2


def test_branch_name_does_not_identify_a_model_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A moving branch cannot authorize long-lived vector reuse."""
    monkeypatch.setenv("CGR_EMBEDDING_REVISION", "main")
    assert embedding_revision() is None


@pytest.mark.parametrize(
    ("legacy_revision", "max_length", "reuses"),
    [
        (None, 512, False),
        ("b" * 40, 512, False),
        ("a" * 40, 256, False),
        ("a" * 40, 512, True),
    ],
)
def test_legacy_vectors_require_confirmed_revision_and_original_token_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    legacy_revision: str | None,
    max_length: int,
    reuses: bool,
) -> None:
    """Promote known legacy vectors and compute unknown vectors with the model."""
    legacy = embedder.EmbeddingCache(tmp_path / cs.EMBEDDING_CACHE_FILENAME)
    legacy.put("snippet", [1.0])
    legacy.save()
    monkeypatch.setenv("CGR_EMBEDDING_REVISION", "a" * 40)
    if legacy_revision is not None:
        monkeypatch.setenv("CGR_EMBEDDING_LEGACY_REVISION", legacy_revision)
    inference = MagicMock(return_value=[[2.0]])
    monkeypatch.setattr(embedder, "_unixcoder_embed_batch", inference)

    expected = [[1.0]] if reuses else [[2.0]]
    assert embedder.embed_code_batch(["snippet"], max_length=max_length) == expected
    assert inference.call_count == int(not reuses)
    embedder.get_embedding_cache().save()
    embedder.clear_embedding_cache()
    monkeypatch.delenv("CGR_EMBEDDING_LEGACY_REVISION", raising=False)
    assert embedder.embed_code_batch(["snippet"], max_length=max_length) == expected
    assert inference.call_count == int(not reuses)


def test_imported_legacy_vectors_stay_bound_to_the_first_declared_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A later declaration must not reinterpret vectors imported for another model."""
    legacy = embedder.EmbeddingCache(tmp_path / cs.EMBEDDING_CACHE_FILENAME)
    legacy.put("seed", [1.0])
    legacy.put("second snippet", [2.0])
    legacy.save()
    monkeypatch.setenv("CGR_EMBEDDING_REVISION", "a" * 40)
    monkeypatch.setenv("CGR_EMBEDDING_LEGACY_REVISION", "a" * 40)
    assert embedder.embed_code("seed") == [1.0]
    embedder.get_embedding_cache().save()
    embedder.clear_embedding_cache()

    monkeypatch.setenv("CGR_EMBEDDING_REVISION", "b" * 40)
    monkeypatch.setenv("CGR_EMBEDDING_LEGACY_REVISION", "b" * 40)
    inference = MagicMock(return_value=[3.0])
    monkeypatch.setattr(embedder, "_unixcoder_embed_code", inference)
    assert embedder.embed_code("second snippet") == [3.0]
    inference.assert_called_once_with("second snippet", None)
