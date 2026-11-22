"""Describe an embedding space without importing an inference runtime."""

import json
import os
import re

from . import constants as cs
from .config import settings


def effective_max_length(max_length: int | None) -> int:
    """Resolve the token limit passed to UniXcoder."""
    return settings.EMBEDDING_MAX_LENGTH if max_length is None else max_length


def embedding_revision() -> str | None:
    """Return an immutable model commit when the configuration supplies one."""
    revision = os.environ.get("CGR_EMBEDDING_REVISION", "")
    return revision.lower() if re.fullmatch(r"[0-9a-fA-F]{40}", revision) else None


def legacy_embedding_revision() -> str | None:
    """Permit migration only for a confirmed match to the local model revision."""
    revision = embedding_revision()
    if (
        settings.EMBEDDING_PROVIDER == cs.EmbeddingProvider.UNIXCODER
        and revision
        and os.environ.get("CGR_EMBEDDING_LEGACY_REVISION", "").lower() == revision
    ):
        return revision
    return None


def embedding_namespace(max_length: int | None = None) -> str:
    """Identify the provider and settings that determine its output vectors."""
    revision = embedding_revision()
    if settings.EMBEDDING_PROVIDER == cs.EmbeddingProvider.OPENAI:
        return json.dumps(
            [
                cs.EmbeddingProvider.OPENAI,
                settings.OPENAI_EMBEDDING_MODEL,
                settings.OPENAI_EMBEDDING_BASE_URL,
                settings.OPENAI_EMBEDDING_DIMENSIONS,
                revision,
            ]
        )
    return json.dumps(
        [
            cs.EmbeddingProvider.UNIXCODER,
            cs.UNIXCODER_MODEL,
            effective_max_length(max_length),
            revision,
            None if revision else os.environ.get("CGR_UNIXCODER_MODEL_PATH"),
        ]
    )
