"""Reconcile Qdrant points against CGR's current source and graph bindings."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, TypedDict, cast

from loguru import logger
from qdrant_client import models

from . import constants as cs
from . import embedder
from . import vector_store as vectors
from .config import settings
from .graph_updater import (
    _QDRANT_EMBEDDING_PROVENANCE_VERSION,
    _embeddings_marker,
    _flush_embedding_batch,
    _load_parser_fingerprint,
    _save_embeddings_owed,
)
from .types_defs import EmbeddingPoint, EmbeddingSymbol, ResultRow
from .utils.source_extraction import extract_source_lines, validate_source_location

if TYPE_CHECKING:
    from .graph_updater import GraphUpdater

PROVENANCE_KEY = f"cgr_embedding_provenance_v{_QDRANT_EMBEDDING_PROVENANCE_VERSION}"


class _Provenance(TypedDict):
    model: str
    input: str
    snippet: str
    ast: bool
    disk_equal: bool


class _FileVersion(NamedTuple):
    sha256: str
    stat: tuple[int, int, int, int]


def _stat_key(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _file_version(path: Path, graph_hash: str | None) -> _FileVersion | None:
    """Reject working-tree edits that landed after the graph's source scan."""
    try:
        stat = _stat_key(path)
        data = path.read_bytes()
        if _stat_key(path) != stat:
            return None
    except OSError:
        return None
    if (
        graph_hash is not None
        and hashlib.md5(data, usedforsecurity=False).hexdigest() != graph_hash
    ):
        return None
    return _FileVersion(hashlib.sha256(data).hexdigest(), stat)


def _unchanged(path: Path, version: _FileVersion) -> bool:
    try:
        return _stat_key(path) == version.stat
    except OSError:
        return False


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _retrieve(ids: Sequence[str]) -> Iterator[models.Record]:
    if not ids:
        return
    client = vectors.get_qdrant_client()
    size = vectors._RETRIEVE_BATCH_SIZE
    for start in range(0, len(ids), size):
        yield from client.retrieve(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            ids=ids[start : start + size],
            with_payload=[
                vectors.PAYLOAD_NODE_ID,
                vectors.PAYLOAD_QUALIFIED_NAME,
                cs.PAYLOAD_PROJECT,
                PROVENANCE_KEY,
            ],
            with_vectors=False,
        )


def _stored_provenance(
    payload: Mapping[str, Any], project: str, symbol: EmbeddingSymbol, model: str
) -> _Provenance | None:
    if (
        payload.get(cs.PAYLOAD_PROJECT) != project
        or payload.get(vectors.PAYLOAD_QUALIFIED_NAME) != symbol.qualified_name
    ):
        return None
    saved = payload.get(PROVENANCE_KEY)
    if (
        isinstance(saved, dict)
        and saved.get("model") == model
        and isinstance(saved.get("input"), str)
        and isinstance(saved.get("snippet"), str)
        and isinstance(saved.get("ast"), bool)
        and isinstance(saved.get("disk_equal"), bool)
    ):
        return cast(_Provenance, saved)
    return None


def _put_metadata(changes: dict[str, tuple[int, _Provenance]]) -> None:
    """Rebind proven vectors and update provenance without moving vector data."""
    if not changes:
        return
    client = vectors.get_qdrant_client()
    entries = list(changes.items())
    size = settings.QDRANT_BATCH_SIZE
    for start in range(0, len(entries), size):
        operations = [
            models.SetPayloadOperation(
                set_payload=models.SetPayload(
                    payload={vectors.PAYLOAD_NODE_ID: node_id, PROVENANCE_KEY: tag},
                    points=[point_id],
                )
            )
            for point_id, (node_id, tag) in entries[start : start + size]
        ]
        client.batch_update_points(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            update_operations=operations,
            wait=True,
        )


def _store(
    project: str, points: Sequence[EmbeddingPoint], tags: Mapping[int, _Provenance]
) -> int:
    """Commit each vector and its complete provenance in one upsert."""
    if not points:
        return 0
    vectors._check_vector_dims(
        cs.VectorStoreBackend.QDRANT, (embedding for _, embedding, _ in points)
    )
    records = [
        models.PointStruct(
            id=vectors.embedding_point_id(project, symbol),
            vector=embedding,
            payload={
                vectors.PAYLOAD_NODE_ID: node_id,
                vectors.PAYLOAD_QUALIFIED_NAME: symbol.qualified_name,
                cs.PAYLOAD_PROJECT: project,
                PROVENANCE_KEY: tags[node_id],
            },
        )
        for node_id, embedding, symbol in points
    ]
    return vectors._store_batch(records, lambda: vectors._upsert_with_retry(records))


def _verified_ids(
    project: str,
    expected: Mapping[int, EmbeddingSymbol],
    tags: Mapping[int, _Provenance],
) -> set[int]:
    nodes = vectors._node_by_point(project, expected)
    verified = set()
    for point in _retrieve(list(nodes)):
        node_id = nodes[str(point.id)]
        payload = point.payload or {}
        if (
            payload.get(vectors.PAYLOAD_NODE_ID) == node_id
            and _stored_provenance(
                payload, project, expected[node_id], tags[node_id]["model"]
            )
            == tags[node_id]
        ):
            verified.add(node_id)
    return verified


def reconcile_qdrant_deltas(
    updater: GraphUpdater,
    rows: list[ResultRow],
    current: Mapping[int, EmbeddingSymbol],
    model: str,
) -> None:
    """Refresh one project's vectors and settle its existing recovery marker."""
    project = updater.project_name
    marker = _embeddings_marker(updater.state_dir, project)
    if not _save_embeddings_owed(marker, project, True):
        raise OSError("could not record pending embeddings before Qdrant update")

    points = {
        str(point.id): point.payload or {}
        for point in _retrieve(list(vectors._node_by_point(project, current)))
    }
    parser = updater._pending_parser_fingerprint or _load_parser_fingerprint(
        updater.state_dir / cs.PARSER_FINGERPRINT_FILENAME
    )
    hashes = updater._pending_hash_cache[1] if updater._pending_hash_cache else {}
    files: dict[Path, _FileVersion | None] = {}
    overlays: dict[Path, str | None] = {}
    tags: dict[int, _Provenance] = {}
    metadata: dict[str, tuple[int, _Provenance]] = {}
    expected: dict[int, EmbeddingSymbol] = {}
    stored_ids: set[int] = set()
    pending: list[tuple[int, EmbeddingSymbol, str]] = []
    attempted = stored = extracted = 0
    source_debt = False

    def flush() -> None:
        nonlocal attempted, stored
        attempted += len(pending)
        stored += _flush_embedding_batch(
            pending,
            expected,
            stored_ids,
            embedder.embed_code_batch,
            lambda batch: _store(project, batch, tags),
        )

    for row in rows:
        parsed = updater._parse_embedding_result(row)
        if parsed is None:
            continue
        raw_path = parsed[cs.KEY_PATH]
        start, end = parsed[cs.KEY_START_LINE], parsed[cs.KEY_END_LINE]
        valid, relative_path = validate_source_location(raw_path, start, end)
        if not valid or relative_path is None or start is None or end is None:
            continue
        path = updater.repo_path / relative_path
        file_key = relative_path.as_posix()
        if path not in files:
            files[path] = _file_version(path, hashes.get(file_key))
            overlay = updater._delombok_overlay.get(file_key)
            overlays[path] = (
                hashlib.sha256(overlay).hexdigest() if overlay is not None else None
            )
        if (version := files[path]) is None:
            source_debt = True
            continue

        node_id = parsed[cs.KEY_NODE_ID]
        symbol = current[node_id]
        point_id = vectors.embedding_point_id(project, symbol)
        payload = points.get(point_id, {})
        saved = _stored_provenance(payload, project, symbol, model)
        signature = _sha256(
            json.dumps([parser, raw_path, start, end, version.sha256, overlays[path]])
        )
        has_ast = path in updater.ast_cache
        if (
            parser is not None
            and saved is not None
            and saved["input"] == signature
            and (saved["ast"] == has_ast or (not has_ast and saved["disk_equal"]))
        ):
            tag = saved
        else:
            extracted += 1
            candidate = updater._embedding_candidate(row)
            if candidate is None:
                source_debt |= saved is not None
                continue
            source = candidate[2]
            disk_equal = has_ast and extract_source_lines(path, start, end) == source
            if not _unchanged(path, version):
                source_debt = True
                continue
            tag = _Provenance(
                model=model,
                input=signature,
                snippet=_sha256(source),
                ast=has_ast,
                disk_equal=disk_equal,
            )
            if saved is None or saved["snippet"] != tag["snippet"]:
                tags[node_id] = tag
                pending.append(candidate)
                if len(pending) >= settings.QDRANT_BATCH_SIZE:
                    flush()
                continue

        tags[node_id] = tag
        expected[node_id] = symbol
        stored_ids.add(node_id)
        if saved != tag or payload.get(vectors.PAYLOAD_NODE_ID) != node_id:
            metadata[point_id] = (node_id, tag)

    flush()
    _put_metadata(metadata)
    pruned = updater._delete_stale_embeddings(
        current, stored_ids, vectors.delete_stale_embeddings
    )
    verified = updater._reconcile_embeddings(
        expected, lambda name, symbols: _verified_ids(name, symbols, tags)
    )
    if attempted:
        embedder.get_embedding_cache().save()
    updater._record_embeddings_pending(source_debt or not (pruned and verified))
    logger.info(
        "Qdrant embedding delta: graph_rows={} existing={} extracted={} embedded={} "
        "upserted={} payload_updates={} reused={}",
        len(rows),
        len(points),
        extracted,
        attempted,
        stored,
        len(metadata),
        len(expected) - attempted,
    )
