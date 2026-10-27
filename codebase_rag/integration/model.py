"""Captured source identities and graph contract validation; no indexed-state writes."""

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

VERSION = "python-snapshot-1"
KINDS = {"Repository", "Module", "Class", "Function", "CallSite"}
RELATIONS = {"CONTAINS", "IMPORTS", "INHERITS", "REFERENCES", "CALLS"}
STATUSES = {"structural", "candidate"}


def canonical(value):
    """Serialize identity-bearing values with stable keys, never normalize source text."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    """SHA-256 bytes or canonical UTF-8 text."""
    return hashlib.sha256(
        value if isinstance(value, bytes) else value.encode()
    ).hexdigest()


def git(path, *args):
    """Read Git metadata without optional index refresh writes."""
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    return subprocess.check_output(
        ["git", "-C", str(path), *args],
        env=env,
        text=True,
        encoding="utf-8",
        timeout=15,
    ).strip()


def capture(repositories):
    """Read explicit Python files only; source paths must stay inside each repository."""
    sources = []
    repos = []
    search = []
    for entry in repositories:
        root = Path(entry["path"]).resolve()
        record = dict(
            name=entry["name"],
            path=str(root),
            head=git(root, "rev-parse", "HEAD"),
            branch=git(root, "branch", "--show-current"),
            status=git(root, "status", "--short", "--branch"),
            python_clean=not bool(
                git(
                    root, "status", "--porcelain", "--untracked-files=all", "--", "*.py"
                )
            ),
        )
        pyproject = root / "pyproject.toml"
        record["dependency_pins"] = {}
        if pyproject.is_file():
            data = tomllib.loads(pyproject.read_text(encoding="utf-8-sig"))
            for dep, binding in (
                data.get("tool", {}).get("uv", {}).get("sources", {}).items()
            ):
                if isinstance(binding, dict) and "trpg-rules-engine" in binding.get(
                    "git", ""
                ):
                    record["dependency_pins"][dep] = binding.get("rev")
        for rel in entry.get("source_roots", ["src"]):
            directory = (root / rel).resolve()
            directory.relative_to(root)
            search.append(str(directory))
        for rel in sorted(set(entry["files"])):
            path = (root / rel).resolve()
            path.relative_to(root)
            if path.suffix != ".py" or path.is_symlink():
                raise ValueError("Explicit non-symlink Python files required")
            raw = path.read_bytes()
            module = entry.get("modules", {}).get(rel)
            if not module:
                selected = next(
                    (
                        r
                        for r in entry.get("source_roots", ["src"])
                        if Path(rel).is_relative_to(r)
                    ),
                    None,
                )
                module = (
                    (Path(rel).relative_to(selected).as_posix() if selected else rel)
                    .removesuffix(".py")
                    .replace("/", ".")
                )
            sources.append(
                dict(
                    repository=entry["name"],
                    path=Path(rel).as_posix(),
                    module=module,
                    absolute=str(path),
                    text=raw.decode("utf-8-sig"),
                    file_sha256=digest(raw),
                    git_head=record["head"],
                    git_status=record["status"],
                )
            )
        repos.append(record)
    return sources, repos, list(dict.fromkeys(search))


def snapshot_identity(sources, repositories, config):
    """Bind actual source/config/dependency/parser and adapter bytes, not branch alone."""
    code = {
        p.name: digest(p.read_bytes())
        for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    try:
        jedi_version = importlib.metadata.version("jedi")
    except importlib.metadata.PackageNotFoundError:
        jedi_version = "absent"
    record = dict(
        platform=sys.platform,
        distribution_requirements=sorted(
            importlib.metadata.requires("code-graph-rag") or []
        ),
        schema=1,
        adapter=VERSION,
        adapter_files=code,
        python=sys.version.split()[0],
        jedi=jedi_version,
        sources=[
            {k: s[k] for k in ("repository", "path", "module", "file_sha256")}
            for s in sources
        ],
        repositories=[
            {
                k: r[k]
                for k in ("name", "head", "status", "python_clean", "dependency_pins")
            }
            for r in repositories
        ],
        config=config,
    )
    return digest(canonical(record)), record


def verify_sources(bundle):
    """Prevent publication if any captured source or dependency identity drifted."""
    for source in bundle["sources"]:
        if digest(Path(source["absolute"]).read_bytes()) != source["file_sha256"]:
            raise ValueError("Captured source changed")
    for repo in bundle["repositories"]:
        if (
            git(repo["path"], "rev-parse", "HEAD") != repo["head"]
            or git(repo["path"], "status", "--short", "--branch") != repo["status"]
        ):
            raise ValueError("Repository identity changed")


def validate(bundle):
    """Reject duplicate identities, dangling edges, missing evidence or wrong call types."""
    nodes = bundle["nodes"]
    byid = {n["node_id"]: n for n in nodes}
    if len(byid) != len(nodes):
        raise ValueError("Duplicate node identity")
    ids = set()
    captured = {(s["repository"], s["path"]): s for s in bundle["sources"]}
    snapshot = bundle["snapshot_id"]
    for n in nodes:
        if n["kind"] not in KINDS or n["snapshot_id"] != snapshot:
            raise ValueError("Invalid node contract")
        if n["kind"] not in {"Repository", "CallSite"}:
            source = captured[n["repository"], n["path"]]
            lines = source["text"].splitlines(keepends=True)
            if n["kind"] == "Module":
                exact = source["text"]
            elif n["subtype"] == "lambda":
                selected = lines[n["start"] - 1 : n["end"]]
                if len(selected) == 1:
                    exact = selected[0][n["column"] : n["end_column"]]
                else:
                    selected[0] = selected[0][n["column"] :]
                    selected[-1] = selected[-1][: n["end_column"]]
                    exact = "".join(selected)
            else:
                exact = "".join(lines[n["start"] - 1 : n["end"]])
            if exact != n["source"] or digest(exact) != n["source_sha256"]:
                raise ValueError("Captured source/span fingerprint mismatch")
            parent = byid.get(n["parent"])
            if (
                parent
                and parent["kind"] != "Repository"
                and not (parent["start"] <= n["start"] <= n["end"] <= parent["end"])
            ):
                raise ValueError("Invalid lexical owner range")
    for edge in bundle["edges"]:
        if edge["edge_id"] in ids:
            raise ValueError("Duplicate relationship identity")
        ids.add(edge["edge_id"])
        if (
            edge["kind"] not in RELATIONS
            or edge["source"] not in byid
            or edge["target"] not in byid
        ):
            raise ValueError("Dangling relation")
        if edge["status"] not in STATUSES or not edge.get("evidence"):
            raise ValueError("Relation evidence required")
        if edge["kind"] == "CALLS":
            site = byid[edge["source"]]
            if (
                site["kind"] != "CallSite"
                or not site["active_call"]
                or byid[edge["target"]]["kind"] not in {"Class", "Function"}
            ):
                raise ValueError("Invalid CALLS semantics")
            if edge.get("exhaustive") is not False:
                raise ValueError("No exhaustiveness proof available")
    return dict(
        nodes=len(nodes),
        edges=len(ids),
        callsites=sum(n["kind"] == "CallSite" for n in nodes),
    )
