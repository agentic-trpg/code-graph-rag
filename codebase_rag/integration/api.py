"""Versioned Python-only CGR interface, independent of chat/agent/CLI internals."""

from .analyze import Analyzer
from .flow import Flow
from .model import canonical, capture, digest, validate

API_VERSION = 1


def analyze_sources(sources, repositories, search_paths, mode="default"):
    """One canonical node catalog, branch-aware defaults and optional real Jedi evidence."""
    if mode not in {"default", "jedi"}:
        raise ValueError("mode must be default or jedi")
    analyzer = Analyzer(
        sources,
        repositories,
        search_paths,
        {"mode": mode, "max_seconds": 120, "max_requests": 20000},
    )
    bundle = analyzer.run(resolve=mode == "jedi")
    catalog = analyzer.catalog
    flow = Flow(catalog)
    values = flow.run()
    existing = {
        (
            e["source"],
            e["target"],
            e["kind"],
            e["evidence"]["line"],
            e["evidence"]["column"],
        )
        for e in catalog.edges
    }
    for request in catalog.requests:
        site = catalog.byid.get(request.get("site_id"))
        owner = catalog.byid[request["owner_id"]]
        if not request["active"]:
            continue
        value = values.get(flow.position(request["expr"], owner, request["role"]))
        if value is None:
            value = flow.evaluate(
                request["expr"], owner, flow.env[owner["node_id"]], True
            )
        targets = []
        unknown = any(tag in {"unknown", "external"} for tag, _ in value)
        for tag, ident in value:
            if tag not in {"function", "class", "module"}:
                continue
            node = catalog.byid[ident]
            relation = request["relation"]
            if relation == "CALLS" and node["kind"] not in {"Function", "Class"}:
                continue
            if relation == "INHERITS" and node["kind"] != "Class":
                continue
            if node["repository"] != owner["repository"]:
                source_repo = next(
                    r for r in repositories if r["name"] == owner["repository"]
                )
                target_repo = next(
                    r for r in repositories if r["name"] == node["repository"]
                )
                dep = (
                    "dnd5e-srd-data"
                    if node["qualname"].startswith("dnd5e_srd_data")
                    else "dnd5e-engine"
                )
                if not (
                    source_repo["dependency_pins"].get(dep) == target_repo["head"]
                    and target_repo["python_clean"]
                ):
                    if site:
                        site["reasons"].append("dependency_version_boundary")
                    unknown = True
                    continue
            targets.append(ident)
            source = request.get("site_id", request["owner_id"])
            key = (source, ident, relation, request["line"], request["column"])
            if key not in existing:
                catalog.edge(
                    source,
                    ident,
                    relation,
                    dict(
                        provider="cgr-python-flow-v1",
                        line=request["line"],
                        column=request["column"],
                        phase=request["phase"],
                        expression=request["expression"],
                        unknown_sources=unknown,
                    ),
                    caller_id=request["owner_id"],
                )
                existing.add(key)
        if site:
            site["external_candidates"] = sorted(
                ident for tag, ident in value if tag == "external"
            )
            if site["external_candidates"]:
                site["reasons"].append("external_or_excluded_definition")
            site["candidates"] = sorted(set(site["candidates"]) | set(targets))
            site["unknown_sources"] = unknown
            site["reasons"] = sorted(
                set(
                    site["reasons"]
                    + (["unknown_value_origin"] if unknown else [])
                    + ["candidate_set_not_proven_exhaustive"]
                )
            )
            site["resolution_status"] = (
                "candidate"
                if site["candidates"]
                else "boundary"
                if "dependency_version_boundary" in site["reasons"]
                else "unresolved"
            )
    bundle["edges"] = sorted(
        {e["edge_id"]: e for e in catalog.edges}.values(),
        key=lambda edge: edge["edge_id"],
    )
    bundle["validation"] = validate(bundle)
    bundle["api_version"] = API_VERSION
    return bundle


def build_snapshot(repositories, mode="default", previous=None):
    """Capture explicit files and report real add/change/delete deltas; revalidate all links.

    Upstream parser fingerprint/provenance are reused, not mistaken for online
    database isolation. Relationship analysis is rerun across the fixed small
    scope so changing a callee invalidates callers; no stale old graph is adopted.
    """
    from codebase_rag.parser_fingerprint import compute_parser_fingerprint

    sources, repos, search = capture(repositories)
    bundle = analyze_sources(sources, repos, search, mode)
    hashes = {s["repository"] + ":" + s["path"]: s["file_sha256"] for s in sources}
    old = (previous or {}).get("file_hashes", {})
    bundle["source_snapshot_id"] = digest(
        canonical(
            dict(
                files=hashes,
                repositories=[
                    {k: r[k] for k in ("name", "head", "status")} for r in repos
                ],
            )
        )
    )
    bundle["file_hashes"] = hashes
    bundle["changes"] = dict(
        added=sorted(hashes.keys() - old.keys()),
        deleted=sorted(old.keys() - hashes.keys()),
        modified=sorted(k for k in hashes.keys() & old.keys() if hashes[k] != old[k]),
        unchanged=sorted(k for k in hashes.keys() & old.keys() if hashes[k] == old[k]),
    )
    bundle["upstream_parser_fingerprint"] = compute_parser_fingerprint()
    return bundle
