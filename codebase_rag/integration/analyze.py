"""Jedi 0.20.1 definition candidates mapped only by verified captured AST positions."""

import importlib.metadata
import os
import sys
import tempfile
import time
from pathlib import Path

from .model import snapshot_identity, validate
from .parser import Catalog


class Analyzer:
    """No custom points-to engine, old Graph fallback or name-based target substitution."""

    def __init__(
        self, sources, repositories, search_paths, config=None, include_calls=True
    ):
        self.sources = sources
        self.repositories = repositories
        self.search_paths = search_paths
        self.config = config or {}
        self.snapshot_id, self.identity = snapshot_identity(
            sources, repositories, self.config
        )
        self.catalog = Catalog(sources, repositories, self.snapshot_id, include_calls)
        self.observations = []

    def candidate(self, name, owner_repo, path_map):
        """full_name is evidence/display only; actual unique definition anchor decides identity."""
        raw = dict(
            name=name.name,
            full_name=name.full_name,
            type=name.type,
            line=name.line,
            column=name.column,
            module_path=(
                "external://"
                + str(name.module_name)
                + "/"
                + Path(name.module_path).name
            )
            if name.module_path
            else None,
            description=name.description,
            path_mapping="external physical prefix omitted; not a captured definition",
        )
        if raw["full_name"] and raw["full_name"].endswith("classmethod.__get__"):
            return None, raw, "rejected_classmethod_descriptor_artifact"
        if not name.module_path:
            return None, raw, "no_source_path"
        mapped = path_map.get(os.path.normcase(str(Path(name.module_path).resolve())))
        if not mapped:
            return None, raw, "external_or_excluded_definition"
        raw["module_path"] = "captured://" + mapped["repository"] + "/" + mapped["path"]
        raw["path_mapping"] = (
            "Jedi captured-mirror path; physical temporary prefix omitted"
        )
        ids = self.catalog.anchors.get((mapped["absolute"], name.line, name.column), [])
        if len(ids) != 1:
            return (
                None,
                raw,
                "ambiguous_definition_position"
                if ids
                else "unmapped_definition_position",
            )
        node = self.catalog.byid[ids[0]]
        if node["repository"] != owner_repo:
            owner = next(r for r in self.repositories if r["name"] == owner_repo)
            target = next(
                r for r in self.repositories if r["name"] == node["repository"]
            )
            dep = (
                "dnd5e-srd-data"
                if mapped["module"].startswith("dnd5e_srd_data")
                else "dnd5e-engine"
            )
            pin = owner["dependency_pins"].get(dep)
            if not (pin and pin == target["head"] and target["python_clean"]):
                return None, raw, "dependency_version_boundary"
        if name.type not in {"function", "class", "module"}:
            return None, raw, "non_callable_or_value_definition"
        return node, raw, None

    def run(self, resolve=True):
        """Resolve against a captured-source mirror so imports cannot read newer live source."""
        c = self.catalog
        if resolve:
            import jedi

            if importlib.metadata.version("jedi") != "0.20.1":
                raise ValueError("Jedi 0.20.1 required")
            with tempfile.TemporaryDirectory(prefix="lct-captured-") as directory:
                mirror = Path(directory)
                path_map = {}
                scripts = {}
                roots = []
                for repo in self.repositories:
                    base = Path(repo["path"])
                    for path in self.search_paths:
                        try:
                            relative = Path(path).relative_to(base)
                        except ValueError:
                            continue
                        roots.append(str(mirror / repo["name"] / relative))
                for source in self.sources:
                    p = mirror / source["repository"] / source["path"]
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(source["text"], encoding="utf-8", newline="")
                    path_map[os.path.normcase(str(p.resolve()))] = source
                    source["_mirror"] = str(p)
                jedi.settings.cache_directory = str(mirror / "jedi-cache")
                project = jedi.Project(
                    str(mirror),
                    sys_path=list(dict.fromkeys(roots)) + sys.path,
                    smart_sys_path=False,
                    load_unsafe_extensions=False,
                )
                for source in self.sources:
                    scripts[source["absolute"]] = jedi.Script(
                        code=source["text"], path=source["_mirror"], project=project
                    )
                start = time.monotonic()
                limit = self.config.get("max_seconds", 120)
                requests = self.config.get("max_requests", 10000)
                for index, request in enumerate(c.requests):
                    site = c.byid.get(request.get("site_id"))
                    raw = []
                    reasons = []
                    targets = []
                    rejected = []
                    if not request["active"]:
                        reasons.append("deferred_expression_not_runtime_call")
                    elif request["role"] == "decorator_factory_application":
                        reasons.append("decorator_returned_callable_not_resolved")
                    elif index >= requests or time.monotonic() - start > limit:
                        reasons.append("inference_resource_budget")
                    else:
                        script = scripts[request["source"]["absolute"]]
                        try:
                            inferred = script.infer(
                                request["end_line"], request["end_column"]
                            )
                            descriptor = False
                            for value in inferred:
                                node, evidence, reason = self.candidate(
                                    value, request["source"]["repository"], path_map
                                )
                                evidence["api"] = "infer"
                                raw.append(evidence)
                                if reason:
                                    reasons.append(reason)
                                    rejected.append(
                                        dict(reason=reason, evidence=evidence)
                                    )
                                    descriptor |= (
                                        reason
                                        == "rejected_classmethod_descriptor_artifact"
                                    )
                                elif node:
                                    targets.append((node, evidence))
                            if descriptor:
                                # This narrow correction uses actual source anchors, never spelling coincidence.
                                for value in script.goto(
                                    request["end_line"],
                                    request["end_column"],
                                    follow_imports=True,
                                    follow_builtin_imports=False,
                                ):
                                    node, evidence, reason = self.candidate(
                                        value, request["source"]["repository"], path_map
                                    )
                                    evidence["api"] = "goto_after_descriptor_rejection"
                                    raw.append(evidence)
                                    if node and node["subtype"] == "classmethod":
                                        targets.append((node, evidence))
                                    else:
                                        rejected.append(
                                            dict(
                                                reason=reason
                                                or "goto_not_verified_classmethod",
                                                evidence=evidence,
                                            )
                                        )
                        except Exception as exc:
                            reasons.append("jedi_error:" + type(exc).__name__)
                    accepted = {}
                    kind = request["relation"]
                    for node, evidence in targets:
                        allowed = (
                            {"Class", "Function"}
                            if kind == "CALLS"
                            else {"Class"}
                            if kind == "INHERITS"
                            else {"Module", "Class", "Function"}
                        )
                        if node["kind"] not in allowed:
                            reasons.append("candidate_kind_incompatible")
                            rejected.append(
                                dict(
                                    reason="candidate_kind_incompatible",
                                    evidence=evidence,
                                )
                            )
                            continue
                        accepted[node["node_id"]] = node
                        c.edge(
                            request.get("site_id", request["owner_id"]),
                            node["node_id"],
                            kind,
                            dict(
                                provider="jedi-0.20.1",
                                line=request["line"],
                                column=request["column"],
                                phase=request["phase"],
                                expression=request["expression"],
                                raw=evidence,
                                caller_id=request["owner_id"],
                            ),
                            caller_id=request["owner_id"],
                        )
                    reasons.append(
                        "candidate_set_not_proven_exhaustive"
                        if accepted
                        else "no_accepted_definition_candidate"
                    )
                    if site:
                        site.update(
                            candidates=sorted(accepted),
                            raw=sorted(raw, key=lambda item: str(sorted(item.items()))),
                            rejected=sorted(
                                rejected, key=lambda item: str(sorted(item.items()))
                            ),
                            reasons=sorted(set(reasons)),
                            resolution_status="candidate"
                            if accepted
                            else "deferred"
                            if not request["active"]
                            else "boundary"
                            if "dependency_version_boundary" in reasons
                            else "unresolved",
                        )
                    else:
                        self.observations.append(
                            dict(
                                kind=kind,
                                source=request["owner_id"],
                                line=request["line"],
                                column=request["column"],
                                expression=request["expression"],
                                phase=request["phase"],
                                active=request["active"],
                                targets=sorted(accepted),
                                raw=raw,
                                rejected=rejected,
                                reasons=sorted(set(reasons)),
                            )
                        )
                for source in self.sources:
                    source.pop("_mirror", None)
        bundle = dict(
            snapshot_id=self.snapshot_id,
            identity=self.identity,
            nodes=c.nodes,
            edges=list({e["edge_id"]: e for e in c.edges}.values()),
            observations=self.observations,
            sources=self.sources,
            repositories=self.repositories,
        )
        bundle["validation"] = validate(bundle)
        return bundle
