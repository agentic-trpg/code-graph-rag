"""Apply canonical Python CALLS/REFERENCES to the existing GraphUpdater sink."""

from pathlib import Path

from .api import analyze_sources
from .model import canonical, digest


def python_source(processor, qualified_name):
    """Only suppress the legacy Python call emitter, never another language."""
    matches = [
        (qn, path)
        for qn, path in processor.module_qn_to_file_path.items()
        if qualified_name == qn or qualified_name.startswith(qn + ".")
    ]
    return bool(
        matches
        and max(matches, key=lambda pair: len(pair[0]))[1].suffix in {".py", ".pyi"}
    )


def emit(processor, file_path):
    """Reuse the updater's exact parsed module map and configured frontend mode."""
    from codebase_rag import constants as cs
    from codebase_rag.parsers.py_frontend import resolve_python_frontend

    bundle = getattr(processor, "_python_accuracy_bundle", None)
    if bundle is None:
        sources = []
        root = processor.repo_path.resolve()
        name = processor.project_name
        for module, path in processor.module_qn_to_file_path.items():
            path = Path(path).resolve()
            if path.suffix not in {".py", ".pyi"} or not path.is_file():
                continue
            raw = path.read_bytes()
            sources.append(
                dict(
                    repository=name,
                    path=path.relative_to(root).as_posix(),
                    module=module.removeprefix(name + "."),
                    absolute=str(path),
                    text=raw.decode("utf-8-sig"),
                    file_sha256=digest(raw),
                    git_head="native-updater-source-capture",
                    git_status="identity supplied by updater provenance",
                )
            )
        repos = [
            dict(
                name=name,
                path=str(root),
                head="native-updater-source-capture",
                branch="unknown",
                status="identity supplied by updater provenance",
                python_clean=True,
                dependency_pins={},
            )
        ]
        mode = (
            "jedi" if resolve_python_frontend() == cs.PythonFrontend.JEDI else "default"
        )
        bundle = analyze_sources(
            sources,
            repos,
            [str(root)] + [str(Path(s["absolute"]).parent) for s in sources],
            mode,
        )
        processor._python_accuracy_bundle = bundle
    byid = {n["node_id"]: n for n in bundle["nodes"]}
    relative = (
        Path(file_path).resolve().relative_to(processor.repo_path.resolve()).as_posix()
    )

    def spec(n):
        if n["kind"] == "Module":
            return (
                "Module",
                "qualified_name",
                processor.project_name + "." + n["qualname"],
            )
        if n["kind"] == "CallSite":
            return ("CallSite", "qualified_name", n["node_id"])
        local = n["qualname"].rsplit(".", 1)[-1]
        if n["subtype"] != "lambda" and n["occurrence"].rsplit(".", 1)[-1] != "1":
            local += "#" + n["occurrence"].rsplit(".", 1)[-1]
        qualified = spec(byid[n["parent"]])[2] + "." + local
        label = (
            "Method"
            if byid[n["parent"]]["kind"] == "Class"
            else "Function"
            if n["kind"] == "Function"
            else "Class"
        )
        return (
            label,
            "qualified_name",
            qualified,
        )

    module = next(
        n for n in bundle["nodes"] if n["kind"] == "Module" and n["path"] == relative
    )
    processor.ingestor.ensure_node_batch(
        "Module",
        dict(
            qualified_name=spec(module)[2],
            python_callsite_evidence=canonical(
                [
                    n
                    for n in bundle["nodes"]
                    if n["kind"] == "CallSite" and n["path"] == relative
                ]
            ),
        ),
    )
    for n in bundle["nodes"]:
        if n["path"] != relative:
            continue
        if n["kind"] not in {"Class", "Function"}:
            continue
        label, _, qn = spec(n)
        processor.ingestor.ensure_node_batch(
            label,
            dict(
                qualified_name=qn,
                name=n["qualname"].rsplit(".", 1)[-1],
                path=n["path"],
                absolute_path=str(processor.repo_path / n["path"]),
                start_line=n["definition_line"],
                start_col=n["column"],
                name_start_line=n["definition_line"],
                name_start_col=n["name_column"],
                end_line=n["end"],
                docstring=n["docstring"],
            ),
        )
    for e in bundle["edges"]:
        if e["kind"] not in {"CALLS", "REFERENCES"}:
            continue
        source = byid[e["source"]]
        if source["path"] != relative:
            continue
        if source["kind"] == "CallSite":
            source = byid[source["execution_owner"]]
        target = byid[e["target"]]
        kind = (
            "INSTANTIATES"
            if e["kind"] == "CALLS" and target["kind"] == "Class"
            else e["kind"]
        )
        processor.ingestor.ensure_relationship_batch(
            spec(source),
            kind,
            spec(target),
            properties=dict(
                line=e["evidence"]["line"],
                col=e["evidence"]["column"],
                phase=e["evidence"]["phase"],
                resolution="candidate",
                exhaustive=False,
            ),
        )
