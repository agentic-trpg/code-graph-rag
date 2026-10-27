"""Compare a manually frozen source-only corpus, preserving misses and raw uncertainty."""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .api import build_snapshot


def evaluate(fixture, mode):
    """No imports of fixture code and no inference-derived expected labels."""
    truth = json.loads((fixture / "truth.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="cgr-frozen-") as directory:
        root = Path(directory)
        for name in ["sample.py", "helpers.py", "registry.py"]:
            shutil.copyfile(fixture / name, root / name)
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "add",
                "--",
                "sample.py",
                "helpers.py",
                "registry.py",
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "-qm",
                "Fixture",
            ],
            check=True,
        )
        bundle = build_snapshot(
            [
                dict(
                    name="fixture",
                    path=str(root),
                    files=["sample.py", "helpers.py", "registry.py"],
                    source_roots=["."],
                )
            ],
            mode=mode,
        )
    byid = {n["node_id"]: n for n in bundle["nodes"]}

    def label(node):
        module = byid.get(node.get("module_id"), node)["qualname"]
        qual = (
            "<module>"
            if node["kind"] == "Module"
            else node["qualname"].replace("<lambda>#1", "<lambda>")
        )
        return module + ":" + qual

    text = (fixture / "sample.py").read_text(encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for expected in truth["sites"]:
        token = (
            "reference:" if expected["kind"] == "reference" else "case:"
        ) + expected["id"]
        line = next(i + 1 for i, s in enumerate(text.splitlines()) if token in s)
        if expected["kind"] == "reference":
            selected = [
                e
                for e in bundle["edges"]
                if e["kind"] == "REFERENCES"
                and e["evidence"]["line"] == line
                and byid[e["source"]]["path"] == expected["file"]
            ]
            targets = {label(byid[e["target"]]) for e in selected}
            site = None
        else:
            sites = [
                n
                for n in bundle["nodes"]
                if n["kind"] == "CallSite"
                and n["path"] == expected["file"]
                and n["start"] == line
                and (
                    not expected.get("expression")
                    or n["expression"] == expected["expression"]
                )
            ]
            site = sites[0] if sites else None
            targets = {label(byid[t]) for t in site["candidates"]} if site else set()
        want = set(expected["targets"])
        forbidden = targets & set(expected["forbidden"])
        rows.append(
            dict(
                case=expected["id"],
                mode=mode,
                line=line,
                expected=sorted(want),
                actual=sorted(targets),
                missing=sorted(want - targets),
                extra=sorted(targets - want) if expected["complete"] else [],
                forbidden=sorted(forbidden),
                site_recorded=site is not None
                if expected["kind"] != "reference"
                else True,
                expected_owner=expected["context"],
                actual_owner=label(byid[site["execution_owner"]]) if site else None,
                raw=site.get("raw", []) if site else [],
                reasons=site.get("reasons", []) if site else [],
                phase=site.get("phase") if site else None,
                unknown_sources=site.get("unknown_sources") if site else None,
                exhaustive=site.get("exhaustive") if site else None,
                complete_truth=expected["complete"],
            )
        )
    closed = [r for r in rows if r["complete_truth"]]
    tp = sum(len(set(r["expected"]) & set(r["actual"])) for r in closed)
    fp = sum(len(r["extra"]) for r in closed)
    fn = sum(len(r["missing"]) for r in closed)
    return dict(
        mode=mode,
        manual_sites=len(rows),
        closed_sites=len(closed),
        tp=tp,
        fp=fp,
        fn=fn,
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        forbidden=sum(len(r["forbidden"]) for r in rows),
        missing_call_positions=sum(not r["site_recorded"] for r in rows),
        definite_claims=sum(r["exhaustive"] is True for r in rows),
        rows=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = [evaluate(args.fixtures, mode) for mode in ["default", "jedi"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    sys.stdout.write(
        json.dumps([{k: v for k, v in r.items() if k != "rows"} for r in result]) + "\n"
    )


if __name__ == "__main__":
    main()
