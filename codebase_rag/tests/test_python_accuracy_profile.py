"""Real native-updater and canonical API regressions; no API/model or server needed."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from codebase_rag import constants as cs
from codebase_rag.config import settings
from codebase_rag.graph_updater import GraphUpdater
from codebase_rag.integration.api import build_snapshot
from codebase_rag.parser_loader import load_parsers

FIXTURES = Path(__file__).parent / "fixtures/python_accuracy"


@pytest.fixture(params=[cs.PythonFrontend.HEURISTIC, cs.PythonFrontend.JEDI])
def mode(request, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "PYTHON_FRONTEND", request.param)
    import jedi

    monkeypatch.setattr(jedi.settings, "cache_directory", str(tmp_path / "jedi-cache"))
    return request.param


def native(tmp_path, text):
    repo = tmp_path / "fixture"
    repo.mkdir()
    (repo / "sample.py").write_text(text, encoding="utf-8")
    parsers, queries = load_parsers()
    sink = MagicMock()
    GraphUpdater(
        sink,
        repo,
        parsers,
        queries,
        project_name="fixture",
        skip_embeddings=True,
        state_dir=tmp_path / "state",
    ).run(force=True)
    return {
        (
            str(c.args[0][2]).removeprefix("fixture.sample."),
            str(c.args[1]),
            str(c.args[2][2]).removeprefix("fixture.sample."),
        )
        for c in sink.ensure_relationship_batch.call_args_list
        if str(c.args[1]) in {"CALLS", "REFERENCES", "INSTANTIATES"}
    }


def test_references_do_not_invoke_and_dict_dispatch_remains(tmp_path, mode):
    text = "def target():pass\ndef other():pass\nvalues=[target,other]\ndef fixed():\n t={'a':target,'b':other}\n t['a']()\ndef dynamic(k):\n t={'a':target,'b':other}\n t[k]()\n"
    edges = native(tmp_path, text)
    assert ("fixture.sample", "CALLS", "target") not in edges
    assert ("fixture.sample", "REFERENCES", "target") in edges
    assert ("fixed", "CALLS", "target") in edges
    assert ("fixed", "CALLS", "other") not in edges
    assert {("dynamic", "CALLS", "target"), ("dynamic", "CALLS", "other")} <= edges


def test_branch_join_is_not_last_assignment_and_unconditional_kills(tmp_path, mode):
    edges = native(
        tmp_path,
        "def target():pass\ndef other():pass\ndef branch(flag):\n if flag: cb=target\n else: cb=other\n cb()\ndef overwrite():\n cb=target\n cb=other\n cb()\n",
    )
    assert {("branch", "CALLS", "target"), ("branch", "CALLS", "other")} <= edges
    assert ("overwrite", "CALLS", "other") in edges
    assert ("overwrite", "CALLS", "target") not in edges


def test_execution_owners_and_anonymous_nodes(tmp_path, mode):
    edges = native(
        tmp_path,
        "def target():pass\ndef make():pass\ndef factory():\n return lambda:target()\ndef outer(v=make()):\n def inner():target()\n inner()\na,b=(lambda:target()),(lambda:target())\n",
    )
    assert ("factory", "CALLS", "target") not in edges
    assert ("factory.<lambda>#1", "CALLS", "target") in edges
    assert ("outer", "CALLS", "target") not in edges
    assert ("outer.inner", "CALLS", "target") in edges
    assert ("outer", "CALLS", "make") not in edges
    assert ("fixture.sample", "CALLS", "make") in edges
    assert ("<lambda>#1", "CALLS", "target") in edges and (
        "<lambda>#2",
        "CALLS",
        "target",
    ) in edges


def test_class_scope_and_real_methods(tmp_path, mode):
    edges = native(
        tmp_path,
        "class Base:\n def work(self):pass\nclass A(Base):\n def bad(self):work()\n def good(self):self.work()\n def parent(self):super().work()\n @classmethod\n def construct(cls):pass\n @staticmethod\n def static():pass\ndef caller():\n obj=A()\n A.work(obj)\n A.construct()\n A.static()\n",
    )
    assert ("A.bad", "CALLS", "Base.work") not in edges
    assert ("A.good", "CALLS", "Base.work") in edges
    assert ("A.parent", "CALLS", "Base.work") in edges
    assert ("caller", "CALLS", "A.construct") in edges
    assert ("caller", "CALLS", "A.static") in edges


def test_frozen_corpus_candidates_and_unknowns(tmp_path, mode):
    repo = tmp_path / "fixture"
    repo.mkdir()
    for name in ["sample.py", "helpers.py", "registry.py"]:
        (repo / name).write_bytes((FIXTURES / name).read_bytes())
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "add", "--", "sample.py", "helpers.py", "registry.py"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
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
    entry = dict(
        name="fixture",
        path=str(repo),
        files=["sample.py", "helpers.py", "registry.py"],
        source_roots=["."],
    )
    bundle = build_snapshot(
        [entry], mode="jedi" if mode == cs.PythonFrontend.JEDI else "default"
    )
    byid = {n["node_id"]: n for n in bundle["nodes"]}
    sites = [n for n in byid.values() if n["kind"] == "CallSite"]
    text = (repo / "sample.py").read_text()
    truth = json.loads((FIXTURES / "truth.json").read_text())
    checked = []
    for expected in truth["sites"]:
        token = (
            "reference:" if expected["kind"] == "reference" else "case:"
        ) + expected["id"]
        line = next(i + 1 for i, t in enumerate(text.splitlines()) if token in t)
        if expected["kind"] == "reference":
            assert not any(
                n["start"] == line and n["path"] == "sample.py" for n in sites
            )
            checked.append(expected["id"])
            continue
        candidates = [
            n
            for n in sites
            if n["start"] == line
            and n["path"] == "sample.py"
            and (
                not expected.get("expression")
                or n["expression"] == expected["expression"]
            )
        ]
        assert candidates, expected["id"]
        site = candidates[0]
        targets = {byid[i]["qualname"] for i in site["candidates"]}
        for bad in expected["forbidden"]:
            assert bad.split(":", 1)[1] not in targets, expected["id"]
        if expected["id"] == "callback_mixed":
            assert "target" in targets and site["unknown_sources"]
        assert site["exhaustive"] is False
        checked.append(expected["id"])
    assert len(checked) == 39
    assert len({n["node_id"] for n in byid.values()}) == len(byid)
    assert {e["kind"] for e in bundle["edges"]} >= {
        "CALLS",
        "REFERENCES",
        "IMPORTS",
        "INHERITS",
        "CONTAINS",
    }
