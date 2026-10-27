"""Snapshot source identity, unique declarations and cross-process reproducibility."""

import os
import subprocess
import sys

import pytest

from codebase_rag.integration.api import build_snapshot
from codebase_rag.integration.model import verify_sources


def repository(tmp_path):
    root = tmp_path / "fixture"
    root.mkdir()
    (root / "m.py").write_text(
        "def one(): pass\ndef one(): return 1\na,b=(lambda:one()),(lambda:one())\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "--", "m.py"], check=True)
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
    return root, [
        dict(name="fixture", path=str(root), files=["m.py"], source_roots=["."])
    ]


def test_unique_redefinitions_lambdas_and_dirty_identity(tmp_path):
    root, spec = repository(tmp_path)
    before = build_snapshot(spec)
    functions = [n for n in before["nodes"] if n["kind"] == "Function"]
    assert len(functions) == 4 and len({n["node_id"] for n in functions}) == 4
    (root / "m.py").write_text(
        (root / "m.py").read_text(encoding="utf-8") + "\n# dirty capture\n",
        encoding="utf-8",
    )
    after = build_snapshot(spec)
    assert after["source_snapshot_id"] != before["source_snapshot_id"]
    assert after["changes"]["added"] == ["fixture:m.py"]
    with pytest.raises(ValueError, match="changed"):
        verify_sources(before)


@pytest.mark.parametrize("mode", ["default", "jedi"])
def test_result_identity_is_independent_of_python_hash_seed(tmp_path, mode):
    root, _ = repository(tmp_path)
    script = """
import sys
from codebase_rag.integration.api import build_snapshot
from codebase_rag.integration.model import canonical,digest
b=build_snapshot([dict(name='fixture',path=sys.argv[1],files=['m.py'],source_roots=['.'])],mode=sys.argv[2])
x=dict(nodes=sorted(b['nodes'],key=lambda n:n['node_id']),edges=sorted(b['edges'],key=lambda e:e['edge_id']),observations=sorted(b['observations'],key=canonical))
print(b['snapshot_id'],digest(canonical(x)))
"""
    results = []
    for seed in ["1", "2"]:
        result = subprocess.run(
            [sys.executable, "-B", "-c", script, str(root), mode],
            env=dict(os.environ, PYTHONHASHSEED=seed),
            check=True,
            capture_output=True,
            text=True,
            timeout=90,
        )
        results.append(result.stdout.strip())
    assert results[0] == results[1]
