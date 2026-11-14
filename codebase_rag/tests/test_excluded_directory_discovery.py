import os
from pathlib import Path

from codebase_rag.parsers.go.module_paths import discover_go_module_paths
from codebase_rag.parsers.go_frontend import find_go_module
from codebase_rag.parsers.java_generated import discover_generated_source_roots
from codebase_rag.parsers.js_ts.module_paths import discover_js_workspace_packages
from codebase_rag.parsers.structure_processor import StructureProcessor
from codebase_rag.utils.path_utils import walk_eligible_files


def _write_project(path: Path, name: str) -> None:
    """Write Go, JavaScript and Java discovery markers in the same directory."""
    generated = path / "target/generated-sources/annotations"
    generated.mkdir(parents=True)
    (generated / "Example.java").write_text("class Example {}", encoding="utf-8")
    (path / "go.mod").write_text(f"module example.org/{name}\n", encoding="utf-8")
    (path / "package.json").write_text(f'{{"name":"{name}"}}', encoding="utf-8")
    (path / "pom.xml").write_text("<project/>", encoding="utf-8")


def test_repository_discovery_prunes_before_descent(
    tmp_path, monkeypatch, mock_ingestor
):
    """Hold results and directory visits fixed as excluded projects accumulate."""
    _write_project(tmp_path / "app", "app")
    _write_project(tmp_path / "build/keep", "keep")
    excluded = tmp_path / ".worktrees"
    excluded.mkdir()
    excludes = frozenset({".worktrees/", "app/target/"})
    unignores = frozenset({"/build/keep/**", "/.worktrees/**"})
    processor = StructureProcessor(
        mock_ingestor, tmp_path, "example", {}, unignores, excludes
    )

    visited = []
    scandir = os.scandir

    def record(path):
        visited.append(Path(path).relative_to(tmp_path))
        return scandir(path)

    monkeypatch.setattr(os, "scandir", record)

    def discover():
        """Run the real discovery entry points and capture their directory visits."""
        visited.clear()
        assert processor._directories_to_derive(None) >= {
            tmp_path,
            tmp_path / "app",
            tmp_path / "build/keep",
        }
        assert [
            item[0] for item in discover_go_module_paths(tmp_path, excludes, unignores)
        ] == ["example.org/keep", "example.org/app"]
        assert find_go_module(tmp_path, excludes, unignores) == tmp_path / "app/go.mod"
        assert discover_js_workspace_packages(tmp_path, excludes, unignores) == [
            ("keep", tmp_path / "build/keep"),
            ("app", tmp_path / "app"),
        ]
        assert discover_generated_source_roots(tmp_path, excludes, unignores) == [
            ("build", "keep", "target", "generated-sources", "annotations")
        ]
        assert list(walk_eligible_files(tmp_path, excludes, unignores))
        return tuple(visited)

    before = discover()
    for index in range(20):
        _write_project(excluded / str(index), f"excluded{index}")
    assert discover() == before
    assert all(path.parts[:1] != (".worktrees",) for path in before)
    visited.clear()
    assert processor._directories_to_derive({"app", ".worktrees/0"}) == {
        tmp_path,
        tmp_path / "app",
    }
    assert not visited
