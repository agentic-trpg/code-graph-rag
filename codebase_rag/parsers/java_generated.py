"""Annotation-processor generated-source discovery (issue #1140, tier 2).

Java codebases ship a large slice of their API surface as generated code
(Lombok, Dagger, MapStruct, AutoValue); the checked-in tree hides it under
`target/` and `build/`, which the walk prunes wholesale, so every call into a
generated method dangles. This module finds the standard generated-source
locations NEXT TO a build file (never by walking inside the pruned dirs
generally), so the indexer can carve them out of the prune, register them as
Java import-probe roots, and stamp their modules `generated: true`.
"""

from __future__ import annotations

from pathlib import Path

from ..utils.path_utils import (
    is_symlink_entry,
    is_walked_dir,
    should_skip_rel_file,
    walk_repo_dirs,
)

_BUILD_FILES = ("pom.xml", "build.gradle", "build.gradle.kts")
_MAVEN_GENERATED = ("target", "generated-sources")
_GRADLE_GENERATED = ("build", "generated", "sources", "annotationProcessor")


def _build_dirs(
    repo_path: Path,
    exclude_paths: frozenset[str] | None,
    unignore_paths: frozenset[str] | None,
) -> list[Path]:
    """Find reachable directories containing an eligible build file."""
    return [
        Path(directory)
        for directory, parts, _, prefix, filenames in walk_repo_dirs(
            repo_path, exclude_paths, unignore_paths
        )
        if any(
            name in filenames
            and not should_skip_rel_file(
                f"{prefix}{name}", parts, exclude_paths, unignore_paths
            )
            and not is_symlink_entry(Path(directory) / name)
            for name in _BUILD_FILES
        )
    ]


def _package_roots(
    repo_path: Path,
    generated_dir: Path,
    depth: int,
    exclude_paths: frozenset[str] | None,
) -> list[Path]:
    """Find package roots below a known generator output, respecting excludes."""
    # The processor-named (maven: annotations/, protobuf/) or
    # sourceSet-named (gradle: java/main/) level(s) under the generated dir
    # are the package-tree roots the import probe needs.
    relative = generated_dir.relative_to(repo_path)
    # The standard output path may be under a built-in ignore. An explicit
    # exclusion anywhere above or within it still wins.
    rescue = frozenset({f"/{relative.as_posix()}/**"})
    if not is_walked_dir(relative.parts, exclude_paths, rescue):
        return []
    if (
        any(
            is_symlink_entry(path)
            for path in (generated_dir, *generated_dir.parents)
            if path.is_relative_to(repo_path) and path != repo_path
        )
        or not generated_dir.is_dir()
    ):
        return []
    roots = [generated_dir]
    for _ in range(depth):
        roots = [
            child
            for parent in roots
            for child in sorted(parent.iterdir())
            if is_walked_dir(child.relative_to(repo_path).parts, exclude_paths, rescue)
            and not is_symlink_entry(child)
            and child.is_dir()
        ]
    return roots


def discover_generated_source_roots(
    repo_path: Path,
    exclude_paths: frozenset[str] | None = None,
    unignore_paths: frozenset[str] | None = None,
) -> list[tuple[str, ...]]:
    """Find on-disk Java generated-source roots as sorted relative path parts."""
    repo_path = repo_path.resolve()
    roots: list[tuple[str, ...]] = []
    for build_dir in _build_dirs(repo_path, exclude_paths, unignore_paths):
        for output, depth in ((_MAVEN_GENERATED, 1), (_GRADLE_GENERATED, 2)):
            roots.extend(
                root.relative_to(repo_path).parts
                for root in _package_roots(
                    repo_path, build_dir.joinpath(*output), depth, exclude_paths
                )
            )
    return sorted(set(roots))


def unignore_patterns_for(roots: list[tuple[str, ...]]) -> frozenset[str]:
    # Concrete anchored patterns only: a `**/target/...` wildcard would defeat
    # the prune under every ignored directory in the repo.
    return frozenset(f"{'/'.join(root)}/**" for root in roots)


def generated_prefixes_for(roots: list[tuple[str, ...]]) -> list[str]:
    return [f"{'/'.join(root)}/" for root in roots]


def generator_hint(rel_path: str, prefixes: list[str]) -> str | None:
    # The processor name is the path segment naming the generated tree's
    # subdirectory: maven's target/generated-sources/<processor>/..., or
    # gradle's .../annotationProcessor/java/<sourceSet>/... where the fixed
    # `annotationProcessor` segment is the honest hint.
    for prefix in prefixes:
        if rel_path.startswith(prefix):
            parts = prefix.rstrip("/").split("/")
            if "annotationProcessor" in parts:
                return "annotationProcessor"
            return parts[-1]
    return None
