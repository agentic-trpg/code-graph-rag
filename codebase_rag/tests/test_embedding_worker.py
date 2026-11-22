"""Exercise the UniXcoder worker across CLI processes with a deterministic model."""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from codebase_rag import embedding_worker

_BACKEND = """
import json
import os

def record(op, value, max_length, batch_size=None):
    with open(os.environ["WORKER_TEST_EVENTS"], "a", encoding="utf-8") as events:
        events.write(json.dumps([os.getpid(), op, value, max_length, batch_size]) + "\\n")

def _unixcoder_embed_code(code, max_length):
    record("code", code, max_length)
    if code == "crash":
        os._exit(17)
    if code == "error":
        raise ValueError("private inference details")
    return [float(len(code)), float(max_length)]

def _unixcoder_embed_batch(snippets, max_length, batch_size):
    record("batch", snippets, max_length, batch_size)
    return [[float(len(code)), float(max_length), float(batch_size)] for code in snippets]
"""


class WorkerFixture:
    """Run independent clients against the production worker with a stub model."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.namespace = directory / "workers"
        self.events_file = directory / "events.jsonl"
        package = directory / "codebase_rag"
        package.mkdir()
        (package / "__init__.py").touch()
        (package / "config.py").write_text(
            "from types import SimpleNamespace\n"
            "settings = SimpleNamespace(EMBEDDING_DEVICE=None)\n",
            encoding="utf-8",
        )
        shutil.copyfile(embedding_worker.__file__, package / "embedding_worker.py")
        (package / "unixcoder_runtime.py").write_text(_BACKEND, encoding="utf-8")
        (package / "unixcoder.py").touch()
        self.env = {
            **os.environ,
            "PYTHONPATH": str(directory),
            "CGR_UNIXCODER_WORKER_DIR": str(self.namespace),
            "CGR_UNIXCODER_MODEL_PATH": str(directory / "model"),
            "CGR_EMBEDDING_REVISION": "worker-test",
            "WORKER_TEST_EVENTS": str(self.events_file),
        }

    def start(self, source: str) -> subprocess.Popen[str]:
        """Start a separate client without importing a model in that process."""
        return subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from codebase_rag.embedding_worker import "
                f"embed_code, embed_batch, stop_worker, WorkerError\n{source}",
            ],
            cwd=self.directory,
            env=self.env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    @staticmethod
    def finish(client: subprocess.Popen[str]) -> str:
        """Return client output and report failures with their stderr."""
        try:
            output, error = client.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            client.kill()
            client.communicate()
            raise
        assert client.returncode == 0, error
        return output.strip()

    def run(self, source: str) -> str:
        """Run a client program to completion."""
        with self.start(source) as client:
            return self.finish(client)

    def events(self) -> list[list]:
        """Read the inputs and process identities recorded by inference."""
        return [
            json.loads(line)
            for line in self.events_file.read_text(encoding="utf-8").splitlines()
        ]


@pytest.fixture
def worker(tmp_path: Path) -> Iterator[WorkerFixture]:
    """Reserve a test-owned worker namespace for every case."""
    fixture = WorkerFixture(tmp_path)
    try:
        yield fixture
    finally:
        if fixture.namespace.exists():
            fixture.run("stop_worker()")


def test_reuse_across_cli_processes(worker: WorkerFixture) -> None:
    """Use one worker for concurrently started clients and preserve argument order."""
    assert worker.run("print(embed_batch([], 25, 2))") == "[]"
    assert not worker.namespace.exists()

    with (
        worker.start("print(embed_code('hé', 24))") as one,
        worker.start("print(embed_batch(['abc', 'a', 'abc'], 25, 2))") as batch,
    ):
        assert worker.finish(one) == "[2.0, 24.0]"
        assert (
            worker.finish(batch)
            == "[[3.0, 25.0, 2.0], [1.0, 25.0, 2.0], [3.0, 25.0, 2.0]]"
        )

    assert worker.run("print(embed_code('z', 1))") == "[1.0, 1.0]"
    events = worker.events()
    assert len({event[0] for event in events}) == 1
    assert sorted(event[1:] for event in events[:2]) == [
        ["batch", ["abc", "a", "abc"], 25, 2],
        ["code", "hé", 24, None],
    ]
    assert events[2][1:] == ["code", "z", 1, None]

    assert worker.run("print(stop_worker())") == "True"
    assert not list(worker.namespace.rglob("endpoint.json"))
    assert worker.run("print(stop_worker())") == "False"


def test_failed_inference_is_not_replayed(worker: WorkerFixture) -> None:
    """Surface failures once and allow a later request to replace a dead worker."""
    for value in ("error", "crash"):
        output = worker.run(
            f"try:\n    embed_code({value!r}, 20)\n"
            "except WorkerError as error:\n    print(error)\n"
            "else:\n    raise AssertionError('Expected WorkerError')"
        )
        assert "UniXcoder worker" in output
        assert "private inference details" not in output

    assert worker.run("print(embed_code('new', 20))") == "[3.0, 20.0]"
    events = worker.events()
    assert [event[2] for event in events] == ["error", "crash", "new"]
    assert events[0][0] == events[1][0] != events[2][0]


def test_installed_implementation_change_selects_new_namespace(
    worker: WorkerFixture,
) -> None:
    """Open a different worker namespace after replacing implementation in place."""
    probe = (
        "from codebase_rag.embedding_worker import _scope\n"
        "print(_scope().directory.name)"
    )
    before = worker.run(probe)
    assert worker.run(probe) == before
    for filename in ("embedding_worker.py", "unixcoder_runtime.py", "unixcoder.py"):
        path = worker.directory / "codebase_rag" / filename
        with path.open("a", encoding="utf-8") as source:
            source.write("\n# In-place installation update\n")
        after = worker.run(probe)
        assert after != before
        before = after
    assert not worker.namespace.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows discovery uses DPAPI")
def test_windows_discovery_protects_port_and_key(tmp_path: Path) -> None:
    """Keep credentials out of Windows discovery JSON and refuse corrupted data."""
    endpoint = tmp_path / "endpoint.json"
    key = bytes(range(32))
    port = 50491
    embedding_worker._write_endpoint(endpoint, port, key)

    record = json.loads(endpoint.read_bytes())
    assert set(record) == {"pid", "protected"}
    assert record["pid"] == os.getpid()
    assert embedding_worker._read_endpoint(endpoint) == (port, key)

    # Diagnostics can be changed independently of the authenticated connection record.
    record["pid"] = 0
    endpoint.write_text(json.dumps(record), encoding="utf-8")
    assert embedding_worker._read_endpoint(endpoint) == (port, key)

    moved = endpoint.with_name("different-endpoint.json")
    moved.write_bytes(endpoint.read_bytes())
    with pytest.raises(OSError):
        embedding_worker._read_endpoint(moved)

    tampered = bytearray(base64.b64decode(record["protected"]))
    tampered[len(tampered) // 2] ^= 1
    record["protected"] = base64.b64encode(tampered).decode("ascii")
    endpoint.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises((OSError, ValueError)):
        embedding_worker._read_endpoint(endpoint)
