"""Keep UniXcoder weights available across short-lived CGR commands.

The embedder uses this module for UniXcoder inference when
CGR_UNIXCODER_WORKER_DIR names a private local directory. The worker process
starts at the first inference request and exits after an idle interval.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import ExitStack, suppress
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, cast

from filelock import FileLock, Timeout

if TYPE_CHECKING:
    from typing import BinaryIO


_HOST = "127.0.0.1"
_READY_SECONDS = 8.0
_INFERENCE_SECONDS = 300.0
_IDLE_SECONDS = 300.0
_HANDSHAKE_SECONDS = 0.75
_POLL_SECONDS = 0.05
_NONCE_SIZE = 32
_MAX_ENDPOINT_SIZE = 16 * 1024
_MAX_FRAME_SIZE = 128 * 1024 * 1024
_ENDPOINT_FILENAME = "endpoint.json"
_DEVICE_VARIABLES = (
    "CGR_EMBEDDING_DEVICE",
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "PYTORCH_ENABLE_MPS_FALLBACK",
)


class WorkerError(RuntimeError):
    """Report a worker failure without exposing inputs or credentials."""


@dataclass(frozen=True)
class _Scope:
    """Hold namespace paths and the effective launch configuration."""

    directory: Path
    model_path: str | None
    device: str | None

    @property
    def endpoint(self) -> Path:
        """Return the path to the current worker's discovery record."""
        return self.directory / _ENDPOINT_FILENAME

    def request_lock(self) -> FileLock:
        """Serialize startup, requests and explicit shutdown across callers."""
        return FileLock(
            self.directory / "request.lock",
            timeout=_READY_SECONDS + _INFERENCE_SECONDS,
        )

    def lifetime_lock(self) -> FileLock:
        """Reserve the namespace while the worker owns its endpoint."""
        return FileLock(self.directory / "worker.lock", timeout=0)


@cache
def _implementation_digest() -> str:
    """Fingerprint the packaged inference implementation once per process."""
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for filename in ("embedding_worker.py", "unixcoder_runtime.py", "unixcoder.py"):
        digest.update(hashlib.sha256((package / filename).read_bytes()).digest())
    return digest.hexdigest()


def _scope() -> _Scope:
    """Resolve the selected namespace without creating files or launching a worker."""
    from .config import settings

    root = os.environ["CGR_UNIXCODER_WORKER_DIR"]
    configured_model = os.environ.get("CGR_UNIXCODER_MODEL_PATH")
    model = str(Path(configured_model).resolve()) if configured_model else None
    device = settings.EMBEDDING_DEVICE
    identity = [
        "unixcoder-worker-v1",
        str(Path(__file__).resolve()),
        _implementation_digest(),
        sys.executable,
        sys.prefix,
        model,
        os.environ.get("CGR_EMBEDDING_REVISION"),
        device,
        *(os.environ.get(name) for name in _DEVICE_VARIABLES),
    ]
    digest = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:32]
    return _Scope(Path(root).resolve() / digest, model, device)


def _prepare_directory(directory: Path) -> None:
    """Prepare the discovery directory with POSIX owner-only permissions."""
    directory.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name != "nt":
        root = directory.parent.stat()
        metadata = directory.lstat()
        if (
            root.st_uid != os.getuid()
            or root.st_mode & 0o022
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o077
        ):
            raise WorkerError("UniXcoder worker directory must be private")


def _windows_discovery(payload: bytes, path: Path, *, protect: bool) -> bytes:
    """Apply current-user DPAPI encryption or decryption to a discovery record."""
    if sys.platform != "win32":
        raise WorkerError("DPAPI discovery protection requires Windows")
    import ctypes
    from ctypes import wintypes

    class DataBlob(ctypes.Structure):
        """Hold a Windows DATA_BLOB byte count and buffer pointer."""

        size: int
        buffer: int
        _fields_ = [("size", wintypes.DWORD), ("buffer", ctypes.c_void_p)]

    crypt32 = ctypes.WinDLL("Crypt32.dll", use_last_error=True)
    transform = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    blob_pointer = ctypes.POINTER(DataBlob)
    transform.argtypes = (
        blob_pointer,
        ctypes.c_void_p,
        blob_pointer,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        blob_pointer,
    )
    transform.restype = wintypes.BOOL
    release = ctypes.WinDLL("Kernel32.dll").LocalFree
    release.argtypes = (ctypes.c_void_p,)
    release.restype = ctypes.c_void_p
    entropy = os.path.normcase(os.fspath(path)).encode("utf-8")
    data_buffer = ctypes.create_string_buffer(payload)
    entropy_buffer = ctypes.create_string_buffer(entropy)
    data = DataBlob(len(payload), ctypes.addressof(data_buffer))
    binding = DataBlob(len(entropy), ctypes.addressof(entropy_buffer))
    result = DataBlob()
    ui_forbidden = 0x01
    if not transform(
        ctypes.byref(data),
        None,
        ctypes.byref(binding),
        None,
        None,
        ui_forbidden,
        ctypes.byref(result),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(result.buffer, result.size)
    finally:
        release(result.buffer)


def _write_endpoint(path: Path, port: int, key: bytes) -> None:
    """Publish the protected discovery record atomically after bind."""
    record: dict[str, int | str] = {"port": port, "key": key.hex()}
    if os.name == "nt":
        protected = _windows_discovery(json.dumps(record).encode(), path, protect=True)
        record = {"protected": base64.b64encode(protected).decode("ascii")}
    # The PID is diagnostic data only. Clients authorize peers by possession of the key.
    record["pid"] = os.getpid()
    pending = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    )
    temporary = Path(pending.name)
    try:
        with pending:
            json.dump(record, pending)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_endpoint(path: Path) -> tuple[int, bytes]:
    """Decode a bounded discovery record and require current-user DPAPI on Windows."""
    with path.open("rb") as discovery:
        raw = discovery.read(_MAX_ENDPOINT_SIZE + 1)
    if len(raw) > _MAX_ENDPOINT_SIZE:
        raise ValueError("UniXcoder worker discovery exceeded the size limit")
    record = json.loads(raw)
    if os.name == "nt":
        protected = base64.b64decode(record["protected"], validate=True)
        record = json.loads(_windows_discovery(protected, path, protect=False))
    port = record["port"]
    key = bytes.fromhex(record["key"])
    if not isinstance(port, int) or not 1 <= port <= 65535 or len(key) != _NONCE_SIZE:
        raise ValueError("UniXcoder worker discovery is invalid")
    return port, key


def _proof(key: bytes, role: bytes, client: bytes, server: bytes) -> bytes:
    """Sign fresh handshake nonces for the specified endpoint role."""
    return hmac.digest(key, role + client + server, "sha256")


def _authenticate_server(
    connection: socket.socket, reader: BinaryIO, key: bytes
) -> None:
    """Authenticate the server before sending code or command data."""
    client = secrets.token_bytes(_NONCE_SIZE)
    connection.sendall(client)
    server = reader.read(_NONCE_SIZE)
    proof = reader.read(_NONCE_SIZE)
    if len(server) != _NONCE_SIZE or not hmac.compare_digest(
        proof, _proof(key, b"server", client, server)
    ):
        raise WorkerError("UniXcoder worker authentication failed")
    connection.sendall(_proof(key, b"client", client, server))


def _authenticate_client(
    connection: socket.socket, reader: BinaryIO, key: bytes
) -> bool:
    """Confirm that the client can read the private discovery record."""
    client = reader.read(_NONCE_SIZE)
    if len(client) != _NONCE_SIZE:
        return False
    server = secrets.token_bytes(_NONCE_SIZE)
    connection.sendall(server + _proof(key, b"server", client, server))
    return hmac.compare_digest(
        reader.read(_NONCE_SIZE), _proof(key, b"client", client, server)
    )


def _send(connection: socket.socket, value: object) -> None:
    """Send one bounded newline-delimited JSON message."""
    payload = json.dumps(value, separators=(",", ":")).encode("utf-8") + b"\n"
    if len(payload) > _MAX_FRAME_SIZE:
        raise WorkerError("UniXcoder worker message exceeded the size limit")
    connection.sendall(payload)


def _receive(reader: BinaryIO) -> dict[str, object]:
    """Read a complete bounded JSON object from a peer."""
    payload = reader.readline(_MAX_FRAME_SIZE + 1)
    if len(payload) > _MAX_FRAME_SIZE or not payload.endswith(b"\n"):
        raise WorkerError("UniXcoder worker sent an incomplete or oversized message")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise WorkerError("UniXcoder worker sent an invalid message")
    return value


def _connect(scope: _Scope, owned: ExitStack) -> tuple[socket.socket, BinaryIO] | None:
    """Find and authenticate the existing worker, before dispatching any request."""
    with ExitStack() as attempt:
        try:
            port, key = _read_endpoint(scope.endpoint)
            connection = attempt.enter_context(
                socket.create_connection((_HOST, port), _HANDSHAKE_SECONDS)
            )
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            reader = attempt.enter_context(connection.makefile("rb"))
            _authenticate_server(connection, reader, key)
        except (OSError, WorkerError, ValueError, TypeError, KeyError):
            return None
        owned.enter_context(attempt.pop_all())
        return connection, reader


def _launch(scope: _Scope) -> subprocess.Popen[bytes]:
    """Launch a detached child that imports the model only on an inference request."""
    launch_env = os.environ.copy()
    if scope.model_path is not None:
        launch_env["CGR_UNIXCODER_MODEL_PATH"] = scope.model_path
    if scope.device is not None:
        launch_env["CGR_EMBEDDING_DEVICE"] = scope.device
    process = subprocess.Popen(
        [
            sys.executable,
            "-P",
            "-m",
            "codebase_rag.embedding_worker",
            str(scope.directory),
        ],
        env=launch_env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=os.name != "nt",
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            if os.name == "nt"
            else 0
        ),
    )
    threading.Thread(target=process.wait, daemon=True).start()
    return process


def _ready_worker(scope: _Scope, owned: ExitStack) -> tuple[socket.socket, BinaryIO]:
    """Start a replacement only when no worker holds the lifetime lock."""
    deadline = time.monotonic() + _READY_SECONDS
    process = None
    try:
        while time.monotonic() < deadline:
            if (connection := _connect(scope, owned)) is not None:
                return connection
            if process is not None:
                if process.poll() is not None:
                    raise WorkerError("UniXcoder worker exited before becoming ready")
            else:
                try:
                    with scope.lifetime_lock():
                        scope.endpoint.unlink(missing_ok=True)
                except Timeout:
                    pass
                else:
                    process = _launch(scope)
            time.sleep(_POLL_SECONDS)
        raise WorkerError("UniXcoder worker did not become ready")
    except BaseException:
        if process is not None and process.poll() is None:
            with suppress(OSError):
                process.kill()
        raise


def _request(payload: dict[str, object]) -> object:
    """Submit an inference request once and return the decoded result."""
    scope = _scope()
    _prepare_directory(scope.directory)
    try:
        with scope.request_lock(), ExitStack() as owned:
            connection, reader = _ready_worker(scope, owned)
            return _exchange(connection, reader, payload)
    except (OSError, Timeout) as error:
        raise WorkerError(
            "UniXcoder worker was unavailable during inference"
        ) from error


def _exchange(
    connection: socket.socket, reader: BinaryIO, payload: dict[str, object]
) -> object:
    """Read a response within the bounded inference window."""
    connection.settimeout(_INFERENCE_SECONDS)
    _send(connection, payload)
    response = _receive(reader)
    if "result" not in response:
        raise WorkerError(
            f"UniXcoder worker operation failed ({response.get('error')})"
        )
    return response["result"]


def embed_code(code: str, max_length: int) -> list[float]:
    """Embed a single cache miss with the stock UniXcoder single-snippet path."""
    return cast(
        "list[float]",
        _request({"op": "code", "code": code, "max_length": max_length}),
    )


def embed_batch(
    snippets: list[str], max_length: int, batch_size: int
) -> list[list[float]]:
    """Embed ordered cache misses with the caller's original batch settings."""
    if not snippets:
        return []
    return cast(
        "list[list[float]]",
        _request(
            {
                "op": "batch",
                "snippets": snippets,
                "max_length": max_length,
                "batch_size": batch_size,
            }
        ),
    )


def stop_worker() -> bool:
    """Stop the worker for the current namespace, if present, without starting one."""
    if not os.environ.get("CGR_UNIXCODER_WORKER_DIR"):
        return False
    scope = _scope()
    if not scope.directory.is_dir():
        return False
    _prepare_directory(scope.directory)
    try:
        with scope.request_lock(), ExitStack() as owned:
            if (connection := _connect(scope, owned)) is None:
                return False
            _exchange(*connection, {"op": "stop"})
            deadline = time.monotonic() + _READY_SECONDS
            while time.monotonic() < deadline:
                try:
                    with scope.lifetime_lock():
                        return True
                except Timeout:
                    time.sleep(_POLL_SECONDS)
            raise WorkerError("UniXcoder worker did not shut down")
    except (OSError, Timeout) as error:
        raise WorkerError("UniXcoder worker could not be stopped") from error


def _dispatch(request: dict[str, object]) -> object:
    """Call the stock inference path with authenticated, schema-checked inputs."""
    from .unixcoder_runtime import _unixcoder_embed_batch, _unixcoder_embed_code

    max_length = request["max_length"]
    if not isinstance(max_length, int):
        raise ValueError("max_length must be an integer")
    if request["op"] == "code" and isinstance(code := request["code"], str):
        return _unixcoder_embed_code(code, max_length)
    if (
        request["op"] == "batch"
        and isinstance(snippets := request["snippets"], list)
        and all(isinstance(snippet, str) for snippet in snippets)
        and isinstance(batch_size := request["batch_size"], int)
    ):
        return _unixcoder_embed_batch(snippets, max_length, batch_size)
    raise ValueError("Unknown UniXcoder worker operation")


def _handle(
    connection: socket.socket, key: bytes, deadline: float
) -> tuple[bool, bool]:
    """Return whether an authenticated request was received and requested shutdown."""
    remaining = max(0, deadline - time.monotonic())
    if not remaining:
        return False, False
    connection.settimeout(min(_HANDSHAKE_SECONDS, remaining))
    with connection.makefile("rb") as reader:
        try:
            if not _authenticate_client(connection, reader, key):
                return False, False
            connection.settimeout(_READY_SECONDS)
            request = _receive(reader)
        except (OSError, ValueError, WorkerError):
            return False, False
    connection.settimeout(_INFERENCE_SECONDS)
    if request.get("op") == "stop":
        _send(connection, {"result": True})
        return True, True

    # A dead client must not leave a worker stuck in native inference indefinitely.
    watchdog = threading.Timer(_INFERENCE_SECONDS, os._exit, args=(1,))
    watchdog.daemon = True
    watchdog.start()
    try:
        try:
            _send(connection, {"result": _dispatch(request)})
        except Exception as error:
            _send(connection, {"error": type(error).__name__})
    except OSError:
        pass
    finally:
        watchdog.cancel()
    return True, False


def _serve(listener: socket.socket, key: bytes) -> None:
    """Serve sequential requests and reclaim the process once it has gone idle."""
    deadline = time.monotonic() + _IDLE_SECONDS
    while (remaining := deadline - time.monotonic()) > 0:
        listener.settimeout(remaining)
        try:
            connection, _ = listener.accept()
        except TimeoutError:
            return
        with connection:
            handled, stop = _handle(connection, key, deadline)
        if stop:
            return
        if handled:
            deadline = time.monotonic() + _IDLE_SECONDS


def _run(directory: Path) -> None:
    """Own the endpoint for one worker lifetime and withdraw it before exit."""
    scope = _Scope(directory, None, None)
    with (
        scope.lifetime_lock(),
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener,
    ):
        if os.name == "nt":
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        listener.bind((_HOST, 0))
        listener.listen()
        key = secrets.token_bytes(_NONCE_SIZE)
        _write_endpoint(scope.endpoint, listener.getsockname()[1], key)
        try:
            _serve(listener, key)
        finally:
            scope.endpoint.unlink(missing_ok=True)


if __name__ == "__main__":
    _run(Path(sys.argv[1]))
