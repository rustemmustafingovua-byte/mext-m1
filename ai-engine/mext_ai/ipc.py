"""Newline-delimited JSON-RPC 2.0 over a Unix domain socket.

Mirror of the Rust implementation in ``core-daemon/src/ipc.rs``. Blocking
sockets, one thread per connection — the engine only serves a handful of local
clients (Rust bridge, dashboard proxies, CLI).
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any, Callable

__all__ = ["RpcError", "RpcClient", "RpcServer"]


class RpcError(RuntimeError):
    """Remote error returned by the peer (or a transport failure)."""


class RpcClient:
    """Blocking JSON-RPC client with automatic reconnect."""

    def __init__(self, path: str | os.PathLike[str], timeout: float = 2.0) -> None:
        self.path = str(path)
        self.timeout = float(timeout)
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self._reader = None
        self._next_id = 0

    # -- connection management -------------------------------------------
    def _connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self.path)
        self._sock = sock
        self._reader = sock.makefile("rb")

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        if self._reader is not None:
            try:
                self._reader.close()
            except OSError:
                pass
            self._reader = None
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def reachable(self) -> bool:
        try:
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            probe.settimeout(0.25)
            probe.connect(self.path)
            probe.close()
            return True
        except OSError:
            return False

    # -- calls ------------------------------------------------------------
    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        last: Exception | None = None
        for attempt in range(2):
            try:
                return self._call_once(method, params or {})
            except (OSError, RpcError, ValueError) as exc:
                last = exc
                with self._lock:
                    self._close_locked()
                if attempt == 0:
                    time.sleep(0.02)
        raise RpcError(str(last)) if last is not None else RpcError("rpc failed")

    def _call_once(self, method: str, params: dict[str, Any]) -> Any:
        with self._lock:
            for attempt in range(2):
                if self._sock is None:
                    try:
                        self._connect()
                    except OSError:
                        if attempt == 1:
                            raise
                        time.sleep(0.02)
                        continue
                self._next_id += 1
                request = {
                    "jsonrpc": "2.0",
                    "id": self._next_id,
                    "method": method,
                    "params": params,
                }
                try:
                    assert self._sock is not None and self._reader is not None
                    self._sock.sendall(json.dumps(request).encode() + b"\n")
                    line = self._reader.readline()
                except OSError:
                    self._close_locked()
                    if attempt == 1:
                        raise
                    continue
                if not line:
                    self._close_locked()
                    if attempt == 1:
                        raise RpcError("peer closed the connection")
                    continue
                response = json.loads(line)
                if "error" in response:
                    message = (response.get("error") or {}).get("message", "rpc error")
                    raise RpcError(str(message))
                return response.get("result")
        raise RpcError("unreachable")


class RpcServer:
    """Unix-socket JSON-RPC server; one thread per client connection."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        handler: Callable[[str, dict[str, Any]], Any],
        name: str = "mext-ai",
    ) -> None:
        self.path = Path(path)
        self.handler = handler
        self.name = name
        self._sock: socket.socket | None = None
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        if len(str(self.path).encode()) > 100:
            raise OSError(
                f"Unix socket path is too long ({len(str(self.path).encode())} bytes, max 104): "
                f"{self.path} — use a shorter state dir (MEXT_HOME)"
            )
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(str(self.path))
        os.chmod(self.path, 0o600)
        sock.listen(32)
        sock.settimeout(0.5)
        self._sock = sock
        self._thread = threading.Thread(target=self._accept_loop, name=f"{self.name}-rpc", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------------
    def _accept_loop(self) -> None:
        while not self._stopped.is_set():
            assert self._sock is not None
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(
                target=self._serve_connection, args=(conn,), name=f"{self.name}-conn", daemon=True
            ).start()

    def _serve_connection(self, conn: socket.socket) -> None:
        buffer = b""
        with conn:
            conn.settimeout(None)
            while not self._stopped.is_set():
                try:
                    chunk = conn.recv(65536)
                except OSError:
                    return
                if not chunk:
                    return
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line.strip():
                        continue
                    response = self._dispatch(line)
                    try:
                        conn.sendall(json.dumps(response).encode() + b"\n")
                    except OSError:
                        return

    def _dispatch(self, raw: bytes) -> dict[str, Any]:
        req_id: Any = None
        try:
            request = json.loads(raw)
            req_id = request.get("id")
            method = request.get("method") or ""
            params = request.get("params") or {}
            result = self.handler(method, params)
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        except Exception as exc:  # noqa: BLE001 — the boundary must never crash
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": f"{type(exc).__name__}: {exc}"},
            }
