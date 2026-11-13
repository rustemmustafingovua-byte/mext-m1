import shutil
import tempfile
import threading
from pathlib import Path

import pytest

from mext_ai.ipc import RpcClient, RpcError, RpcServer


@pytest.fixture()
def sock_path():
    # Unix sockets are limited to 104 bytes; pytest's tmp_path is far too long.
    directory = Path(tempfile.mkdtemp(prefix="mt-", dir="/tmp"))
    try:
        yield directory / "s.sock"
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_roundtrip_and_errors(sock_path):
    def handler(method, params):
        if method == "ping":
            return {"pong": True}
        if method == "echo":
            return {"echo": params}
        raise ValueError(f"unknown {method}")

    server = RpcServer(sock_path, handler)
    server.start()
    client = RpcClient(sock_path, timeout=2.0)
    try:
        assert client.call("ping") == {"pong": True}
        assert client.call("echo", {"a": 1}) == {"echo": {"a": 1}}
        with pytest.raises(RpcError):
            client.call("nope")
        # The connection must survive a remote error.
        assert client.call("ping") == {"pong": True}
    finally:
        client.close()
        server.stop()


def test_concurrent_clients(sock_path):
    server = RpcServer(sock_path, lambda m, p: {"n": p.get("n")})
    server.start()
    results = []

    def worker(n):
        client = RpcClient(sock_path, timeout=2.0)
        results.append(client.call("echo", {"n": n})["n"])
        client.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    server.stop()
    assert sorted(results) == list(range(8))


def test_client_without_server_raises(sock_path):
    client = RpcClient(sock_path, timeout=0.2)
    assert not client.reachable()
    with pytest.raises(RpcError):
        client.call("ping")
