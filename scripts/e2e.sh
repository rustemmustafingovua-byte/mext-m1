#!/usr/bin/env bash
# End-to-end check: boots a real daemon on a throwaway workspace, drives it
# through the CLI and the HTTP API, asserts invariants, then shuts it down.
#
#   bash scripts/e2e.sh              # Rust core only (fast, deterministic)
#   bash scripts/e2e.sh --with-ai    # + MLX neural loader and LanceDB RAG
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-$ROOT/core-daemon/target/release/mext}"
PY="${PY:-$ROOT/.venv/bin/python}"
PORT="${PORT:-8899}"
WITH_AI=0
[[ "${1:-}" == "--with-ai" ]] && WITH_AI=1

say()  { printf '\033[1;36m▸\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m✓\033[0m %s\n' "$*"; }
bad()  { printf '\033[1;31m✗\033[0m %s\n' "$*"; }
die()  { bad "$*"; exit 1; }

[[ -x "$BIN" ]] || cargo build --release --manifest-path "$ROOT/core-daemon/Cargo.toml"

WS="$(mktemp -d /tmp/mext-e2e-ws.XXXXXX)"
STATE="$(mktemp -d /tmp/mext-e2e-home.XXXXXX)"
LOG="$STATE/daemon.log"
export MEXT_HOME="$STATE"
DAEMON=""

cleanup() {
  if [[ -n "$DAEMON" ]]; then
    kill "$DAEMON" 2>/dev/null || true
    wait "$DAEMON" 2>/dev/null || true
  fi
  rm -rf "$WS" "$STATE"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
say "preparing a throwaway workspace"
mkdir -p "$WS/src" "$WS/docs"
for i in $(seq 0 15); do
  printf 'def handler_%s(payload):\n    """module %s"""\n    return payload + %s\n' "$i" "$i" "$i" > "$WS/src/mod_$i.py"
  {
    printf '# Document %s\n\n' "$i"
    for _ in $(seq 1 40); do
      printf 'The compressed pool keeps LZ4 frames in RAM and evicts them with ARC. '
    done
    printf '\n'
  } > "$WS/docs/doc_$i.md"
done
ok "$(find "$WS" -type f | wc -l | tr -d ' ') files"

# ---------------------------------------------------------------------------
say "starting the daemon on :$PORT (MEXT_HOME=$STATE)"
ARGS=(start --workspace "$WS" --ram-limit 256M --port "$PORT" --demo)
if [[ $WITH_AI -eq 0 ]]; then
  ARGS+=(--no-ai)
else
  ARGS+=(--moe)   # exercise router-guided expert paging too
fi
"$BIN" "${ARGS[@]}" >"$LOG" 2>&1 &
DAEMON=$!

for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then break; fi
  if ! kill -0 "$DAEMON" 2>/dev/null; then
    cat "$LOG" >&2
    die "daemon died during startup"
  fi
  sleep 0.25
done
curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null || { cat "$LOG" >&2; die "health endpoint never came up"; }
ok "daemon pid $DAEMON, socket $STATE/core.sock"

# ---------------------------------------------------------------------------
say "CLI surface"
"$BIN" open "$WS/src/mod_0.py" | head -3
"$BIN" predict "$WS/src/mod_0.py" -k 3 >/dev/null
"$BIN" warm "$WS" --limit 40 >/dev/null
"$BIN" status --json > "$STATE/status.json"
"$PY" - "$STATE/status.json" <<'PY'
import json, sys
snap = json.load(open(sys.argv[1]))
assert snap["version"], "no version"
assert snap["pool"]["files"] > 0, "warm did nothing"
assert snap["predictor"]["observations"] > 0, "predictor saw nothing"
print("  status: pool files =", snap["pool"]["files"], "| ratio =", round(snap["pool"]["ratio"], 2))
PY
ok "open / predict / warm / status"

# ---------------------------------------------------------------------------
say "HTTP API + invariants"
PORT="$PORT" WITH_AI="$WITH_AI" "$PY" - <<'PY'
import json
import os
import sys
import time
import urllib.request

PORT = os.environ["PORT"]
BASE = f"http://127.0.0.1:{PORT}"
WITH_AI = os.environ["WITH_AI"] == "1"
failures = []


def check(name, fn):
    try:
        detail = fn()
        print(f"  ✓ {name}" + (f" · {detail}" if detail else ""))
    except Exception as exc:  # noqa: BLE001
        failures.append(name)
        print(f"  ✗ {name}: {type(exc).__name__}: {exc}")


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=20) as response:
        return json.load(response)


def post(path, payload):
    request = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main():
    health = get("/api/health")
    assert health["ok"] is True
    assert health["workspace"], "no workspace"
    return "v" + health["version"]


def snapshot_shape():
    snap = get("/api/snapshot")
    for key in (
        "memory",
        "pressure",
        "pool",
        "predictor",
        "activity",
        "latency",
        "events",
        "neural",
        "rag",
    ):
        assert key in snap, f"missing {key}"
    assert snap["memory"]["total"] > 0
    assert snap["pool"]["budget_bytes"] > 0
    assert snap["pressure"]["level"] in ("green", "yellow", "red", "unknown")
    assert isinstance(snap["events"], list)
    assert "warm_p95_ms" in snap["latency"]
    return f"RAM {snap['memory']['total'] // 2**30} GiB, pressure {snap['pressure']['level']}"


def observe_and_predict():
    result = post("/api/observe", {"path": "src/mod_3.py"})
    assert result["accepted"] is True, result
    assert result["key"] == "src/mod_3.py"
    prediction = get("/api/predict?current=src/mod_3.py&k=3")
    assert prediction["predictions"], "no predictions"
    return f"next={prediction['predictions'][0]['path']} ({prediction['predictions'][0]['score']:.2f})"


def warm_via_http():
    result = post("/api/prefetch", {"paths": ["docs/doc_1.md", "src/mod_1.py"]})
    assert result["warmed"] + result["hits"] >= 1, result
    return f"warmed={result['warmed']} hits={result['hits']}"


def budget_is_hard():
    post("/api/budget", {"size": "64M"})
    time.sleep(1.2)
    snap = get("/api/snapshot")
    requested = 64 * 1024 * 1024
    budget = snap["pool"]["budget_bytes"]
    # macOS pressure may shrink further, but the manual ceiling must stick and
    # must never bounce back to --ram-limit (256M here).
    assert budget <= requested, budget
    assert budget >= 1024 * 1024, budget
    assert snap["pool"]["resident_bytes"] <= budget * 1.02, snap["pool"]["resident_bytes"]
    return (
        f"cap {budget / 2**20:.1f} MiB (requested 64, pressure-shrunk), "
        f"resident {snap['pool']['resident_bytes'] / 2**20:.1f} MiB"
    )


def rag_degrades_cleanly():
    typed = get("/api/rag/type?prefix=compressed")
    if WITH_AI:
        deadline = time.time() + 150
        while time.time() < deadline:
            if typed.get("available"):
                break
            time.sleep(1.5)
            typed = get("/api/rag/type?prefix=compressed")
        assert typed.get("available") is True, typed
        query = get("/api/rag/query?q=compressed%20pool%20ARC&k=3")
        assert query["results"], query
        return f"{len(query['results'])} results, {query['latency_ms']:.0f} ms, hit={query['prefetch_hit']}"
    assert typed.get("available") is False, typed
    return "AI disabled -> available:false (no crash)"


def neural_reports():
    snap = get("/api/snapshot")
    neural = snap["neural"]
    if not WITH_AI:
        assert neural.get("available") is False
        return "AI disabled -> available:false"
    assert neural.get("available") is True, neural.get("error")
    assert neural["cold_bytes"] > 0 and neural["hot_bytes"] > 0
    assert neural.get("experts_total", 0) >= 8, neural
    return (
        f"{neural['mode']} · hot {neural['hot_bytes'] / 2**20:.1f} MiB / "
        f"cold {neural['cold_bytes'] / 2**20:.1f} MiB · "
        f"experts {neural['experts_resident']}/{neural['experts_total']} · "
        f"prefetch {neural['prefetch_hit_rate'] * 100:.0f}%"
    )


def sse_stream_works():
    request = urllib.request.Request(BASE + "/api/stream", headers={"Accept": "text/event-stream"})
    with urllib.request.urlopen(request, timeout=15) as response:
        payload = None
        for raw in response:
            line = raw.decode().strip()
            if line.startswith("data:"):
                payload = line[5:].strip()
                break
    assert payload, "no SSE frame"
    snap = json.loads(payload)
    assert snap["pool"]["budget_bytes"] > 0
    return f"first frame has {len(payload)} bytes"


def prometheus_works():
    with urllib.request.urlopen(BASE + "/metrics", timeout=15) as response:
        text = response.read().decode()
    assert "mext_pool_resident_bytes" in text
    assert "mext_pressure_info{" in text
    assert "# TYPE mext_predictor_hit_rate gauge" in text
    assert "mext_pool_warm_latency_ms_bucket{" in text
    assert "mext_rag_query_latency_ms_count" in text
    return f"{len(text.splitlines())} lines"


def gguf_endpoint_survives():
    # No GGUF fixture in e2e: the endpoint must answer, not crash, and must not
    # pretend to have warmed anything.
    payload = get("/api/gguf/inspect?path=/definitely/missing.gguf")
    assert isinstance(payload, dict)
    assert payload.get("available") is False or "error" in payload, payload
    return "missing file reported cleanly"


def stats_are_sane():
    snap = get("/api/snapshot")
    pool = snap["pool"]
    if pool["compressed_bytes"]:
        assert pool["ratio"] >= 1.0, pool["ratio"]
        assert pool["overhead_bytes"] >= 0
    assert snap["activity"]["opens"] > 0
    assert snap["latency"]["warm_samples"] > 0
    return (
        f"ratio {pool['ratio']:.2f}×, warm p95 {snap['latency']['warm_p95_ms']:.1f} ms, "
        f"evictions {pool['evictions']}, released {pool['released_bytes'] // 1024} KiB"
    )


for name, fn in [
    ("health", main),
    ("snapshot shape", snapshot_shape),
    ("observe + predict", observe_and_predict),
    ("http prefetch", warm_via_http),
    ("hard budget via API", budget_is_hard),
    ("rag endpoint", rag_degrades_cleanly),
    ("neural metrics", neural_reports),
    ("pool accounting", stats_are_sane),
    ("gguf endpoint", gguf_endpoint_survives),
    ("SSE /api/stream", sse_stream_works),
    ("prometheus /metrics", prometheus_works),
]:
    check(name, fn)

if failures:
    print(f"\n{len(failures)} check(s) failed: {failures}", file=sys.stderr)
    sys.exit(1)
PY
ok "HTTP API verified"

# ---------------------------------------------------------------------------
say "clean shutdown"
"$BIN" stop >/dev/null
for _ in $(seq 1 40); do
  kill -0 "$DAEMON" 2>/dev/null || break
  sleep 0.25
done
kill -0 "$DAEMON" 2>/dev/null && die "daemon did not exit after `mext stop`"
DAEMON=""
grep -qi "panicked" "$LOG" && die "panic found in the daemon log"
if find "$STATE" -name predictor.json -print -quit | grep -q .; then
  ok "predictor state persisted ($(find "$STATE" -name predictor.json | sed "s|$STATE/||"))"
else
  bad "predictor.json missing after shutdown"
  exit 1
fi

if "$BIN" profiles list 2>/dev/null | grep -q "mext-e2e"; then
  ok "profile visible via \`mext profiles list\`"
else
  bad "mext profiles list does not show the workspace profile"
  exit 1
fi
ok "stopped cleanly, no panics"

echo
printf '\033[1;32mE2E PASS\033[0m · ai %s\n\n' "$([[ $WITH_AI -eq 1 ]] && echo on || echo off)"
