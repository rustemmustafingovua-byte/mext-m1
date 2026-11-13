#!/usr/bin/env bash
# MEXT-M1 rebuild-and-run dev loop.
#
#   ./scripts/dev.sh                         # watch + debug daemon loop
#   ./scripts/dev.sh --once                  # debug build only
#   ./scripts/dev.sh --review                # release build + all checks/tests
#   ./scripts/dev.sh --no-ai --port 8913     # Rust core only on a custom port
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REVIEW=0
ONCE=0
NO_AI=0
WORKSPACE=""
RAM_LIMIT="2G"
PORT="8770"

usage() {
  sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --review)   REVIEW=1; shift ;;
    --once)     ONCE=1; shift ;;
    --no-ai)    NO_AI=1; shift ;;
    --workspace)
      [[ $# -ge 2 ]] || { echo "missing value for --workspace" >&2; exit 2; }
      WORKSPACE="$2"; shift 2 ;;
    --ram-limit)
      [[ $# -ge 2 ]] || { echo "missing value for --ram-limit" >&2; exit 2; }
      RAM_LIMIT="$2"; shift 2 ;;
    --port)
      [[ $# -ge 2 ]] || { echo "missing value for --port" >&2; exit 2; }
      PORT="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "unknown flag: $1" >&2; usage ;;
  esac
done

if [[ -z "$WORKSPACE" ]]; then
  WORKSPACE="$ROOT"
fi
WORKSPACE_ABS="$(cd "$WORKSPACE" && pwd)" || {
  echo "workspace does not exist: $WORKSPACE" >&2
  exit 1
}

# Prefer the repo-local venv, otherwise any python3 on PATH.
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="$(command -v python3 || command -v python || true)"
fi
if [[ -z "${PYTHON:-}" ]]; then
  echo "python3 not found" >&2
  exit 1
fi

MANIFEST="$ROOT/core-daemon/Cargo.toml"
DEBUG_BIN="$ROOT/core-daemon/target/debug/mext"
RELEASE_BIN="$ROOT/core-daemon/target/release/mext"

# ---------------------------------------------------------------- review mode
if [[ $REVIEW -eq 1 ]]; then
  echo "[review] release build"
  cargo build --release --manifest-path "$MANIFEST"

  echo "[review] cargo fmt --check"
  cargo fmt --check --manifest-path "$MANIFEST"

  echo "[review] cargo test"
  cargo test --manifest-path "$MANIFEST"

  if command -v node >/dev/null 2>&1; then
    echo "[review] node --check dashboard/app.js"
    node --check "$ROOT/dashboard/app.js"
  else
    echo "[review] node not found, skipping dashboard/app.js check"
  fi

  echo "[review] pytest"
  "$PYTHON" -m pytest "$ROOT/ai-engine" -q

  if "$PYTHON" -c "import mlx" >/dev/null 2>&1; then
    echo "[review] mext-ai selftest"
    "$PYTHON" -m mext_ai selftest
  else
    echo "[review] mlx not available, skipping mext-ai selftest"
  fi

  echo "[review] OK"
  exit 0
fi

# ---------------------------------------------------------------- once mode
if [[ $ONCE -eq 1 ]]; then
  echo "[once] debug build"
  cargo build --manifest-path "$MANIFEST"
  exit 0
fi

# ---------------------------------------------------------------- watch mode
export MEXT_HOME="${MEXT_HOME:-/tmp/mext-dev}"
mkdir -p "$MEXT_HOME"

DAEMON_PID=""

build_debug() {
  cargo build --manifest-path "$MANIFEST"
}

start_daemon() {
  if [[ -n "$DAEMON_PID" ]] && kill -0 "$DAEMON_PID" 2>/dev/null; then
    echo "[dev] stopping previous daemon (pid $DAEMON_PID)"
    kill "$DAEMON_PID" 2>/dev/null || true
    wait "$DAEMON_PID" 2>/dev/null || true
  fi

  local args=("$DEBUG_BIN" start --workspace "$WORKSPACE_ABS" --ram-limit "$RAM_LIMIT" --port "$PORT")
  if [[ $NO_AI -eq 1 ]]; then
    args+=(--no-ai)
  fi

  echo "[dev] ${args[*]}"
  "${args[@]}" &
  DAEMON_PID=$!
  echo "[dev] daemon pid $DAEMON_PID on port $PORT"
}

stop_daemon() {
  if [[ -n "$DAEMON_PID" ]] && kill -0 "$DAEMON_PID" 2>/dev/null; then
    echo "[dev] stopping daemon (pid $DAEMON_PID)"
    kill "$DAEMON_PID" 2>/dev/null || true
    wait "$DAEMON_PID" 2>/dev/null || true
  fi
  DAEMON_PID=""
}

rebuild() {
  echo "[dev] change detected"
  if build_debug; then
    start_daemon
  else
    echo "[dev] build failed; daemon left as-is"
  fi
}

shutdown() {
  echo "[dev] shutting down"
  stop_daemon
  exit 0
}

trap rebuild USR1
trap shutdown INT TERM HUP

# Initial build + start.
if build_debug; then
  start_daemon
else
  echo "[dev] initial build failed" >&2
  stop_daemon
  exit 1
fi

# Use cargo-watch if available; otherwise fall back to a debounced mtime loop.
if command -v cargo-watch >/dev/null 2>&1; then
  echo "[dev] using cargo-watch (send SIGUSR1 to $$ on changes)"
  echo "$$" > "$MEXT_HOME/dev.pid"
  cargo watch \
    --watch "$ROOT/core-daemon/src" \
    --watch "$ROOT/dashboard" \
    --watch "$ROOT/ai-engine/mext_ai" \
    --postpone \
    --shell "kill -USR1 \$(cat '$MEXT_HOME/dev.pid')" &
  WATCHER_PID=$!

  while kill -0 "$WATCHER_PID" 2>/dev/null; do
    wait "$WATCHER_PID" || true
  done
else
  echo "[dev] cargo-watch not found, using debounced mtime loop (400 ms)"

  # Portable file mtime snapshot hash.
  snapshot() {
    {
      find "$ROOT/core-daemon/src" -type f -print0 2>/dev/null | xargs -0 stat -f '%m %N' 2>/dev/null || true
      find "$ROOT/dashboard" -type f -print0 2>/dev/null | xargs -0 stat -f '%m %N' 2>/dev/null || true
      find "$ROOT/ai-engine/mext_ai" -type f -print0 2>/dev/null | xargs -0 stat -f '%m %N' 2>/dev/null || true
    } | shasum -a 256 | awk '{print $1}'
  }

  LAST=$(snapshot)
  while true; do
    sleep 0.4
    CURRENT=$(snapshot)
    if [[ "$CURRENT" != "$LAST" ]]; then
      rebuild
      LAST=$(snapshot)
    fi
  done
fi
