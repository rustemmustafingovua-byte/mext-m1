#!/usr/bin/env bash
# Benchmark: how much RAM does the working set need, with and without MEXT?
#
# Phase 1 — offline pool benchmark (compression ratio, latency, eviction).
# Phase 2 — live daemon: measure the daemon's own RSS and macOS pressure after
#           it warms a workspace, plus what happens when the budget is lowered.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BIN="$ROOT/core-daemon/target/release/mext"
WORKSPACE="${1:-$ROOT}"
RAM="${RAM_LIMIT:-2G}"
FILES="${FILES:-400}"
PORT="${PORT:-8791}"

if [[ ! -x "$BIN" ]]; then
  echo "▸ building release binary"
  cargo build --release --manifest-path "$ROOT/core-daemon/Cargo.toml"
fi

echo
echo "════════ phase 1 · offline: page cache vs LZ4 pool ════════"
"$BIN" bench "$WORKSPACE" --ram-limit "$RAM" --files "$FILES" --passes 3

echo "════════ phase 2 · live daemon on port $PORT ════════"
LOG="$(mktemp -t mext-bench)"
"$BIN" start --workspace "$WORKSPACE" --ram-limit "$RAM" --port "$PORT" --no-ai >"$LOG" 2>&1 &
DAEMON=$!
cleanup() {
  kill "$DAEMON" 2>/dev/null || true
  wait "$DAEMON" 2>/dev/null || true
  rm -f "$LOG"
}
trap cleanup EXIT

# Give FSEvents a moment, then feed the daemon a burst of real file opens.
sleep 1.2
COUNT=0
while IFS= read -r -d '' file; do
  [[ $COUNT -ge 60 ]] && break
  "$BIN" open "$file" >/dev/null 2>&1 || true
  COUNT=$((COUNT + 1))
done < <(find "$WORKSPACE" -type f \( -name '*.rs' -o -name '*.py' -o -name '*.md' -o -name '*.toml' \) \
        -not -path '*/target/*' -not -path '*/.venv/*' -not -path '*/.git/*' -print0 2>/dev/null)

sleep 2
echo
"$BIN" status | sed 's/^/  /'

echo
echo "  macOS pressure level : $(sysctl -n kern.memorystatus_vm_pressure_level)  (1=green 2=yellow 4=red)"
echo "  daemon RSS           : $(ps -o rss= -p "$DAEMON" | awk '{printf "%.1f MiB", $1/1024}')"
echo "  swap usage           : $(sysctl -n vm.swapusage)"

echo
echo "  lowering the cap to 256M while the pool is warm…"
"$BIN" status >/dev/null 2>&1 || true
curl -s -X POST "http://127.0.0.1:$PORT/api/budget" \
  -H 'Content-Type: application/json' -d '{"size":"256M"}' >/dev/null || true
sleep 1.5
"$BIN" status | sed -n '3,4p' | sed 's/^/  /'
echo "  (resident should drop immediately and 'released' should grow — that is madvise(MADV_DONTNEED))"
echo
