#!/usr/bin/env bash
# Run llama.cpp with MEXT as a page warmer.
#
#   bash scripts/llama-with-mext.sh \
#     --model ~/models/qwen2.5-coder-7b-q4_k_m.gguf \
#     --runner ~/llama.cpp/build/bin/llama-cli \
#     -- -m ~/models/qwen2.5-coder-7b-q4_k_m.gguf -p "def fib(n):" -n 128
#
# A background `mext-ai gguf prewarm` pages in the layer weights at a controlled
# pace while llama.cpp starts and generates, so its first reads hit RAM instead
# of a cold SSD. Warm pages are shared page cache: no copies, no hooks.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL=""
RUNNER=""
PACE_MS="10"
LAYERS="all"
DRY_RUN=0
RUNNER_ARGS=()

say()  { printf '\033[1;36m▸\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m✓\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m✗\033[0m %s\n' "$*" >&2; exit 2; }

usage() {
  cat <<'EOF'
usage: bash scripts/llama-with-mext.sh --model FILE.gguf --runner PATH [options] -- [runner args]

  --model FILE     GGUF checkpoint to warm (required)
  --runner PATH    llama-cli / llama-server binary (default: llama-cli from PATH)
  --pace-ms N      pause between layers in the warmer (default: 10)
  --layers SPEC    `all` or 0,1,2 / 0-5 (default: all)
  --dry-run        print the commands, start nothing
  -h, --help       this text

  env: MEXT_HOME (default ~/.mext) — warmer log is $MEXT_HOME/gguf-warmer.log
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) [[ $# -ge 2 ]] || die "--model needs a file"; MODEL="$2"; shift 2 ;;
    --runner) [[ $# -ge 2 ]] || die "--runner needs a path"; RUNNER="$2"; shift 2 ;;
    --pace-ms) [[ $# -ge 2 ]] || die "--pace-ms needs a number"; PACE_MS="$2"; shift 2 ;;
    --layers) [[ $# -ge 2 ]] || die "--layers needs a spec"; LAYERS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; RUNNER_ARGS=("$@"); break ;;
    *) die "unknown flag: $1 (try --help)" ;;
  esac
done

[[ -n "$MODEL" ]] || die "--model is required"
[[ -f "$MODEL" ]] || die "model file not found: $MODEL"
MODEL="$(cd "$(dirname "$MODEL")" && pwd)/$(basename "$MODEL")"
[[ "$PACE_MS" =~ ^[0-9]+$ ]] || die "--pace-ms must be numeric"

if [[ -z "$RUNNER" ]]; then
  RUNNER="$(command -v llama-cli || command -v llama-server || true)"
fi
[[ -n "$RUNNER" ]] || die "--runner is required (no llama-cli on PATH)"
[[ -x "$RUNNER" ]] || die "runner is not executable: $RUNNER"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="$(command -v python3 || true)"
fi
[[ -n "$PY" ]] || die "python3 not found (run ./setup.sh)"

export MEXT_HOME="${MEXT_HOME:-$HOME/.mext}"
mkdir -p "$MEXT_HOME"
WARMER_LOG="$MEXT_HOME/gguf-warmer.log"

WARMER_CMD=("$PY" -m mext_ai gguf prewarm "$MODEL" --pace-ms "$PACE_MS" --layers "$LAYERS")

if [[ $DRY_RUN -eq 1 ]]; then
  printf 'warmer: MEXT_HOME=%q PYTHONPATH=%q' "$MEXT_HOME" "$ROOT/ai-engine"
  printf ' %q' "${WARMER_CMD[@]}"
  printf '\nrunner: %q' "$RUNNER"
  if [[ ${#RUNNER_ARGS[@]} -gt 0 ]]; then
    printf ' %q' "${RUNNER_ARGS[@]}"
  fi
  printf '\n'
  ok "dry-run: nothing started"
  exit 0
fi

say "warming $MODEL (pace ${PACE_MS}ms) → $WARMER_LOG"
PYTHONPATH="$ROOT/ai-engine" "${WARMER_CMD[@]}" >/dev/null 2>"$WARMER_LOG" &
WARMER_PID=$!

cleanup() {
  if kill -0 "$WARMER_PID" 2>/dev/null; then
    kill "$WARMER_PID" 2>/dev/null || true
    wait "$WARMER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

say "starting $(basename "$RUNNER")"
set +e
if [[ ${#RUNNER_ARGS[@]} -gt 0 ]]; then
  "$RUNNER" "${RUNNER_ARGS[@]}"
else
  "$RUNNER"
fi
STATUS=$?
set -e
cleanup
trap - EXIT INT TERM
exit "$STATUS"
