#!/usr/bin/env bash
# MEXT-M1 installer / environment checker for Apple Silicon macOS.
#
#   ./setup.sh                 full setup (Rust release build + Python venv + MLX + LanceDB)
#   ./setup.sh --debug         build the Rust core without optimisations
#   ./setup.sh --skip-ai       Rust core only (no MLX/LanceDB)
#   ./setup.sh --with-real-model  also install mlx-lm (local fp16/4-bit checkpoints)
#   ./setup.sh --python /path/to/python3.12
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PROFILE="--release"
SKIP_AI=0
WITH_REAL=0
PYTHON="${MEXT_PYTHON:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug) PROFILE=""; shift ;;
    --skip-ai) SKIP_AI=1; shift ;;
    --with-real-model) WITH_REAL=1; shift ;;
    --python) PYTHON="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

say()  { printf '\033[1;36m▸\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m✓\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m✗\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
say "checking platform"
[[ "$(uname -s)" == "Darwin" ]] || die "macOS only (uses madvise/MADV_DONTNEED + FSEvents)"
if [[ "$(uname -m)" != "arm64" ]]; then
  warn "not Apple Silicon ($(uname -m)) — the code is portable but tuned for M1/M2/M3"
fi
ok "$(sw_vers -productName) $(sw_vers -productVersion) · $(uname -m) · $(sysctl -n hw.memsize | awk '{printf "%.0f GiB", $1/1073741824}') RAM"

# ---------------------------------------------------------------------------
say "checking Rust"
if ! command -v cargo >/dev/null 2>&1; then
  die "cargo not found — install it with: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
fi
ok "cargo $(cargo --version | awk '{print $2}')"

# ---------------------------------------------------------------------------
if [[ $SKIP_AI -eq 0 ]]; then
  say "checking Python"
  if [[ -z "$PYTHON" ]]; then
    for candidate in python3.12 python3.13 python3.11 python3; do
      if command -v "$candidate" >/dev/null 2>&1; then PYTHON="$(command -v "$candidate")"; break; fi
    done
  fi
  [[ -n "$PYTHON" ]] || die "no python3 found (need 3.10+ for MLX/LanceDB)"
  "$PYTHON" - <<'PY' || die "Python 3.10+ is required"
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"found {sys.version.split()[0]}")
print(f"python {sys.version.split()[0]} at {sys.executable}")
PY

  if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    say "creating .venv (this also keeps Homebrew Python clean)"
    "$PYTHON" -m venv "$ROOT/.venv"
  else
    ok "reusing $ROOT/.venv"
  fi
  VENV_PY="$ROOT/.venv/bin/python"

  EXTRAS="dev,pdf"
  [[ $WITH_REAL -eq 1 ]] && EXTRAS="dev,pdf,real-model"
  say "installing Python deps (mlx, lancedb, pyarrow, psutil, pypdf, pytest${WITH_REAL:+, mlx-lm})"
  "$VENV_PY" -m pip install --quiet --upgrade pip setuptools wheel
  "$VENV_PY" -m pip install --quiet -e "$ROOT/ai-engine[$EXTRAS]"
  ok "$("$VENV_PY" - <<'PY'
mods = []
for name in ("mlx.core", "lancedb", "pyarrow", "psutil"):
    try:
        mod = __import__(name, fromlist=["_"])
        mods.append(f"{name.split('.')[0]} {getattr(mod, '__version__', 'ok')}")
    except Exception as exc:
        mods.append(f"{name.split('.')[0]} FAILED ({exc})")
print(" · ".join(mods))
PY
)"
fi

# ---------------------------------------------------------------------------
say "building Rust core $PROFILE"
# shellcheck disable=SC2086
cargo build $PROFILE --manifest-path "$ROOT/core-daemon/Cargo.toml"
BIN_DIR="$ROOT/core-daemon/target/release"
[[ -n "$PROFILE" ]] || BIN_DIR="$ROOT/core-daemon/target/debug"
ok "built $BIN_DIR/mext"

# ---------------------------------------------------------------------------
say "linking CLI into ~/.local/bin"
mkdir -p "$HOME/.local/bin"
ln -sf "$BIN_DIR/mext" "$HOME/.local/bin/mext"
ln -sf "$BIN_DIR/mextd" "$HOME/.local/bin/mextd"
ok "mext -> $BIN_DIR/mext"
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) warn "add ~/.local/bin to PATH:  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.zshrc" ;;
esac

# ---------------------------------------------------------------------------
echo
echo "────────────────────────────────────────────────────────────────────"
echo " MEXT-M1 is ready."
echo
echo " Try it (safe on an 8 GB M1 — 2 GB cap, synthetic activity):"
echo "   mext start --workspace \"\$PWD\" --ram-limit 2G --demo"
echo "   open http://localhost:8765"
echo
echo " Then, with your own project:"
echo "   mext start --workspace ~/code/my-project --ram-limit 3G \\"
echo "        --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
echo
echo " Verify everything:  mext doctor && make test"
echo " Roll a face-off:    make bench"
echo "────────────────────────────────────────────────────────────────────"
