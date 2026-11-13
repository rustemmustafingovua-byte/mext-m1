#!/usr/bin/env bash
# Install a prebuilt MEXT-M1 release (macOS arm64) into ~/.local/bin.
#
#   bash scripts/install-release.sh                     # latest release
#   MEXT_VERSION=v0.6.0 bash scripts/install-release.sh # pinned
#   MEXT_PREFIX=/usr/local/bin bash scripts/install-release.sh
#
# The tarball is verified against the published .sha256 before installation.
set -euo pipefail

REPO="${MEXT_REPO:-rustemmustafingovua-byte/mext-m1}"
VERSION="${MEXT_VERSION:-latest}"
PREFIX="${MEXT_PREFIX:-$HOME/.local/bin}"

say()  { printf '\033[1;36m▸\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m✓\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m✗\033[0m %s\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == "Darwin" ]] || die "MEXT-M1 releases are macOS-only"
[[ "$(uname -m)" == "arm64" ]] || die "MEXT-M1 releases require Apple Silicon (arm64)"
command -v curl >/dev/null || die "curl is required"
command -v tar >/dev/null || die "tar is required"

api="https://api.github.com/repos/${REPO}/releases"
if [[ "$VERSION" == "latest" ]]; then
  say "resolving the latest release"
  VERSION="$(curl -fsSL "${api}/latest" | grep -o '"tag_name": *"[^"]*"' | head -1 | cut -d'"' -f4)"
  [[ -n "$VERSION" ]] || die "cannot resolve the latest release (rate limit?)"
fi
[[ "$VERSION" == v* ]] || VERSION="v${VERSION}"

NAME="mext-m1-${VERSION}-aarch64-apple-darwin"
BASE="https://github.com/${REPO}/releases/download/${VERSION}"
TMP="$(mktemp -d -t mext-install)"
trap 'rm -rf "$TMP"' EXIT

say "downloading ${NAME}.tar.gz"
curl -fsSL "${BASE}/${NAME}.tar.gz" -o "${TMP}/${NAME}.tar.gz"
curl -fsSL "${BASE}/${NAME}.tar.gz.sha256" -o "${TMP}/${NAME}.tar.gz.sha256" \
  || die "checksum file is missing"

say "verifying checksum"
( cd "$TMP" && shasum -a 256 -c "${NAME}.tar.gz.sha256" )

tar -xzf "${TMP}/${NAME}.tar.gz" -C "$TMP"
mkdir -p "$PREFIX"
install -m 0755 "${TMP}/${NAME}/mext" "$PREFIX/mext"
install -m 0755 "${TMP}/${NAME}/mextd" "$PREFIX/mextd"
ok "installed $VERSION into $PREFIX"

case ":$PATH:" in
  *":$PREFIX:"*) ;;
  *) printf '\n  add to PATH:  export PATH="%s:$PATH"\n' "$PREFIX" ;;
esac

cat <<'EOF'

Next steps:
  mext doctor
  mext start --workspace ~/code/project --ram-limit 2G
  open http://localhost:8765

The Python AI engine (MLX/LanceDB) is not bundled: run ./setup.sh from a git
checkout once, or install it manually with
  pip install -e "ai-engine[dev,pdf]"
EOF
