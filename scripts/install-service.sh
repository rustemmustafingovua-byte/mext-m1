#!/usr/bin/env bash
# Install / uninstall / inspect the MEXT-M1 launchd user agent.
#
#   ./scripts/install-service.sh --workspace DIR --ram-limit 2G --port 8765
#   ./scripts/install-service.sh --dry-run
#   ./scripts/install-service.sh --status
#   ./scripts/install-service.sh --uninstall
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LABEL="com.mext.daemon"
WORKSPACE=""
RAM_LIMIT="2G"
PORT="8765"
DRY_RUN=0
UNINSTALL=0
STATUS=0

usage() {
  cat <<USAGE
Install the MEXT-M1 launchd user agent.

  --workspace DIR   workspace to watch (default: repo root)
  --ram-limit SIZE  compressed pool cap, e.g. 2G (default: 2G)
  --port PORT       dashboard port (default: 8765)
  --label LABEL     launchd label (default: com.mext.daemon)
  --dry-run         print the plist and lint it; do not install
  --status          read-only launchctl print of the service
  --uninstall       unload and remove the plist
  -h, --help        show this help
USAGE
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace)
      [[ $# -ge 2 ]] || { echo "missing value for --workspace" >&2; exit 2; }
      WORKSPACE="$2"; shift 2 ;;
    --ram-limit)
      [[ $# -ge 2 ]] || { echo "missing value for --ram-limit" >&2; exit 2; }
      RAM_LIMIT="$2"; shift 2 ;;
    --port)
      [[ $# -ge 2 ]] || { echo "missing value for --port" >&2; exit 2; }
      PORT="$2"; shift 2 ;;
    --label)
      [[ $# -ge 2 ]] || { echo "missing value for --label" >&2; exit 2; }
      LABEL="$2"; shift 2 ;;
    --dry-run)    DRY_RUN=1; shift ;;
    --uninstall)  UNINSTALL=1; shift ;;
    --status)     STATUS=1; shift ;;
    -h|--help)    usage ;;
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

PLIST_DIR="$HOME/Library/LaunchAgents"
PLIST_PATH="$PLIST_DIR/$LABEL.plist"
BIN="$ROOT/core-daemon/target/release/mext"
MEXT_HOME="${MEXT_HOME:-$HOME/.mext}"
LOG="$MEXT_HOME/service.log"

status() {
  launchctl print "gui/$UID/$LABEL" || true
}

uninstall() {
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] would unload and remove $PLIST_PATH"
    return
  fi
  if launchctl print "gui/$UID/$LABEL" >/dev/null 2>&1; then
    echo "[install] unloading $LABEL"
    launchctl bootout "gui/$UID/$LABEL" || true
  fi
  rm -f "$PLIST_PATH"
  echo "[install] uninstalled $LABEL"
}

plist() {
  cat <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>

  <key>ProgramArguments</key>
  <array>
    <string>$BIN</string>
    <string>start</string>
    <string>--workspace</string>
    <string>$WORKSPACE_ABS</string>
    <string>--ram-limit</string>
    <string>$RAM_LIMIT</string>
    <string>--port</string>
    <string>$PORT</string>
  </array>

  <key>RunAtLoad</key>
  <true/>

  <key>KeepAlive</key>
  <dict>
    <key>SuccessfulExit</key>
    <false/>
  </dict>

  <key>EnvironmentVariables</key>
  <dict>
    <key>MEXT_HOME</key>
    <string>$MEXT_HOME</string>
  </dict>

  <key>WorkingDirectory</key>
  <string>$ROOT</string>

  <key>StandardOutPath</key>
  <string>$LOG</string>

  <key>StandardErrorPath</key>
  <string>$LOG</string>
</dict>
</plist>
PLIST
}

# ---------------------------------------------------------------- entrypoints
if [[ $STATUS -eq 1 ]]; then
  status
  exit 0
fi

if [[ $UNINSTALL -eq 1 ]]; then
  uninstall
  exit 0
fi

mkdir -p "$MEXT_HOME"

if [[ $DRY_RUN -eq 1 ]]; then
  TMP=$(mktemp /tmp/com.mext.daemon.XXXXXX.plist)
  plist > "$TMP"
  echo "[dry-run] generated plist (linting with plutil):"
  plutil -lint "$TMP"
  cat "$TMP"
  rm -f "$TMP"
  exit 0
fi

if [[ ! -x "$BIN" ]]; then
  echo "release binary not found: $BIN" >&2
  echo "run: make build" >&2
  exit 1
fi

mkdir -p "$PLIST_DIR"
plist > "$PLIST_PATH"
chmod 644 "$PLIST_PATH"
echo "[install] wrote $PLIST_PATH"

if launchctl print "gui/$UID/$LABEL" >/dev/null 2>&1; then
  echo "[install] service already loaded; kicking it to pick up the new plist"
  launchctl kickstart -k "gui/$UID/$LABEL" || true
else
  echo "[install] loading service with launchctl bootstrap"
  launchctl bootstrap "gui/$UID" "$PLIST_PATH" || true
fi
