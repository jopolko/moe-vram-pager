#!/usr/bin/env bash
# One-command build for the whole project: the core moe-vram-pager server
# (llama-server + the embedded webui, including the /pentest panel) plus
# every pentest-appliance component (nmap/Metasploit/ZAP/theHarvester/
# MetasploitMCP, secrets, systemd units). Then restarts everything so
# what's running matches what was just built.
#
# Unlike tools/setup_pentest_appliance.sh's own build step (which skips
# building if build/bin/llama-server already exists - fine for a true
# first-time install), this always rebuilds from source, including forcing
# a fresh webui embed (rm -rf tools/ui/dist first) - so it's also the right
# thing to run after pulling/making code or UI changes, not just once.
#
# Usage: ./setup.sh [--yes] [--skip-zap] [--skip-osint] [--no-restart]
#   All flags except --no-restart are passed straight through to
#   tools/setup_pentest_appliance.sh - see that script's own usage comment.
#   --no-restart: build/install only, don't restart the running services.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NO_RESTART=0
PASSTHROUGH_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --no-restart) NO_RESTART=1 ;;
        *) PASSTHROUGH_ARGS+=("$arg") ;;
    esac
done

log() { printf '\n\033[1;36m==>\033[0m %s\n' "$1"; }
ok()  { printf '    \033[1;32m✓\033[0m %s\n' "$1"; }

# --- 1. Build the core server, forcing a fresh webui embed ------------------

log "Building llama-server (forcing a fresh webui embed)"
rm -rf "$REPO_DIR/tools/ui/dist"
cmake -B "$REPO_DIR/build" -S "$REPO_DIR"
cmake --build "$REPO_DIR/build" --target llama-server -j"$(nproc)"
ok "built build/bin/llama-server"

# --- 2. Everything else (tools, systemd units, secrets, pentest deps) -------

log "Setting up pentest appliance components"
"$REPO_DIR/tools/setup_pentest_appliance.sh" "${PASSTHROUGH_ARGS[@]}"

# --- 3. Restart so the freshly built binary is what's actually running -----

if [ "$NO_RESTART" = 1 ]; then
    log "Skipping restart (--no-restart passed) - run tools/pentest_appliance.sh restart when ready"
else
    log "Restarting the full stack"
    "$REPO_DIR/tools/pentest_appliance.sh" restart
fi

log "Done. Open http://127.0.0.1:8080/#/pentest for the GUI panel."
