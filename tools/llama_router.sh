#!/usr/bin/env bash
# Start/stop/restart just the core moe-vram-pager server (llama-moe-router.
# service - llama-server with MoE VRAM streaming + the webui, including the
# /pentest panel). Scoped to that one unit on purpose: if only the model
# server hangs (a stuck generation, a wedged HTTP handler, ...), this is the
# fast fix - no need to also bounce the whole Metasploit/ZAP/pentest-ui-api
# chain via tools/pentest_appliance.sh just to recover the LLM.
#
# Requires the passwordless-sudo rule from PENTEST_APPLIANCE.md section 6.1
# (same rule tools/pentest_appliance.sh depends on) - without it, sudo
# blocks waiting for a password this script has no terminal to read.
#
# Usage: tools/llama_router.sh start|stop|restart|status

set -euo pipefail

SERVICE=llama-moe-router.service

status() {
    state=$(systemctl is-active "$SERVICE" 2>/dev/null || true)
    printf "  %-24s %s\n" "$SERVICE" "$state"
}

case "${1:-}" in
    start)   echo "Starting $SERVICE..."; sudo systemctl start "$SERVICE"; status ;;
    stop)    echo "Stopping $SERVICE..."; sudo systemctl stop "$SERVICE"; status ;;
    restart) echo "Restarting $SERVICE..."; sudo systemctl restart "$SERVICE"; status ;;
    status)  status ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}" >&2
        exit 1
        ;;
esac
