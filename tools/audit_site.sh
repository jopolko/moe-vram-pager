#!/usr/bin/env bash
# One-shot wrapper: "audit <target>" -> recon run -> PDF report, no manual
# steps in between. Stops at recon on purpose - exploitation always stays a
# separate, explicitly-confirmed run (tools/pentest_agent.py --phase exploit
# --confirm-exploitation), this script never touches that path.
#
# Usage:
#   tools/audit_site.sh "joshuaopolko.com (authorized scope: web app only)"
set -euo pipefail

SCOPE="${1:?Usage: audit_site.sh \"<target/scope description>\"}"
CLIENT="${2:-}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PYTHON="${PENTEST_PYTHON:-/home/josh/MetasploitMCP/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "error: python not found at $PYTHON (set PENTEST_PYTHON to override)" >&2
    exit 1
fi

echo "== checking dependencies are up =="
if ! curl -sf http://127.0.0.1:8080/health > /dev/null; then
    echo "error: llama-server router not responding on 127.0.0.1:8080" >&2
    exit 1
fi
if ! curl -sf -m 2 http://127.0.0.1:8085/sse -o /dev/null 2>/dev/null; then
    # SSE endpoint streams, so a plain curl "succeeding" isn't the right test;
    # just check the port is open instead.
    if ! (exec 3<>/dev/tcp/127.0.0.1/8085) 2>/dev/null; then
        echo "error: MetasploitMCP not responding on 127.0.0.1:8085" >&2
        exit 1
    fi
    exec 3<&- 3>&-
fi

echo "== phase 1: recon =="
"$PYTHON" tools/pentest_agent.py --target "$SCOPE" --max-iterations 200

RECON_LOG=$(ls -t runs/*-recon.json 2>/dev/null | head -1)
if [ -z "$RECON_LOG" ]; then
    echo "error: no recon log found in runs/ after the run - something went wrong above" >&2
    exit 1
fi

echo "== generating report from $RECON_LOG =="
mkdir -p reports
OUT="reports/$(date -u +%Y%m%dT%H%M%SZ)-audit.pdf"
"$PYTHON" tools/pentest_report.py "$RECON_LOG" --out "$OUT" \
    --client "$CLIENT" --engagement "$SCOPE"

echo
echo "Recon-only audit complete."
echo "Report: $OUT"
echo "Recon log: $RECON_LOG"
echo
echo "To proceed to active exploitation of these findings (requires explicit review + confirmation):"
echo "  $PYTHON tools/pentest_agent.py --target \"$SCOPE\" --phase exploit --confirm-exploitation --resume-from $RECON_LOG"
