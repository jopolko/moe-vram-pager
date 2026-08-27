# Pentest appliance passoff

Read this on restart to pick up where the last session left off. Keep it
updated as state changes; overwrite stale sections rather than appending a
history log.

## Where things live

- Repo root / working directory for everything below: `/home/josh/moe-vram-pager`
  (no separate pentest folder - the appliance scripts, `tools/pentest_agent.py`,
  and `runs/` all live in this one repo).
- Run logs: `runs/<UTC timestamp>-<phase>.json` (recon / exploit / osint).
- Design doc for the fix below: `ontology.md` (same directory as this file).
- Appliance docs: `PENTEST_APPLIANCE.md`.
- Honeytarget network: `172.30.0.0/24` via `br-60d6c60908b0`, this host is
  `172.30.0.100`. Never touch the shared `eth0` IP - see
  [[moe-vram-pager-pentest-honeypot]] memory. Current live target used in
  recent runs: `172.30.0.50` (DVWA + Samba box).

## Current state (as of 2026-08-20)

- Ran a full recon -> exploit pass against `172.30.0.50`:
  - `runs/20260820T185401Z-recon.json`
  - `runs/20260820T190304Z-exploit.json` (finished, `outcome: "stopped"`,
    ~2.5 hours, no session/shell obtained, no proven exploitation)
- Root cause of that run's failure: the model hallucinated a nonexistent
  Metasploit module (`auxiliary/scanner/http/dvwa_login`,
  `auxiliary/scanner/http/fingerprint_webapp`) and looped on it for most of
  the run with no grounding or loop-breaking to stop it. Full writeup in
  `ontology.md`.
- **Fix implemented, NOT yet verified live:**
  - `~/MetasploitMCP/MetasploitMCP.py` - added `validate_module` tool
    (ground-truth module-name check against the live MSF module tree, with
    close-match suggestions on miss).
  - `~/moe-vram-pager/tools/pentest_agent.py` - dispatch-time gate on
    `run_exploit`/`run_auxiliary_module`/`run_post_module`: validates the
    module name before the real call, 2-strike counter permanently blocks a
    repeated bad name for the rest of the run.
  - Both files verified with `python3 -m py_compile` only. **Not yet run
    end-to-end against a live target.**

## Next step on resume

1. Restart the appliance chain if it's not already up:
   `tools/pentest_appliance.sh start` (chain: msf-db -> msfrpcd ->
   metasploit-mcp -> zap -> pentest-ui-api -> llama-moe-router). Check with
   `ps aux | grep -E "pentest_ui_api|MetasploitMCP|llama-server"`.
2. Continue from `runs/20260820T185401Z-recon.json`: run the exploit phase
   against `172.30.0.50` with `--resume-from` that recon log, to confirm the
   ontology fix actually steers the model away from re-looping on a fake
   module name:
   ```
   .venv/bin/python tools/pentest_agent.py \
       --target "172.30.0.50 (authorized honeytarget scope)" \
       --phase exploit --confirm-exploitation \
       --resume-from runs/20260820T185401Z-recon.json \
       --model llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GGUF:Q6_K \
       --log-file runs/<new-timestamp>-exploit.json
   ```
3. Watch the new run's log for `validate_module` calls and any
   `"blocked": true` events - that's the gate working. If the model still
   loops on a *different* wrong name each time (no exact repeat), that's the
   known gap noted in `ontology.md` ("doesn't solve" section) - not a bug in
   what shipped, just an uncovered case worth deciding whether to handle
   next (e.g. exposing a browsable `list_auxiliary_modules` tool instead of
   only validate-a-guess).
4. If this run also fails to get a session, the DVWA box's actual
   vulnerable surface hasn't been confirmed at all yet - worth manually
   checking what DVWA security level the honeytarget is running before
   assuming Metasploit even has coverage for it (DVWA is usually exploited
   via its own web app logic, not off-the-shelf MSF modules - the ZAP
   active-scan-only findings from the first run support this).
