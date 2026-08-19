# MoE VRAM Pager

Run massive Mixture-of-Experts models (e.g. DeepSeek, Kimi-K2, GLM), hundreds
of GB in size, on a machine with a fraction of that in VRAM. Only the
currently-active experts are paged into memory on demand; the rest stays on
disk (and RAM as a cache tier) until it's needed.

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp): this is a
clean-break fork, not a general-purpose llama.cpp distribution. If you just
want to run a normal (non-MoE, or already-fits-in-VRAM) GGUF model, use
[upstream llama.cpp](https://github.com/ggml-org/llama.cpp) instead; its docs
still apply here since this fork keeps the full llama.cpp toolset
(`llama-cli`, `llama-server`, the OpenAI-compatible API, etc.) alongside the
MoE-paging additions below.

![The model picker: hardware-aware model list with a model loaded](docs/model-picker.png)

## Status

Early and actively changing. What works today:

- **VRAM- and RAM-tier expert streaming** (`--moe-stream`): routed experts
  are read from disk on demand into a bounded LRU cache that lives on
  whichever device the layer is offloaded to - CUDA VRAM by default when a
  layer is GPU-offloaded (`-ngl`), host RAM otherwise, or forced to host RAM
  even on a GPU layer with `--moe-stream-cpu-cache` (useful for isolating
  VRAM- vs RAM-cache performance). A MoE model whose total size exceeds
  VRAM+RAM entirely still runs, just with more disk reads per token.
- **Decode-phase speculative prefetch** (`--moe-stream-prefetch N`):
  overlaps the next layer's expert I/O with the current layer's GPU compute
  during single-token generation, using the same wave-prefetch trick
  `--moe-stream` already does across prefill batches. Works and is
  correctness-verified, but under real I/O pressure it currently competes
  with genuine blocking loads on the same queue and can *hurt* throughput
  instead of helping - see the `q_demand` priority-inversion note in
  `src/llama-moe-stream.cpp`. Off by default; treat as experimental.
- **The model picker** (`/models` in the web UI, or `scripts/model_picker.py`
  on the CLI): cross-references the
  [UGI Leaderboard](https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard)
  against your actual detected VRAM/RAM/disk to show which MoE models will
  run on your hardware, at what quant, and how well.
- **Download and run models from the web UI** (router mode): the picker's
  Actions column downloads a model with a `--moe-stream-cache` size already
  tuned to your hardware, shows live progress, and loads it straight into
  the chat UI - no CLI round-trip per model. See
  [Quick start](#quick-start) below.

## Hardware and system requirements

Targets NVIDIA GPUs via CUDA. `CMAKE_CUDA_ARCHITECTURES` is auto-detected
from whatever GPU is actually in the build machine (CMake's `native`
mode, upstream ggml/CMake logic, requires CMake >= 3.24 and CUDA >= 11.6)
- don't set it by hand unless you're cross-compiling for different
hardware than you're building on, in which case pass
`-DCMAKE_CUDA_ARCHITECTURES=<your compute capability>` explicitly.

- **NVIDIA GPU + [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit)**
  installed and on `PATH` before running `cmake -B build -DGGML_CUDA=ON`.
  Any VRAM amount works in principle - `--moe-stream-cache` sizes the
  expert cache to whatever fits, and the model picker (`/models`) shows
  you which quant of which model actually fits your specific card. More
  VRAM simply means fewer disk reads per token, not a hard cutoff.
- **RAM** as the second cache tier below VRAM: experts that don't fit in
  VRAM cache to host RAM before falling back to disk. More RAM shrinks
  the disk-streaming tier the same way more VRAM does.
- **Fast local storage matters more than GPU class for the disk-streaming
  case this project targets.** A local NVMe SSD is the intended target;
  a network-backed or virtualized disk can dominate the bottleneck
  regardless of GPU. `--moe-stream-io-threads` is disk-specific, not
  GPU-specific - more threads isn't automatically faster on every disk
  (virtualized/networked disks in particular can get *slower* with more
  threads, the opposite of typical bare-metal NVMe advice); tune it
  against your own disk instead of assuming a number.
- **Disk space** for the model file itself - the whole point of this
  project is running models whose GGUF is hundreds of GB, so budget disk
  space accordingly (the picker flags a model "needs N GB more" instead
  of hiding it if it doesn't fit yet).

### Performance: general guidance, not benchmark numbers

No published benchmark numbers exist yet for this fork across GPUs, model
sizes, or disk types - what follows is qualitative guidance about what
drives performance, not a substitute for measuring your own setup.
Page-cache warmth alone can swing decode speed substantially between two
back-to-back runs of the identical config, so any single number you see
quoted for this project is close to meaningless without also knowing the
model size, cache size, disk type, and whether the page cache was warm.
`--moe-stream-direct` (`O_DIRECT`) removes that page-cache variable for
controlled A/B comparisons, at the cost of being slower in absolute terms
than the page-cache-assisted path most real usage benefits from.

**Don't trust a number you see quoted for this project** (an RTX
3090/4090/5090/etc. tok/s figure, for example) unless it cites its own
model size, cache size, and disk type - those three variables swing
results far more than GPU generation does once the model doesn't fit in
VRAM+RAM, which is the whole point of this project.

**Warning on huge models specifically**: this is the regime the project
targets but hasn't stress-tested end-to-end yet. Expect disk I/O, not GPU
compute, to be the dominant bottleneck once the working set exceeds
VRAM+RAM - a model that's merely large but still fits in VRAM+RAM will
feel close to native llama.cpp speed, while one that genuinely exceeds
both will bottleneck on storage throughput and page-cache pressure per
token, however fast the GPU is. If you hit this case, the levers worth
tuning are `--moe-stream-io-threads` (verify against your own disk, don't
assume higher is better), `--moe-stream-cache` (bigger cache, fewer
misses, if you have the VRAM/RAM to spare), and whether your storage is
local NVMe vs. anything virtualized or network-backed.

## Quick start

```sh
# Configure - enable CUDA and the web UI. CMAKE_CUDA_ARCHITECTURES is
# auto-detected from the GPU in this machine, no need to set it by hand.
cmake -B build \
  -DGGML_CUDA=ON \
  -DLLAMA_BUILD_UI=ON \
  -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --target llama-server -j"$(nproc)"

# Start in router mode - no model on the command line. Downloads/loads
# happen from the web UI instead. --models-preset is where per-model
# --moe-stream-cache sizes (computed by the picker) get remembered; the
# file doesn't need to exist yet, it's created on first download.
./build/bin/llama-server \
  --models-preset ~/.cache/llama-moe-pager/models-preset.ini \
  --moe-stream \
  --moe-stream-io-threads 2 \
  --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080/#/models`, find a model that fits your
hardware, click **Download**, then **Load** once it's ready, then **Chat**.
That's the whole flow - no manual `-hf` flags, no restarting the server per
model.

Prefer the old single-model workflow (one model, one command, no picker
UI)? That still works:

```sh
./build/bin/llama-server \
  -hf <org>/<model>-GGUF \
  -ngl 99 \
  --moe-stream \
  --moe-stream-cache 8 \
  --moe-stream-io-threads 2 \
  --host 0.0.0.0 --port 8080
```

### `--moe-stream` flags

| Flag | Meaning |
|---|---|
| `--moe-stream` | Enable on-demand disk-streaming of routed experts. |
| `--moe-stream-cache <NG\|Ns>` | Expert cache size: a memory budget in GiB (`8`) or an exact slot count per layer (`64s`). Default: auto-sized from available VRAM/RAM. In router mode the picker sets this per model for you via `--models-preset`. |
| `--moe-stream-cpu-cache` | Force the expert cache into host RAM even on a GPU-offloaded layer. Mainly useful for isolating VRAM- vs RAM-cache performance; normally the cache follows the layer's own device automatically. |
| `--moe-stream-io-threads <N>` | Parallel disk-read threads for streaming. Tune low (1-2) on Hyper-V-backed virtual disks (e.g. WSL2) - more threads measured *slower* there; verify against your own disk rather than assuming higher is better. |
| `--moe-stream-direct` | Use `O_DIRECT` for expert reads, bypassing the page cache. Useful for clean A/B benchmarking (removes page-cache warmth as a confound); real disk I/O without it may be faster in practice since a warm page cache helps. |
| `--moe-stream-prefetch <N>` | Experimental, off by default - see [Status](#status). |
| `--models-preset <path>` | Router mode only. INI file of per-model CLI overrides, written to automatically by the picker's Download button; safe to point at a path that doesn't exist yet. |

## The model picker

`/models` reads your live hardware (VRAM via `ggml_backend_dev_memory()`,
RAM, free disk) and cross-references it against the UGI Leaderboard's MoE
models to show, per model:

- **VRAM / SSD**: active- and total-parameter footprint at the best quant
  that fits, computed as params (B) x bits-per-weight / 8. VRAM and RAM
  never exclude a model outright - a model's expected speed falls into one
  of four informational tiers (VRAM-resident, spills into a RAM cache, or
  the actual target case for `--moe-stream`: a working set bigger than
  VRAM+RAM, streamed from disk on cache misses), but none of those tiers
  stop it from running. Only disk space is a hard constraint, and even
  there a model that doesn't fit *right now* stays visible, tagged "needs
  N GB more," instead of disappearing, so you can judge what to free up for
  the one you actually want. Only a model bigger than the drive's total
  capacity (no amount of deleting fixes that) is excluded outright.
- **UGI / Willingness**: the leaderboard's score for how much knowledge and
  reasoning a model demonstrates on sensitive topics instead of refusing,
  and its separate pure-refusal-rate sub-score. See
  [Why derestricted models](#why-derestricted-models) below.
- **Sortable columns, ranked UGI-first**: the candidate pool is ranked
  smartest-first before your hardware's `top` cutoff applies, then every
  column is independently sortable client-side, no re-fetch. **Derestricted
  finetunes only** filters that same pool to models specifically known to
  have refusal behavior removed - a curated allowlist plus live HF tags, not
  a name-keyword guess.

The same ranking logic is also available as a CLI: `scripts/model_picker.py
--help`.

## Why derestricted models

Every model on Hugging Face is aligned to someone's content policy before it
reaches you, usually a commercial lab's, tuned for a general consumer
audience and a legal team's risk tolerance, not for a security researcher's
actual job. That alignment doesn't only block clearly harmful requests; it
also hedges, refuses, or moralizes on entirely legitimate work: analyzing a
malware sample, explaining how a CVE is actually exploited, writing a YARA
rule that has to describe the pattern it detects, drafting a phishing
template for an authorized red-team engagement, or discussing a
controversial topic without a disclaimer paragraph bolted onto every answer.

That's a real cost, not a moral question. A model that won't engage with the
actual content of your work isn't "safer" for a working security
professional, it's slower and less useful, and it pushes you toward ad hoc
jailbreaks or a worse local model that will just answer. Heavy alignment
also isn't neutral: a model tuned to avoid certain framings encodes its
trainers' choices, not an absence of bias, so a derestricted variant is
often a more direct read on what the base model actually knows.

This project treats derestricted (a.k.a. "abliterated," "uncensored")
finetunes as a first-class ranking criterion, not an afterthought:

- **Derestricted finetunes only** filters to models specifically known to
  have refusal behavior removed, cross-referenced against a curated
  allowlist and live HF tags, not a name-keyword guess.
- **UGI** and **Willingness** come straight from the
  [UGI Leaderboard](https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard)
  ("Uncensored General Intelligence"): UGI measures how much knowledge and
  reasoning a model actually demonstrates on sensitive topics instead of
  refusing, Willingness measures how rarely it refuses at all. Both are
  measured, not guessed.
- Everything downloads and runs entirely on your own hardware. No
  engagement data, no client work, no malware sample, no exploit code ever
  leaves the machine, unlike a hosted API that logs prompts by default and
  applies its own refusal layer regardless of what you've already screened
  for.

If your work is authorized penetration testing, malware analysis,
red-teaming, CTF, or any other legitimate security research where "the
model refused to discuss the attack technique it was supposed to help you
analyze" is a recurring problem, that's exactly the gap this exists to
close.

## Pentest appliance

Optional layer on top of the base MoE VRAM Pager: a local-LLM-driven
recon/exploit agent, wired into real offensive-security and OSINT
platforms, with a hard-gated osint/recon/exploit phase split. Full
from-scratch setup (systemd units, per-tool install, API keys, architecture
diagram) lives in [PENTEST_APPLIANCE.md](PENTEST_APPLIANCE.md); summary
below.

An uncensored model chosen by the picker above (see
[Why derestricted models](#why-derestricted-models)) drives a tool-calling
loop (`tools/pentest_agent.py`) against an authorized target. A `/pentest`
panel in the web UI (`tools/pentest_ui_api.py` sidecar + SvelteKit
frontend) starts, monitors, restarts, and stops runs, and compiles a PDF
report when one finishes.

> [!WARNING]
> Only scan systems you own or have explicit legal/contractual
> authorization to test. Unauthorized scanning and exploitation is illegal
> in most jurisdictions regardless of intent - get written scope
> confirmation before pointing this at anything.
>
> Remote exploits that need a reverse connection or a spawned shell
> (Metasploit reverse/bind payloads, callback-based modules) require a
> Linux host with a direct internet-routable connection - no NAT, no
> firewall in the way of the callback path. Behind NAT/CGNAT or a firewall,
> point exploitation at connectionless proof (check-only confirmation, or a
> command-execution module that writes and independently reads back a
> marker file) instead of a full session.

### Setup

```bash
tools/setup_pentest_appliance.sh
```

One idempotent script for the whole stack: builds `llama-server`, installs
nmap with the raw-socket capabilities `nmap_scan` needs, offers to install
Metasploit Framework and OWASP ZAP (each asks for confirmation first, since
both installers do more than drop a single package), clones and patches
[MetasploitMCP](https://github.com/GH05TCREW/MetasploitMCP), writes the
secrets file (prompts for `MSF_PASSWORD`/`ZAP_API_KEY`, blank = generate a
random one), installs the systemd units and a scoped passwordless-sudo
rule, starts everything, and verifies it's up. Safe to re-run any time -
every step checks current state first and skips what's already done. See
[PENTEST_APPLIANCE.md](PENTEST_APPLIANCE.md) for exactly what each step
does and how to run any piece by hand instead.

### Start / stop / restart

```bash
tools/pentest_appliance.sh start     # msf-db -> msfrpcd -> metasploit-mcp -> zap -> pentest-ui-api -> llama-moe-router
tools/pentest_appliance.sh stop      # reverse order
tools/pentest_appliance.sh restart
tools/pentest_appliance.sh status
```

Wraps the six systemd services in the dependency order the Metasploit
chain needs, stopping in reverse so nothing loses a dependency mid-
teardown. Requires the passwordless-sudo rule the setup script installs -
without it, `sudo` blocks mid-sequence waiting for a password with no
terminal to read it from.

Ordered from most passive (never touches the target) to most aggressive
(runs real exploits against it):

| Platform | Role | Reached via |
|---|---|---|
| [theHarvester](https://github.com/laramies/theHarvester) | Passive OSINT aggregator: emails, subdomains/hosts, IPs from Certificate Transparency, search engines, breach indexes, wayback history | local checkout, run via `uv` - no API key needed |
| Google dorking | Curated Google Hacking Database-style queries (exposed admin panels, leaked config/backup files, directory listings, exposed `.git`, ...) | Brave Search API if `BRAVE_API_KEY` is set (real operator precision), else DuckDuckGo HTML search (zero-config, looser matching) - never contacts google.com or the target |
| [NIST NVD](https://nvd.nist.gov/) | CVE lookup by keyword or exact CPE 2.3 string | `services.nvd.nist.gov` REST API |
| Passive origin OSINT | Find a real origin IP behind a CDN/WAF | Certificate Transparency logs (crt.sh) + DNS, cross-checked against Cloudflare's published ranges |
| nmap | Port/service/OS scanning, NSE scripts | local binary, `cap_net_raw`/`cap_net_admin`/`cap_net_bind_service` via setcap (no root) |
| [OWASP ZAP](https://www.zaproxy.org/) (Zed Attack Proxy) | Web app spidering (recon), then active scan with real attack payloads (exploit) | ZAP's REST API directly (it has no MCP server of its own) |
| [Metasploit Framework](https://github.com/rapid7/metasploit-framework) | Exploitation, payload generation, session control - the most aggressive tier | `msfrpcd`, its local RPC daemon (port 55553, loopback-only) |
| [MetasploitMCP](https://github.com/GH05TCREW/MetasploitMCP) | Wraps `msfrpcd`'s RPC API as an MCP server | MCP session over SSE (port 8085), separate repo cloned by the setup script |

`pentest_ui_api.py` (the sidecar behind the `/pentest` panel) and
`pentest_report.py` (PDF generation) run from the same Python venv as
MetasploitMCP, on FastAPI/Uvicorn/Pydantic/ReportLab/`requests`/`mcp` - see
`tools/requirements-pentest-agent.txt`.

### Phase gating

OSINT, recon, and active exploitation are separate runs, split at the code
level, not just by prompt:

- **OSINT** (optional, most passive) - only `theharvester_scan` and
  `google_dork_search` are exposed; every request goes to a third-party
  public data source, never to the target itself. Run it ahead of recon if
  you want its findings available to resume into a later run.
- **Direct Recon** (default, safe to run unattended) - only read-only/
  scanning tools are exposed to the model at all; exploitation tool
  schemas are never sent, so the model can't call them mid-run no matter
  what it decides.
- **Exploit** (opt-in, human-gated) - requires `--phase exploit` **and**
  `--confirm-exploitation` together, one flag alone isn't enough. Typically
  run with `--resume-from` pointing at the recon run's log, so the model
  has the human-reviewed findings as context instead of starting blind.
- **Always-on backstop** - a destructive-command regex (`rm`, `cp`/`mv`
  overwriting an existing file, `mkfs`, `dd ... of=/dev/...`,
  `shutdown`/`reboot`, `iptables -F`, appending to a server config file
  like `.htaccess`/`nginx.conf`/`httpd.conf`, ...) intercepts matching
  `send_session_command` calls before they ever reach the target, even
  inside exploit phase - a hard code-level gate the command never gets
  past, not a prompt the model is just asked nicely to follow. Appending
  to a new marker file, or to an existing inert plain-text file (`echo`/
  `cat`/`tee` with `>>`), is deliberately not blocked - that's the actual
  connectionless-proof technique the model is told to use, and reverse/bind
  sessions are preferred over it whenever the target is actually reachable.
- **Never scans CDN/edge infrastructure** - before nmap, ZAP, or any
  Metasploit module ever touches a host, that host is resolved and checked
  against Cloudflare's published ranges and known second-CDN CNAME chains
  (Akamai, CloudFront, Fastly, Azure Front Door, ...); a target domain that
  resolves to one, or an "origin IP" candidate that turns out to be one, is
  refused rather than scanned, and the refusal is logged (`safety_skip`
  events, shown live in the `/pentest` UI). This is a code-level gate, not
  a prompt instruction, and applies the same way whether the host came from
  `--target`, `find_origin_ip`, or a resumed prior run.

### API keys / accounts

The core appliance (model inference, the webui, expert streaming) needs
none of this - it's fully local. These only apply if you set up the
optional pentest layer:

| Variable | Required? | What it is |
|---|---|---|
| `MSF_PASSWORD` | Yes, for Metasploit | `msfrpcd`'s local RPC auth (loopback-only, port 55553). Not a real credential - pick anything, or let the setup script generate one; nothing external to coordinate with. |
| `ZAP_API_KEY` | Yes, for ZAP | Same idea, local REST API auth for the ZAP daemon (loopback, port 8090). |
| `BRAVE_API_KEY` | No | A real external account: Brave Search API (free tier, 2,000 queries/month), used by `google_dork_search` for real operator precision. Without it, dorking still works with zero config via DuckDuckGo's HTML results, just noisier. |
| `NVD_API_KEY` | No | A real external account: NIST NVD's API key, raises the rate limit on `cve_lookup`/`cve_lookup_cpe`. Without it, lookups still work against NVD's public (lower) rate limit. |

All four live in the same secrets file (`/var/secrets/moe-vram-pager.env`),
never in code or git - see
[PENTEST_APPLIANCE.md](PENTEST_APPLIANCE.md#5-secrets) for exactly where to
put them. NIST NVD itself is always contacted for CVE lookups regardless of
whether `NVD_API_KEY` is set - it's a public API, the key just lifts the
rate limit.

### Default chat system prompt

The webui's Settings > General > System Message field ships with a default
covering general assistant behavior plus pentest-tool-specific guidance
(gated to only apply when Metasploit/nmap/ZAP/etc. tools are actually in the
tool list). Source of truth is
[`tools/ui/src/lib/constants/default-system-prompt.ts`](tools/ui/src/lib/constants/default-system-prompt.ts)
(`DEFAULT_SYSTEM_PROMPT`) - once you've saved your own copy in the GUI it's
persisted to localStorage and stops tracking code changes, so paste the
block below over your saved copy any time this file is updated upstream:

```
You are the assistant running on this user's self-hosted MoE VRAM Pager appliance: a local
llama.cpp fork with on-demand VRAM-tier MoE expert streaming, serving a locally-run
open-weight model with no cloud LLM API and no telemetry. If pentest tools are attached,
NIST NVD and optionally Brave Search are reached directly for CVE lookups/dorking - say so
if asked, don't claim total isolation. Say plainly what you are and where you run if
asked.

CAPABILITIES AND HONESTY
- Only use tools actually in your tool list this turn - don't imply a capability you don't
  currently have.
- Text-only, no vision/image input. Say so if asked to look at an image.
- If you don't know something and can't check, say so - a wrong guess stated as fact is
  worse than "I don't know."
- Be concise. Match response length to the question; save headers/bullets for genuinely
  multi-part answers.

TOOL USE
- Pick the tool that matches the question - check its description first, don't guess from
  its name.
- If a call fails or returns nothing useful, say so and try a different approach or stop -
  don't retry blindly or fabricate a result.
- Order calls so each result informs the next; stop once you have enough to answer.

---

SECURITY ASSESSMENT MODE
(Only applies when Metasploit, nmap_scan, zap_spider_scan, zap_active_scan, cve_lookup,
find_origin_ip, theharvester_scan, google_dork_search, or raw_tcp_send are in your tool
list. Otherwise ignore this section.)

You are an authorized penetration tester. Only operate against hosts the user explicitly
named in-scope - never assume scope, never pivot, never act on anything outside the
reviewed scope (mention it instead).

Work recon before exploitation, verify before acting, least-destructive method first.
Don't work a findings list one item at a time - combine related low/medium findings into a
real chain (weak cookie flags plus missing headers can mean a working XSS-to-cookie-theft
path; a version banner plus a matching cve_lookup hit is a targeted exploit attempt, not a
shotgunned one). Before any exploit module, run it with check_vulnerability=true first - a
lightweight probe that's a real, reportable finding on its own even with no session. Only
go further with payload_name/payload_options if a session/shell adds evidence the check
didn't. If a module doesn't support check, skip straight to exploitation.

The goal is proving code execution happened. Prefer a reverse/bind session when you have
real reachability back to yourself - it gives you
list_active_sessions/send_session_command for real verification (id, whoami, cat a file),
stronger evidence than one marker write. Set LHOST/LPORT for any reverse payload or it
fails validation before reaching the target - use the address you were given, don't guess
or leave it blank. Only fall back to a connectionless payload (list_payloads arch='cmd',
e.g. cmd/unix/generic with a CMD option) when reverse/bind connectivity genuinely isn't
possible (NAT, CGNAT, an outbound firewall) or the module has no session support: write
ONE marker to a new web/FTP-readable file, or append to an existing plain-text file a tool
can already read, then verify with a separate read (HTTP GET, FTP RETR) - not through the
same session. Never target a server config file for this (.htaccess, nginx.conf,
httpd.conf, web.config, php.ini, etc.), even to append - a malformed edit there can break
the whole site. If the read-back 403s/404s/comes back empty, don't assume execution failed
- server config can block the read path even after real execution; try another extension
or already-web-served directory before concluding, and report the two cases (didn't run
vs. ran but blocked) distinctly.

zap_active_scan sends real attack payloads (SQLi, XSS, path traversal) - use it like any
exploit tool, in-scope only. When cve_lookup names a specific technique, use raw_tcp_send
to send that exact request and pull back real evidence, not just the CVE ID. If a WAF
looks like naive string-matching, standard encoding variants (URL/double-encoding, case,
path tricks like ..;/ or %2e%2e%2f) are fair game to confirm the underlying flaw is real.
raw_tcp_send is for requests the other tools don't already send, not a substitute for
them.

Once you have a session, act like the attacker actually would to prove impact -
send_session_command with id/whoami/hostname/uname -a, or cat a file, is real evidence,
not a read-only stand-in. A session also sidesteps the .htaccess/server-config read-back
problem entirely - cat the marker straight through the shell instead of an external HTTP
read. Destructive commands (rm, cp/mv overwriting an existing file, mkfs, dd to a device,
shutdown/reboot, appending to a server config file, etc.) are blocked at the tool level
regardless of privilege - a hard code-level gate the command never gets past, not a rule
you're asked to follow, so proving you could have run them is exactly as strong a finding
as doing it. Appending to a new marker file, or an existing inert plain-text file, with
echo/cat/tee and >> is fine and is the intended proof technique.

Report every outcome honestly:
  - EXPLOITED - session/shell obtained, with commands run and their output as evidence
  - PROVEN WITHOUT A SESSION - connectionless execution confirmed via independent
    read-back
  - LIKELY EXECUTED, READ-BACK BLOCKED - marker command's own output suggests it ran, but
    verification was blocked by the target's own access control (e.g. .htaccess), not by
    the exploit failing - don't edit target config to force the read, don't round up or
    down; name the block if you can, it helps remediation
  - CONFIRMED VULNERABLE (check only) - module's check confirmed it; no further
    exploitation needed
  - ATTEMPTED, NOT EXPLOITABLE - plausible but the target is patched/mitigated; report as
    a real negative finding
  - BLOCKED BY A FILTER/WAF - the flaw may still be real, distinguish this from "not
    vulnerable"
  - INCONCLUSIVE / TOOL ERROR - timed out, errored, or ambiguous; say so, don't round up
    or down
  - OUT OF SCOPE - noticed but not part of the reviewed scope; name it, don't act on it
  - HOST/SERVICE UNREACHABLE - say so rather than silently moving on

When done (every finding worked through, or no safe next step left), reply with a
plain-text summary (host, service, vulnerability, action, outcome, evidence) and stop
calling tools. No markdown headers for a short list - plain paragraphs read better than
bullet soup. ASCII punctuation only (no em/en dashes, curly quotes, non-breaking spaces).
```

## Repo layout

- `src/llama-moe-stream.{cpp,h}`: the expert-streaming cache itself.
- `tools/server/server-model-picker.{cpp,h}`: the `/model-picker/*` API
  backing the web picker (hardware-fit ranking, and preparing a per-model
  `--moe-stream-cache` override before a router-mode download).
- `tools/ui/src/routes/models/`: the picker's frontend (SvelteKit) -
  ranking table plus, in router mode, download/load/delete actions.
- `scripts/model_picker.py`: CLI mirror of the ranking logic.
- `common/preset.{cpp,h}`: INI preset read/write for router mode's
  `--models-preset`, including the writer the picker uses to persist a
  computed `--moe-stream-cache` value per model.
- `HANDOFF.md`: running technical log: hardware constraints, build
  gotchas, and open engineering threads.
- `tools/pentest_agent.py`: the recon/exploit agent loop and its local
  tools (nmap, CVE lookup, origin-IP OSINT, ZAP, raw TCP/UDP).
- `tools/pentest_ui_api.py`, `tools/ui/src/routes/pentest/`: the `/pentest`
  web UI panel and the sidecar API that runs/streams/stops agent runs.
- `tools/pentest_report.py`: turns a run's JSON log into a PDF report.
- `tools/setup_pentest_appliance.sh`: idempotent, from-scratch install of
  the whole pentest appliance stack.
- `tools/pentest_appliance.sh`: start/stop/restart/status for the six
  appliance systemd services, in dependency order.
- `PENTEST_APPLIANCE.md`: from-scratch setup for the pentest appliance
  layer - see [Pentest appliance](#pentest-appliance) above.

## License

MIT. See [LICENSE](LICENSE).
