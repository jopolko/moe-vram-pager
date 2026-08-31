# Pentest governance ontology

## What this is

`tools/pentest_ontology.py` is a typed object graph plus a governed-action layer
that every pentest tool call passes through, in the spirit of Palantir/Karp's
ontology framing: model the world as typed objects with typed links, and make
**actions** the only sanctioned, validated, audited way state changes.

It generalizes an earlier single-purpose fix (v1, below) into a layer that:

- models the engagement as objects: `Target`, `Service`, `Module`, `Payload`,
  `Option`, `Session`, `Finding`, connected by typed links (`RUNS_ON`,
  `TARGETS`, `COMPATIBLE_WITH`, `USES_PAYLOAD`, `OPENED`, `ON`, `HAS_OPTION`,
  `EVIDENCES`);
- gates every tool call through `check_action(tool, args, ...) -> Verdict`,
  which runs live ground-truth resolvers, graph-query preconditions, a phase
  gate, and a per-run strike counter;
- seeds the graph from recon output and renders a compact `KNOWN ENVIRONMENT`
  block into the exploit-phase prompt so the model proposes real, compatible
  things in the first place.

## The origin story (v1): the DVWA hallucination loop

Run `runs/20260820T190304Z-exploit.json` (2026-08-20, target 172.30.0.50,
model `Qwen3.6-35B-A3B-uncensored-heretic`) spent ~2.5 hours almost entirely
retrying `auxiliary/scanner/http/dvwa_login` and
`auxiliary/scanner/http/fingerprint_webapp` - **neither module exists in
Metasploit**. Every attempt returned the identical error:

```
[-] No results from search
[-] Failed to load module: auxiliary/scanner/http/dvwa_login
```

The model never adapted. It re-issued the same nonexistent module name roughly
every 2 minutes for the whole exploit phase. Root causes:

1. **No ground truth was queryable** for the module the model needed.
   `list_exploits` only searched Metasploit's `exploit/` tree; the guessed
   module lives in `auxiliary/`, so the search came back empty and the model
   had no way to find out it was hallucinating.
2. **No validation gate before dispatch.** `run_auxiliary_module` was called
   directly with whatever `module_name` the model produced; the failure
   round-tripped through a full Metasploit console `use` before returning.
3. **No loop detection.** Nothing tracked "you already tried this exact
   (tool, module_name) pair and it failed identically."

v1 fixed exactly that: a live `validate_module` MCP tool, a dispatch-time gate,
and a 2-strike counter (`MODULE_NAME_FAILURE_LIMIT`).

## Why v1 needed generalizing

The next failure was a different shape, and slipped through every v1 guard:

- a hallucinated **payload** (`cmd/linux/http/x64/reverse_tcp`) with
  `LHOST: "auto"` - there was no gate for `payload_name` at all;
- it came through the **llama.cpp webui chat mode**, which talks straight to
  the MCP server on port 8085 and never runs the agent loop, so none of v1's
  guards (all in `pentest_agent.py`) executed.

Two lessons: the gate has to cover more than module names, and it has to sit at
**both** entry points, not just the agent loop.

## The object graph

`OntologyGraph` (one per run / per MCP session, `RLock`-guarded) holds:

| Object | Key fields |
|---|---|
| `Target` | host, in_scope, arch, os_family, source |
| `Service` | target_key, port, proto, name, product, version, cpes, url |
| `Module` | fullname, module_type, exists, required_options, suggestions |
| `Payload` | fullname, exists, required_options, suggestions |
| `Option` | name, value, source |
| `Session` | sid, session_type, target_key, via_module |
| `Finding` | severity, title, tool, detail, refs |

Links are `(src_type, src_key, Rel, dst_key)` tuples in a set. Queries the
preconditions use: `host_in_scope(host) -> bool | None` (`None` = no scope info,
so preconditions warn rather than block), `find_target`, `known_services`,
`session_ids`, `observed_arch`.

The graph is **seeded** before the exploit phase: `seed_target`,
`seed_scope(graph, authorized_hosts)` (marks the graph scope-aware, so unknown
hosts return `False` not `None`), and `seed_from_recon_run(graph, run_json)`
(reuses `pentest_report.extract_findings_from_event` for findings, an nmap port
regex for services, and guesses arch/os from version banners).

## Governed actions

`ACTIONS: dict[str, ActionSpec]` in code (not JSON - preconditions are async
graph queries and live RPC calls, not expressible as data; and the registry
never enters model context, so the ctx-budget argument that favors a live gate
over a prompt list does not apply). Each `ActionSpec` has a minimum phase,
an ordered tuple of preconditions, an optional effects function, and a strike
kind. Coverage:

| tool | min phase | preconditions | effects |
|---|---|---|---|
| `run_exploit` | exploit | target in scope, module exists, payload exists, payload compatible, required options present, reverse-route (warn) | parse `session N opened` -> add Session |
| `run_auxiliary_module` | recon | module exists, recon-safe auxiliary, target in scope | upsert Module |
| `run_post_module` | exploit | module exists, session exists | upsert Module |
| `generate_payload` | exploit | payload exists, required options present | upsert Payload |
| `send_session_command` | exploit | session exists, not destructive (`DESTRUCTIVE_COMMAND_RE`) | - |
| `terminate_session` | exploit | session exists | remove Session |
| `start_listener` | exploit | lhost sane (warn) | - |
| `nmap_scan` / `raw_tcp_send` / `nping_send` | recon | scan target in scope, not CDN edge | nmap: parse ports -> add Service |
| `zap_spider_scan` | recon | scan url in scope | - |
| `zap_active_scan` | exploit | scan url in scope | - |
| `cve_lookup` | any | - | parse CVE lines -> add Finding |
| `list_active_sessions` | any | - | reconcile Session set |

Tools not in `ACTIONS` -> `Verdict(allowed=True)` (matches the old `else`
passthrough).

## `check_action` and `Verdict`

```
async def check_action(tool_name, args, *, phase=None, scope=None, graph=None,
                       session_key=None, resolver=None, cdn_check=None,
                       strikes=None, dict_shape=False) -> Verdict
```

1. Unknown tool -> allow.
2. Strike short-circuit **before any resolver call**: if this exact call is at
   its strike limit, return blocked ("permanently blocked this run").
3. Phase gate (skipped entirely when `phase is None`, i.e. chat mode).
4. Run preconditions in order (sync or async); the first `block` wins ->
   record a strike, return a blocked `Verdict` carrying its message and
   suggestions.
5. No block -> collect any warnings, clear the prior strike, attach effects,
   return an allowed `Verdict`.
6. **Any exception anywhere in 2-5 -> stderr log + `Verdict(allowed=True)`.**
   Fail-open is absolute: a resolver hiccup, an unexpected pymetasploit3
   attribute, a bug in a precondition - none of it can ever block a legitimate
   call.

`Verdict.render_block()` returns an `ERROR: ...`-shaped string (or a
`{"status":"blocked","message":...}` dict for the msf tools whose return
annotation is `dict`) so the model reads it as the tool's own output.
`render_warnings()` prefixes `NOTE:`. `apply_effects(graph, result_text)`
mutates the graph after a successful dispatch; every effect is
exception-swallowed.

## Live resolvers

`Resolver` protocol: `async module(module_type, name)` and
`async payload(name, module=None)`, both returning `None` on failure (caller
fails open). `_TtlCache` keeps positive results 300s, negative 60s.

- `DirectMsfResolver` (server side, in `pentest_tools_mcp.py`): lazily
  `import MetasploitMCP as msf` and `await msf.validate_module/validate_payload`
  directly - same process, same live client, zero logic duplication.
- `McpSessionResolver` (agent side): `await mcp_session.call_tool(...)` and
  parse `.content[].text` JSON.

`validate_payload` (added this cycle to `~/MetasploitMCP/MetasploitMCP.py`,
mirrors `validate_module`): live `_get_module_object('payload', name)` check,
`difflib` suggestions from `client.modules.payloads`, `required_options` from
the payload object, and - when a `module_name` is given - a compatibility check
against `exploit_obj.payloads`, returning `compatible_payloads` when incompatible.

## Two enforcement points

| point | phase / scope | adds |
|---|---|---|
| `check_action()` in `pentest_agent.py` dispatch loop | real phase + authorized scope + seeded per-run graph | phase gate, in-scope enforcement, CDN-edge refusal, `KNOWN ENVIRONMENT` prompt block |
| `_govern()` wrapper on every tool in `pentest_tools_mcp.py` | `phase=None`, `scope=None` | covers webui chat mode; existence / compatibility / required-options / destructive-command / session checks; scope checks degrade to warnings |

`_govern` uses `functools.wraps` + an explicit `wrapper.__wrapped__ = fn` so
FastMCP's `Tool.from_function` (which calls `inspect.signature(fn,
eval_str=True)` and follows `__wrapped__`) still sees the original name, doc,
annotations, and JSON schema. Session isolation for concurrent MCP clients is a
`contextvars.ContextVar` set per SSE connection; the documented fallback is a
single `"chat"` bucket (over-shares for a single-user appliance, never
corrupts).

## The strike counter, generalized

`StrikeLedger`: `dict[(session_key, canonical_sig), int]` with
`record/clear/count/at_limit`. `_canonical_sig(tool, args, kind)` keys on:
module name (type prefix stripped), payload name (normalized), scan host
(canonicalized), or `sha1(command)[:12]` for session commands. Limits:
`module_name`/`payload_name` = 2, `scan_target`/`session_cmd` = 3, default 4.
This replaces v1's `pentest_agent.module_name_failures`.

## Launch-time prompt rendering

`render_ontology_context(graph, phase, max_chars=1200)` returns "" when the
graph has nothing useful, else a compact block: `KNOWN ENVIRONMENT` (scope
hosts, each target + arch + services), `PREFERRED PAYLOADS for <arch>` (from
the static `PREFERRED_PAYLOADS_BY_ARCH` table, guidance not a live query),
`OPEN SESSIONS`. Appended to `user_content` (not the system prompt) so
`--system-prompt-file` cannot clobber it.

This is the "don't hallucinate in the first place" half. The gate catches bad
proposals; the rendered context reduces how often the model makes them, by
handing it the real arch and the real preferred payload tokens up front.

## Ontology vs. a system message

Why a live layer rather than a list of valid names in the system prompt:

- **Static vs. live.** A prompt list is frozen at construction. Resolvers query
  `msfrpcd` at the moment of use, via the same lookup the real call would do.
- **Trusted vs. verified.** A system message is text the model must read,
  remember, and apply correctly; nothing checks that it did. The gate is
  enforced on every call regardless of the model's compliance.
- **Always-loaded vs. pay-per-use.** Every module/payload name in context on
  every request costs tokens whether or not it is needed. The gate costs
  nothing until a name is actually proposed.
- **No feedback loop vs. a built-in one.** A static list cannot catch a model
  that misreads it or hallucinates past it. The strike counter is server-side
  state, not something the model has to notice.

## Why this lets you run a smaller model

The 2.5-hour loop was not a "smart enough" problem: a bigger model would very
likely have guessed `dvwa_login` too, since it is a plausible-sounding name.
What a bigger model buys you is noticing the repeated identical failure and
deciding to stop - exactly the implicit self-monitoring a small/quantized model
is worst at once its context fills with tool schemas and findings.

The ontology moves that decision into deterministic code:

- **The model no longer has to notice it is wrong** - the resolver tells it
  directly, with real alternatives.
- **The model no longer has to notice it is looping** - the strike counter does
  that mechanically; a model that ignores the suggestions gets hard-blocked
  after 2 tries, minutes of waste instead of hours.
- **Less context spent per wrong guess** - a blocked call is one short error
  string, not a full console round trip's worth of output.

## What this does and doesn't solve

**Solves:** the nonexistent-module and nonexistent-payload loops; the chat-mode
bypass; scanning a CDN edge instead of an origin; running an exploit before
recon; destructive session commands; acting on a session that does not exist;
missing required options; module/payload incompatibility.

**Doesn't solve:** a model hallucinating a *different* wrong name every time
(no exact repeat to strike - the suggestions list is the only mitigation);
semantically bad but schema-valid option *values* (wrong RHOSTS within scope,
a bad credential guess); anything the live resolver itself gets wrong (it fails
open by design).

## Files

- `tools/pentest_ontology.py` - the layer (objects, graph, `ACTIONS`,
  `check_action`, `Verdict`, resolvers, `StrikeLedger`, seeding, rendering,
  `dump_ontology_text`).
- `tools/pentest_ontology_tables.json` - optional overlay for operator-tunable
  tables (strike limits, preferred payloads). Merged at import; absent is fine.
- `tools/dump_ontology.py` - CLI, prints the human-readable reference snapshot.
- `tools/pentest_tools_mcp.py` - `_govern` wrapper + contextvar + registration.
- `tools/pentest_agent.py` - dispatch-loop guard chain replaced with
  `check_action`; run-scoped graph/strike setup; exploit-phase prompt hook;
  `DESTRUCTIVE_COMMAND_RE` / `MODULE_NAME_TOOLS` /
  `RECON_SAFE_AUXILIARY_PREFIXES` relocated here and re-imported for back-compat.
- `~/MetasploitMCP/MetasploitMCP.py` - `validate_payload` tool (next to
  `validate_module`).
- `tools/tests/test_pentest_ontology.py` (+ `_ontology_fakes.py`, `conftest.py`)
  - 20 cases, no live `msfrpcd`. Run `python -m pytest tools/tests` or
  `python tools/tests/test_pentest_ontology.py` standalone.

Regenerate the desktop reference after any registry or table change:

```bash
python tools/dump_ontology.py > /mnt/c/Users/josh/Desktop/pentest-ontology-reference.txt
```

Code stays authoritative; edits to the `.txt` do not affect runtime.
