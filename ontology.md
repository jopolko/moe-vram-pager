# Module ontology: fixing the DVWA hallucination loop

## The problem this fixes

Run `runs/20260820T190304Z-exploit.json` (2026-08-20, target 172.30.0.50,
model `Qwen3.6-35B-A3B-uncensored-heretic`) spent ~2.5 hours almost entirely
retrying `auxiliary/scanner/http/dvwa_login` and
`auxiliary/scanner/http/fingerprint_webapp` - **neither module exists in
Metasploit**. Every attempt returned the identical error:

```
[-] No results from search
[-] Failed to load module: auxiliary/scanner/http/dvwa_login
```

The model never adapted. It re-issued the same nonexistent module name
roughly every 2 minutes for the whole exploit phase. Root causes:

1. **No ground truth was ever queryable for the module the model needed.**
   `list_exploits` (the only enumeration tool MetasploitMCP exposed) only
   searches Metasploit's `exploit/` tree via `client.modules.exploits`. The
   module the model was guessing at lives in `auxiliary/` -
   `list_exploits(search_term="dvwa")` came back empty not because DVWA has
   no matching module, but because the tool was searching the wrong tree
   entirely. The model had no way to ever find out.
2. **No validation gate before dispatch.** `run_auxiliary_module` was called
   directly with whatever `module_name` the model produced; the failure
   round-tripped through a full Metasploit console `use` command before
   coming back as an error string.
3. **No loop detection.** Nothing in the agent tracked "you already tried
   this exact (tool, module_name) pair and it failed identically" - so nothing
   ever escalated, corrected, or blocked the repeat.

## The fix: a queryable module ontology

"Ontology" here means: **the actual, current, authoritative set of module
names Metasploit knows about**, made cheaply queryable by both the agent's
dispatch code and the model itself - instead of the model relying on
training-data memory of module names (which drifts, and apparently
hallucinates DVWA-specific modules that were never real).

### 1. `validate_module` - new MetasploitMCP tool

`MetasploitMCP.py`, next to `list_exploits`. Given `module_type`
(`exploit`/`auxiliary`/`post`) and a `module_name`, it:

- Tries the exact same lookup `run_exploit`/`run_auxiliary_module`/
  `run_post_module` use internally (`client.modules.use` via
  `_get_module_object`). If that succeeds, `{"exists": true}` - a positive
  result here means the follow-up real call will find the module too.
- If not found, pulls the **full real module list** for that type
  (`client.modules.auxiliary`, `.exploits`, or `.post` - the same lists MSF
  itself uses, not a cached/stale copy) and returns up to 5 close string
  matches via `difflib.get_close_matches`, so the caller gets real
  alternatives instead of a bare "not found."

This is the ontology's source of truth: it is never mirrored, cached, or
allowed to drift, because it's read live from `msfrpcd` on every call.

### 2. Dispatch-time gate - `pentest_agent.py`

`MODULE_NAME_TOOLS` maps each module-invoking tool to its module type:

```python
MODULE_NAME_TOOLS = {
    "run_exploit": "exploit",
    "run_auxiliary_module": "auxiliary",
    "run_post_module": "post",
}
```

Before any of these three tools reaches the real (slow, console-backed) MCP
call, the dispatch loop now calls `validate_module` first:

- **Exists** → proceeds to the real call exactly as before, clears any prior
  failure count for that name.
- **Doesn't exist** → the real tool is never called at all (saves the round
  trip that used to take 30-120s through the Metasploit console). The model
  gets back the close-match suggestions plus an explicit strike count.
- **`validate_module` itself errors** (RPC hiccup, timeout) → fails open,
  falls through to the real call so a genuine infrastructure blip never
  blocks a legitimate attempt.

### 3. Loop breaker - `MODULE_NAME_FAILURE_LIMIT = 2`

A `module_name_failures: dict[(tool, module_name), int]` counter lives for
the duration of one run. Two strikes on the exact same (tool, module_name)
pair and the third attempt is blocked **before even calling
`validate_module`** - no RPC round trip, no console call, just an immediate
error telling the model this name is permanently blocked for the rest of the
run and to pick something else or move on.

This is what actually caps the failure mode: even if the model ignores the
suggestions and tries the identical hallucinated name a third time, the
appliance stops it cold instead of burning another 2 hours.

## What this does and doesn't solve

**Solves:**
- The specific loop from this run (retrying a nonexistent module forever).
- The blind spot where `list_exploits` couldn't see `auxiliary/` modules at
  all, so the model had no way to self-correct even if it tried to check
  first.
- Wasted wall-clock time: invalid names now fail in well under a second
  instead of a full console round trip.

**Doesn't solve:**
- A model hallucinating a *different* wrong module name every time (no
  exact repeat to catch). The suggestions list is the mitigation here - it's
  handed back on every failure, not just after repeats - but it only helps
  if the model actually reads and uses it.
- Bad *arguments* to a module that does exist (wrong RHOSTS, bad option
  values, etc.) - this ontology only validates the module name, not its
  option schema.
- `list_post_module`-style enumeration still doesn't exist as a tool (there's
  no `list_auxiliary_modules`/`list_post_modules` browsing tool, only the
  validate-a-specific-name check). Worth adding if the model needs to browse
  rather than validate a guess - not done here since it wasn't the failure
  mode in this run.

## Why this also lets you drop to a smaller model

The 2.5-hour loop wasn't really a "smart enough" problem - a bigger model
would very likely have made the exact same wrong guess for `dvwa_login`,
since that name is a plausible-sounding module that simply doesn't exist.
What a bigger model buys you is noticing the repeated identical failure and
deciding to stop; a smaller/quantized model is exactly the kind that keeps
grinding on the same guess because it has less capacity left over for
"wait, this isn't working, let me change strategy" once the rest of its
context is full of tool schemas and recon findings.

This ontology removes the need for that judgment call entirely:

- **The model no longer has to notice it's wrong** - `validate_module`
  tells it directly, with real alternatives, instead of the model needing to
  infer "not found" means "give up" from a Metasploit console error string.
- **The model no longer has to notice it's looping** - the strike counter
  does that mechanically. A weak model that ignores the suggestions and
  retries anyway gets hard-blocked after 2 tries, capped at minutes of waste
  instead of hours, no self-awareness required on the model's part.
- **Less context spent per wrong guess** - a blocked/invalid call now costs
  one short error string instead of a full console `use` round trip's worth
  of output, which matters more on a small model's smaller context budget.

In short: the smaller the model, the more of its job used to be "notice you
should stop," which is precisely the kind of implicit self-monitoring small
models are worst at. Moving that decision into deterministic code (the
gate + strike counter) means a much weaker/cheaper model can drive this
appliance about as reliably as a stronger one, at least for this failure
class - the model just needs to call tools and read short, blunt error
messages, not manage its own retry discipline.

## Ontology vs. a system message

Why not just list valid module names in the system prompt instead of building
a live check? Because a system message and this ontology behave completely
differently on every axis that mattered for this failure:

- **Static vs. live.** A system-prompt list is frozen at prompt construction
  and never updated for the rest of the run. `validate_module` queries
  `msfrpcd` at the moment of use, via the same `client.modules.use` lookup the
  real call would do, so it can never go stale.
- **Trusted vs. verified.** A system message is just text the model has to
  read, remember, and correctly apply, nothing checks whether it did. The
  ontology is a gate the dispatch loop enforces on every call, the model's
  compliance is irrelevant.
- **Always-loaded vs. pay-per-use.** A list of every exploit/auxiliary/post
  module name would sit in context on every single request whether or not
  it's ever needed. The ontology costs nothing until a module name is
  actually proposed.
- **No feedback loop vs. built-in one.** A static list can't catch a model
  that misreads it or hallucinates past it. The ontology's strike counter
  (`module_name_failures`) catches exactly that case, since it's server-side
  state, not something the model has to notice on its own.

### The enforcement mechanism, concretely

The model never asks for validation, it doesn't know the gate exists. It
just emits a normal tool call, e.g. `run_auxiliary_module(module_name=
"auxiliary/scanner/http/dvwa_login")`. What actually happens in
`pentest_agent.py`'s dispatch loop (~line 2550):

1. The loop sees `name in MODULE_NAME_TOOLS` and, before ever touching the
   real MCP tool, checks `module_name_failures[(tool, module_name)]` against
   `MODULE_NAME_FAILURE_LIMIT`. If already at the limit, the call is blocked
   outright, no RPC round trip at all, just an immediate "permanently
   blocked" error string handed back as the tool result.
2. Otherwise it calls `validate_module` itself (the redirect is silent, the
   model believes it called the real tool).
3. `validate_module` tries the real lookup; if that fails it pulls the live
   module list for that type and returns close matches via
   `difflib.get_close_matches`.
4. The dispatch loop reads `exists`:
   - **True** -> proceeds to the real tool call exactly as if no gate
     existed, and clears any prior strike for that name.
   - **False** -> the real tool is *never called*. The loop builds an error
     string with the suggestions and strike count and feeds it back to the
     model as if it were the tool's own output.

This is why the guarantee holds even against a model that ignores
everything it's told: `run_exploit`/`run_auxiliary_module`/`run_post_module`
have no code path to execution that doesn't pass through this `elif` branch
first. It isn't advice the model can skip, it's the only route from tool
call to dispatch. The one deliberate exception is failing open when
`validate_module` itself errors (RPC hiccup, timeout), that defaults to
`{"exists": True}` so a genuine infrastructure blip doesn't false-block, but
there's no equivalent leniency once a name has actually been checked and
found fake.

## Files changed

- `~/MetasploitMCP/MetasploitMCP.py` - added `import difflib` and the
  `validate_module` tool (after `list_exploits`).
- `~/moe-vram-pager/tools/pentest_agent.py` - added `MODULE_NAME_TOOLS`,
  `MODULE_NAME_FAILURE_LIMIT`, the `module_name_failures` tracker, and the
  dispatch-time gate (new `elif` branch, sits between the
  `send_session_command` destructive-pattern check and the `cve_lookup`
  branch in the phase-2 tool dispatch chain).

Both files verified with `python3 -m py_compile`. Not yet run end-to-end
against a live target - worth a short exploit-phase run against the DVWA
honeytarget to confirm the model gets steered toward a real module (or a
non-Metasploit tool like `raw_tcp_send`/`zap_active_scan`) instead of
re-looping.
