# Contributing

MoE VRAM Pager is a clean-break fork of [llama.cpp](https://github.com/ggml-org/llama.cpp),
maintained by one person, built primarily by pairing with Claude Code. It is
not run like a large open-source project with tiered maintainers and a
formal review board, and the docs shouldn't pretend otherwise. This file
replaces the upstream `CONTRIBUTING.md`, which described a process (a
three-tier contributor hierarchy, a ban on AI-generated PRs) that doesn't
apply to a personal fork - upstream's own [AGENTS.md](AGENTS.md) says as
much: "Private forks are exempt."

## Scope: what's ours, what's upstream's

- **Fork-specific** (the actual point of this repo): `src/llama-moe-stream.*`,
  `tools/server/server-model-picker.*`, `tools/ui/src/routes/models/`,
  `common/preset.*`, `common/hf-cache.*`, `scripts/model_picker.py`, and the
  `--moe-stream*` / `--models-preset` flags wired through `common/arg.cpp`.
  Bugs, features, and PRs against this code belong here.
- **Everything else** is still upstream llama.cpp, carried along so the
  full toolset (`llama-cli`, `llama-server`, the OpenAI-compatible API,
  quantization, etc.) keeps working. A bug that reproduces on unmodified
  upstream code isn't a bug in this fork - report it to
  [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) instead,
  where it'll actually get triaged. If you're not sure which side a bug is
  on, open an issue here anyway and it'll get pointed the right way.

## Building and testing changes

See [Quick start](README.md#quick-start) for the build command. There's no
formal test suite for the fork-specific code yet; the practical way to
validate a change is:

1. Build `llama-server` and run it in router mode.
2. Exercise the actual code path through the web UI (`/#/models` for picker
   changes, the chat UI for streaming/cache changes) - not just a
   compile-and-hope. `--moe-stream` behavior in particular is easy to break
   in ways that only show up under real I/O pressure; see the `q_demand`
   history in [HANDOFF.md](HANDOFF.md) for an example of a regression that
   looked correct until it was actually load-tested.
3. For anything touching the expert cache or eviction logic, check both a
   pressured cache (small `--moe-stream-cache`) and a roomy one - behavior
   that's fine under one and wrong under the other is exactly the kind of
   bug this subsystem tends to produce.

## AI-assisted contributions

This project is built with AI assistance and doesn't pretend otherwise. If
you use AI tools in a contribution, that's expected here, not a violation
of anything - just hold yourself to the same bar the maintainer does:
understand what you're submitting well enough to explain any line of it,
and don't paste in something you haven't actually read.

## Code style

- Comments explain *why*, not *what*. If a comment just restates the code
  in English, delete it. Write one when there's a non-obvious constraint, a
  workaround for a specific bug, or a reason a naive approach was rejected -
  see almost any comment in `src/llama-moe-stream.cpp` for the target
  density and tone.
- No em dashes or en dashes anywhere - commas, periods, colons, or "to" for
  ranges instead. This is enforced by hand, not a linter; PRs that add them
  will get asked to fix it.
- Match the surrounding file's formatting rather than introducing a new
  style mid-file.

## Reporting issues

Open a GitHub issue on this repo. Include what you ran (the exact command
or UI action), what you expected, and what actually happened. For anything
performance-related, include your hardware (GPU, VRAM, whether you're on
WSL2 - the Hyper-V-backed virtual disk there has real, measured I/O
differences from bare-metal Linux, see the `--moe-stream-io-threads` note
in the [README](README.md)).

## Security issues

Do not open a public issue for a security vulnerability. See
[SECURITY.md](SECURITY.md).
