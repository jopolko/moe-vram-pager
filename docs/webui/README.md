# Web UI

Technical reference for the SvelteKit web UI's main sections, one page per
sidebar destination:

- [Chat](chat.md) - the default chat interface, model selector, and
  conversation search.
- [MCP Servers](mcp-servers.md) - managing MCP tool-server connections used
  by both chat and the pentest appliance.
- [Models](models.md) - the hardware-fit catalog, ad-hoc/Ollama GGUF
  loading, and the live "Loaded:" model switcher.
- [Pentest](pentest.md) - the autonomous offensive-security agent: scope,
  phases, port/context sizing, and what each control actually forces vs.
  merely suggests to the model.
- [Settings](settings.md) - the ten settings sections and what each
  controls.

The UI is a single SvelteKit SPA (hash-routed: `#/`, `#/models`, etc),
built with `npm run build` in `tools/ui/` and embedded directly into the
`llama-server` binary at compile time (see the `llama-ui-assets` CMake
target) - there is no separate static file server. A frontend-only change
still requires a C++ rebuild (`cmake --build . --target llama-server`) to
take effect, unless the server is started with `--path <dist-dir>` to serve
assets straight from disk for faster iteration during development.
