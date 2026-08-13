# Security Policy

## Reporting a vulnerability

Open a [private security advisory](../../security/advisories/new) on this
repo (`jopolko/moe-vram-pager`), not on upstream llama.cpp's. This is a
personal project maintained by one person, so skip the multi-week embargo
process; a clear description and, if you have one, a minimal reproduction
is enough. If GitHub advisories aren't working for you, open a regular
issue and mark clearly that it's security-sensitive without including
exploit details in the public body.

## Scope

Two categories, handled differently:

### 1. Fork-specific code (report here)

The code this fork actually adds on top of llama.cpp:

- `src/llama-moe-stream.{cpp,h}` - the expert-streaming cache: paging,
  eviction, and the worker-thread I/O queue.
- `tools/server/server-model-picker.{cpp,h}` - fetches remote data (the UGI
  leaderboard CSV, a derestricted-model term list, an architecture map, live
  HF tags, HF search results) over HTTP and writes it to a local disk cache;
  also drives real model downloads in router mode.
- `common/download.cpp`, `common/hf-cache.{cpp,h}` - the actual file
  download and Hugging Face cache management.
- `common/preset.{cpp,h}` - INI read/write for `--models-preset`, including
  the picker's automatic per-model `--moe-stream-cache` writes.

Things worth specifically checking in a report against this surface: path
handling around the HF cache directory and `--models-preset` file, anything
that could turn a crafted HF API response into unexpected local disk
writes, and resource exhaustion (disk space, thread/connection counts) in
the download and tag-fetch paths.

### 2. Everything else (report upstream)

The rest of the tree is still llama.cpp. Model-loading bugs (GGUF parsing,
tensor-op backends, the inference engine itself) are almost certainly
present in unmodified upstream too and should go to
[ggml-org/llama.cpp's security policy](https://github.com/ggml-org/llama.cpp/security/policy)
instead, where the people who actually own that code can fix it. If you're
not sure which side of the line a bug falls on, report it here anyway.

## What this is not

**Model output content is not a software vulnerability.** This project
deliberately makes it easy to find and run derestricted ("uncensored")
models, on the position that refusal behavior is a content-policy choice
made by a model's trainers, not a security property of the inference
software running it - see
[Why derestricted models](README.md#why-derestricted-models). A model
answering a question you didn't want it to isn't a bug in this repo. If
you're looking to report objectionable model output, this isn't the place;
if you're looking to report an actual software vulnerability (memory
safety, injection, unintended file access, etc.), that's exactly what this
policy covers.

## Using this fork securely

The general guidance is the same as upstream llama.cpp, worth restating
here because router mode adds a few new specifics:

- **Untrusted GGUF files**: only load models from sources you trust. GGUF
  parsing bugs are a real, historical vulnerability class in llama.cpp; a
  malicious file is a malicious file regardless of which UI downloaded it.
- **Router mode spawns child processes**: each loaded model runs as a
  separate `llama-server` subprocess. Don't expose the router's HTTP API
  (`--host 0.0.0.0` or beyond) to an untrusted network - it can download
  and load arbitrary models on request.
- **`--models-preset` is a local trust boundary**: it's a plain INI file
  the picker writes to automatically. Don't point it at a path writable by
  another user or process you don't control.
- **The HF cache is shared, not sandboxed**: downloaded models land in the
  standard Hugging Face hub cache (`~/.cache/huggingface/hub` by default,
  see the info icon next to Free space in the picker UI), the same
  directory any other HF-based tool on the machine reads from. Treat it
  with the same trust level you'd give any other tool that shares that
  cache.
