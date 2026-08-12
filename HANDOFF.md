# Handoff — read this, then just say "continue"

Written because WSL2's memory ballooned to 14.6GB (mostly reclaimable page
cache Windows wasn't getting back fast enough — host was down to 0.7GB free
while native installers were running) and the fix was `wsl --shutdown` from
Windows, which kills whatever WSL2 session wrote this file. Everything below
is what a fresh session needs to pick up cleanly.

## The project

**MoE VRAM Pager** — run Mixture-of-Experts GGUF models bigger than your
VRAM (and ideally bigger than RAM+VRAM combined) by streaming routed
experts from SSD, with a real GPU-resident (VRAM) cache tier — not just a
RAM cache. Apache/MIT-compatible clean-break project, no ties to upstream
git history. Eventually: a custom browser UI (own design, not a copy of
Colibri's), a model picker that cross-references the UGI-Leaderboard,
and a "load any GGUF + audit whether your hardware can run it" flow.

## How we got to this specific foundation (don't re-litigate)

- Rejected forking **BigMoeOnEdge** (Helldez/BigMoeOnEdge, Apache-2.0):
  it's deliberately CPU-only — streamed experts rebind raw pointers into
  the CPU `mul_mat_id` kernel, no path to GPU memory at all, and its own
  docs say so explicitly (`docs/limitations.md`).
- Rejected **Colibri** (JustVugg/colibri — verify this is the real one,
  not the impostor `uv-genai/colibri` which has 0 stars vs 24k+ and was
  created two weeks later, or forks of it): it DOES do real GPU-VRAM
  expert caching already, but it's not a general GGUF runtime — each
  model architecture is hand-implemented, and Qwen3 MoE (the model we
  actually want to run) is only on its roadmap, not supported.
- Landed on **mainline llama.cpp + an open PR**: `freedomljc/llama.cpp`
  branch `feat/moe-streaming-core` (PR ggml-org/llama.cpp#25294, not yet
  merged as of 2026-08-12), which adds `--moe-stream` — on-demand
  disk-streaming of routed experts into a **host-RAM** LRU cache. Its
  cache is built on `ggml_backend_buffer` and already branches on
  `ggml_backend_buffer_is_host()`, i.e. it's structured with a non-host
  (GPU) buffer in mind even though only the RAM tier is wired up — the
  PR author explicitly deferred VRAM tiering to "a separate PR." That's
  the extension point: **our actual job is to build that separate PR
  ourselves**, i.e. make the expert cache allocate from a CUDA backend
  buffer instead of host memory.
- Base commit imported: `freedomljc/llama.cpp@1248fd8fa8cfebaece5ea992e4d951c1e18bb9d5`.

## Repo locations (two checkouts, one source of truth)

- **`/home/josh/moe-vram-pager`** (WSL2, ext4) — **source of truth**. Real
  git history starts at commit `14be909` (clean-break initial import).
  Git identity (local config, not global): `Josh Opolko
  <joshua.opolko@gmail.com>`. Develop and commit here.
- **`C:\Users\josh\moe-vram-pager`** (Windows native NTFS) — **build/run
  target only, intentionally has no `.git`**. Sync from WSL with
  `scripts/sync-to-windows.sh` (excludes `build/`, `.git`). Build and run
  natively here so the process isn't capped by WSL2's 16GB
  `.wslconfig` limit — WSL2 dynamic memory doesn't release page cache
  back to Windows fast enough under load, confirmed the hard way tonight.

Workflow going forward: **edit/commit in WSL, sync + build + run on
Windows**, with `wsl --shutdown` before any real (non-dev) run so the
full ~24GB host RAM is available to the native process.

## Hardware

- GPU: GTX 1080 Ti, 11GB VRAM, **Pascal, compute capability 6.1** →
  always build with `CMAKE_CUDA_ARCHITECTURES=61`.
- Host RAM: **24GB total** (not 32GB+ — don't over-assume headroom).
  `.wslconfig`: `memory=16GB, processors=8, swap=8GB`. That leaves only
  ~8GB for Windows itself, most of which Windows wants for its own use —
  don't assume raising the WSL cap buys much.
- Driver: 582.28, reports max CUDA 13.0 — installed CUDA Toolkit
  **13.0** specifically (not the newer 13.3 winget offers by default,
  which could exceed driver support).
- Disk (WSL2 virtual disk, `/dev/sdd` ext4): measured for real with
  `scripts/bench-odirect.py` (O_DIRECT random reads, not confounded by
  page cache) — **2.5-3GB/s at 16MiB blocks**, genuinely fast, WSL2's
  Hyper-V virtual disk is NOT a meaningful I/O bottleneck here.
  Counterintuitive finding: **more I/O threads made it slower**,
  especially at larger block sizes (8 threads/64MiB: 0.9GB/s vs
  1 thread/64MiB: 2.2GB/s) — opposite of typical bare-metal NVMe advice.
  Tune `--moe-stream-io-threads` low (1-2) on this box, verify against
  the real model rather than assuming higher is better.

## Windows toolchain — status when session ended

Installed via `winget`, **confirmed complete**:
- CMake (`Kitware.CMake`)
- Visual Studio 2022 BuildTools (`Microsoft.VisualStudio.2022.BuildTools`)
  with C++ workload (`Microsoft.VisualStudio.Workload.VCTools` +
  `Microsoft.VisualStudio.Component.VC.Tools.x86.x64`) — winget reported
  "Successfully installed".

**Still in progress when session ended, status unknown:**
- NVIDIA CUDA Toolkit **13.0** specifically, not the newer default
  (`winget install --id Nvidia.CUDA --version 13.0 -e --silent
  --accept-package-agreements --accept-source-agreements`) — was
  mid-download of `cuda_13.0.2_windows.exe`, no completion confirmation
  seen. **Check `nvcc --version` from a fresh terminal first**, re-run
  the winget command above if it's not there.

## WSL2 build — known-good, already committed

`build/bin/llama-cli` built successfully in WSL with CUDA. Exact working
config (see commit `c6e19a7` for the CMakeLists fix this needed):

```
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build --config Release -j 12
```

Why the CMakeLists edits were needed: `tools/ui`'s asset pipeline doesn't
produce a `loading.html` that `llama-ui-embed`'s validator requires, so
the UI build fails even with `LLAMA_BUILD_UI=OFF` (that target ran
unconditionally). `llama-cli` doesn't need the UI, only `server-context`
— so `tools/CMakeLists.txt` now always builds `server`+`cli`, gating only
`ui` behind `LLAMA_BUILD_SERVER`; `tools/server/CMakeLists.txt` splits
`server-context` (always built) from `llama-server`/`llama-server-impl`
(UI-dependent, still gated). Full detail in commit `c6e19a7`'s message.

**On native Windows (MSVC), this UI failure may not even occur** — it
might be specific to this fork branch's npm/prebuilt-fetch path in this
environment. Worth testing with `LLAMA_BUILD_SERVER=ON` first before
assuming the same CMakeLists patch is needed there; keep it either way
since it's harmless and already committed.

## Model

Target test model: **Qwen3-30B-A3B**, official quant repo
`Qwen/Qwen3-30B-A3B-GGUF`, file `Qwen3-30B-A3B-Q4_K_M.gguf` (~18.6GB).
Chosen because BigMoeOnEdge itself benchmarks it and it comfortably fits
the disk/RAM/VRAM budget here.

**Not downloaded — start fresh.** Two attempts in WSL both failed/were
killed (`huggingface-cli download ... --local-dir .` doesn't resume from
`--local-dir` mode, restarts from 0% every time; got to 71% before being
killed to relieve memory pressure). **Download directly onto Windows
native storage this time** (e.g. `C:\Users\josh\models\`), not WSL —
avoids the copy step and matches where it'll actually run.

## Other research findings worth remembering

- **UGI-Leaderboard integration is real and buildable**: the data isn't
  behind the rendered Dash app, it's a plain CSV directly in the HF Space
  repo — `https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard`,
  file `ugi-leaderboard-data.csv`. Columns that matter: `Active
  Parameters` vs `Total Parameters` (MoE detector — Active < Total means
  sparse), `Architecture` (cross-reference against what our engine
  supports), `UGI 🏆` (ranking score), `Model Link`. Most rows point at
  the original safetensors repo, not a GGUF — need a lookup step to find
  a community GGUF quant (bartowski/unsloth/mradermacher naming
  conventions) before download.
- **Hardware-fit audit should be a shared primitive**, not duplicated
  between the leaderboard-picker flow and a "paste any GGUF" flow: read
  just the GGUF header via an HTTP range request (architecture,
  `expert_count`/`expert_used_count`, file size, shard count) without
  downloading the whole file, compare against detected local
  VRAM/RAM/disk, report comfortable/workable/marginal/won't-fit before
  committing to a multi-hour download. Works the same for a 1080 Ti+15GB
  box or someone else's 5090+64GB box.
- **Browser UI direction**: inspired by Colibri's dashboard (`./coli
  web` — live tok/s, TTFT, VRAM/RAM/disk tier usage, plus flashy
  "expert cortex" and 3D "atlas" views) but our own design, sitting on
  top of an OpenAI-compatible API (llama-server's, once the UI build
  issue is sorted or bypassed) plus our own telemetry additions once the
  VRAM-cache work exists to have telemetry worth showing.
- **Distribution/Reddit strategy** (for later, not now): no hard OS-split
  data exists for the local-LLM crowd, but hardware mentions in
  community threads are all consumer NVIDIA gaming cards → Windows is
  likely the largest single segment, Linux the louder technical one.
  What actually drives Reddit spread is friction-to-first-run (a
  prebuilt binary beats "clone and compile"), so the plan is prebuilt
  releases for both Windows and Linux, not picking one — llama.cpp is
  genuinely cross-platform (unlike Colibri, which looks Linux/macOS-only
  given its Makefile-only build).

## Immediate next steps on resume

1. Confirm VS Build Tools + CUDA Toolkit 13.0 actually finished installing
   on Windows (check, don't assume — session ended mid-install).
2. `cd /home/josh/moe-vram-pager && ./scripts/sync-to-windows.sh`
3. Build natively on Windows (`C:\Users\josh\moe-vram-pager`) via
   cmake + MSVC + CUDA, same flags as the WSL build above (adjust for
   MSVC generator).
4. Download Qwen3-30B-A3B-Q4_K_M.gguf directly to Windows native storage.
5. First real test: `llama-cli.exe --moe-stream` against it (RAM-only
   cache, no VRAM tier yet) — verify it actually works before touching
   any GPU-cache code.
6. Then start the real engineering: extend the `moe-stream` cache in
   `src/llama-moe-stream.cpp`/`.h` to allocate from a CUDA
   `ggml_backend_buffer` instead of host memory.
