# Handoff

Running technical log for this fork. Read this before trusting any claim
about what's built vs. not - it's been wrong before (see "corrections"
below) and the fix each time was reading the actual code/logs, not this
file's prior narrative.

## The project

**MoE VRAM Pager** - run Mixture-of-Experts GGUF models bigger than your
VRAM (and ideally bigger than RAM+VRAM combined) by streaming routed
experts from SSD, with a real GPU-resident (VRAM) cache tier, not just a
RAM cache. Clean-break project, no ties to upstream git history.

## How we got to this foundation

- Rejected forking **BigMoeOnEdge** (Helldez/BigMoeOnEdge, Apache-2.0):
  it's deliberately CPU-only, no path to GPU memory at all.
- Rejected **Colibri** (JustVugg/colibri): does real GPU-VRAM expert
  caching, but each model architecture is hand-implemented and Qwen3 MoE
  (our target) was only on its roadmap, not supported.
- Landed on **mainline llama.cpp + an open PR**: `freedomljc/llama.cpp`
  branch `feat/moe-streaming-core` (PR ggml-org/llama.cpp#25294, not yet
  merged as of 2026-08-12), which added `--moe-stream`. Base commit
  imported: `freedomljc/llama.cpp@1248fd8fa8cfebaece5ea992e4d951c1e18bb9d5`.

## Corrections to earlier versions of this file

An earlier version of this file claimed the VRAM-resident expert cache
"doesn't exist yet" and framed building it as the project's main open
task. **That was wrong.** Verified directly via `-lv 4` server logs
(`alloc_bufs: CUDA0 expert cache size = ...`): the cache already allocates
from whichever device the layer is offloaded to - CUDA VRAM by default
under `-ngl`, host RAM otherwise. `llama_moe_stream_select_buft()` in
`src/llama-model.cpp` is where that device selection happens. Don't
re-litigate this; if a future session doubts it, run with `-lv 4` and
grep for `expert cache size` instead of trusting old notes (including
this one).

## Current architecture (accurate as of 2026-08-12)

- `src/llama-moe-stream.{cpp,h}`: the streaming cache. Per-layer slot
  table, LFU-with-LRU-tiebreak eviction (`pick_victim_locked`), a
  `q_demand` work queue serviced by `n_io_threads` workers. Two prefetch
  mechanisms share that queue: prefill "wave" preloading (pre-existing)
  and decode-phase speculative prefetch (`--moe-stream-prefetch`, added
  this session, see below).
- Cache device: chosen per-layer by `llama_moe_stream_select_buft()` in
  `src/llama-model.cpp`, following `-ngl`'s own per-layer device
  assignment unless `--moe-stream-cpu-cache` forces host RAM.
- Minimum cache size: multi-pass expert GEMMs require at least
  `3*n_expert_used` resident slots (`llm_graph_context::build_moe_ffn` in
  `src/llama-graph.cpp` hard-aborts below that). `llama_moe_stream_resolve_slots()`
  in `src/llama-model.cpp` now clamps up to that floor automatically
  (explicit, budget-derived, and auto-default paths all go through the
  same clamp) - it didn't always; a too-small explicit or budget-derived
  value used to reach the hard abort directly. If you see that abort
  message again, this clamp regressed.
- **Known real bug, not yet fixed**: `q_demand` is a single FIFO queue.
  Genuine blocking demand-loads (the GPU is stalled waiting) and
  best-effort speculative loads (wave preload, decode prefetch) all
  `push_back` onto it with no priority distinction. Under real I/O
  pressure (small cache, `--moe-stream-direct` to remove page-cache
  confounding) this measurably hurts: decode-phase prefetch dropped
  throughput from 3.68 to 1.46 tok/s in a controlled A/B (`--moe-stream-cache 24s`,
  a Qwen3-30B-A3B model, `top-4` prefetch) because a real stall can land
  behind several already-queued speculative guesses. The fix (proposed,
  not yet applied): make the four genuinely-blocking enqueue call sites
  `push_front` instead of `push_back`, leaving the two speculative sites
  (`push_back`) as lower priority. Same single queue, no new subsystem.
- Router mode (`tools/server/server-models.{cpp,h}`, upstream llama.cpp
  feature, not something we built): manages multiple model child
  processes, on-demand download with real SSE progress
  (`GET /models/sse`), load/unload, delete. `--models-preset <path>` is
  an INI file of per-model CLI-flag overrides; `common_preset_write_ini_section()`
  (`common/preset.cpp`, added this session) is how the picker persists a
  computed `--moe-stream-cache` size per model before the router loads
  it, since the engine's own `-fit` auto-sizer has no idea
  `--moe-stream-cache` exists. `--models-preset` tolerates a missing file
  at startup (treated as empty, not fatal) so router mode can start with
  zero pre-setup.
- Model picker (`tools/server/server-model-picker.{cpp,h}`,
  `tools/ui/src/routes/models/`): cross-references the UGI Leaderboard CSV
  against detected hardware. In router mode its Actions column drives the
  whole download -> load -> chat flow via the router's own API plus the
  one new `POST /model-picker/prepare-download` endpoint.

## Hardware (this dev box)

- GPU: GTX 1080 Ti, 11GB VRAM, Pascal, compute capability 6.1 - build with
  `CMAKE_CUDA_ARCHITECTURES=61`.
- Host RAM: ~16-17GB as seen by `ggml_backend_dev_memory()` under WSL2.
- Disk: WSL2 virtual disk. Free-space reporting needed a WSL-specific fix
  (`disk_free_gb()` in `server-model-picker.cpp`) - the virtual ext4
  filesystem's self-reported free space can be far larger than the real
  host `/mnt/c` free space it still has to grow into; capped by whichever
  is smaller when running under WSL2.
- I/O threads: more threads measured *slower* here on this virtual disk
  (opposite of typical bare-metal NVMe advice) - tune
  `--moe-stream-io-threads` low (1-2), verify against your own disk rather
  than assuming higher is better.
- Page cache matters a lot for benchmarking: two runs back-to-back on the
  same expert data differ hugely in speed purely from Linux page-cache
  warmth (8.32 vs 5.27 tok/s decode, same config, only run order
  different). `--moe-stream-direct` (`O_DIRECT`) removes that confound
  for controlled A/B comparisons, at the cost of being much slower in
  absolute terms than the page-cache-assisted path most real usage
  benefits from.

## Model used for testing

`Qwen/Qwen3-30B-A3B-GGUF`, `Qwen3-30B-A3B-Q4_K_M.gguf` (~18.6GB, 128
experts, top-8 routing). Small enough to iterate fast; not representative
of the actual target use case (hundreds-of-GB models where the disk-tier
and cache-pressure code paths actually matter). The `q_demand` priority
bug above was only reproducible by artificially shrinking the cache
(`--moe-stream-cache 24s`, the minimum this model allows) to simulate the
pressure a genuinely huge model would create naturally - worth retesting
against a real large model once one's been downloaded through the new
picker flow.

## Open threads / next steps

1. Fix the `q_demand` priority inversion (see above) - the actual next
   engineering task, and the reason decode-phase prefetch isn't proven to
   help yet despite being correctness-verified and demonstrably firing.
2. Retest decode-phase prefetch against a real disk-streaming-tier model
   (hundreds of GB) once one exists locally, now that download-from-the-UI
   removes the manual-download friction that blocked this earlier.
3. `scripts/model_picker.py` (the Python CLI mirror) has not been updated
   with the C++ side's newest fit-tier/quant logic changes in a while -
   diff it against `server-model-picker.cpp` before trusting it's still
   in sync.
