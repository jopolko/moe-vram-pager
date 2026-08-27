#!/usr/bin/env python3
"""Sweep --moe-stream-* flags against a real model and compare throughput/hit-rate.

Restarts llama-server once per config (moe-stream stats only flush to the log on
clean shutdown - see clean_up() in tools/server/server.cpp), drives a fixed
decode-heavy workload over HTTP, sends SIGINT, then parses the stats block that
llama_moe_stream::print_stats() (src/llama-moe-stream.cpp) writes to the server log.

No existing harness in this repo sweeps --moe-stream-* or parses that log text -
llama-bench has its own cmd_params struct with no moe_stream fields at all, so it
silently ignores these flags. Hence this script talks to llama-server directly.

Example:
    tools/bench_moe_stream.py \\
        --model /home/josh/models/Qwen3-235B-A22B-GGUF/Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00001-of-00003.gguf \\
        --server build/bin/llama-server \\
        --out bench-moe-stream-results.csv
"""

import argparse
import csv
import logging
import re
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from time import sleep, time
from typing import Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("bench_moe_stream")

HEALTH_TIMEOUT_S = 1800  # a 150GB+ model can take a long time to open/prepare on first load
SIGINT_TIMEOUT_S = 30

# Matched against llama_moe_stream::print_stats() (src/llama-moe-stream.cpp:554-576) log
# lines verbatim, e.g.:
#   "moe stream: remap calls = 123, expert hits = 100, misses = 23 (5 cold), hit rate = 81.30%"
#   "moe stream: load stall = 45.20 ms total (0.367 ms per remap call)"
#   "moe stream: decode prefetch: top-8, 12 speculative loads issued"
#   "moe stream: ram cache: 4.0 GiB budget, hits = 3, misses = 1, hit rate = 75.00%"
# re.search takes the first match in the log, and the expert-cache line always prints
# before the ram-cache line (if any), so the plain patterns below correctly bind to the
# expert-cache stats even without excluding the ram-cache line explicitly.
STATS_PATTERNS = {
    "n_calls":             r"remap calls\s*=\s*(\d+)",
    "n_hit":                r"expert hits\s*=\s*(\d+)",
    "n_miss":               r"misses\s*=\s*(\d+)\s*\(\d+\s*cold\)",
    "hit_rate_pct":         r"hit rate\s*=\s*([\d.]+)%",
    "t_stall_ms_per_call":  r"\(([\d.]+)\s*ms per remap call\)",
    "n_prefetch_issued":    r"(\d+)\s*speculative loads issued",
    "ram_hit_rate_pct":     r"ram cache:.*?hit rate\s*=\s*([\d.]+)%",
}


@dataclass
class Config:
    name: str
    prefetch: int = 0          # --moe-stream-prefetch
    cache: Optional[str] = None  # --moe-stream-cache value, e.g. "24s" or "48G"; None = auto
    io_threads: Optional[int] = None
    direct: bool = False


@dataclass
class Result:
    config: Config
    tok_s: float = 0.0
    n_tokens: int = 0
    wall_s: float = 0.0
    stats: dict = field(default_factory=dict)
    log_tail: str = ""


def default_grid() -> list[Config]:
    # Deliberately not a full cross product - a 150GB+ model restart per config is
    # expensive. Covers the comparisons Phase 1 of the plan actually needs answered:
    # does decode prefetch help at scale, under both a roomy and a pressured cache.
    grid = []
    for cache_name, cache_val in [("roomy", None), ("pressured", "24s")]:
        for prefetch in (0, 8):
            grid.append(Config(
                name=f"prefetch{prefetch}_{cache_name}",
                prefetch=prefetch,
                cache=cache_val,
            ))
    # io-threads spot check, roomy cache, prefetch off (isolate the one variable)
    for io_threads in (1, 2, 9):
        grid.append(Config(name=f"io{io_threads}_roomy", io_threads=io_threads))
    # direct-io A/B for the page-cache confound, roomy cache, prefetch off
    grid.append(Config(name="direct_roomy", direct=True))
    return grid


def build_server_cmd(
    server_bin: str, model: str, port: int, ctx: int, ngl: int, cfg: Config,
    force_cpu_cache: bool = False,
) -> list[str]:
    cmd = [
        server_bin, "-m", model, "--port", str(port), "--host", "127.0.0.1",
        "-c", str(ctx), "-ngl", str(ngl), "--moe-stream",
    ]
    if force_cpu_cache:
        # this model's per-layer expert slab size is large enough that even the hard
        # 3*n_expert_used slot floor (llama-model.cpp:1296-1304) exceeds an 11GB card at
        # full GPU offload - confirmed by an actual OOM here, not a guess. Routing the
        # cache into host RAM keeps the same eviction/prefetch code path (buft-agnostic)
        # while letting dense/attention weights still run on GPU.
        cmd += ["--moe-stream-cpu-cache"]
    if cfg.prefetch:
        cmd += ["--moe-stream-prefetch", str(cfg.prefetch)]
    if cfg.cache:
        cmd += ["--moe-stream-cache", cfg.cache]
    if cfg.io_threads:
        cmd += ["--moe-stream-io-threads", str(cfg.io_threads)]
    if cfg.direct:
        cmd += ["--moe-stream-direct"]
    return cmd


def wait_healthy(address: str, process: subprocess.Popen) -> None:
    t0 = time()
    n_failures = 0
    while time() - t0 < HEALTH_TIMEOUT_S:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(f"server exited during startup, code={exit_code}")
        try:
            r = requests.get(f"{address}/health", timeout=5)
            if r.status_code == 200:
                return
        except requests.ConnectionError:
            n_failures += 1
        sleep(2)
    raise RuntimeError(f"server not healthy after {HEALTH_TIMEOUT_S}s")


def run_workload(address: str, n_predict: int, n_requests: int) -> tuple[int, float]:
    # Sequential, single-stream: this measures the streaming pager's own decode
    # throughput, not server request concurrency - concurrent requests would
    # confound expert-cache contention with scheduler effects, which is a
    # different question than the one Phase 1 asks.
    total_tokens = 0
    t0 = time()
    for i in range(n_requests):
        payload = {
            "prompt": f"Write a detailed technical explanation of topic number {i}, covering its history, mechanism, and practical tradeoffs.",
            "ignore_eos": True,
            "cache_prompt": False,
            "n_predict": n_predict,
            "seed": 1000 + i,
        }
        r = requests.post(f"{address}/completion", json=payload, timeout=1800)
        r.raise_for_status()
        data = r.json()
        total_tokens += data.get("tokens_predicted", n_predict)
    wall_s = time() - t0
    return total_tokens, wall_s


def stop_and_collect_log(process: subprocess.Popen, log_path: str) -> str:
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=SIGINT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        logger.warning("server did not exit after SIGINT within %ds, killing", SIGINT_TIMEOUT_S)
        process.kill()
        process.wait()
    with open(log_path, "r", errors="replace") as f:
        return f.read()


def parse_stats(log: str) -> dict:
    out = {}
    for key, pattern in STATS_PATTERNS.items():
        m = re.search(pattern, log, re.IGNORECASE)
        if m:
            out[key] = m.group(1)
    return out


def run_one(server_bin: str, model: str, port: int, ctx: int, ngl: int,
            n_predict: int, n_requests: int, log_dir: str, cfg: Config,
            force_cpu_cache: bool = False) -> Result:
    address = f"http://127.0.0.1:{port}"
    log_path = f"{log_dir}/{cfg.name}.log"
    cmd = build_server_cmd(server_bin, model, port, ctx, ngl, cfg, force_cpu_cache)
    logger.info("=== %s ===\n%s", cfg.name, " ".join(cmd))

    with open(log_path, "w") as fout:
        process = subprocess.Popen(cmd, stdout=fout, stderr=subprocess.STDOUT)
        try:
            wait_healthy(address, process)
            n_tokens, wall_s = run_workload(address, n_predict, n_requests)
        finally:
            log = stop_and_collect_log(process, log_path)

    stats = parse_stats(log)
    tail = "\n".join(log.splitlines()[-40:])
    tok_s = n_tokens / wall_s if wall_s > 0 else 0.0
    logger.info("%s: %.2f tok/s, stats=%s", cfg.name, tok_s, stats)
    return Result(config=cfg, tok_s=tok_s, n_tokens=n_tokens, wall_s=wall_s, stats=stats, log_tail=tail)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to the (first split of the) GGUF file")
    ap.add_argument("--server", default="build/bin/llama-server")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--n-requests", type=int, default=4)
    ap.add_argument("--log-dir", default="bench-moe-stream-logs")
    ap.add_argument("--out", default="bench-moe-stream-results.csv")
    ap.add_argument("--only", help="comma-separated config names to run, default: full default grid")
    ap.add_argument(
        "--force-cpu-cache", action="store_true",
        help="pass --moe-stream-cpu-cache to every run (needed when the expert-cache "
             "floor for this model doesn't fit VRAM at the chosen -ngl)",
    )
    args = ap.parse_args()

    import os
    os.makedirs(args.log_dir, exist_ok=True)

    grid = default_grid()
    if args.only:
        wanted = set(args.only.split(","))
        grid = [c for c in grid if c.name in wanted]
        missing = wanted - {c.name for c in grid}
        if missing:
            logger.error("unknown config name(s): %s", ", ".join(missing))
            sys.exit(1)

    results: list[Result] = []
    for cfg in grid:
        try:
            results.append(run_one(
                args.server, args.model, args.port, args.ctx, args.ngl,
                args.n_predict, args.n_requests, args.log_dir, cfg, args.force_cpu_cache))
        except Exception:
            logger.exception("config %s failed, continuing with the rest", cfg.name)

    fieldnames = ["name", "prefetch", "cache", "io_threads", "direct", "tok_s", "n_tokens", "wall_s"] + list(STATS_PATTERNS.keys())
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = {
                "name": r.config.name, "prefetch": r.config.prefetch, "cache": r.config.cache,
                "io_threads": r.config.io_threads, "direct": r.config.direct,
                "tok_s": f"{r.tok_s:.3f}", "n_tokens": r.n_tokens, "wall_s": f"{r.wall_s:.2f}",
            }
            row.update(r.stats)
            w.writerow(row)

    logger.info("wrote %s (%d rows)", args.out, len(results))


if __name__ == "__main__":
    main()
