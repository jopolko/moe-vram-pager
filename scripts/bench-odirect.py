#!/usr/bin/env python3
"""Quick O_DIRECT random-read benchmark, mimicking expert-slice read patterns."""
import os
import sys
import time
import random
import mmap
import concurrent.futures

PATH = sys.argv[1] if len(sys.argv) > 1 else "testfile"
FILE_SIZE = os.path.getsize(PATH)
ALIGN = 4096


def aligned_buffer(size):
    buf = mmap.mmap(-1, size + ALIGN)
    addr = mmap.mmap.tell if False else None
    return buf


def random_reads(block_size, n_reads, n_threads):
    max_off = (FILE_SIZE - block_size) // ALIGN * ALIGN
    offsets = [random.randint(0, max_off // ALIGN) * ALIGN for _ in range(n_reads)]

    def worker(offs):
        fd = os.open(PATH, os.O_RDONLY | os.O_DIRECT)
        buf = bytearray(block_size + ALIGN)
        mv = memoryview(buf)
        total = 0
        try:
            for off in offs:
                n = os.preadv(fd, [mv[:block_size]], off)
                total += n
        finally:
            os.close(fd)
        return total

    chunks = [offsets[i::n_threads] for i in range(n_threads)]
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as ex:
        results = list(ex.map(worker, chunks))
    dt = time.perf_counter() - t0
    total_bytes = sum(results)
    mb = total_bytes / 1e6
    print(f"  block={block_size//1024//1024}MiB  threads={n_threads:2d}  "
          f"reads={n_reads:4d}  total={mb:8.0f}MB  time={dt:6.2f}s  "
          f"throughput={mb/dt:8.1f}MB/s  iops={n_reads/dt:6.1f}/s  "
          f"avg_latency={dt/n_reads*1000:6.1f}ms")


print(f"file: {PATH} ({FILE_SIZE/1e9:.2f} GB)")
for block_mb in (16, 64):
    block_size = block_mb * 1024 * 1024
    n_reads = max(20, int(2e9 / block_size))
    for threads in (1, 2, 4, 8):
        random_reads(block_size, n_reads, threads)
