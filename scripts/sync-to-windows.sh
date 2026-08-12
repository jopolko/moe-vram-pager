#!/usr/bin/env bash
# Sync source (not build/) from the WSL2 dev checkout to the native Windows checkout.
# Dev happens here in WSL; this pushes changes to C:\Users\josh\moe-vram-pager for a
# native MSVC+CUDA build and run, with WSL2 out of the loop at runtime.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="/mnt/c/Users/josh/moe-vram-pager"

rsync -a --delete \
    --exclude='build' \
    --exclude='build2' \
    --exclude='.git' \
    "$SRC"/ "$DST"/

echo "synced $SRC -> $DST"
