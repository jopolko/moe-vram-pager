#!/usr/bin/env bash
# Pull gemma-2-2b weights + the Gemma Scope SAEs used in Phase 1.
# HF_TOKEN is sourced from the secrets file, never printed.
set -euo pipefail

set -a; source /var/secrets/nowservingto.env; set +a
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
export HF_HUB_ENABLE_HF_TRANSFER=1

cd "$(dirname "$0")/.."
source .venv/bin/activate

# Keep the model cache inside this module (gitignored), not ~/.cache.
export HF_HOME="$PWD/hf_cache"
mkdir -p "$HF_HOME"

# NOTE: each --include takes ONE pattern. Trailing positional args are read as
# explicit filenames and silently disable --include/--exclude.
echo "=== gemma-2-2b (fp16 safetensors only) ==="
hf download google/gemma-2-2b \
  --include "*.safetensors" --include "*.json" --include "tokenizer.model"

echo "=== gemma-scope-2b-pt-res-canonical: layer 12, width 16k ==="
hf download google/gemma-scope-2b-pt-res \
  --include "layer_12/width_16k/canonical/*"

echo "=== done. cache: $HF_HOME ==="
du -sh "$HF_HOME"
