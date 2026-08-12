#!/usr/bin/env python3
"""
Regenerate arch-map.json from conversion/__init__.py's TEXT_MODEL_MAP.

arch-map.json is fetched at runtime by the (planned) native model-picker
server endpoint, translating a leaderboard model's HF architecture class
name (e.g. "Qwen3MoeForCausalLM") to this fork's GGUF architecture name
(e.g. "qwen3moe"). TEXT_MODEL_MAP is the only place that mapping exists,
and it's Python, so it can't be read directly by the compiled binary,
this script is the sync step between the two.

Run this (and commit the result) whenever conversion/__init__.py changes.
The pre-commit hook does this automatically and aborts the commit for
review if arch-map.json wasn't already up to date.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
CONVERSION_INIT = REPO_ROOT / "conversion" / "__init__.py"
OUTPUT_PATH = REPO_ROOT / "arch-map.json"


def load_arch_map() -> dict[str, str]:
    src = CONVERSION_INIT.read_text()
    m = re.search(r"TEXT_MODEL_MAP:\s*dict\[str,\s*str\]\s*=\s*(\{.*?\n\})", src, re.S)
    if not m:
        raise RuntimeError(f"could not find TEXT_MODEL_MAP in {CONVERSION_INIT}")
    return ast.literal_eval(m.group(1))


def main() -> int:
    arch_map = load_arch_map()
    new_content = json.dumps(arch_map, indent=2, sort_keys=True) + "\n"

    old_content = OUTPUT_PATH.read_text() if OUTPUT_PATH.exists() else None
    if old_content == new_content:
        return 0

    OUTPUT_PATH.write_text(new_content)
    print(f"regenerated {OUTPUT_PATH} ({len(arch_map)} architectures)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
