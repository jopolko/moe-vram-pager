#!/usr/bin/env python3
r"""Live per-token interpretability sidecar for the webui's #/interp Live view.

Companion to `interp_ui_api.py` (which serves the batch `run exp*` result
envelopes). This one keeps a loaded fp16 model in memory and streams a token at
a time over SSE, so it must run inside the interpretability venv:

    interpretability/.venv/bin/python tools/interp_live_api.py --port 8088

Equivalently, with the venv + HF token handled for you:

    cd interpretability && ./interp serve            # or  .\interp.ps1 serve

The implementation lives in the package (`obench_interp.live_server`) because it
imports nnsight; this file is just the tools/ entry point, matching the layout
of the other sidecars.
"""
from obench_interp.live_server import main

if __name__ == "__main__":
    raise SystemExit(main())
