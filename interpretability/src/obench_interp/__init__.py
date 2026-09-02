"""openbench-toolkit interpretability pipeline.

Decoupled from the llama.cpp / CUDA / GGUF inference path. Runs models via
HuggingFace transformers in fp16 for mechanistic-interpretability analysis
(TransformerLens / SAELens / nnsight).
"""

__version__ = "0.0.1"
