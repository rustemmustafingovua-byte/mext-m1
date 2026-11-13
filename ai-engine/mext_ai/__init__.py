"""MEXT-M1 AI engine.

Two predictive layers on top of the Rust core:

* :mod:`mext_ai.neural` — MLX model runner that keeps "hot" weights in unified
  memory and streams "cold" weights from SSD ahead of the current layer.
* :mod:`mext_ai.rag` — disk-backed LanceDB index with a streaming query
  prefetcher that warms IVF clusters while the user is still typing.
"""

__version__ = "0.6.1"

__all__ = ["__version__"]
