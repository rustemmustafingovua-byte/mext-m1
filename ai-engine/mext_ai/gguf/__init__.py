"""GGUF toolkit: header/metadata/tensor-table parsing + predictive page warming.

The reader parses GGUF v1/v2/v3 headers, the metadata key/value table and the
tensor info table *without* loading any weights — everything comes from a
read-only ``mmap`` of the file. The warmer drives ``MADV_WILLNEED`` / page
touches so that the OS pages in the weights for the transformer layers a model
runner is about to consume.
"""

from .reader import (
    GgufError,
    GgufFile,
    GgufFormatError,
    GgufTruncatedError,
    TensorInfo,
)
from .warmer import LayerPageWarmer

__all__ = [
    "GgufError",
    "GgufFile",
    "GgufFormatError",
    "GgufTruncatedError",
    "TensorInfo",
    "LayerPageWarmer",
]
