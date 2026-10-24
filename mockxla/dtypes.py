"""Dtypes and shapes.

A `Shape` in XLA is dims + element type, and it is the ONLY thing the compiler
knows about a value at compile time. There is no data. That is why XLA is a
static-shape compiler and why `jit` retraces when your input shape changes.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np


@dataclasses.dataclass(frozen=True)
class DType:
    name: str          # how it prints in HLO text: f32, bf16, s32, pred
    bits: int          # what the COST MODEL cares about (stage 3)
    np_name: str       # what numpy should actually use to execute

    @property
    def np(self):
        return np.dtype(self.np_name)

    def __str__(self):
        return self.name


# bf16 executes as f32 because numpy has no bf16. `bits` still says 16, so the
# stage-3 cost model prices the traffic correctly even though execution upcasts.
F32 = DType("f32", 32, "float32")
F16 = DType("f16", 16, "float16")
BF16 = DType("bf16", 16, "float32")
S32 = DType("s32", 32, "int32")
PRED = DType("pred", 8, "bool")

_BY_NP = {np.dtype("float32"): F32, np.dtype("float16"): F16,
          np.dtype("int32"): S32, np.dtype("int64"): S32, np.dtype("bool"): PRED,
          np.dtype("float64"): F32}


def dtype_of(x) -> DType:
    """DType of a numpy array or python scalar."""
    if isinstance(x, bool):
        return PRED
    if isinstance(x, int):
        return S32
    if isinstance(x, float):
        return F32
    return _BY_NP[np.asarray(x).dtype]


@dataclasses.dataclass(frozen=True)
class Shape:
    dims: tuple[int, ...]
    dtype: DType

    def __post_init__(self):
        object.__setattr__(self, "dims", tuple(int(d) for d in self.dims))

    @property
    def rank(self) -> int:
        return len(self.dims)

    @property
    def num_elements(self) -> int:
        return math.prod(self.dims) if self.dims else 1

    @property
    def size_bytes(self) -> int:
        """Stage 3's cost model is built on this number."""
        return self.num_elements * self.dtype.bits // 8

    def with_dtype(self, dtype: DType) -> Shape:
        return Shape(self.dims, dtype)

    def __str__(self):
        return f"{self.dtype}[{','.join(str(d) for d in self.dims)}]"


def shape_of(x) -> Shape:
    """Shape of a concrete numpy array or python scalar."""
    arr = np.asarray(x)
    return Shape(arr.shape, dtype_of(x))
