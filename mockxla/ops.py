"""The op registry.

THE central design decision of mock-xla: an op is DATA, not code. Each entry is

    name -> (shape_rule, impl, cost_rule)

so growing the op set toward "enough to write DeepSeek V3" is filling in a
table, not writing compiler code. `cost_rule` is unused in stage 1 and is what
stage 3's fusion cost model will read.

FIDELITY NOTE: like real HLO, binary ops here require IDENTICAL shapes. There is
no implicit broadcasting. The frontend (numpy.py) inserts explicit `broadcast`
instructions, which is exactly what jax.lax does on the way to HLO -- and it is
why HLO dumps are full of broadcasts you never wrote.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

from .dtypes import PRED, Shape

# ---------------------------------------------------------------- registry


# What does shape_rule do here?
# What is Callable?
@dataclasses.dataclass
class OpDef:
    name: str
    shape_rule: Callable[..., Shape]
    impl: Callable[..., np.ndarray]
    cost_rule: Callable | None = None   # stage 3

# Map of op names to it's definition above
REGISTRY: dict[str, OpDef] = {}


def register(name, shape_rule, impl, cost_rule=None):
    REGISTRY[name] = OpDef(name, shape_rule, impl, cost_rule)


def lookup(name) -> OpDef:
    if name not in REGISTRY:
        raise KeyError(f"no such op: {name!r}. registered: {sorted(REGISTRY)}")
    return REGISTRY[name]


# ---------------------------------------------------------------- shape rules


def _same(*shapes, **_):
    """Elementwise: all operands must agree exactly, result matches."""
    first = shapes[0]
    for s in shapes[1:]:
        if s.dims != first.dims:
            raise TypeError(
                f"elementwise shape mismatch: {first} vs {s}. HLO has no implicit "
                f"broadcasting -- the frontend must insert a broadcast.")
        if s.dtype is not first.dtype:
            raise TypeError(
                f"elementwise dtype mismatch: {first} vs {s}. HLO has no implicit "
                f"promotion either -- the frontend must insert a convert.")
    return first


def _compare_shape(lhs, rhs, **_):
    _same(lhs, rhs)
    return lhs.with_dtype(PRED)


def _select_shape(pred, on_true, on_false, **_):
    if pred.dims != on_true.dims:
        raise TypeError(f"select: pred {pred} vs operands {on_true}")
    _same(on_true, on_false)
    return on_true


def _broadcast_shape(operand, *, shape, dimensions, **_):
    """`dimensions[i]` is the output dim that operand dim i maps to."""
    if len(dimensions) != operand.rank:
        raise TypeError(f"broadcast: {len(dimensions)} dimensions for rank {operand.rank}")
    for i, d in enumerate(dimensions):
        if shape.dims[d] != operand.dims[i]:
            raise TypeError(f"broadcast: operand dim {i}={operand.dims[i]} -> "
                            f"output dim {d}={shape.dims[d]}")
    return shape


def _reshape_shape(operand, *, shape, **_):
    if operand.num_elements != shape.num_elements:
        raise TypeError(f"reshape: {operand} has {operand.num_elements} elements, "
                        f"{shape} has {shape.num_elements}")
    return shape


def _transpose_shape(operand, *, permutation, **_):
    if sorted(permutation) != list(range(operand.rank)):
        raise TypeError(f"transpose: {permutation} is not a permutation of rank {operand.rank}")
    return Shape(tuple(operand.dims[p] for p in permutation), operand.dtype)


def _reduce_shape(operand, init, *, dimensions, computation, **_):
    if init.rank != 0:
        raise TypeError(f"reduce: init must be scalar, got {init}")
    dims = tuple(d for i, d in enumerate(operand.dims) if i not in set(dimensions))
    return Shape(dims, operand.dtype)


def _dot_shape(lhs, rhs, *, lhs_contracting, rhs_contracting,
               lhs_batch=(), rhs_batch=(), **_):
    """XLA's output dim order: batch dims, then lhs free dims, then rhs free dims."""
    for lc, rc in zip(lhs_contracting, rhs_contracting):
        if lhs.dims[lc] != rhs.dims[rc]:
            raise TypeError(f"dot: contracting dims disagree, lhs dim {lc}="
                            f"{lhs.dims[lc]} vs rhs dim {rc}={rhs.dims[rc]}")
    for lb, rb in zip(lhs_batch, rhs_batch):
        if lhs.dims[lb] != rhs.dims[rb]:
            raise TypeError(f"dot: batch dims disagree, {lhs.dims[lb]} vs {rhs.dims[rb]}")
    lhs_skip = set(lhs_contracting) | set(lhs_batch)
    rhs_skip = set(rhs_contracting) | set(rhs_batch)
    dims = ([lhs.dims[b] for b in lhs_batch]
            + [d for i, d in enumerate(lhs.dims) if i not in lhs_skip]
            + [d for i, d in enumerate(rhs.dims) if i not in rhs_skip])
    return Shape(tuple(dims), lhs.dtype)


def _nullary_shape(*, shape, **_):
    return shape


def _constant_shape(*, value, shape, **_):
    return shape


def _convert_shape(operand, *, dtype, **_):
    return operand.with_dtype(dtype)


# ---------------------------------------------------------------- impls

REDUCERS = {
    "add": (np.add, 0.0),
    "max": (np.maximum, -np.inf),
    "min": (np.minimum, np.inf),
    "mul": (np.multiply, 1.0),
}


def _reduce_impl(operand, init, *, dimensions, computation, **_):
    fn, _ = REDUCERS[computation]
    return fn.reduce(operand, axis=tuple(dimensions), initial=init)


_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _dot_impl(lhs, rhs, *, lhs_contracting, rhs_contracting,
              lhs_batch=(), rhs_batch=(), **_):
    """Lower dot_general to an einsum by assigning a letter per dimension.

    Batch and contracting pairs SHARE a letter (that is what makes them batch or
    contracting); every free dim gets its own.
    """
    lsub = [None] * lhs.ndim
    rsub = [None] * rhs.ndim
    nxt = 0
    for lb, rb in zip(lhs_batch, rhs_batch):
        lsub[lb] = rsub[rb] = _LETTERS[nxt]; nxt += 1
    for lc, rc in zip(lhs_contracting, rhs_contracting):
        lsub[lc] = rsub[rc] = _LETTERS[nxt]; nxt += 1
    for i in range(lhs.ndim):
        if lsub[i] is None:
            lsub[i] = _LETTERS[nxt]; nxt += 1
    for i in range(rhs.ndim):
        if rsub[i] is None:
            rsub[i] = _LETTERS[nxt]; nxt += 1

    lhs_skip = set(lhs_contracting) | set(lhs_batch)
    rhs_skip = set(rhs_contracting) | set(rhs_batch)
    out = ([lsub[b] for b in lhs_batch]
           + [lsub[i] for i in range(lhs.ndim) if i not in lhs_skip]
           + [rsub[i] for i in range(rhs.ndim) if i not in rhs_skip])
    spec = f"{''.join(lsub)},{''.join(rsub)}->{''.join(out)}"
    return np.einsum(spec, lhs, rhs)


def _broadcast_impl(operand, *, shape, dimensions, **_):
    # Place the operand's dims at their target positions, then broadcast the rest.
    if operand.ndim == 0:
        return np.broadcast_to(operand, shape.dims).copy()
    expanded = [1] * len(shape.dims)
    for i, d in enumerate(dimensions):
        expanded[d] = operand.shape[i]
    return np.broadcast_to(operand.reshape(expanded), shape.dims).copy()


def _iota_impl(*, shape, dimension, **_):
    out = np.arange(shape.dims[dimension], dtype=shape.dtype.np)
    view = [1] * len(shape.dims)
    view[dimension] = shape.dims[dimension]
    return np.broadcast_to(out.reshape(view), shape.dims).copy()


_DIRECTIONS = {"lt": np.less, "le": np.less_equal, "gt": np.greater,
               "ge": np.greater_equal, "eq": np.equal, "ne": np.not_equal}


def _compare_impl(lhs, rhs, *, direction, **_):
    return _DIRECTIONS[direction](lhs, rhs)


# ---------------------------------------------------------------- the table

_UNARY = {
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "tanh": np.tanh,
    "negate": np.negative, "abs": np.abs, "floor": np.floor, "sign": np.sign,
    "rsqrt": lambda x: 1.0 / np.sqrt(x),
    "logistic": lambda x: 1.0 / (1.0 + np.exp(-x)),
}
_BINARY = {
    "add": np.add, "subtract": np.subtract, "multiply": np.multiply,
    "divide": np.divide, "maximum": np.maximum, "minimum": np.minimum,
    "power": np.power,
}

for _name, _fn in _UNARY.items():
    register(_name, _same, (lambda fn: lambda x, **kw: fn(x))(_fn))
for _name, _fn in _BINARY.items():
    register(_name, _same, (lambda fn: lambda a, b, **kw: fn(a, b))(_fn))

register("parameter", _nullary_shape, None)  # bound by the interpreter, not computed
register("constant", _constant_shape, lambda *, value, shape, **kw: np.asarray(value, dtype=shape.dtype.np))
register("iota", _nullary_shape, _iota_impl)
register("broadcast", _broadcast_shape, _broadcast_impl)
register("reshape", _reshape_shape, lambda x, *, shape, **kw: x.reshape(shape.dims))
register("transpose", _transpose_shape, lambda x, *, permutation, **kw: np.transpose(x, permutation))
register("reduce", _reduce_shape, _reduce_impl)
register("dot_general", _dot_shape, _dot_impl)
register("compare", _compare_shape, _compare_impl)
register("select", _select_shape, lambda p, t, f, **kw: np.where(p, t, f))
register("convert", _convert_shape, lambda x, *, dtype, **kw: x.astype(dtype.np))


# ---------------------------------------------------------------- collectives
#
# Stage 3. Their `impl` is None because a collective cannot be evaluated from
# one device's arrays alone -- the SPMD interpreter special-cases them, the way
# the interpreter already special-cases `parameter`.
#
# `replica_groups` is the only place the mesh axis survives into the IR. Axis
# NAMES do not exist in HLO; the partitioner flattens them to device-id groups,
# so group size and stride are all you have to recover which axis a collective
# is on. (We also keep an `axis` attr purely so dumps stay readable -- real HLO
# has no such luxury.)


def _all_gather_shape(operand, *, dimensions, group_size, **_):
    d = dimensions[0]
    dims = list(operand.dims)
    dims[d] *= group_size
    return Shape(tuple(dims), operand.dtype)


def _reduce_scatter_shape(operand, *, dimensions, group_size, **_):
    d = dimensions[0]
    dims = list(operand.dims)
    if dims[d] % group_size:
        raise TypeError(f"reduce-scatter: dim {d}={dims[d]} not divisible by {group_size}")
    dims[d] //= group_size
    return Shape(tuple(dims), operand.dtype)


def _dynamic_slice_shape(operand, *, dimensions, group_size, **_):
    return _reduce_scatter_shape(operand, dimensions=dimensions, group_size=group_size)


register("all-reduce", _same, None)
register("all-gather", _all_gather_shape, None)
register("reduce-scatter", _reduce_scatter_shape, None)
# A simplification: real dynamic-slice takes its offsets as OPERANDS (that is
# what makes it dynamic). Here the offset is implied by the device's coordinate
# on `axis`, which is the only form the partitioner ever emits. Keeping the name
# matters because ReduceScatterCreator pattern-matches on it.
register("dynamic-slice", _dynamic_slice_shape, None)
