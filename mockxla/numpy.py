"""The jax.numpy-shaped frontend.

This layer exists for exactly one reason: numpy has implicit broadcasting and
HLO does not. Every function here adapts loose numpy semantics onto the strict
IR by inserting explicit `constant` / `broadcast` / `reshape` instructions.

That adaptation is why an HLO dump contains broadcasts and reshapes you never
wrote, and it is the same job jax.lax does under jax.numpy.
"""

from __future__ import annotations

import numpy as np

from .dtypes import F32, PRED, S32, Shape, dtype_of
from .trace import Tracer, current_builder

# ---------------------------------------------------------------- plumbing

# What purpose does this function serve?
def _as_tracer(x, like: Tracer | None = None) -> Tracer:
    """Lift a python/numpy scalar into a `constant` (+ `broadcast` if needed)."""
    if isinstance(x, Tracer):
        return x
    b = current_builder()
    arr = np.asarray(x)
    dtype = like.dtype if (like is not None and arr.ndim == 0) else dtype_of(x)
    scalar = b.emit("constant", [], value=arr.tolist(), shape=Shape(arr.shape, dtype))
    return scalar

# Can we walk through this function line by line so I can understand the semantics of the broadcast op?
def _broadcast_to(x: Tracer, dims: tuple[int, ...]) -> Tracer:
    """Right-aligned numpy-style broadcast, expressed as an HLO `broadcast`."""
    if x.dims == tuple(dims):
        return x
    offset = len(dims) - x.ndim
    if offset < 0:
        raise TypeError(f"cannot broadcast {x.shape} to {dims}")
    mapping = []
    for i, d in enumerate(x.dims):
        out = offset + i
        if d != dims[out] and d != 1:
            raise TypeError(f"cannot broadcast {x.shape} to {dims}")
        mapping.append(out)
    # A size-1 operand dim cannot map onto a larger output dim in HLO, so squeeze
    # those away with a reshape first -- another artifact you see in real dumps.
    squeezed = [i for i, d in enumerate(x.dims) if d == 1 and dims[offset + i] != 1]
    if squeezed:
        keep = [i for i in range(x.ndim) if i not in squeezed]
        x = reshape(x, tuple(x.dims[i] for i in keep))
        mapping = [offset + i for i in keep]
    b = current_builder()
    return b.emit("broadcast", [x], shape=Shape(tuple(dims), x.dtype),
                  dimensions=tuple(mapping))


def _binop(name: str, a, b_):
    """The implicit-broadcast adapter: lift scalars, align ranks, then emit."""
    if not isinstance(a, Tracer) and not isinstance(b_, Tracer):
        raise TypeError(f"{name}: at least one operand must be traced")
    like = a if isinstance(a, Tracer) else b_
    x, y = _as_tracer(a, like), _as_tracer(b_, like)
    dims = np.broadcast_shapes(x.dims, y.dims)
    x, y = _broadcast_to(x, dims), _broadcast_to(y, dims)
    return current_builder().emit(name, [x, y])


def _unop(name: str, x):
    return current_builder().emit(name, [_as_tracer(x)])


# ---------------------------------------------------------------- elementwise

def add(a, b): return _binop("add", a, b)
def subtract(a, b): return _binop("subtract", a, b)
def multiply(a, b): return _binop("multiply", a, b)
def divide(a, b): return _binop("divide", a, b)
def maximum(a, b): return _binop("maximum", a, b)
def minimum(a, b): return _binop("minimum", a, b)
def power(a, b): return _binop("power", a, b)

def exp(x): return _unop("exp", x)
def log(x): return _unop("log", x)
def sqrt(x): return _unop("sqrt", x)
def rsqrt(x): return _unop("rsqrt", x)
def tanh(x): return _unop("tanh", x)
def negate(x): return _unop("negate", x)
def abs(x): return _unop("abs", x)       # noqa: A001 -- mirrors jnp.abs
def sigmoid(x): return _unop("logistic", x)


def silu(x):
    """x * sigmoid(x). The activation in DeepSeek's SwiGLU."""
    return multiply(x, sigmoid(x))


def relu(x):
    return maximum(x, 0.0)


# ---------------------------------------------------------------- shape ops


def reshape(x, dims):
    x = _as_tracer(x)
    dims = tuple(dims) if isinstance(dims, (tuple, list)) else (dims,)
    return current_builder().emit("reshape", [x], shape=Shape(dims, x.dtype))


def transpose(x, permutation=None):
    x = _as_tracer(x)
    if permutation is None:
        permutation = tuple(reversed(range(x.ndim)))
    return current_builder().emit("transpose", [x], permutation=tuple(permutation))


def broadcast_to(x, dims):
    return _broadcast_to(_as_tracer(x), tuple(dims))


def expand_dims(x, axis):
    x = _as_tracer(x)
    dims = list(x.dims)
    dims.insert(axis if axis >= 0 else len(dims) + axis + 1, 1)
    return reshape(x, tuple(dims))


def iota(dims, dimension=0, dtype=S32):
    return current_builder().emit("iota", [], shape=Shape(tuple(dims), dtype),
                                  dimension=dimension)


def convert(x, dtype):
    return current_builder().emit("convert", [_as_tracer(x)], dtype=dtype)


# ---------------------------------------------------------------- reductions


def _normalize_axes(x: Tracer, axis):
    if axis is None:
        return tuple(range(x.ndim))
    if isinstance(axis, int):
        axis = (axis,)
    return tuple(a if a >= 0 else a + x.ndim for a in axis)


def _reduce(x, axis, computation, init, keepdims):
    x = _as_tracer(x)
    axes = _normalize_axes(x, axis)
    b = current_builder()
    init_t = b.emit("constant", [], value=init, shape=Shape((), x.dtype))
    out = b.emit("reduce", [x, init_t], dimensions=axes, computation=computation)
    if keepdims:
        # HLO's reduce DROPS the reduced dims. keepdims is a frontend convenience
        # implemented as a reshape -- which is why dumps show reshape-after-reduce.
        dims = tuple(1 if i in axes else d for i, d in enumerate(x.dims))
        out = reshape(out, dims)
    return out


def sum(x, axis=None, keepdims=False):      # noqa: A001
    return _reduce(x, axis, "add", 0.0, keepdims)


def max(x, axis=None, keepdims=False):      # noqa: A001
    return _reduce(x, axis, "max", -np.inf, keepdims)


def min(x, axis=None, keepdims=False):      # noqa: A001
    return _reduce(x, axis, "min", np.inf, keepdims)


def mean(x, axis=None, keepdims=False):
    x = _as_tracer(x)
    axes = _normalize_axes(x, axis)
    n = 1
    for a in axes:
        n *= x.dims[a]
    return divide(sum(x, axis=axes, keepdims=keepdims), float(n))


# ---------------------------------------------------------------- contraction


def dot_general(lhs, rhs, lhs_contracting, rhs_contracting,
                lhs_batch=(), rhs_batch=()):
    return current_builder().emit(
        "dot_general", [_as_tracer(lhs), _as_tracer(rhs)],
        lhs_contracting=tuple(lhs_contracting), rhs_contracting=tuple(rhs_contracting),
        lhs_batch=tuple(lhs_batch), rhs_batch=tuple(rhs_batch))


def matmul(a, b):
    """numpy-style matmul: contract a's last dim with b's first (or second-to-last).

    Leading dims of `a` are free (not batched), which is the usual
    (tokens, d_in) @ (d_in, d_out) shape of a linear layer.
    """
    a, b = _as_tracer(a), _as_tracer(b)
    rc = 0 if b.ndim == 1 else b.ndim - 2
    return dot_general(a, b, lhs_contracting=(a.ndim - 1,), rhs_contracting=(rc,))


def einsum_like_batched_matmul(a, b):
    """(..., m, k) x (..., k, n) -> (..., m, n), batching over all leading dims."""
    a, b = _as_tracer(a), _as_tracer(b)
    nb = a.ndim - 2
    if b.ndim - 2 != nb:
        raise TypeError(f"batched matmul: rank mismatch {a.shape} vs {b.shape}")
    batch = tuple(range(nb))
    return dot_general(a, b, lhs_contracting=(a.ndim - 1,), rhs_contracting=(nb,),
                       lhs_batch=batch, rhs_batch=batch)


# ---------------------------------------------------------------- comparison


def _cmp(direction, a, b):
    like = a if isinstance(a, Tracer) else b
    x, y = _as_tracer(a, like), _as_tracer(b, like)
    dims = np.broadcast_shapes(x.dims, y.dims)
    x, y = _broadcast_to(x, dims), _broadcast_to(y, dims)
    return current_builder().emit("compare", [x, y], direction=direction)


def less(a, b): return _cmp("lt", a, b)
def less_equal(a, b): return _cmp("le", a, b)
def greater(a, b): return _cmp("gt", a, b)
def greater_equal(a, b): return _cmp("ge", a, b)
def equal(a, b): return _cmp("eq", a, b)


def where(pred, on_true, on_false):
    pred = _as_tracer(pred)
    # `like` must come from a VALUE operand, never from pred: pred is PRED, and
    # adopting its dtype silently turned -1e30 into the boolean True.
    like = on_true if isinstance(on_true, Tracer) else (
        on_false if isinstance(on_false, Tracer) else None)
    t = _as_tracer(on_true, like)
    f = _as_tracer(on_false, like)
    dims = np.broadcast_shapes(pred.dims, t.dims, f.dims)
    pred = _broadcast_to(pred, dims)
    t, f = _broadcast_to(t, dims), _broadcast_to(f, dims)
    return current_builder().emit("select", [pred, t, f])


select = where


# ---------------------------------------------------------------- composites


def softmax(x, axis=-1):
    """The numerically-stable form, which is why dumps show max/sub/exp/sum/div."""
    m = max(x, axis=axis, keepdims=True)
    e = exp(subtract(x, m))
    return divide(e, sum(e, axis=axis, keepdims=True))


def log_softmax(x, axis=-1):
    m = max(x, axis=axis, keepdims=True)
    shifted = subtract(x, m)
    return subtract(shifted, log(sum(exp(shifted), axis=axis, keepdims=True)))


def rms_norm(x, weight, eps=1e-6):
    """DeepSeek / Llama RMSNorm."""
    ms = mean(multiply(x, x), axis=-1, keepdims=True)
    return multiply(multiply(x, rsqrt(add(ms, eps))), weight)
