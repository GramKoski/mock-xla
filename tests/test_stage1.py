"""Stage 1 tests. Run: python tests/test_stage1.py  (or pytest)"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mockxla.numpy as mnp
from mockxla import F32, ConcretizationError, Shape, jit
from mockxla.interp import execute


def _check(fn, ref, *args, tol=1e-5):
    got = jit(fn)(*args)
    want = ref(*args)
    assert got.shape == np.asarray(want).shape, f"{got.shape} vs {np.shape(want)}"
    err = np.abs(got - want).max()
    assert err < tol, f"{fn.__name__}: max err {err:.3e}"
    return got


# ---------------------------------------------------------------- shapes

def test_shape_basics():
    s = Shape((8, 128), F32)
    assert s.rank == 2 and s.num_elements == 1024 and s.size_bytes == 4096
    assert str(s) == "f32[8,128]"
    from mockxla import BF16
    assert Shape((1024,), BF16).size_bytes == 2048   # bits=16 even though it runs as f32


def test_dot_shape_rule():
    from mockxla.ops import lookup
    rule = lookup("dot_general").shape_rule
    # (8,64) x (64,176) contracting 1 vs 0
    out = rule(Shape((8, 64), F32), Shape((64, 176), F32),
               lhs_contracting=(1,), rhs_contracting=(0,))
    assert out.dims == (8, 176)
    # batched: (4,8,64) x (4,64,16), batch 0
    out = rule(Shape((4, 8, 64), F32), Shape((4, 64, 16), F32),
               lhs_contracting=(2,), rhs_contracting=(1,),
               lhs_batch=(0,), rhs_batch=(0,))
    assert out.dims == (4, 8, 16), out.dims
    # mismatched contraction must be caught at TRACE time
    try:
        rule(Shape((8, 64), F32), Shape((32, 176), F32),
             lhs_contracting=(1,), rhs_contracting=(0,))
    except TypeError:
        pass
    else:
        raise AssertionError("expected a contracting-dim TypeError")


def test_no_implicit_broadcast_in_ir():
    """The IR is strict; the frontend is what inserts broadcasts."""
    from mockxla.ops import lookup
    try:
        lookup("add").shape_rule(Shape((8, 64), F32), Shape((64,), F32))
    except TypeError as e:
        assert "implicit broadcasting" in str(e)
    else:
        raise AssertionError("elementwise ops must require identical shapes")


# ---------------------------------------------------------------- tracing

def test_branch_on_tracer_raises():
    @jit
    def bad(x):
        if x:                      # noqa: SIM103
            return x
        return x
    try:
        bad(np.zeros((4,), "float32"))
    except ConcretizationError as e:
        assert "Python branch" in str(e)
    else:
        raise AssertionError("branching on a tracer must raise")


def test_cache_keys_on_shape():
    f = jit(lambda x: mnp.exp(x))
    f(np.zeros((4,), "float32"))
    f(np.zeros((4,), "float32"))
    assert f.cache_size() == 1
    f(np.zeros((8,), "float32"))
    assert f.cache_size() == 2, "a new input shape is a new program"


def test_validate_catches_use_before_def():
    f = jit(lambda x: mnp.add(x, 1.0))
    m = f.lower(np.zeros((4,), "float32")).module
    m.validate()
    order = list(m.entry.instructions)
    # reverse the dict order so an operand follows its user
    m.entry.instructions = {k: m.entry.instructions[k] for k in reversed(order)}
    try:
        m.validate()
    except ValueError as e:
        assert "before its definition" in str(e)
    else:
        raise AssertionError("validate() should reject a non-topological order")


# ---------------------------------------------------------------- numerics

def test_elementwise_and_scalars():
    _check(lambda x: mnp.add(mnp.multiply(x, 2.0), 1.0),
           lambda x: x * 2.0 + 1.0, np.random.randn(4, 5).astype("float32"))
    _check(lambda x: 1.0 - x, lambda x: 1.0 - x, np.random.randn(6).astype("float32"))
    _check(lambda x: -x / 3.0, lambda x: -x / 3.0, np.random.randn(6).astype("float32"))


def test_reductions_and_keepdims():
    x = np.random.randn(4, 7).astype("float32")
    _check(lambda a: mnp.sum(a, axis=1), lambda a: a.sum(1), x)
    _check(lambda a: mnp.max(a, axis=0), lambda a: a.max(0), x)
    _check(lambda a: mnp.mean(a, axis=-1, keepdims=True),
           lambda a: a.mean(-1, keepdims=True), x)
    _check(lambda a: mnp.sum(a), lambda a: a.sum(), x)


def test_shape_ops():
    x = np.random.randn(2, 3, 4).astype("float32")
    _check(lambda a: mnp.transpose(a, (2, 0, 1)), lambda a: a.transpose(2, 0, 1), x)
    _check(lambda a: mnp.reshape(a, (6, 4)), lambda a: a.reshape(6, 4), x)
    _check(lambda a: mnp.broadcast_to(mnp.sum(a, axis=2), (2, 3)),
           lambda a: a.sum(2), x)


def test_matmul_and_batched():
    a = np.random.randn(8, 64).astype("float32")
    b = np.random.randn(64, 16).astype("float32")
    _check(lambda x, y: mnp.matmul(x, y), lambda x, y: x @ y, a, b, tol=1e-4)

    a = np.random.randn(4, 8, 64).astype("float32")
    b = np.random.randn(4, 64, 16).astype("float32")
    _check(lambda x, y: mnp.einsum_like_batched_matmul(x, y),
           lambda x, y: np.einsum("bmk,bkn->bmn", x, y), a, b, tol=1e-4)


def test_softmax_and_select():
    x = np.random.randn(4, 9).astype("float32") * 3.0

    def ref_softmax(a):
        e = np.exp(a - a.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)
    _check(lambda a: mnp.softmax(a), ref_softmax, x)

    _check(lambda a: mnp.where(mnp.greater(a, 0.0), a, 0.0),
           lambda a: np.where(a > 0, a, 0.0), x)


def test_causal_mask_via_iota():
    """iota + compare + select: the pattern attention needs in stage 1b."""
    n = 6
    scores = np.random.randn(n, n).astype("float32")

    def masked(s):
        r = mnp.iota((n, n), dimension=0)
        c = mnp.iota((n, n), dimension=1)
        return mnp.where(mnp.greater_equal(r, c), s, -1e30)

    def ref(s):
        m = np.tril(np.ones((n, n), bool))
        return np.where(m, s, -1e30)
    _check(masked, ref, scores)


def test_rms_norm():
    x = np.random.randn(5, 32).astype("float32")
    w = np.random.randn(32).astype("float32")

    def ref(a, g):
        return a * (1.0 / np.sqrt((a * a).mean(-1, keepdims=True) + 1e-6)) * g
    _check(lambda a, g: mnp.rms_norm(a, g), ref, x, w)


def test_impl_and_shape_rule_agree():
    """execute() asserts this per instruction; make sure the check is live."""
    x = np.random.randn(3, 4).astype("float32")
    m = jit(lambda a: mnp.softmax(a)).lower(x).module
    execute(m, x)   # raises AssertionError if any shape rule disagrees with numpy


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    np.random.seed(0)
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
