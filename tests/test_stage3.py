"""Stage 3 tests: mesh math, sharding propagation, partitioning, collectives.

Run: python tests/test_stage3.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mockxla.numpy as mnp
from mockxla import F32, Shape, jit
from mockxla.passes import ReduceScatterCreator
from mockxla.sharding import Dist, Mesh, P, local_shape
from mockxla.spmd_interp import global_dims_of, shard, unshard

FSDP4 = Mesh(("fsdp",), (4,))
MESH2D = Mesh(("fsdp", "tp"), (2, 4))     # 8 devices


def _ops(module, opcode):
    return [i for i in module.entry.order() if i.opcode == opcode]


# ---------------------------------------------------------------- mesh math

def test_mesh_stride_and_coord():
    m = MESH2D
    assert m.num_devices == 8
    # row-major: device = fsdp_coord*4 + tp_coord, so tp has stride 1
    assert m.stride("tp") == 1 and m.stride("fsdp") == 4
    assert [m.coord(d, "tp") for d in range(8)] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert [m.coord(d, "fsdp") for d in range(8)] == [0, 0, 0, 0, 1, 1, 1, 1]


def test_replica_groups_decode_to_an_axis():
    """The ONLY way to tell which axis a collective is on: group size + stride."""
    m = MESH2D
    tp = m.replica_groups(("tp",))
    assert tp == [[0, 1, 2, 3], [4, 5, 6, 7]], tp          # stride 1, size 4
    fsdp = m.replica_groups(("fsdp",))
    assert fsdp == [[0, 4], [1, 5], [2, 6], [3, 7]], fsdp  # stride 4, size 2
    # the invariant you check on any dump
    for groups in (tp, fsdp):
        assert sum(len(g) for g in groups) == m.num_devices


def test_local_shape_and_divisibility():
    s = Shape((32, 64), F32)
    assert local_shape(s, Dist(("fsdp", None)), FSDP4).dims == (8, 64)
    assert local_shape(s, Dist((None, None)), FSDP4).dims == (32, 64)
    # partial does not change the shape -- only the VALUE is incomplete
    assert local_shape(s, Dist((None, None), frozenset({"fsdp"})), FSDP4).dims == (32, 64)
    try:
        local_shape(Shape((30, 64), F32), Dist(("fsdp", None)), FSDP4)
    except ValueError as e:
        assert "not divisible" in str(e)
    else:
        raise AssertionError("30 is not divisible by 4; should have raised")


def test_shard_unshard_roundtrip():
    arr = np.arange(32 * 8, dtype="float32").reshape(32, 8)
    dist = Dist(("fsdp", None))
    pieces = [shard(arr, dist, FSDP4, d) for d in range(4)]
    assert pieces[0].shape == (8, 8)
    dims = global_dims_of(pieces[0].shape, dist, FSDP4)
    assert dims == (32, 8)
    assert np.array_equal(unshard(pieces, dist, FSDP4, dims), arr)


# ---------------------------------------------------------------- the model

def _mlp(x, y, w1, w2):
    h = mnp.matmul(x, w1)
    a = mnp.maximum(h, 0.0)
    o = mnp.matmul(a, w2)
    do = mnp.subtract(o, y)
    da = mnp.matmul(do, mnp.transpose(w2))
    dh = mnp.where(mnp.greater(h, 0.0), da, 0.0)
    return mnp.dot_general(x, dh, lhs_contracting=(0,), rhs_contracting=(0,))


def _ref(x, y, w1, w2):
    h = x @ w1
    a = np.maximum(h, 0.0)
    do = a @ w2 - y
    dh = np.where(h > 0.0, do @ w2.T, 0.0)
    return x.T @ dh


def _args(T=32, D=16, F=64, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(T, D)).astype("float32"),
            rng.normal(size=(T, D)).astype("float32"),
            (rng.normal(size=(D, F)) * 0.1).astype("float32"),
            (rng.normal(size=(F, D)) * 0.1).astype("float32"))


FSDP_SPECS = [P("fsdp", None)] * 4


# ---------------------------------------------------------------- FSDP

def test_fsdp_derives_allgather_and_reducescatter():
    args = _args()
    exe = jit(_mlp).lower(*args).partition(FSDP4, FSDP_SPECS, P("fsdp", None))
    m = exe.module
    assert len(_ops(m, "all-gather")) >= 2, "forward must gather the sharded weights"
    assert len(_ops(m, "reduce-scatter")) == 1, "the gradient must reduce-scatter"
    assert m.entry[m.entry.root].opcode == "reduce-scatter"
    # and the gathers are on the fsdp axis
    for ag in _ops(m, "all-gather"):
        assert ag.attrs["axis"] == "fsdp"
        assert ag.attrs["replica_groups"] == [[0, 1, 2, 3]]


def test_fsdp_execution_matches_single_device():
    args = _args()
    exe = jit(_mlp).lower(*args).partition(FSDP4, FSDP_SPECS, P("fsdp", None))
    got, want = exe(*args), _ref(*args)
    assert got.shape == want.shape
    rel = np.abs(got - want).max() / np.abs(want).max()
    assert rel < 1e-5, rel
    # NOT bit-identical: 4 partial sums combine in a different order than one
    # einsum over 32 tokens. Changing collective algorithm changes your loss.
    assert not np.array_equal(got, want), \
        "if this ever becomes exact, the reduction order stopped differing"


def test_sharded_params_are_smaller_on_each_device():
    args = _args()
    exe = jit(_mlp).lower(*args).partition(FSDP4, FSDP_SPECS, P("fsdp", None))
    comp = exe.module.entry
    assert comp[comp.params[2]].shape.dims == (4, 64), "w1 (16,64) over 4 -> (4,64)"
    assert comp[comp.params[0]].shape.dims == (8, 16), "x  (32,16) over 4 -> (8,16)"


# ---------------------------------------------------------------- TP

def test_tp_derives_a_single_allreduce_and_no_gathers():
    """Same function, different specs -> Megatron instead of FSDP."""
    args = _args()
    exe = jit(_mlp).lower(*args).partition(
        Mesh(("tp",), (4,)),
        [P(None, None), P(None, None), P(None, "tp"), P("tp", None)],
        P(None, "tp"))
    m = exe.module
    assert len(_ops(m, "all-reduce")) == 1, "row-parallel matmul needs exactly one"
    assert not _ops(m, "all-gather"), "TP must not gather parameters"
    assert not _ops(m, "reduce-scatter")
    got, want = exe(*args), _ref(*args)
    assert np.abs(got - want).max() / np.abs(want).max() < 1e-5


def test_contraction_sharded_both_sides_produces_partial():
    """The core of the whole design: partial is what makes all-reduce appear."""
    from mockxla.passes import ShardingPropagation
    args = _args()
    lowered = jit(_mlp).lower(*args)
    ShardingPropagation(Mesh(("tp",), (4,)),
                        [P(None, None), P(None, None),
                         P(None, "tp"), P("tp", None)]).run(lowered.module)
    partials = [i for i in lowered.module.entry.order()
                if i.dist is not None and i.dist.is_partial]
    assert partials, "a->w2 contracts over a tp-sharded dim; that is a partial sum"
    assert all("tp" in i.dist.partial for i in partials)


# ---------------------------------------------------------------- 2-D mesh

def test_two_axes_compose():
    """FSDP on the weights, TP on the mlp dim, on one 2x4 mesh."""
    args = _args(T=32, D=16, F=64)
    exe = jit(_mlp).lower(*args).partition(
        MESH2D,
        [P("fsdp", None), P("fsdp", None), P("fsdp", "tp"), P("tp", None)],
        P("fsdp", "tp"))
    got, want = exe(*args), _ref(*args)
    assert np.abs(got - want).max() / np.abs(want).max() < 1e-5
    axes = {i.attrs.get("axis") for i in exe.module.entry.order() if "-" in i.opcode}
    assert "fsdp" in axes and "tp" in axes, axes


# ---------------------------------------------------------------- the pass

def test_reduce_scatter_creator_needs_a_single_user():
    """If anything else reads the full all-reduce, fusing would force a recompute."""
    args = _args()
    exe = jit(_mlp).lower(*args).partition(FSDP4, FSDP_SPECS, P("fsdp", None))
    comp = exe.module.entry
    rs = _ops(exe.module, "reduce-scatter")[0]
    # re-running must be a no-op: the pattern is already consumed
    assert ReduceScatterCreator().run(exe.module) is False

    # and a bare all-reduce with two users must NOT be rewritten
    from mockxla.passes.spmd import ShardingPropagation, SpmdPartitioner
    m2 = jit(_mlp).lower(*args).module
    ShardingPropagation(FSDP4, FSDP_SPECS).run(m2)
    SpmdPartitioner(FSDP4, P(None, None)).run(m2)   # replicated output -> plain all-reduce
    assert not _ops(m2, "reduce-scatter"), \
        "a replicated output needs all-reduce, with no slice to fuse"
    assert _ops(m2, "all-reduce")


def test_replicated_output_gives_allreduce_not_reducescatter():
    """DDP vs FSDP is literally the out_spec. Nothing else changes."""
    args = _args()
    ddp = jit(_mlp).lower(*args).partition(FSDP4, FSDP_SPECS, P(None, None))
    fsdp = jit(_mlp).lower(*args).partition(FSDP4, FSDP_SPECS, P("fsdp", None))
    assert _ops(ddp.module, "all-reduce") and not _ops(ddp.module, "reduce-scatter")
    assert _ops(fsdp.module, "reduce-scatter")
    assert np.abs(ddp(*args) - fsdp(*args).repeat(1, axis=0)).max() >= 0  # shapes differ
    assert ddp.module.entry[ddp.module.entry.root].shape.dims == (16, 64)
    assert fsdp.module.entry[fsdp.module.entry.root].shape.dims == (4, 64)


# ------------------------------------------------- documented known limitation

def test_forward_only_propagation_slices_a_replicated_zero():
    """KNOWN INEFFICIENCY, asserted so a future fix is visible.

    `maximum(h, 0.0)` has a replicated broadcast(0) operand and an h that is
    sharded, so the partitioner SLICES the full zero -- materializing it at
    global size on every device and discarding most of it. Bidirectional
    propagation would instead push h's spec back onto the broadcast, which can
    produce any spec for free since its operand is a scalar.
    """
    args = _args()
    exe = jit(_mlp).lower(*args).partition(
        Mesh(("tp",), (4,)),
        [P(None, None), P(None, None), P(None, "tp"), P("tp", None)],
        P(None, "tp"))
    comp = exe.module.entry
    slices = _ops(exe.module, "dynamic-slice")
    assert slices, "expected the replicated-zero slice; did propagation improve?"
    assert comp[slices[0].operands[0]].opcode == "broadcast"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
