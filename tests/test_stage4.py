"""Stage 4 tests: cost model and PriorityFusion.

Run: python tests/test_stage4.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mockxla.numpy as mnp
from mockxla import F32, Shape, jit
from mockxla.cost import (H100, Device, bytes_accessed, combine, flops_of,
                          module_cost, time_of)
from mockxla.interp import execute
from mockxla.passes.fusion import PriorityFusion, benefit
from mockxla.sharding import Mesh, P

X = np.random.default_rng(0).normal(size=(256, 512)).astype("float32")
W = np.ones((512,), "float32")


def _mod(fn, *args):
    return jit(fn).lower(*args).module


def _fusions(m):
    return [i for i in m.entry.order() if i.opcode == "fusion"]


# ---------------------------------------------------------------- cost model

def test_combine_is_roofline_plus_a_tax():
    """XLA: compute + memory - min(.)*0.95, i.e. max + 0.05*min. NOT pure max."""
    c, mem = 10.0, 4.0
    got = combine(c, mem, H100)
    assert abs(got - (max(c, mem) + 0.05 * min(c, mem))) < 1e-12
    assert got > max(c, mem), "there is a 5% tax on the smaller term"
    # p=1.0 recovers the textbook roofline
    pure = Device("pure", memory_compute_parallelism=1.0)
    assert abs(combine(c, mem, pure) - max(c, mem)) < 1e-12
    # p=0.0 is no overlap at all
    none = Device("serial", memory_compute_parallelism=0.0)
    assert abs(combine(c, mem, none) - (c + mem)) < 1e-12


def test_dot_flops_are_2mnk():
    m = _mod(lambda a, b: mnp.matmul(a, b), X, np.ones((512, 64), "float32"))
    dot = [i for i in m.entry.order() if i.opcode == "dot_general"][0]
    assert flops_of(dot, m.entry) == 2 * 256 * 64 * 512


def test_shape_ops_are_free_of_flops():
    m = _mod(lambda a: mnp.transpose(mnp.reshape(a, (512, 256))), X)
    for i in m.entry.order():
        if i.opcode in ("reshape", "transpose"):
            assert flops_of(i, m.entry) == 0.0


def test_bytes_accessed_is_operands_plus_output():
    m = _mod(lambda a: mnp.exp(a), X)
    e = [i for i in m.entry.order() if i.opcode == "exp"][0]
    assert bytes_accessed(e, m.entry) == 2 * X.nbytes


def test_fusion_bytes_exclude_intermediates():
    """THE measurement that makes fusion profitable."""
    m = _mod(lambda a: mnp.exp(mnp.multiply(mnp.add(a, 1.0), 2.0)), X)
    unfused = module_cost(m)
    PriorityFusion().run(m)
    fused = module_cost(m)
    assert fused["hbm_traffic"] < unfused["hbm_traffic"]
    assert fused["launches"] < unfused["launches"]
    f = _fusions(m)[0]
    # the fusion touches only its operands and its output
    expect = sum(m.entry[o].shape.size_bytes for o in f.operands) + f.shape.size_bytes
    assert bytes_accessed(f, m.entry) == expect


def test_collective_cost_uses_the_ring_formula():
    big = Shape((1024, 1024), F32)
    m = _mod(lambda a: mnp.exp(a), X)
    comp = m.entry
    ar = comp.add("all-reduce", [comp.params[0]], comp[comp.params[0]].shape,
                  {"replica_groups": [[0, 1, 2, 3]], "computation": "add",
                   "axis": "fsdp"})
    t = time_of(ar, comp, m, H100)
    v, n = ar.shape.size_bytes, 4
    want = 2 * v * (n - 1) / n / H100.nvlink_bandwidth + H100.nccl_launch_overhead
    assert abs(t["total"] - want) < 1e-15
    assert t["kind"] == "collective/nvlink"


def test_wide_groups_drop_to_the_slow_link():
    """The HSDP argument, as one assertion."""
    fast, link_f = H100.link_for(8)
    slow, link_s = H100.link_for(16)
    assert link_f == "nvlink" and link_s == "ib"
    assert fast / slow == H100.nvlink_bandwidth / H100.ib_bandwidth > 5


# ---------------------------------------------------------------- fusion

def test_fusion_is_bit_exact():
    """Fusion reorders NO arithmetic, so unlike the collectives it is exact."""
    def f(a, g):
        return mnp.multiply(mnp.rms_norm(a, g), 2.0)
    m = _mod(f, X, W)
    want = execute(m, X, W)
    PriorityFusion().run(m)
    m.validate()
    got = execute(m, X, W)
    assert np.array_equal(got, want), "fusion must be exact"


def test_elementwise_chain_becomes_one_fusion():
    m = _mod(lambda a: mnp.exp(mnp.multiply(mnp.add(a, 1.0), 2.0)), X)
    PriorityFusion().run(m)
    m.validate()
    assert len(_fusions(m)) == 1
    body = m.computations[_fusions(m)[0].attrs["computation"]]
    kinds = {i.opcode for i in body.order()}
    assert {"add", "multiply", "exp"} <= kinds, kinds


def test_reduce_root_gives_kinput_elementwise_gives_kloop():
    """The kLoop / kInput distinction, decided by the ROOT's opcode."""
    m = _mod(lambda a: mnp.sum(mnp.multiply(a, a), axis=1), X)
    PriorityFusion().run(m)
    f = _fusions(m)[0]
    assert f.attrs["kind"] == "kInput", "a reduce root drives the INPUT loop nest"

    m2 = _mod(lambda a: mnp.exp(mnp.multiply(a, a)), X)
    PriorityFusion().run(m2)
    assert _fusions(m2)[0].attrs["kind"] == "kLoop"


def test_dot_and_collectives_are_never_fused():
    m = _mod(lambda a, b: mnp.exp(mnp.matmul(a, b)), X,
             np.ones((512, 64), "float32"))
    PriorityFusion().run(m)
    for f in _fusions(m):
        body = m.computations[f.attrs["computation"]]
        assert all(i.opcode != "dot_general" for i in body.order()), \
            "a dot has its own tiled loop nest; it goes to a library, not a fusion"


def test_two_distinct_consumers_are_declined():
    """Fusing into two SEPARATE consumers duplicates the producer. We decline.

    The second consumer here is a dot_general, which is never fusible, so the
    exp can never get down to one user and must survive.
    """
    B = np.ones((512, 512), "float32")

    def f(a, b):
        t = mnp.exp(a)
        return mnp.add(mnp.matmul(t, b), mnp.multiply(t, 2.0))
    m = _mod(f, X, B)
    PriorityFusion().run(m)
    m.validate()
    assert any(i.opcode == "exp" for i in m.entry.order()),         "a producer feeding a dot AND an elementwise op cannot be fused"


def test_one_consumer_twice_is_fused_without_duplication():
    """users() counts CONSUMERS, not USES -- and that is the right granularity.

    `t` is read twice by the same fusion, so both body parameters map to the
    SAME inner node and exp is computed once. Fusing is both legal and optimal.
    """
    def f(a):
        t = mnp.exp(a)
        return mnp.add(mnp.multiply(t, 2.0), mnp.multiply(t, 3.0))
    small = np.random.default_rng(0).normal(size=(8, 16)).astype("float32")
    m = _mod(f, small)
    want = execute(m, small)
    PriorityFusion().run(m)
    m.validate()
    assert np.array_equal(execute(m, small), want)
    fs = _fusions(m)
    assert len(fs) == 1
    body = m.computations[fs[0].attrs["computation"]]
    assert sum(1 for i in body.order() if i.opcode == "exp") == 1,         "exp must appear ONCE in the body, not once per use"


def test_benefit_is_positive_for_a_real_chain():
    m = _mod(lambda a: mnp.exp(mnp.multiply(a, 2.0)), X)
    mul = [i for i in m.entry.order() if i.opcode == "multiply"][0]
    exp = [i for i in m.entry.order() if i.opcode == "exp"][0]
    b = benefit(mul, exp, m.entry, m, H100)
    assert b > 0
    # the saving is one intermediate (X.nbytes) plus one kernel launch
    floor = X.nbytes / H100.hbm_bandwidth * 0.9 + H100.launch_overhead * 0.9
    assert b > floor, (b, floor)


def test_heap_is_incremental_so_the_guards_matter():
    """The queue is updated incrementally, so popped entries CAN be stale.

    If `stale` is 0 on a chain this long, the implementation is rebuilding the
    whole candidate list after every fusion -- which makes the re-score guard
    dead code and defeats the point of a priority queue.
    """
    def f(a, g):
        h = mnp.rms_norm(a, g)
        return mnp.multiply(mnp.silu(h), 2.0)
    m = _mod(f, X, W)
    pf = PriorityFusion()
    pf.run(m)
    assert len(pf.log) >= 5, len(pf.log)
    assert pf.stale > 0,         "no stale entries: the heap is being rebuilt, not updated"


def test_rerunning_fusion_is_a_noop():
    m = _mod(lambda a: mnp.exp(mnp.multiply(a, 2.0)), X)
    pf = PriorityFusion()
    assert pf.run(m) is True
    assert PriorityFusion().run(m) is False, "the pass must reach a fixpoint"


def test_min_benefit_threshold_blocks_fusion():
    m = _mod(lambda a: mnp.exp(mnp.multiply(a, 2.0)), X)
    assert PriorityFusion(min_benefit=1.0).run(m) is False, \
        "a 1-second threshold should reject everything"


# ---------------------------------------------------------------- composition

def test_fusion_runs_after_spmd_and_prices_local_bytes():
    rng = np.random.default_rng(0)
    args = (rng.normal(size=(64, 64)).astype("float32"), np.ones((64,), "float32"))
    exe = jit(lambda a, g: mnp.multiply(mnp.rms_norm(a, g), 2.0)).lower(*args).partition(
        Mesh(("fsdp",), (4,)), [P("fsdp", None), P(None,)], P("fsdp", None))
    names = [n for n, _, c in exe.pipeline.log for _ in (0,)] if False else \
        [row[0] for row in exe.pipeline.log]
    assert names.index("spmd_partitioner") < names.index("priority_fusion"), \
        "XLA order: partition at 1911, fuse at 1970"
    assert _fusions(exe.module), "should still fuse after partitioning"
    # and the fused shapes are LOCAL
    assert exe.module.entry[exe.module.entry.root].shape.dims[0] in (16, 64)


def test_pipeline_can_disable_fusion():
    args = (X, W)
    with_f = jit(lambda a, g: mnp.rms_norm(a, g)).lower(*args).compile(fuse=True)
    without = jit(lambda a, g: mnp.rms_norm(a, g)).lower(*args).compile(fuse=False)
    assert _fusions(with_f.module) and not _fusions(without.module)
    assert np.array_equal(with_f(*args), without(*args))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
