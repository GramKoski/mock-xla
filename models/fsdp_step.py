"""Stage 3 acceptance model: a 2-layer MLP, forward AND backward, under FSDP.

THE BACKWARD PASS IS HAND-WRITTEN, on purpose. There is no autodiff in mock-xla
because `grad` is a JAX-level transform that runs ABOVE the compiler -- by the
time HLO exists, the backward ops are already ordinary nodes in the graph. So
writing them out by hand is not a workaround, it is what the compiler actually
receives.

What the partitioner should derive from four PartitionSpecs:

  FORWARD   h = x @ w1,  contraction over D which is sharded on w1 ONLY
            -> all-gather w1              == FSDP's forward parameter gather
  BACKWARD  dw1 = x^T @ dh, contraction over T sharded on BOTH sides
            -> partial={fsdp}; dw1 must END sharded
            -> all-reduce + dynamic-slice
            -> ReduceScatterCreator folds that into ONE reduce-scatter
                                          == FSDP's gradient reduce-scatter

Run:  python models/fsdp_step.py
      MOCKXLA_DUMP=./dumps python models/fsdp_step.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mockxla.numpy as mnp
from mockxla import jit
from mockxla.sharding import Mesh, P


# ------------------------------------------------------------------ the model

def mlp_dw1(x, y, w1, w2):
    """Forward, then the hand-written backward, returning dL/dw1.

    L = sum((relu(x @ w1) @ w2 - y)^2), so dL/do = 2(o - y); the 2 is dropped
    since it changes nothing about the sharding.
    """
    h = mnp.matmul(x, w1)                                   # (T,F)
    a = mnp.maximum(h, 0.0)                                 # relu
    o = mnp.matmul(a, w2)                                   # (T,D)

    do = mnp.subtract(o, y)                                 # (T,D)
    da = mnp.matmul(do, mnp.transpose(w2))                  # (T,F)
    dh = mnp.where(mnp.greater(h, 0.0), da, 0.0)            # relu'
    dw1 = mnp.dot_general(x, dh, lhs_contracting=(0,),      # (D,F)
                          rhs_contracting=(0,))             # contract over TOKENS
    return dw1


def reference(x, y, w1, w2):
    h = x @ w1
    a = np.maximum(h, 0.0)
    o = a @ w2
    do = o - y
    da = do @ w2.T
    dh = np.where(h > 0.0, da, 0.0)
    return x.T @ dh


# ------------------------------------------------------------------ driver

def main():
    rng = np.random.default_rng(0)
    T, D, F = 32, 16, 64
    x = rng.normal(size=(T, D)).astype("float32")
    y = rng.normal(size=(T, D)).astype("float32")
    w1 = (rng.normal(size=(D, F)) * 0.1).astype("float32")
    w2 = (rng.normal(size=(F, D)) * 0.1).astype("float32")
    args = (x, y, w1, w2)

    mesh = Mesh(("fsdp",), (4,))
    in_specs = [
        P("fsdp", None),     # x   sharded on TOKENS  (data parallel)
        P("fsdp", None),     # y   sharded on TOKENS
        P("fsdp", None),     # w1  sharded on EMBED   (FSDP)
        P("fsdp", None),     # w2  sharded on MLP     (FSDP)
    ]
    out_spec = P("fsdp", None)   # dw1 must come back SHARDED -> reduce-scatter

    lowered = jit(mlp_dw1).lower(*args)
    print(f"--- global program ({len(lowered.module.entry.instructions)} instrs) ---")
    print(lowered.as_text())

    exe = lowered.partition(mesh, in_specs, out_spec,
                            dump_dir=os.environ.get("MOCKXLA_DUMP"))
    print(f"--- partitioned for {mesh} ---")
    print(exe.as_text())
    print(exe.pipeline.report())

    counts: dict = {}
    for i in exe.module.entry.order():
        if "-" in i.opcode:
            counts[i.opcode] = counts.get(i.opcode, 0) + 1
    print("\ncollectives XLA inserted that you never wrote:")
    for i in exe.module.entry.order():
        if "-" in i.opcode:
            groups = i.attrs.get("replica_groups")
            print(f"  {i.opcode:<16} axis={i.attrs.get('axis'):<6} "
                  f"local{i.shape.dims}  replica_groups={groups}")

    got = exe(*args)
    want = reference(*args)
    err = np.abs(got - want).max()
    rel = err / np.abs(want).max()
    print(f"\npartitioned vs single-device: max abs err {err:.3e}  rel {rel:.3e}")
    assert rel < 1e-5, (err, rel)
    # NOT bit-identical, and that is the lesson: the all-reduce/reduce-scatter
    # sums 4 partial sums in a different order than one einsum over 32 tokens.
    # Changing collective ALGORITHM changes your loss bitwise -- which is why
    # bisecting a loss spike needs a fixed collective implementation.
    print("bit-identical:", bool(np.array_equal(got, want)),
          "  <- reduction ORDER differs; see the comment in this file")

    # ---- the same function, partitioned as MEGATRON instead ----------------
    print()
    print("=" * 70)
    print("SAME FUNCTION, different PartitionSpecs -> tensor parallel")
    print("=" * 70)
    tp_mesh = Mesh(("tp",), (4,))
    tp = jit(mlp_dw1).lower(*args).partition(
        tp_mesh,
        in_specs=[P(None, None),      # x   replicated
                  P(None, None),      # y   replicated
                  P(None, "tp"),      # w1  COLUMN-parallel (split the mlp dim)
                  P("tp", None)],     # w2  ROW-parallel    (split the mlp dim)
        out_spec=P(None, "tp"))
    for i in tp.module.entry.order():
        if "-" in i.opcode:
            print(f"  {i.opcode:<16} axis={i.attrs.get('axis')} local{i.shape.dims}")
    tp_err = np.abs(tp(*args) - want).max() / np.abs(want).max()
    print(f"  rel err {tp_err:.3e}")
    print()
    print("column-parallel then row-parallel gives exactly ONE all-reduce and")
    print("ZERO parameter gathers. Nothing in the partitioner knows the word")
    print("'Megatron' -- it is the same resharding rules, different specs.")
    print("OK")


if __name__ == "__main__":
    main()
