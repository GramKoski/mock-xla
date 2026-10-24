"""Stage 2 demo: watch the passes work.

Two models:
  1. `redundant` -- deliberately full of things the passes should remove, so the
     before/after is dramatic.
  2. the real SwiGLU MLP from mlp.py, to see what a clean model leaves behind.

Run:  python models/passes_demo.py
      MOCKXLA_DUMP=./dumps python models/passes_demo.py && ls dumps/
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mockxla.numpy as mnp
from mockxla import jit
from mockxla.interp import execute, peak_live_bytes
from mockxla.printer import print_module

from mlp import swiglu_mlp


# ------------------------------------------------------------------ model 1

def redundant(x, w):
    """Every line here is something a pass should notice."""
    a = mnp.rms_norm(x, w)          # computed...
    b = mnp.rms_norm(x, w)          # ...twice, identically          -> CSE
    y = mnp.add(a, b)
    y = mnp.multiply(y, 1.0)        # x * 1                          -> algebraic
    y = mnp.add(y, 0.0)             # x + 0                          -> algebraic
    y = mnp.transpose(mnp.transpose(y))   # double transpose         -> algebraic
    y = mnp.reshape(y, y.dims)      # reshape to the same shape      -> algebraic
    dead = mnp.exp(mnp.multiply(x, x))    # never used               -> DCE
    return y


# ------------------------------------------------------------------ driver

def run(name, fn, args, show_metadata=False):
    f = jit(fn)
    lowered = f.lower(*args)
    before = print_module(lowered.module, metadata=show_metadata)
    n_before = len(lowered.module.entry.instructions)
    peak_before = peak_live_bytes(lowered.module)

    exe = lowered.compile(dump_dir=os.environ.get("MOCKXLA_DUMP"))

    print("=" * 76)
    print(f"{name}: {n_before} -> {len(exe.module.entry.instructions)} instructions")
    print("=" * 76)
    if show_metadata:
        print("\n--- BEFORE (with source attribution) ---")
        print(before)
    print("--- AFTER ---")
    print(exe.as_text())
    print(exe.pipeline.report())
    print(f"peak live bytes  {peak_before/1e3:.1f} kB -> "
          f"{peak_live_bytes(exe.module)/1e3:.1f} kB")

    # the whole point: optimizing must not change the answer
    ref = jit(fn).lower(*args).compile(optimize=False)
    got, want = exe(*args), ref(*args)
    err = np.abs(got - want).max()
    print(f"max |optimized - unoptimized| = {err:.3e}")
    assert err == 0.0, err
    print()


def main():
    rng = np.random.default_rng(0)
    T, D, F = 8, 64, 176
    x = rng.normal(size=(T, D)).astype("float32")
    w = (rng.normal(size=(D,)) * 0.1 + 1.0).astype("float32")

    run("redundant", redundant, (x, w), show_metadata=True)

    w_gate = (rng.normal(size=(D, F)) * 0.02).astype("float32")
    w_up = (rng.normal(size=(D, F)) * 0.02).astype("float32")
    w_down = (rng.normal(size=(F, D)) * 0.02).astype("float32")
    run("swiglu_mlp", swiglu_mlp, (x, w, w_gate, w_up, w_down))

    if os.environ.get("MOCKXLA_DUMP"):
        print(f"dumps written to {os.environ['MOCKXLA_DUMP']}/ -- try:")
        print("  diff dumps/*.0000.* dumps/*.0001.*")


if __name__ == "__main__":
    main()
