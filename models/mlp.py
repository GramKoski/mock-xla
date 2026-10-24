"""The stage-1 acceptance model: RMSNorm + SwiGLU MLP.

This is a real DeepSeek/Llama dense sub-block, and it exercises everything
stage 3's fusion pass will care about:

  reduce          (the RMSNorm mean)
  broadcast       (the norm scale back over the feature dim)
  dot_general     (three of them)
  elementwise     (a chain long enough to be worth fusing)

and it needs no gather, no top-k, no RoPE and no attention -- so if this works,
the machinery is sound and the rest of the op set is table-filling.

Run:  python models/mlp.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mockxla.numpy as mnp
from mockxla import jit


# ------------------------------------------------------------------ the model

def swiglu_mlp(x, norm_w, w_gate, w_up, w_down):
    """y = (silu(h @ Wg) * (h @ Wu)) @ Wd   with h = RMSNorm(x)"""
    h = mnp.rms_norm(x, norm_w)
    gate = mnp.silu(mnp.matmul(h, w_gate))
    up = mnp.matmul(h, w_up)
    return mnp.matmul(mnp.multiply(gate, up), w_down)


def reference(x, norm_w, w_gate, w_up, w_down, eps=1e-6):
    """Plain numpy, for the correctness assert."""
    ms = np.mean(x * x, axis=-1, keepdims=True)
    h = x * (1.0 / np.sqrt(ms + eps)) * norm_w
    g = h @ w_gate
    g = g * (1.0 / (1.0 + np.exp(-g)))
    return (g * (h @ w_up)) @ w_down


# ------------------------------------------------------------------ driver

def main():
    rng = np.random.default_rng(0)
    T, D, F = 8, 64, 176
    x = rng.normal(size=(T, D)).astype("float32")
    norm_w = rng.normal(size=(D,)).astype("float32") * 0.1 + 1.0
    w_gate = (rng.normal(size=(D, F)) * 0.02).astype("float32")
    w_up = (rng.normal(size=(D, F)) * 0.02).astype("float32")
    w_down = (rng.normal(size=(F, D)) * 0.02).astype("float32")
    args = (x, norm_w, w_gate, w_up, w_down)

    f = jit(swiglu_mlp)

    print(f.lower(*args).as_text())

    got = f(*args)
    want = reference(*args)
    err = np.abs(got - want).max()
    print(f"max |mockxla - numpy| = {err:.3e}   shape {got.shape}")
    assert err < 1e-5, err

    from mockxla.interp import peak_live_bytes
    from mockxla.printer import stats
    m = f.lower(*args).module
    print(stats(m))
    print(f"naive peak live bytes = {peak_live_bytes(m) / 1e3:.1f} kB")

    # Retracing: a new input shape is a new program.
    f(rng.normal(size=(16, D)).astype("float32"), norm_w, w_gate, w_up, w_down)
    print(f"compile cache entries = {f.cache_size()}  (one per input-shape signature)")
    print("OK")


if __name__ == "__main__":
    main()
