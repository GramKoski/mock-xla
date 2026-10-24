"""Stage 2 tests. Run: python tests/test_stage2.py"""

import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mockxla.numpy as mnp
from mockxla import jit
from mockxla.passes import (CSE, DCE, AlgebraicSimplifier, ConstantFolding,
                            Fixpoint, PassPipeline, default_pipeline,
                            structural_key)


def _module(fn, *args):
    return jit(fn).lower(*args).module


def _n(module):
    return len(module.entry.instructions)


X = np.arange(12, dtype="float32").reshape(3, 4)


# ---------------------------------------------------------------- DCE

def test_dce_removes_unused():
    def f(x):
        _ = mnp.exp(mnp.multiply(x, x))     # dead
        return mnp.add(x, x)
    m = _module(f, X)
    before = _n(m)
    assert DCE().run(m) is True
    assert _n(m) == before - 2, "both dead instructions should go in one sweep"
    assert DCE().run(m) is False, "a second run must report no change"
    m.validate()


def test_dce_keeps_root_and_params():
    m = _module(lambda x: mnp.exp(x), X)
    DCE().run(m)
    assert _n(m) == 2 and m.entry.root is not None


# ---------------------------------------------------------------- CSE

def test_cse_merges_identical():
    def f(x):
        a = mnp.multiply(x, x)
        b = mnp.multiply(x, x)      # identical
        return mnp.add(a, b)
    m = _module(f, X)
    assert CSE().run(m) is True
    DCE().run(m)
    mults = [i for i in m.entry.instructions.values() if i.opcode == "multiply"]
    assert len(mults) == 1, f"expected 1 multiply, got {len(mults)}"
    m.validate()


def test_structural_key_ignores_metadata_and_id():
    def f(x):
        a = mnp.multiply(x, x)
        b = mnp.multiply(x, x)
        return mnp.add(a, b)
    m = _module(f, X)
    ms = [i for i in m.entry.instructions.values() if i.opcode == "multiply"]
    assert len(ms) == 2
    assert ms[0].id != ms[1].id
    assert ms[0].metadata.source_line != ms[1].metadata.source_line
    assert structural_key(ms[0]) == structural_key(ms[1]), \
        "CSE identity must exclude id and metadata"


def test_cse_distinguishes_attrs():
    def f(x):
        return mnp.add(mnp.sum(x, axis=0, keepdims=True),
                       mnp.sum(x, axis=0, keepdims=True))
    m = _module(f, X)
    CSE().run(m); DCE().run(m)
    assert len([i for i in m.entry.instructions.values() if i.opcode == "reduce"]) == 1

    def g(x):
        return mnp.add(mnp.sum(x, axis=0, keepdims=True),
                       mnp.sum(x, axis=1, keepdims=True))        # DIFFERENT dims
    m = _module(g, X)
    CSE().run(m); DCE().run(m)
    assert len([i for i in m.entry.instructions.values() if i.opcode == "reduce"]) == 2, \
        "reduces over different dimensions are not the same computation"


# ---------------------------------------------------------------- algebraic

def _simplifies_away(fn, opcode):
    m = _module(fn, X)
    Fixpoint([AlgebraicSimplifier(), DCE()]).run(m)
    m.validate()
    return all(i.opcode != opcode for i in m.entry.instructions.values())


def test_algebraic_identities():
    assert _simplifies_away(lambda x: mnp.multiply(x, 1.0), "multiply")
    assert _simplifies_away(lambda x: mnp.add(x, 0.0), "add")
    assert _simplifies_away(lambda x: mnp.subtract(x, 0.0), "subtract")
    assert _simplifies_away(lambda x: mnp.divide(x, 1.0), "divide")
    assert _simplifies_away(lambda x: mnp.reshape(x, x.dims), "reshape")
    assert _simplifies_away(lambda x: mnp.transpose(mnp.transpose(x)), "transpose")


def test_algebraic_sees_through_broadcast():
    """`x * 1` is add(x, broadcast(constant(1))) -- the pattern must look through."""
    m = _module(lambda x: mnp.multiply(x, 1.0), X)
    assert any(i.opcode == "broadcast" for i in m.entry.instructions.values())
    assert AlgebraicSimplifier().run(m) is True


def test_algebraic_does_not_overreach():
    """x * 2 must survive."""
    m = _module(lambda x: mnp.multiply(x, 2.0), X)
    AlgebraicSimplifier().run(m)
    assert any(i.opcode == "multiply" for i in m.entry.instructions.values())


def test_triple_transpose_composes():
    def f(x):
        y = mnp.transpose(x)                      # (4,3)
        y = mnp.transpose(y)                      # (3,4)
        return mnp.transpose(y)                   # (4,3)
    m = _module(f, X)
    Fixpoint([AlgebraicSimplifier(), DCE()]).run(m)
    m.validate()
    ts = [i for i in m.entry.instructions.values() if i.opcode == "transpose"]
    assert len(ts) == 1, f"three transposes should compose to one, got {len(ts)}"
    assert m.entry[m.entry.root].shape.dims == (4, 3)


# ---------------------------------------------------------------- folding

def test_constant_folding_folds_real_arithmetic():
    def f(x):
        return mnp.multiply(x, mnp.exp(1.0))    # exp(constant) -> constant
    m = _module(f, X)
    assert any(i.opcode == "exp" for i in m.entry.instructions.values())
    assert ConstantFolding().run(m) is True
    DCE().run(m)
    assert all(i.opcode != "exp" for i in m.entry.instructions.values()),         "exp of a constant should have been folded away"


def test_constant_folding_skips_cheap_ops():
    """Folding broadcast(constant) materializes bytes AND blinds the simplifier."""
    m = _module(lambda x: mnp.multiply(x, 1.0), X)
    assert ConstantFolding().run(m) is False, \
        "broadcast/reshape/transpose must be excluded from folding"


# ---------------------------------------------------------------- pipeline

def test_fixpoint_converges_and_reports():
    def f(x):
        y = mnp.multiply(x, 1.0)
        y = mnp.add(y, 0.0)
        _ = mnp.exp(x)                   # dead
        return y
    m = _module(f, X)
    before = _n(m)
    pipe = default_pipeline()
    pipe.run(m)
    assert _n(m) < before
    assert "before_optimizations" in pipe.report()
    m.validate()


def test_optimization_preserves_semantics():
    def f(x):
        a = mnp.rms_norm(x, mnp.broadcast_to(1.0, (4,)))
        b = mnp.rms_norm(x, mnp.broadcast_to(1.0, (4,)))
        y = mnp.multiply(mnp.add(a, b), 1.0)
        return mnp.add(y, 0.0)
    fast = jit(f).lower(X).compile()
    slow = jit(f).lower(X).compile(optimize=False)
    assert np.abs(fast(X) - slow(X)).max() == 0.0
    assert _n(fast.module) < _n(slow.module)


def test_dump_writes_numbered_files():
    tmp = tempfile.mkdtemp()
    try:
        m = _module(lambda x: mnp.multiply(mnp.exp(x), 1.0), X)
        PassPipeline("p", dump_dir=tmp).add(AlgebraicSimplifier()).add(DCE()).run(m)
        files = sorted(os.listdir(tmp))
        assert len(files) == 3, files
        assert "0000.before_optimizations" in files[0]
        assert "0001.algebraic_simplifier" in files[1]
        assert "0002.dead_code_elimination" in files[2]
        # metadata is written into the dumps -- that is the point
        assert "//" in open(os.path.join(tmp, files[0]), encoding="utf-8").read()
    finally:
        shutil.rmtree(tmp)


def test_metadata_points_at_user_code():
    m = _module(lambda x: mnp.rms_norm(x, mnp.broadcast_to(1.0, (4,))), X)
    md = [i.metadata for i in m.entry.instructions.values() if i.metadata]
    assert md, "every instruction should carry metadata"
    files = {x.source_file for x in md}
    assert all(f.endswith(".py") for f in files)
    assert not any("numpy.py" in f or "trace.py" in f for f in files), \
        "metadata must point at USER code, not at mockxla internals"


def test_validate_runs_after_every_pass():
    class Broken(AlgebraicSimplifier):
        name = "broken"

        def run(self, module):
            comp = module.entry
            order = list(comp.instructions)
            comp.instructions = {k: comp.instructions[k] for k in reversed(order)}
            return True
    try:
        PassPipeline("p").add(Broken()).run(_module(lambda x: mnp.add(x, x), X))
    except ValueError as e:
        assert "before its definition" in str(e)
    else:
        raise AssertionError("the pipeline must validate after each pass")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
