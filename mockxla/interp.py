"""The stage-1 backend: walk the IR and call numpy.

This is the CORRECTNESS half of mock-xla. It is deliberately dumb -- one numpy
call per instruction, every intermediate materialized. It will never be fast and
that is fine: stage 3 reports *simulated* cost from a roofline model, so speed
and correctness are measured by two different mechanisms and neither pollutes
the other.

Dumb-and-correct here is also what makes fusion legible later: an unfused
reference always exists to diff against.
"""

from __future__ import annotations

import numpy as np

from .ir import Module
from .ops import lookup


def execute(module: Module, *args, trace: bool = False) -> np.ndarray:
    comp = module.entry
    if len(args) != len(comp.params):
        raise TypeError(f"{module.name} expects {len(comp.params)} args, got {len(args)}")

    env: dict[str, np.ndarray] = {}

    for instr in comp.order():
        if instr.opcode == "parameter":
            arr = np.asarray(args[instr.attrs["index"]])
            want = instr.shape
            if arr.shape != want.dims:
                raise TypeError(f"{instr.id}: expected {want}, got shape {arr.shape}")
            env[instr.id] = arr.astype(want.dtype.np, copy=False)
        else:
            if instr.opcode == "fusion":
                # Correctness only: the body runs with intermediates
                # materialized. Fusion's BENEFIT is measured analytically by
                # cost.py, which counts only the fusion's boundary bytes -- so
                # performance and correctness stay measured by two separate
                # mechanisms, exactly as in stage 1.
                out = execute_fusion(module, instr,
                                     [env[o] for o in instr.operands])
            else:
                op = lookup(instr.opcode)
                inputs = [env[o] for o in instr.operands]
                out = op.impl(*inputs, **instr.attrs)
            out = np.asarray(out)
            if out.shape != instr.shape.dims:
                raise AssertionError(
                    f"{instr.id} ({instr.opcode}): shape rule said {instr.shape} "
                    f"but the impl produced {out.shape}. The two disagree -- fix one.")
            env[instr.id] = out

        if trace:
            print(f"  {instr.id:>16} {instr.opcode:<12} -> {env[instr.id].shape}")

    return env[comp.root]


def peak_live_bytes(module: Module) -> int:
    """Naive peak memory: buffers stay live until their last use.

    A preview of stage 5's buffer assignment. Nothing is reused and nothing is
    aliased, so this is an upper bound -- and the number fusion will reduce.
    """
    comp = module.entry
    last_use: dict[str, int] = {}
    order = comp.order()
    for i, instr in enumerate(order):
        for o in instr.operands:
            last_use[o] = i
    last_use[comp.root] = len(order)

    live: set[str] = set()
    peak = 0
    for i, instr in enumerate(order):
        live.add(instr.id)
        peak = max(peak, sum(comp[b].shape.size_bytes for b in live))
        live = {b for b in live if last_use.get(b, -1) > i}
    return peak


def execute_fusion(module: Module, instr, operands: list[np.ndarray]) -> np.ndarray:
    """Run a fusion's body. Parameters map positionally to the fusion's operands."""
    body = module.computations[instr.attrs["computation"]]
    env: dict[str, np.ndarray] = {}
    for i in body.order():
        if i.opcode == "parameter":
            env[i.id] = operands[i.attrs["index"]]
        else:
            env[i.id] = np.asarray(
                lookup(i.opcode).impl(*[env[o] for o in i.operands], **i.attrs))
    return env[body.root]
