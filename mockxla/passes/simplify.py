"""DCE, CSE, algebraic simplification, constant folding.

All four are PATTERN REWRITERS with legality conditions and no cost model --
"if pattern P holds and the rewrite is legal, do it," on the author's belief
that the result is better. That is how most of XLA's ~69 optimization passes
work, and it is exactly why CollectivePipeliner could hoist an entire
transformer block without ever asking whether that was a good idea: a pass
that never computes worth has no mechanism to decline.

THE RECURRING ANNOYANCE, and it is real in XLA too: scalars never appear bare.
`x + 0` is `add(x, broadcast(constant(0)))`, so every algebraic pattern has to
look THROUGH a broadcast to find the constant. `_as_scalar_constant` is that
lookthrough, and writing one is the first thing you do in any HLO pass.
"""

from __future__ import annotations

import numpy as np

from ..dtypes import Shape
from ..ir import Computation, Instruction, Module
from ..ops import lookup
from .pipeline import Pass

# --------------------------------------------------------------- structural key


def _hashable(v):
    if isinstance(v, (list, tuple)):
        return tuple(_hashable(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _hashable(x)) for k, x in v.items()))
    if isinstance(v, np.ndarray):
        return ("ndarray", v.shape, v.tobytes())
    return v


def structural_key(instr: Instruction):
    """Identity for CSE: what the instruction COMPUTES, nothing else.

    Includes opcode, operand ids, attrs and result shape. Deliberately EXCLUDES
    the instruction's own id and its metadata -- two instructions from different
    source lines that compute the same value are redundant, and we want to merge
    them. Emmy's `Graph.structural_key()` documents the same exclusions.
    """
    return (instr.opcode,
            tuple(instr.operands),
            tuple(sorted((k, _hashable(v)) for k, v in instr.attrs.items())),
            instr.shape)


# --------------------------------------------------------------- helpers


def _as_scalar_constant(comp: Computation, iid: str):
    """If `iid` is a (possibly broadcast) scalar constant, return its value."""
    instr = comp[iid]
    if instr.opcode == "broadcast":
        inner = comp[instr.operands[0]]
        if inner.opcode == "constant" and inner.shape.rank == 0:
            return inner.attrs["value"]
        return None
    if instr.opcode == "constant" and instr.shape.rank == 0:
        return instr.attrs["value"]
    return None


def _is_value(comp: Computation, iid: str, want: float) -> bool:
    v = _as_scalar_constant(comp, iid)
    return v is not None and not isinstance(v, (list, tuple)) and float(v) == want


# --------------------------------------------------------------- DCE


class DCE(Pass):
    name = "dead_code_elimination"

    def run(self, module: Module) -> bool:
        comp = module.entry
        changed = False
        # Reverse order: removing a node can orphan its operands, and walking
        # backwards picks those up in the same sweep.
        for instr in reversed(list(comp.instructions.values())):
            if instr.id == comp.root or instr.opcode == "parameter":
                continue
            if not comp.users(instr.id):
                del comp.instructions[instr.id]
                changed = True
        return changed


# --------------------------------------------------------------- CSE


class CSE(Pass):
    name = "common_subexpression_elimination"

    def run(self, module: Module) -> bool:
        comp = module.entry
        seen: dict = {}
        changed = False
        # Topological order matters: by the time we key an instruction, its
        # operands have already been canonicalized, so equal computations have
        # literally equal operand ids.
        for instr in list(comp.instructions.values()):
            if instr.opcode == "parameter":
                continue
            key = structural_key(instr)
            first = seen.get(key)
            if first is None:
                seen[key] = instr.id
            else:
                comp.replace_all_uses(instr.id, first)
                changed = True
        return changed


# --------------------------------------------------------------- constant folding


class ConstantFolding(Pass):
    """Evaluate instructions whose operands are all constants.

    The size guard matters: folding a broadcast to [8192,8192] would bake 256 MB
    of literal into the module. Real XLA has the same limit for the same reason.
    """

    name = "constant_folding"
    MAX_ELEMENTS = 1024

    # Ops that are CHEAPER TO RECOMPUTE THAN TO STORE. Folding
    # broadcast(constant(64)) is "correct" -- the result really is a constant --
    # but it replaces one scalar with 8 materialized floats AND blinds
    # _as_scalar_constant, so the algebraic simplifier can no longer see through
    # it. A locally valid rewrite that is globally harmful: the same failure
    # class as CollectivePipeliner hoisting a transformer block. Real XLA's
    # HloConstantFolding carries the same exclusion list.
    SKIP = {"constant", "parameter", "iota", "broadcast",
            "reshape", "transpose", "convert"}

    def run(self, module: Module) -> bool:
        comp = module.entry
        changed = False
        for instr in list(comp.instructions.values()):
            if instr.opcode in self.SKIP:
                continue
            if not instr.operands:
                continue
            if any(comp[o].opcode != "constant" for o in instr.operands):
                continue
            if instr.shape.num_elements > self.MAX_ELEMENTS:
                continue
            op = lookup(instr.opcode)
            if op.impl is None:
                continue
            inputs = [np.asarray(comp[o].attrs["value"], dtype=comp[o].shape.dtype.np)
                      for o in instr.operands]
            out = np.asarray(op.impl(*inputs, **instr.attrs))
            folded = comp.add("constant", [], instr.shape,
                              {"value": out.tolist(), "shape": instr.shape},
                              metadata=instr.metadata)
            # Keep the new constant physically before its users.
            _move_before(comp, folded.id, instr.id)
            comp.replace_all_uses(instr.id, folded.id)
            changed = True
        return changed


def _move_before(comp: Computation, iid: str, before: str) -> None:
    """Reorder the instruction dict so `iid` precedes `before`.

    Necessary because `validate()` enforces def-before-use, and a pass that
    appends a new instruction appends it at the END.
    """
    instr = comp.instructions.pop(iid)
    rebuilt: dict = {}
    for k, v in comp.instructions.items():
        if k == before:
            rebuilt[iid] = instr
        rebuilt[k] = v
    if iid not in rebuilt:
        rebuilt[iid] = instr
    comp.instructions.clear()
    comp.instructions.update(rebuilt)


# --------------------------------------------------------------- algebraic


class AlgebraicSimplifier(Pass):
    name = "algebraic_simplifier"

    def run(self, module: Module) -> bool:
        comp = module.entry
        changed = False
        for instr in list(comp.instructions.values()):
            new = self._simplify(comp, instr)
            if new is not None and new != instr.id:
                comp.replace_all_uses(instr.id, new)
                changed = True
        return changed

    def _simplify(self, comp: Computation, instr: Instruction) -> str | None:
        """Return the id this instruction should be replaced BY, or None."""
        op = instr.opcode
        ops = instr.operands

        if op in ("add", "subtract"):
            if _is_value(comp, ops[1], 0.0):
                return ops[0]                       # x +- 0  ->  x
            if op == "add" and _is_value(comp, ops[0], 0.0):
                return ops[1]                       # 0 + x   ->  x

        if op == "multiply":
            if _is_value(comp, ops[1], 1.0):
                return ops[0]                       # x * 1   ->  x
            if _is_value(comp, ops[0], 1.0):
                return ops[1]                       # 1 * x   ->  x

        if op == "divide" and _is_value(comp, ops[1], 1.0):
            return ops[0]                           # x / 1   ->  x

        if op == "power" and _is_value(comp, ops[1], 1.0):
            return ops[0]

        if op == "reshape":
            inner = comp[ops[0]]
            if inner.shape.dims == instr.shape.dims:
                return ops[0]                       # reshape to the same shape
            if inner.opcode == "reshape":
                # reshape(reshape(x)) -> reshape(x): retarget at the inner source.
                instr.operands[0] = inner.operands[0]
                return None                          # mutated in place; DCE cleans up

        if op == "transpose":
            inner = comp[ops[0]]
            if inner.opcode == "transpose":
                outer_p = instr.attrs["permutation"]
                inner_p = inner.attrs["permutation"]
                composed = tuple(inner_p[p] for p in outer_p)
                if composed == tuple(range(len(composed))):
                    return inner.operands[0]        # transpose(transpose(x)) -> x
                instr.operands[0] = inner.operands[0]
                instr.attrs["permutation"] = composed
                return None

        if op == "convert" and comp[ops[0]].shape.dtype is instr.shape.dtype:
            return ops[0]                           # convert to the same dtype

        return None
