"""HLO-style text output.

This is the primary debugging surface of the whole project. In stage 2 the pass
pipeline dumps this after every pass, and DIFFING consecutive dumps is how you
watch the compiler work -- a collective appearing, a fusion swallowing three ops.

Format deliberately tracks real HLO closely enough that reading an `xla_dump_to`
dump afterwards feels familiar.
"""

from __future__ import annotations

import numpy as np

from .ir import Computation, Instruction, Module

# Attributes that are structural rather than printable (the shape is already in
# the result type; `value` gets special handling; `index` prints positionally).
_HIDDEN = {"shape", "value", "index"}

#: Column where the `// metadata` comment starts. Fixed so dumps stay diffable.
_META_COL = 96


def _fmt_value(v) -> str:
    if isinstance(v, (tuple, list)):
        return "{" + ",".join(str(x) for x in v) + "}"
    if isinstance(v, float):
        if v == float("inf"):
            return "inf"
        if v == float("-inf"):
            return "-inf"
        return f"{v:g}"
    return str(v)


def _fmt_attrs(instr: Instruction) -> str:
    parts = []
    if instr.opcode == "parameter":
        return str(instr.attrs.get("index", 0))
    if instr.opcode == "constant":
        val = instr.attrs["value"]
        arr = np.asarray(val)
        if arr.size <= 8:
            return _fmt_value(val) if arr.ndim else _fmt_value(float(arr) if arr.dtype.kind == "f" else arr.item())
        return f"<{arr.size} values>"
    for k, v in instr.attrs.items():
        if k in _HIDDEN:
            continue
        if isinstance(v, tuple) and len(v) == 0:
            continue
        parts.append(f"{k}={_fmt_value(v)}")
    return ", ".join(parts)


def print_instruction(instr: Instruction, is_root: bool) -> str:
    operands = ", ".join(instr.operands)
    attrs = _fmt_attrs(instr)

    if instr.opcode in ("parameter", "constant", "iota"):
        inner = attrs
    elif attrs:
        inner = f"{operands}), {attrs}" if operands else f"{attrs}"
    else:
        inner = operands

    if instr.opcode in ("parameter", "constant", "iota"):
        body = f"{instr.opcode}({inner})"
    elif attrs and operands:
        body = f"{instr.opcode}({operands}), {attrs}"
    else:
        body = f"{instr.opcode}({inner})"

    prefix = "ROOT " if is_root else ""
    return f"  {prefix}{instr.id} = {instr.shape} {body}"


def print_computation(comp: Computation, metadata: bool = False) -> str:
    params = ", ".join(
        f"{comp[p].id.lstrip('%')}: {comp[p].shape}" for p in comp.params)
    out_shape = comp[comp.root].shape if comp.root else "()"
    lines = [f"ENTRY %{comp.name} ({params}) -> {out_shape} {{"]
    body = [(print_instruction(instr, instr.id == comp.root), instr)
            for instr in comp.order()]
    # FIXED column, deliberately. Padding to max(len(line)) made the metadata
    # column shift whenever the instruction count changed, so every line in two
    # dumps differed textually and `diff` showed the whole file as rewritten.
    # Dumps must be diffable; that is the entire point of writing them.
    width = _META_COL
    for text, instr in body:
        if metadata and instr.metadata is not None:
            lines.append(f"{text:<{width}}   // {instr.metadata.short()}")
        else:
            lines.append(text)
    lines.append("}")
    return "\n".join(lines)


def print_module(module: Module, metadata: bool = False) -> str:
    return (f"HloModule {module.name}\n\n"
            f"{print_computation(module.entry, metadata)}\n")


def stats(module: Module) -> str:
    """One-line summary, useful when diffing dumps across passes."""
    comp = module.entry
    counts: dict[str, int] = {}
    for instr in comp.order():
        counts[instr.opcode] = counts.get(instr.opcode, 0) + 1
    total = len(comp.instructions)
    top = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
    return f"{total} instructions  [{top}]"
