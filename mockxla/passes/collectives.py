"""Collective optimization. XLA: RunCollectiveOptimizationPasses, line 1935.

ReduceScatterCreator is the whole point of this file, and its real doc comment
in xla/backends/gpu/transforms/reduce_scatter_creator.h is one line:

    "Transforms dynamic-slice(all-reduce) to a reduce-scatter."

That is where FSDP's gradient reduce-scatter ACTUALLY comes from. The partitioner
does not emit one: when a partial sum has to end up sharded, the natural thing
to emit is "reduce the whole thing, then keep my slice". This pass then spots
that shape and rewrites it:

    before   all-reduce(g)  ->  2V(N-1)/N bytes       then throw away (N-1)/N of it
    after    reduce-scatter ->   V(N-1)/N bytes       a clean 2x

Which also means a slightly unusual sharding annotation can fail to match the
pattern, and you silently get the all-reduce version at twice the traffic. Worth
knowing, because nothing errors.
"""

from __future__ import annotations

from ..dtypes import Shape
from ..ir import Module
from .pipeline import Pass


class ReduceScatterCreator(Pass):
    name = "reduce_scatter_creator"

    def run(self, module: Module) -> bool:
        comp = module.entry
        changed = False

        for ds in list(comp.order()):
            if ds.opcode != "dynamic-slice":
                continue
            ar = comp[ds.operands[0]]
            if ar.opcode != "all-reduce":
                continue
            # The all-reduce must feed ONLY this slice. If anything else reads
            # the full reduced value, we still need it, so fusing would force a
            # recompute -- the same single-user condition fusion will use.
            if len(comp.users(ar.id)) != 1:
                continue
            # The slice must be over an axis the all-reduce actually reduced,
            # otherwise the two are unrelated and the rewrite is wrong.
            ar_axes = set(ar.attrs.get("axis", "").split(","))
            if ds.attrs["axis"] not in ar_axes:
                continue

            axis = ds.attrs["axis"]
            rs = comp.add("reduce-scatter", [ar.operands[0]], ds.shape,
                          {"replica_groups": ar.attrs["replica_groups"],
                           "computation": ar.attrs["computation"],
                           "dimensions": ds.attrs["dimensions"],
                           "group_size": ds.attrs["group_size"],
                           "axis": axis},
                          metadata=ds.metadata)
            rs.dist = ds.dist
            rs.global_shape = ds.global_shape
            _place_after(comp, rs.id, ar.operands[0])
            comp.replace_all_uses(ds.id, rs.id)
            changed = True

        return changed


def _place_after(comp, iid: str, after: str) -> None:
    """Move `iid` to sit immediately after `after`, preserving def-before-use."""
    instr = comp.instructions.pop(iid)
    rebuilt: dict = {}
    for k, v in comp.instructions.items():
        rebuilt[k] = v
        if k == after:
            rebuilt[iid] = instr
    if iid not in rebuilt:
        rebuilt[iid] = instr
    comp.instructions.clear()
    comp.instructions.update(rebuilt)
