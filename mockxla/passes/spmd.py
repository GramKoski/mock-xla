"""SPMD: sharding propagation and partitioning.

Runs FIRST in the pipeline (XLA: gpu_compiler.cc:1911, before
RunOptimizationPasses at 1928), so every later pass -- simplification, fusion,
the cost model -- sees per-device shapes with collectives already in the graph.

Two passes:

  ShardingPropagation   assigns a Dist to every instruction. Global shapes
                        untouched. Forward inference from annotated parameters,
                        plus explicit constraints.
  SpmdPartitioner       rewrites each instruction's shape to this device's shape
                        and INSERTS COLLECTIVES wherever an operand's actual
                        distribution differs from what the op requires.

Nothing here knows the words "FSDP" or "tensor parallel". Those are names for
patterns that FALL OUT of the resharding rules:

  forward: X[batch:fsdp, embed:*] @ W[embed:fsdp, *]
           contraction dim sharded on ONE side  ->  all-gather W      = FSDP fwd
  backward: dW = X^T @ dY, both sharded on batch
           contraction dim sharded on BOTH sides -> partial={fsdp}
           and W must END sharded -> all-reduce + dynamic-slice
           -> ReduceScatterCreator folds that into reduce-scatter   = FSDP bwd
  TP:      row-parallel W2[mlp:tp, embed:*], contraction sharded both sides
           -> partial={tp} -> all-reduce                            = Megatron

HONEST LIMITATION: propagation here is forward-only plus explicit constraints.
Real XLA's ShardingPropagation is bidirectional and iterative -- it also pushes
requirements BACKWARD from constrained values and runs to a fixpoint. Forward +
constraints is enough to express every parallelism strategy below, but it will
sometimes gather where a smarter inference would not have to.
"""

from __future__ import annotations

from ..dtypes import Shape
from ..ir import Computation, Module
from ..sharding import Dist, Mesh, local_shape
from .pipeline import Pass
from .simplify import _move_before

# ---------------------------------------------------------------- spec algebra


def _merge(a: tuple, b: tuple) -> tuple:
    """Combine two specs dim-by-dim, preferring sharded over replicated.

    Returns a spec in which no axis appears twice -- an axis on two dims is
    nonsense (you cannot split one mesh axis across two tensor dims), and a
    conflict is resolved by dropping the later occurrence, which makes the
    partitioner gather that operand instead.
    """
    out = []
    used: set = set()
    for x, y in zip(a, b):
        pick = x if x is not None else y
        if pick in used:
            pick = None
        if pick is not None:
            used.add(pick)
        out.append(pick)
    return tuple(out)


def _plain(spec: tuple) -> Dist:
    return Dist(spec, frozenset())


# ---------------------------------------------------------------- per-op rules

_ELEMENTWISE = {
    "add", "subtract", "multiply", "divide", "maximum", "minimum", "power",
    "exp", "log", "sqrt", "rsqrt", "tanh", "negate", "abs", "floor", "sign",
    "logistic", "compare", "select",
}

def _rule(instr, dists: list[Dist], mesh: Mesh):
    """How one op interacts with sharding. Returns TWO things:

        required : list[Dist]   -- what each OPERAND must look like for this op
                                   to be computable from local data alone
        out      : Dist         -- what the RESULT then looks like

    Who uses which half:

        ShardingPropagation  uses only `out`, to label every instruction.
                             It DISCARDS `required` -- which is exactly what
                             makes propagation forward-only. Consuming it, by
                             pushing requirements back onto producers and
                             iterating, is what bidirectional propagation means.
        SpmdPartitioner      uses BOTH: reshards any operand whose actual Dist
                             differs from `required`, then stamps `out`.

    `_rule` is PURE and EMITS NOTHING. It only states requirements; `reshard()`
    is the single function in this file that creates instructions. That split is
    why both passes can call the same rule, and why calling it twice is free.

    A requirement of `[*,*]` on an operand that is actually `[fsdp,*]` is how an
    all-gather comes to exist. A requirement that LEAVES both contraction dims
    sharded is how a partial sum -- and therefore an all-reduce or a
    reduce-scatter -- comes to exist.
    """
    op = instr.opcode
    rank = len(instr.global_shape.dims)

    if op in ("parameter", "constant", "iota"):
        return [], instr.dist or Dist.replicated(rank)

    if op in _ELEMENTWISE:
        # One common spec, and NO operand may still be partial: exp(a1+a2) is
        # not exp(a1)+exp(a2), so a partial sum has to be resolved first. (Even
        # for the linear ops we insist, rather than being clever.)
        spec = dists[0].spec
        for d in dists[1:]:
            spec = _merge(spec, d.spec)
        want = [_plain(spec)] * len(dists)
        return want, _plain(spec)

    if op == "broadcast":
        dims = instr.attrs["dimensions"]
        src = dists[0]
        spec = [None] * rank
        for i, d in enumerate(dims):
            spec[d] = src.spec[i]
        return [src.without_partial()], _plain(tuple(spec))

    if op == "transpose":
        perm = instr.attrs["permutation"]
        src = dists[0]
        return [src.without_partial()], _plain(tuple(src.spec[p] for p in perm))

    if op == "reshape":
        return _reshape_rule(instr, dists[0], rank)

    if op == "reduce":
        operand, init = dists[0], dists[1]
        red = set(instr.attrs["dimensions"])
        spec = tuple(a for i, a in enumerate(operand.spec) if i not in red)
        lost = {operand.spec[i] for i in red if operand.spec[i] is not None}
        return ([operand, Dist.replicated(0)],
                Dist(spec, frozenset(operand.partial | lost)))

    if op == "dot_general":
        return _dot_rule(instr, dists)

    if op == "convert":
        src = dists[0]
        return [src.without_partial()], _plain(src.spec)

    raise NotImplementedError(f"no SPMD rule for {op!r}")


def _reshape_rule(instr, src: Dist, rank: int):
    """Only handled case: inserting/removing size-1 dims.

    Anything else (a real re-layout) would need the partitioner to understand
    how elements move, which is genuinely hard. We require the operand
    replicated and produce a replicated result -- correct but a gather. Real
    XLA's ReshapeSharding has partial support and bails to replication too.
    """
    in_dims = instr.attrs.get("_in_dims")
    out_dims = instr.global_shape.dims
    if in_dims is None:
        return [Dist.replicated(src.rank)], Dist.replicated(rank)

    src_nonunit = [(i, d) for i, d in enumerate(in_dims) if d != 1]
    out_nonunit = [(i, d) for i, d in enumerate(out_dims) if d != 1]
    if [d for _, d in src_nonunit] != [d for _, d in out_nonunit]:
        return [Dist.replicated(src.rank)], Dist.replicated(rank)

    spec = [None] * rank
    for (si, _), (oi, _) in zip(src_nonunit, out_nonunit):
        spec[oi] = src.spec[si]
    return [src.without_partial()], _plain(tuple(spec))


def _dot_rule(instr, dists: list[Dist]):
    lhs, rhs = dists
    a = instr.attrs
    lc, rc = a["lhs_contracting"], a["rhs_contracting"]
    lb, rb = a.get("lhs_batch", ()), a.get("rhs_batch", ())

    l_spec = list(lhs.spec)
    r_spec = list(rhs.spec)
    partial: set = set()

    # Batch dims must agree on both sides; take the lhs's choice.
    for i, j in zip(lb, rb):
        pick = l_spec[i] if l_spec[i] is not None else r_spec[j]
        l_spec[i] = r_spec[j] = pick

    # Contracting dims: sharded on BOTH sides by the same axis is the good case
    # -- each device computes a partial sum. Sharded on one side only is a
    # mismatch, so require both replicated, which makes the partitioner gather.
    for i, j in zip(lc, rc):
        al, ar = l_spec[i], r_spec[j]
        if al is not None and al == ar:
            partial.add(al)
        else:
            l_spec[i] = r_spec[j] = None

    # Free dims survive into the output. An axis may not appear twice, so if
    # both sides carry the same axis on a free dim, drop the rhs's.
    lhs_skip, rhs_skip = set(lc) | set(lb), set(rc) | set(rb)
    out_spec = [l_spec[i] for i in lb]
    out_spec += [l_spec[i] for i in range(len(l_spec)) if i not in lhs_skip]
    used = {x for x in out_spec if x is not None} | partial
    for i in range(len(r_spec)):
        if i in rhs_skip:
            continue
        pick = r_spec[i]
        if pick in used:
            r_spec[i] = pick = None
        if pick is not None:
            used.add(pick)
        out_spec.append(pick)

    want = [Dist(tuple(l_spec), frozenset()), Dist(tuple(r_spec), frozenset())]
    return want, Dist(tuple(out_spec), frozenset(partial))


# ---------------------------------------------------------------- propagation


class ShardingPropagation(Pass):
    name = "sharding_propagation"

    def __init__(self, mesh: Mesh, in_specs: list[tuple] | None = None):
        self.mesh = mesh
        self.in_specs = in_specs

    def run(self, module: Module) -> bool:
        comp = module.entry
        # Stash global shapes and reshape input dims before anything is rewritten.
        for instr in comp.order():
            instr.global_shape = instr.shape
            if instr.opcode == "reshape":
                instr.attrs["_in_dims"] = comp[instr.operands[0]].shape.dims

        for k, pid in enumerate(comp.params):
            rank = comp[pid].shape.rank
            # Fully replicated if there is no in_specs for that parameter
            spec = (self.in_specs[k] if self.in_specs and k < len(self.in_specs)
                    else (None,) * rank)
            if len(spec) != rank:
                raise ValueError(
                    f"in_specs[{k}] has {len(spec)} entries for a rank-{rank} "
                    f"parameter {comp[pid].shape}")
            comp[pid].dist = _plain(tuple(spec))
        # This is where we actually set the dist that will be propogated to the output of this instruction based on the 
        # dists of the inputs to the instruction and the rule for that instruction.
        # Note that rules can both change the output dist of an instruction and demand a certain partioning of the input dists
        # Here we just set the output dist
        # Note that later in the SpmdPartioner pass, we look at the rule again to see how the inputs need to be sharded ("want")
        # And insert necessary collectives
        for instr in comp.order():
            if instr.opcode == "parameter":
                continue
            dists = [comp[o].dist for o in instr.operands]
            _, out = _rule(instr, dists, self.mesh)
            instr.dist = out
        return True


# ---------------------------------------------------------------- resharding


def _emit(comp: Computation, before: str, opcode: str, operands: list[str],
          shape: Shape, attrs: dict, dist: Dist, metadata) -> str:
    instr = comp.add(opcode, operands, shape, attrs, metadata=metadata)
    instr.dist = dist
    instr.global_shape = shape
    if before is not None:
        _move_before(comp, instr.id, before)
    return instr.id


def reshard(comp: Computation, mesh: Mesh, before: str, iid: str,
            cur: Dist, want: Dist, metadata=None) -> str:
    """Emit whatever collectives turn `cur` into `want`. The heart of stage 3."""
    if cur == want:
        return iid
    cur_shape = comp[iid].shape

    # 1. Resolve partial sums. If the target wants the result sharded on one of
    #    the partial axes, emit all-reduce THEN dynamic-slice -- the naive form,
    #    which ReduceScatterCreator later folds into one reduce-scatter. This is
    #    exactly the order real XLA's partitioner emits it in.
    if cur.partial:
        axes = tuple(sorted(cur.partial))
        iid = _emit(comp, before, "all-reduce", [iid], cur_shape,
                    {"replica_groups": mesh.replica_groups(axes),
                     "computation": "add", "axis": ",".join(axes)},
                    cur.without_partial(), metadata)
        cur = cur.without_partial()

    # 2. Gather any dim that is sharded now but must be whole.
    for i, axis in enumerate(cur.spec):
        if axis is not None and want.spec[i] is None:
            n = mesh.size(axis)
            dims = list(comp[iid].shape.dims)
            dims[i] *= n
            new_spec = list(cur.spec); new_spec[i] = None
            iid = _emit(comp, before, "all-gather", [iid],
                        Shape(tuple(dims), cur_shape.dtype),
                        {"replica_groups": mesh.replica_groups((axis,)),
                         "dimensions": (i,), "group_size": n, "axis": axis},
                        _plain(tuple(new_spec)), metadata)
            cur = _plain(tuple(new_spec))

    # 3. Slice any dim that is whole now but must be sharded. No communication:
    #    every device already holds the data and just keeps its own piece.
    for i, axis in enumerate(want.spec):
        if axis is not None and cur.spec[i] is None:
            n = mesh.size(axis)
            dims = list(comp[iid].shape.dims)
            dims[i] //= n
            new_spec = list(cur.spec); new_spec[i] = axis
            iid = _emit(comp, before, "dynamic-slice", [iid],
                        Shape(tuple(dims), cur_shape.dtype),
                        {"dimensions": (i,), "group_size": n, "axis": axis},
                        _plain(tuple(new_spec)), metadata)
            cur = _plain(tuple(new_spec))

    if cur != want:
        raise NotImplementedError(f"cannot reshard {cur} -> {want}")
    return iid


# ---------------------------------------------------------------- partitioner


class SpmdPartitioner(Pass):
    name = "spmd_partitioner"

    def __init__(self, mesh: Mesh, out_spec: tuple | None = None):
        self.mesh = mesh
        self.out_spec = out_spec

    def run(self, module: Module) -> bool:
        comp = module.entry
        mesh = self.mesh

        for instr in list(comp.order()):
            if instr.opcode == "parameter":
                instr.shape = local_shape(instr.global_shape, instr.dist, mesh)
                instr.attrs["shape"] = instr.shape
                continue
            if instr.opcode in ("all-reduce", "all-gather", "reduce-scatter",
                                "dynamic-slice"):
                continue                        # already partitioned by reshard()

            dists = [comp[o].dist for o in instr.operands]
            want, out = _rule(instr, dists, mesh)

            for k, (o, w) in enumerate(zip(list(instr.operands), want)):
                got = comp[o].dist
                if got != w:
                    instr.operands[k] = reshard(comp, mesh, instr.id, o, got, w,
                                                instr.metadata)

            instr.dist = out
            instr.shape = local_shape(instr.global_shape, out, mesh)
            if instr.opcode in ("constant", "iota") and "shape" in instr.attrs:
                instr.attrs["shape"] = instr.shape

        # Force the requested output distribution. For an FSDP weight update
        # this is the line that creates the gradient reduce-scatter.
        if self.out_spec is not None:
            root = comp[comp.root]
            want = _plain(tuple(self.out_spec))
            # before=None appends, so the reshard chain lands AFTER the old root
            new = reshard(comp, mesh, None, comp.root, root.dist, want,
                          root.metadata)
            comp.root = new
        return True
