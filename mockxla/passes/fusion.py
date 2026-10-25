"""PriorityFusion, after xla/backends/gpu/transforms/priority_fusion.

The real pass's doc comment is the whole algorithm:

    "PriorityFusion is the main fusion pass for XLA:GPU. It assigns a priority
     to each producer instruction based on the estimated performance benefit of
     fusing it into its consumers... calculated using a performance cost model:

         priority = time_unfused - time_fused"

and from gpu_performance_model.cc, what those two actually are:

    time_unfused = 2 * kKernelLaunchOverhead + producer.exec_time + consumer.exec_time
    time_fused   = 1 * kKernelLaunchOverhead + EstimateRunTimeForFusion(...)

Note the launch overheads: 2 kernels become 1. At 1 us each that is a real term
for small kernels, and it is why XLA fuses things whose memory traffic barely
changes.

WHY A PRIORITY QUEUE rather than one greedy traversal: fusing A into B changes
whether fusing C into AB is profitable. A single pass decides each pair once,
with stale information. So: pop the best candidate, fuse it, then RE-SCORE its
neighbours and push them back. That re-scoring is the entire difference from
the older greedy InstructionFusion, and it is why the pass is named for the
queue rather than for fusion.

WHAT FUSION DOES AND DOES NOT CHANGE
    FLOPs   unchanged (unless a producer is duplicated into several consumers)
    BYTES   every intermediate stops touching HBM  <- the whole benefit
    kernels N launches become 1
"""

from __future__ import annotations

import heapq

from ..cost import Device, H100, bytes_accessed, combine, flops_of
from ..dtypes import Shape
from ..ir import Computation, Module
from .pipeline import Pass

#: Ops whose output can be computed one element at a time from their inputs, so
#: they can live inside somebody else's loop. Matmuls and collectives cannot:
#: a dot has its own tiled loop structure and a collective has to synchronize.
FUSIBLE = {
    "add", "subtract", "multiply", "divide", "maximum", "minimum", "power",
    "exp", "log", "sqrt", "rsqrt", "tanh", "negate", "abs", "floor", "sign",
    "logistic", "compare", "select", "broadcast", "convert",
}
#: Fusible only as the ROOT of a fusion, never in the middle: a reduce's loop
#: nest is driven by its INPUT shape, so producers fuse into it but it cannot
#: fuse into an elementwise consumer without forcing the naive emitter.
#: This is precisely the kLoop / kInput distinction.
ROOT_ONLY = {"reduce"}

NEVER = {"parameter", "constant", "dot_general", "iota", "all-reduce",
         "all-gather", "reduce-scatter", "dynamic-slice"}


# ---------------------------------------------------------------- the estimate


def _instr_time(instr, comp, module, device) -> float:
    from ..cost import time_of
    return time_of(instr, comp, module, device)["total"]


def _fused_time(producer, consumer, comp, module, device: Device) -> float:
    """Cost of the hypothetical fused kernel.

    Bytes: the consumer's operands MINUS the producer's output, PLUS the
    producer's operands, plus the consumer's output. The producer's output is
    what stops existing -- that single term is the saving.
    """
    flops = _body_flops(producer, comp, module) + _body_flops(consumer, comp, module)

    ins = {o for o in consumer.operands if o != producer.id}
    ins |= set(producer.operands)
    mem = sum(comp[o].shape.size_bytes for o in ins) + consumer.shape.size_bytes

    t_c = flops / device.flops_for(consumer.shape.dtype)
    t_m = mem / device.hbm_bandwidth
    return combine(t_c, t_m, device) + device.launch_overhead


def _body_flops(instr, comp, module) -> float:
    if instr.opcode == "fusion":
        body = module.computations[instr.attrs["computation"]]
        return sum(flops_of(i, body) for i in body.order())
    return flops_of(instr, comp)


def benefit(producer, consumer, comp, module, device) -> float:
    """priority = time_unfused - time_fused. Positive means fuse."""
    t_unfused = (_instr_time(producer, comp, module, device)
                 + _instr_time(consumer, comp, module, device))
    return t_unfused - _fused_time(producer, consumer, comp, module, device)


# ---------------------------------------------------------------- legality


def _fusible_into(producer, consumer, comp) -> bool:
    if producer.opcode in NEVER or consumer.opcode in NEVER:
        return False
    if producer.opcode in ROOT_ONLY:
        return False                      # a reduce can only be a fusion's root
    if producer.opcode not in FUSIBLE and producer.opcode != "fusion":
        return False
    if consumer.opcode not in FUSIBLE | ROOT_ONLY and consumer.opcode != "fusion":
        return False
    # Exactly one user, or fusing DUPLICATES the producer into every consumer.
    # That is legal and sometimes profitable, and XLA will do it -- but the cost
    # model then has to price the extra FLOPs, so we decline here and say so.
    if len(comp.users(producer.id)) != 1:
        return False
    return True


# ---------------------------------------------------------------- the rewrite


class _Body:
    """Builds a fused Computation and the outer operand list it needs."""

    def __init__(self, name: str):
        self.comp = Computation(name)
        self.outer: list[str] = []        # outer ids, positionally = body params
        self._param_of: dict[str, str] = {}

    def param_for(self, outer_id: str, shape: Shape) -> str:
        if outer_id not in self._param_of:
            p = self.comp.add("parameter", [], shape,
                              {"index": len(self.outer), "shape": shape})
            self._param_of[outer_id] = p.id
            self.outer.append(outer_id)
        return self._param_of[outer_id]

    def copy_in(self, instr, comp, inner_for: dict) -> str:
        """Copy one outer instruction into the body, mapping its operands."""
        ops = []
        for o in instr.operands:
            if o in inner_for:
                ops.append(inner_for[o])
            else:
                ops.append(self.param_for(o, comp[o].shape))
        new = self.comp.add(instr.opcode, ops, instr.shape, dict(instr.attrs),
                            metadata=instr.metadata)
        return new.id


def _kind_of(body: Computation) -> str:
    """kInput if the root is a reduce (its INPUT shape drives the loop nest),
    else kLoop (the OUTPUT shape drives it). Exactly XLA's distinction, and
    exactly what selects the reduction emitter over the loop emitter."""
    return "kInput" if body.comp_root_opcode() == "reduce" else "kLoop"


def _materialize(comp: Computation, module: Module, consumer, producer,
                 device) -> str:
    """Replace producer+consumer with one `fusion` instruction."""
    name = f"fused_{len(module.computations)}"
    body = _Body(name)
    inner: dict = {}

    members = []
    for instr, is_fusion in ((producer, producer.opcode == "fusion"),
                             (consumer, consumer.opcode == "fusion")):
        if is_fusion:
            members.append(module.computations[instr.attrs["computation"]])
        else:
            members.append(instr)

    # Inline the producer first (its results feed the consumer), then the
    # consumer. Order matters: def before use, inside the body too.
    for outer, mem in ((producer, members[0]), (consumer, members[1])):
        if isinstance(mem, Computation):
            local: dict = {}
            for i in mem.order():
                if i.opcode == "parameter":
                    src = outer.operands[i.attrs["index"]]
                    local[i.id] = (inner[src] if src in inner
                                   else body.param_for(src, comp[src].shape))
                else:
                    ops = [local[o] for o in i.operands]
                    new = body.comp.add(i.opcode, ops, i.shape, dict(i.attrs),
                                        metadata=i.metadata)
                    local[i.id] = new.id
            inner[outer.id] = local[mem.root]
        else:
            inner[outer.id] = body.copy_in(mem, comp, inner)

    body.comp.root = inner[consumer.id]
    module.computations[name] = body.comp

    fused = comp.add("fusion", body.outer, consumer.shape,
                     {"computation": name, "kind": _kind_of(body.comp),
                      "out_shape": consumer.shape},
                     metadata=consumer.metadata)
    # Do we have to check that dist is the same in order to fuse?
    fused.dist = consumer.dist
    fused.global_shape = consumer.global_shape
    # Okay so we add the new computation of the producer (could be a fusion itself) into the consumer op (could also be a fusion)
    # Then we replace all uses of the consumer's data with the fused data. 
    # The old ops (unfused) will be cleaned up by the DCE
    _place_after(comp, fused.id, _last_of(comp, body.outer) or consumer.id)
    comp.replace_all_uses(consumer.id, fused.id)
    return fused.id


def _last_of(comp: Computation, ids: list[str]) -> str | None:
    order = list(comp.instructions)
    present = [i for i in ids if i in order]
    return max(present, key=order.index) if present else None


def _place_after(comp: Computation, iid: str, after: str) -> None:
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


# ---------------------------------------------------------------- the pass


class PriorityFusion(Pass):
    name = "priority_fusion"

    def __init__(self, device: Device | None = None, min_benefit: float = 0.0):
        self.device = device or H100
        self.min_benefit = min_benefit
        # Q: What is this data structure?
        #: One (producer_id, consumer_id, benefit_seconds) per fusion performed,
        #: in the order performed. Purely diagnostic -- report() formats it, and
        #: nothing in the algorithm reads it. Real XLA does the same thing with
        #: VLOG(8) << "Unfused time: " ... << "Fused time: ".
        self.log: list[tuple[str, str, float]] = []
        #: How many popped heap entries turned out to be stale. If this is 0,
        #: the incremental update is wrong (or the graph is trivially small).
        self.stale = 0

    def run(self, module: Module) -> bool:
        comp = module.entry
        changed = False
        self.log = []
        self.stale = 0          # how many popped entries were out of date

        def pair(pid: str, cid: str):
            """Score one producer->consumer pair, or None if not a candidate."""
            if pid not in comp.instructions or cid not in comp.instructions:
                return None
            p, c = comp[pid], comp[cid]
            if not _fusible_into(p, c, comp):
                return None
            b = benefit(p, c, comp, module, self.device)
            return (-b, pid, cid) if b > self.min_benefit else None

        def candidates_for(pid: str):
            """Entries where `pid` is the PRODUCER."""
            if pid not in comp.instructions:
                return []
            users = comp.users(pid)
            if len(users) != 1:
                return []
            e = pair(pid, users[0].id)
            return [e] if e else []

        def affected_by(iid: str):
            """After creating `iid`, only these pairs can have changed:
            iid as a producer, and each of iid's operands as a producer."""
            out = list(candidates_for(iid))
            if iid in comp.instructions:
                for o in comp[iid].operands:
                    out += candidates_for(o)
            return out

        heap = [e for p in comp.order() for e in candidates_for(p.id)]
        heapq.heapify(heap)
        rounds = 0

        while heap:
            negb, pid, cid = heapq.heappop(heap)

            # THESE TWO GUARDS ARE LOAD-BEARING, because the heap is updated
            # INCREMENTALLY: an entry scored N fusions ago is still sitting
            # there. Two ways it goes stale:
            #
            #   deleted    the consumer was absorbed into some other fusion, so
            #              cid is gone entirely
            #   illegal    both still exist, but an earlier replace_all_uses
            #              pointed a second instruction at the producer. Now
            #              users(p) == 2 and fusing would DUPLICATE it. Without
            #              this recheck we would silently duplicate.
            #   repriced   both exist and are legal, but the consumer is now a
            #              bigger fusion with a different byte profile, so the
            #              benefit is not what we recorded.
            fresh = pair(pid, cid)
            if fresh is None or fresh[0] != negb:
                self.stale += 1
                if fresh is None:
                    continue
                # re-scored: push the corrected entry back and re-pop, so the
                # queue order stays honest
                heapq.heappush(heap, fresh)
                continue

            p, c = comp[pid], comp[cid]
            new_id = self._fuse(comp, module, p, c, -negb)
            changed = True
            rounds += 1
            if rounds > 500:
                raise RuntimeError("priority fusion did not settle")

            # Push ONLY what the rewrite could have changed.
            for e in affected_by(new_id):
                heapq.heappush(heap, e)

        return changed

    def _fuse(self, comp, module, producer, consumer, b) -> str:
        self.log.append((producer.id, consumer.id, b))
        new_id = _materialize(comp, module, consumer, producer, self.device)
        del comp.instructions[consumer.id]
        if not comp.users(producer.id):
            del comp.instructions[producer.id]
        return new_id

    def report(self) -> str:
        if not self.log:
            return "priority_fusion: nothing fused"
        head = (f"priority_fusion: {len(self.log)} fusions, "
                f"{self.stale} stale heap entries skipped or repriced")
        lines = [head, f"{'producer':<22}{'into consumer':<22}{'benefit us':>12}"]
        for p, c, b in self.log:
            lines.append(f"{p:<22}{c:<22}{b*1e6:>12.3f}")
        return "\n".join(lines)
