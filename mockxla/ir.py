"""The IR: Module -> Computation -> Instruction.

This mirrors XLA's HloModule / HloComputation / HloInstruction. Three properties
make it an SSA DATAFLOW graph rather than a tape:

  1. Every instruction has a unique id and is assigned exactly once.
  2. Operands are referenced BY ID, not by position in a recorded sequence.
  3. Shapes are attached to instructions, so the graph is self-describing and a
     pass can reason about it without re-running anything.

Rewrite passes mutate this structure in place -- there is no separate "program"
type, exactly as in XLA (`compiler/graph.py` in Emmy makes the same choice).
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .dtypes import Shape
from .metadata import OpMetadata


# The instruction object mostly makes sense. Each 
@dataclasses.dataclass
class Instruction:
    id: str                       # "%dot.7" -- unique within its Computation
    opcode: str                   # "dot_general", "add", "parameter", ...
    operands: list[str]           # ids of other instructions that produce values we use
    shape: Shape                  # the result shape, computed at trace time
    attrs: dict[str, Any] = dataclasses.field(default_factory=dict)
    # NOT part of structural identity: two instructions from different source
    # lines that compute the same thing ARE redundant, so CSE must ignore this.
    metadata: OpMetadata | None = None
    # Stage 3. ShardingPropagation fills these in; SpmdPartitioner reads them.
    # Mirrors XLA, where HloSharding is an attribute ON the instruction.
    dist: object | None = None            # a sharding.Dist
    global_shape: Shape | None = None     # shape BEFORE partitioning

    def __repr__(self):
        return f"{self.id} = {self.shape} {self.opcode}({', '.join(self.operands)})"


class Computation:
    """One function. The entry computation is the jitted function itself."""

    def __init__(self, name: str):
        self.name = name
        self.instructions: dict[str, Instruction] = {}   # insertion-ordered, map of instruction id to Instruction object involved in this computation
        self.params: list[str] = []                      # parameter ids, in order
        self.root: str | None = None
        self._next = 0 

    # -- construction ------------------------------------------------------

    def fresh_id(self, opcode: str) -> str:
        """HLO-style naming: opcode + a GLOBAL monotonically increasing suffix.

        Global, not per-opcode, so the numbers increase as you read a dump top to
        bottom. Real XLA does the same (%add.3, %fusion.12 come from a
        module-wide unique-name counter).
        """
        # Monotonic, NOT len(self.instructions): after a pass deletes a node the
        # length goes back down and ids start colliding (%all-gather.21 and
        # %reduce-scatter.21 both appeared in one module).
        self._next = max(self._next, len(self.instructions))
        n = self._next
        self._next += 1
        return f"%{opcode}.{n}"

    # Add computation 
    def add(self, opcode: str, operands: list[str], shape: Shape,
            attrs: dict | None = None, id: str | None = None,
            metadata: OpMetadata | None = None) -> Instruction:
        iid = id or self.fresh_id(opcode)
        instr = Instruction(iid, opcode, list(operands), shape,
                            dict(attrs or {}), metadata)
        self.instructions[iid] = instr
        if opcode == "parameter":
            self.params.append(iid)
        return instr

    # -- mutation, for rewrite passes --------------------------------------

    def replace_all_uses(self, old: str, new: str) -> int:
        """Point every consumer of `old` at `new`. Returns how many uses moved.

        This is the workhorse of every rewrite pass, and it is only this simple
        BECAUSE of SSA: operands are ids, so redirecting an edge is a string
        swap. Nothing holds a direct reference that could go stale.
        """
        n = 0
        for instr in self.instructions.values():
            for i, op in enumerate(instr.operands):
                if op == old:
                    instr.operands[i] = new
                    n += 1
        if self.root == old:
            self.root = new
        return n

    def remove(self, iid: str) -> None:
        if self.users(iid):
            raise ValueError(f"cannot remove {iid}: still used by "
                             f"{[u.id for u in self.users(iid)]}")
        if iid == self.root:
            raise ValueError(f"cannot remove the root {iid}")
        del self.instructions[iid]

    # -- queries -----------------------------------------------------------

    def __getitem__(self, iid: str) -> Instruction:
        return self.instructions[iid]

    def order(self) -> list[Instruction]:
        """Execution order. Insertion order from tracing is already topological."""
        return list(self.instructions.values())

    def comp_root_opcode(self) -> str:
        """Opcode of the root. Selects kInput vs kLoop for a fused computation."""
        return self.instructions[self.root].opcode if self.root else ""

    def users(self, iid: str) -> list[Instruction]:
        """Computed on demand. XLA maintains an index; at this scale we don't."""
        return [i for i in self.instructions.values() if iid in i.operands]

    def validate(self) -> None:
        """Every operand must be defined BEFORE its user.

        Cheap, and it is the invariant a buggy rewrite pass breaks first -- so
        stage 2 will call this after every pass.
        """
        seen: set[str] = set()
        for instr in self.instructions.values():
            for op in instr.operands:
                if op not in self.instructions:
                    raise ValueError(f"{instr.id} references undefined {op}")
                if op not in seen:
                    raise ValueError(f"{instr.id} uses {op} before its definition")
            seen.add(instr.id)
        if self.root not in self.instructions:
            raise ValueError(f"root {self.root} is not an instruction")


# What is the purpose of the Module class?
#: Global module counter. Real XLA names dumps module_0000.jit_f.<pass>.txt for
#: the same reason: one process can compile the SAME function several times (two
#: parallelism configs, two input shapes), and without a sequence number the
#: later compile silently overwrites the earlier one's dumps.
_MODULE_SEQ = [0]


class Module:
    """A Computation plus whatever sub-computations it refers to."""

    def __init__(self, name: str):
        self.name = name
        self.seq = _MODULE_SEQ[0]
        _MODULE_SEQ[0] += 1
        self.entry = Computation("main")
        #: Sub-computations. THIS is what Module was always for -- in stage 1 it
        #: held one thing and I said so. Stage 4 fills it: every fusion node
        #: owns a fused computation here, exactly as HloModule owns the
        #: computations behind kFusion / while / conditional / reduce.
        self.computations: dict[str, Computation] = {}

    def validate(self) -> None:
        self.entry.validate()

    def as_text(self) -> str:
        from .printer import print_module
        return print_module(self)

    def __str__(self):
        return self.as_text()
