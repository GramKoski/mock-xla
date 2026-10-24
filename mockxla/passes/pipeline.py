"""The pass pipeline, and the dump machinery that makes it legible.

Mirrors XLA's `HloPassPipeline`. Two features carry the whole stage:

  * `dump_dir` writes the module after EVERY pass, numbered. Diffing consecutive
    files is how you watch the compiler work -- a node vanish, a collective
    appear, a fusion swallow three ops. Real XLA has `--xla_dump_to`; this is it.

  * `Fixpoint` re-runs a sub-pipeline until nothing changes, because passes feed
    each other: the simplifier creates dead nodes, DCE removes them, which
    exposes new simplifications. XLA nests pipelines the same way.

A Pass returns True iff it changed the module. That bool is the only contract,
and it is what makes a fixpoint loop possible.
"""

from __future__ import annotations

import os

from ..ir import Module
from ..printer import print_module, stats


class Pass:
    """Base class. Subclasses implement run() and return 'did I change it'."""

    name: str = "pass"

    def run(self, module: Module) -> bool:
        raise NotImplementedError

    def __repr__(self):
        return f"<{self.name}>"


class Fixpoint(Pass):
    """Run a list of passes until none of them report a change."""

    def __init__(self, passes: list[Pass], name="fixpoint", max_iters: int = 10):
        self.passes = passes
        self.name = name
        self.max_iters = max_iters

    def run(self, module: Module) -> bool:
        changed_ever = False
        for _ in range(self.max_iters):
            changed = False
            for p in self.passes:
                changed |= p.run(module)
            changed_ever |= changed
            if not changed:
                return changed_ever
        # Hitting the cap usually means two passes are undoing each other.
        raise RuntimeError(
            f"{self.name} did not converge in {self.max_iters} iterations -- "
            "two passes are probably fighting (one's output is the other's input)")


class PassPipeline:
    def __init__(self, name: str = "pipeline", dump_dir: str | None = None):
        self.name = name
        self.passes: list[Pass] = []
        self.dump_dir = dump_dir or os.environ.get("MOCKXLA_DUMP")
        self.log: list[tuple[str, int, bool]] = []   # (pass, instrs after, changed)

    def add(self, p: Pass) -> PassPipeline:
        self.passes.append(p)
        return self

    # -- dumping -----------------------------------------------------------

    def _dump(self, module: Module, index: int, label: str) -> None:
        if not self.dump_dir:
            return
        os.makedirs(self.dump_dir, exist_ok=True)
        # A traced lambda is named jit_<lambda>, and <> are illegal in Windows
        # filenames. Sanitize rather than crash on a perfectly valid module.
        safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in module.name)
        path = os.path.join(self.dump_dir, f"{safe}.{index:04d}.{label}.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(print_module(module, metadata=True))

    # -- running -----------------------------------------------------------

    def run(self, module: Module) -> Module:
        self._dump(module, 0, "before_optimizations")
        self.log = [("before_optimizations", len(module.entry.instructions), False)]

        for i, p in enumerate(self.passes, start=1):
            changed = p.run(module)
            # Validate after EVERY pass. A broken pass is found at the pass that
            # broke it, not three passes later where the symptom appears.
            module.validate()
            self._dump(module, i, p.name)
            self.log.append((p.name, len(module.entry.instructions), changed))

        return module

    def report(self) -> str:
        """What each pass actually did. The point of the whole exercise."""
        lines = [f"{'pass':<28}{'instrs':>8}{'delta':>8}"]
        prev = None
        for name, count, changed in self.log:
            delta = "" if prev is None else f"{count - prev:+d}"
            mark = "" if prev is None or changed else "  (no change)"
            lines.append(f"{name:<28}{count:>8}{delta:>8}{mark}")
            prev = count
        return "\n".join(lines)


def default_pipeline(dump_dir: str | None = None) -> PassPipeline:
    """The stage-2 pipeline. Ordering mirrors XLA's RunOptimizationPasses:
    simplify, then clean up, repeatedly, then a final DCE."""
    from .simplify import CSE, DCE, AlgebraicSimplifier, ConstantFolding

    pipe = PassPipeline("optimizations", dump_dir)
    pipe.add(Fixpoint([AlgebraicSimplifier(), ConstantFolding(), CSE(), DCE()],
                      name="simplification"))
    pipe.add(DCE())
    return pipe
