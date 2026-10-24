"""Source attribution -- the answer to "where did this instruction come from?"

Real XLA instructions carry OpMetadata (source file, line, op name) and JAX
populates it, so an HLO dump can be mapped back to the Python that produced it:

    metadata={op_name="jit(f)/add" source_file="mlp.py" source_line=33}

Without it, HLO is unreadable: nine of the sixteen instructions in a traced
RMSNorm are broadcasts and reshapes that appear in no source file. With it, every
one of them names the line that caused it.

We capture the first stack frame OUTSIDE the mockxla package, which is the
equivalent of JAX's traceback filtering.
"""

from __future__ import annotations

import dataclasses
import os
import sys

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_DISABLED = bool(os.environ.get("MOCKXLA_NO_METADATA"))


@dataclasses.dataclass(frozen=True)
class OpMetadata:
    source_file: str
    source_line: int
    op_name: str          # the enclosing python function, e.g. "rms_norm"

    def __str__(self):
        return f'op_name="{self.op_name}" source="{self.source_file}:{self.source_line}"'

    def short(self) -> str:
        return f"{self.op_name} @ {self.source_file}:{self.source_line}"


def capture() -> OpMetadata | None:
    """First frame outside this package. None if disabled."""
    if _DISABLED:
        return None
    frame = sys._getframe(1)
    while frame is not None:
        filename = frame.f_code.co_filename
        if not filename.startswith(_PKG_DIR):
            return OpMetadata(os.path.basename(filename), frame.f_lineno,
                              frame.f_code.co_name)
        frame = frame.f_back
    return None
