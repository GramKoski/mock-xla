"""mock-xla: a teaching reimplementation of XLA's shape, IR and pass machinery.

    import mockxla.numpy as mnp
    from mockxla import jit

    @jit
    def f(x, w):
        return mnp.matmul(mnp.rms_norm(x, w), w)

    f(x, w)                      # trace -> compile -> execute on numpy
    print(f.lower(x, w).as_text())   # the HLO
"""

from .dtypes import BF16, F16, F32, PRED, S32, DType, Shape
from .ir import Computation, Instruction, Module
from .ops import REGISTRY, OpDef, register
from .trace import ConcretizationError, Tracer, jit

__all__ = [
    "jit", "Tracer", "ConcretizationError",
    "Module", "Computation", "Instruction",
    "Shape", "DType", "F32", "F16", "BF16", "S32", "PRED",
    "REGISTRY", "OpDef", "register",
]
