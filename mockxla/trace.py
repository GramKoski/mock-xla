"""Tracing: Python -> IR.

`@jit` runs your function once with ABSTRACT values. A Tracer carries a shape
and a dtype and no data, so every operation it participates in can only record
itself into the graph. Three consequences, all of which are the real JAX:

  * shapes must be static -- there is no data to make them depend on
  * `if x > 0` is impossible, so it raises (see Tracer.__bool__)
  * changing an input's shape is a different program, so we retrace and recache

Shape inference runs HERE, at trace time, not as a later pass. That means a
shape error points at the line of YOUR code that caused it.
"""

from __future__ import annotations

import functools

import numpy as np

from .dtypes import Shape, shape_of
from .ir import Computation, Module
from .metadata import capture
from .ops import lookup


class ConcretizationError(Exception):
    pass


# ---------------------------------------------------------------- tracer


class Tracer:
    """An abstract value: a shape and a dtype, plus where it came from."""

    __slots__ = ("_builder", "id", "shape")

    def __init__(self, builder: "Builder", id: str, shape: Shape):
        self._builder = builder         # What is this builder?
        self.id = id
        self.shape = shape

    @property
    def dtype(self):
        return self.shape.dtype

    @property
    def dims(self):
        return self.shape.dims

    @property
    def ndim(self):
        return self.shape.rank

    def __repr__(self):
        return f"Tracer({self.id}: {self.shape})"

    # -- operator sugar; all of it routes through the frontend so that the
    #    implicit-broadcast adapter runs in exactly one place.

    def _n(self):
        from . import numpy as mnp
        return mnp

    # What is this stuff right here?
    def __add__(self, o): return self._n().add(self, o)
    def __radd__(self, o): return self._n().add(o, self)
    def __sub__(self, o): return self._n().subtract(self, o)
    def __rsub__(self, o): return self._n().subtract(o, self)
    def __mul__(self, o): return self._n().multiply(self, o)
    def __rmul__(self, o): return self._n().multiply(o, self)
    def __truediv__(self, o): return self._n().divide(self, o)
    def __rtruediv__(self, o): return self._n().divide(o, self)
    def __neg__(self): return self._n().negate(self)
    def __pow__(self, o): return self._n().power(self, o)

    
    @property
    def T(self):
        return self._n().transpose(self)

    def reshape(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        return self._n().reshape(self, dims)

    def __bool__(self):
        raise ConcretizationError(
            f"cannot use a traced value ({self.shape}) in a Python branch.\n"
            "Control flow and shapes must be decided at trace time, because the "
            "tracer carries no data -- only a shape. Either hoist the condition "
            "out of the jitted function, or use mnp.select for a value-level choice."
        )

    __nonzero__ = __bool__

    def __float__(self):
        raise ConcretizationError("cannot convert a traced value to a Python float")

    def __iter__(self):
        raise ConcretizationError("cannot iterate a traced value")


# ---------------------------------------------------------------- builder


class Builder:
    """Accumulates instructions into a Computation while tracing."""

    def __init__(self, name: str = "jit"):
        self.module = Module(name)
        self.comp: Computation = self.module.entry

    # This feels like it answers my questions about how shape is propogated. So here shape_rule is used as the definition of how an op changes a shape, not a validator.
    # So it returns a tracer with a new shape with the instruction id of what it just outputted.
    def emit(self, opcode: str, operands: list[Tracer], **attrs) -> Tracer:
        """Run the op's shape rule, record the instruction, hand back a Tracer."""
        shapes = [o.shape for o in operands]
        shape = lookup(opcode).shape_rule(*shapes, **attrs)
        instr = self.comp.add(opcode, [o.id for o in operands], shape, attrs,
                              metadata=capture())
        return Tracer(self, instr.id, shape)

    def parameter(self, index: int, shape: Shape) -> Tracer:
        instr = self.comp.add("parameter", [], shape,
                              {"index": index, "shape": shape},
                              metadata=capture())
        return Tracer(self, instr.id, shape)


_STACK: list[Builder] = []


def current_builder() -> Builder:
    if not _STACK:
        raise RuntimeError(
            "no active trace. mockxla.numpy functions only work inside a @jit "
            "function -- there is no eager mode.")
    return _STACK[-1]


# ---------------------------------------------------------------- jit


class Executable:
    """A compiled module: the output of the pass pipeline, ready to execute."""

    pipeline = None          # the PassPipeline that produced it, for .report()

    def __init__(self, module: Module, param_shapes: list[Shape]):
        self.module = module
        self.param_shapes = param_shapes

    def as_text(self, metadata: bool = False) -> str:
        from .printer import print_module
        return print_module(self.module, metadata)

    mesh = None              # set by Lowered.partition()

    def __call__(self, *args):
        if self.mesh is not None:
            from .spmd_interp import execute_spmd
            return execute_spmd(self.module, self.mesh, *args)
        from .interp import execute
        return execute(self.module, *args)


class Lowered:
    """The traced-but-not-yet-optimized module, mirroring jax's .lower()."""

    def __init__(self, module: Module, param_shapes: list[Shape]):
        self.module = module
        self.param_shapes = param_shapes

    def as_text(self, metadata: bool = False) -> str:
        from .printer import print_module
        return print_module(self.module, metadata)

    def partition(self, mesh, in_specs=None, out_spec=None,
                  dump_dir: str | None = None) -> "Executable":
        """Stage 3: partition for `mesh`, then optimize. Returns an Executable
        whose __call__ runs the simulated multi-device interpreter."""
        from .passes import spmd_pipeline
        pipe = spmd_pipeline(mesh, in_specs, out_spec, dump_dir)
        pipe.run(self.module)
        exe = Executable(self.module, self.param_shapes)
        exe.pipeline = pipe
        exe.mesh = mesh
        return exe

    def compile(self, optimize: bool = True,
                dump_dir: str | None = None) -> Executable:
        """Run the pass pipeline. `optimize=False` keeps the raw trace, which is
        what the demos diff against to prove optimization is semantics-preserving."""
        if not optimize:
            return Executable(self.module, self.param_shapes)
        from .passes import default_pipeline
        pipe = default_pipeline(dump_dir)
        pipe.run(self.module)
        exe = Executable(self.module, self.param_shapes)
        exe.pipeline = pipe
        return exe


class Jitted:
    def __init__(self, fn):
        self.fn = fn
        functools.update_wrapper(self, fn)
        self._cache: dict[tuple, Executable] = {}

    # -- tracing -----------------------------------------------------------

    def _trace(self, shapes: list[Shape]) -> Module:
        builder = Builder(f"jit_{self.fn.__name__}")
        _STACK.append(builder)
        try:
            args = [builder.parameter(i, s) for i, s in enumerate(shapes)]
            out = self.fn(*args)
            if not isinstance(out, Tracer):
                raise TypeError(
                    f"{self.fn.__name__} returned {type(out).__name__}; a jitted "
                    "function must return a single traced array in stage 1.")
            builder.comp.root = out.id
        finally:
            _STACK.pop()
        builder.module.validate()
        return builder.module

    def lower(self, *args) -> Lowered:
        shapes = [shape_of(a) for a in args]
        return Lowered(self._trace(shapes), shapes)

    # -- the cached compile-and-run path ----------------------------------

    def __call__(self, *args):
        shapes = [shape_of(a) for a in args]
        key = tuple(shapes)
        exe = self._cache.get(key)
        if exe is None:
            # A new input shape is a NEW PROGRAM. This is why jax retraces.
            exe = self.lower(*args).compile()
            self._cache[key] = exe
        return exe(*args)

    def cache_size(self) -> int:
        return len(self._cache)


def jit(fn) -> Jitted:
    return Jitted(fn)
