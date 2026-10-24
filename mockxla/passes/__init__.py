"""Compiler passes."""

from .pipeline import Fixpoint, Pass, PassPipeline, default_pipeline
from .simplify import (CSE, DCE, AlgebraicSimplifier, ConstantFolding,
                       structural_key)

__all__ = ["Pass", "PassPipeline", "Fixpoint", "default_pipeline",
           "DCE", "CSE", "AlgebraicSimplifier", "ConstantFolding",
           "structural_key"]
