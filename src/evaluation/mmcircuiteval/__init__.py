"""MMCircuitEval evaluation module for FedEDA.

This module provides components for evaluating Vision-Language Models on the
MMCircuitEval benchmark. It wraps the external MMCircuitEval repository with
fixes and FedEDA-specific integrations.
"""

from src.evaluation.mmcircuiteval.evaluator import Evaluator
from src.evaluation.mmcircuiteval.formatter import (
    formatAnswer,
    formatModelOutput,
    formatScore,
)
from src.evaluation.mmcircuiteval.runner import Runner

__all__ = [
    "Evaluator",
    "Runner",
    "formatScore",
    "formatAnswer",
    "formatModelOutput",
]
