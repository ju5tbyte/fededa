"""Output formatting utilities for MMCircuitEval evaluation.

This module provides functions for formatting model outputs, answers, and scores.
It fixes bugs present in the external MMCircuitEval formatter implementation.
"""

import json
from typing import Optional


def formatAnswer(
    answer: Optional[str], explanation: Optional[str], raw_pred: Optional[str] = None
) -> str:
    """Format answer and explanation into evaluation string.

    Args:
        answer: The answer text (can be None).
        explanation: The explanation text (can be None).
        raw_pred: The raw prediction text (can be None).

    Returns:
        Formatted string combining answer and explanation.
    """
    answer = "" if answer is None else answer
    explanation = "" if explanation is None else explanation
    raw_pred = "" if raw_pred is None else raw_pred

    if raw_pred != "":
        answer = raw_pred if answer == "" else answer

    formatted_answer = f"Answer: {answer}."
    if explanation != "":
        formatted_answer += f" Explanation: {explanation}."

    return formatted_answer


def formatModelOutput(out: Optional[str], version: str = "v1") -> tuple[Optional[str], Optional[str]]:
    """Parse model output to extract answer and explanation.

    Supports two output formats:
    - v1: JSON format with 'answer' and 'explanation' keys
    - v2: Markdown format with '### Answer ###' and '### Explanation ###' sections

    Args:
        out: The raw model output string.
        version: Output format version ('v1' or 'v2', default: 'v1').

    Returns:
        Tuple of (answer, explanation). Both can be None if parsing fails.

    Raises:
        NotImplementedError: If version is not 'v1' or 'v2'.
    """
    if out is None:
        return None, None

    if version == "v1":
        # Parse JSON format
        out = out.split("```json")[-1].split("```")[0].strip()
        try:
            pred = json.loads(out)
            return pred.get("answer", None), pred.get("explanation", None)
        except json.JSONDecodeError:
            return None, None

    elif version == "v2":
        # Parse markdown format
        answer = out.split("### Answer ###")[-1].split("### Explanation ###")[0].strip()
        explanation = out.split("### Explanation ###")[-1].strip()

        if answer == "":
            answer = None
        if explanation == "":
            explanation = None

        return answer, explanation

    else:
        raise NotImplementedError(f"Unsupported version: {version}")


def formatScore(
    score: dict[str, float],
    bleu_weight: float = 1.0,
    rouge_weight: float = 1.0,
    emb_weight: float = 1.0,
    llm_weight: float = 2.0,
) -> float:
    """Compute weighted aggregated score from individual metrics.

    This function fixes two bugs in the external implementation:
    1. Adds default parameter values to support single-argument calls
    2. Uses 'emb' key instead of 'embedding' to match evaluator output

    Args:
        score: Dictionary with 'bleu', 'rouge', 'emb', 'llm' keys.
        bleu_weight: Weight for BLEU metric (default: 1.0).
        rouge_weight: Weight for ROUGE metric (default: 1.0).
        emb_weight: Weight for embedding similarity metric (default: 1.0).
        llm_weight: Weight for LLM-based metric (default: 2.0).

    Returns:
        Weighted average score in the range [0, 1].

    Raises:
        KeyError: If required keys are missing from score dictionary.
    """
    total_score = (
        bleu_weight * score["bleu"]
        + rouge_weight * score["rouge"]
        + emb_weight * score["emb"]  # FIXED: was 'embedding' in external code
        + llm_weight * score["llm"]
    )
    total_weight = bleu_weight + rouge_weight + emb_weight + llm_weight

    return total_score / total_weight
