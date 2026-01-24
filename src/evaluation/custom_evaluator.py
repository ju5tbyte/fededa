"""Custom evaluator using UniEval, embedding similarity, and LLM scoring."""

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class CustomEvaluator:
    """Custom evaluator using UniEval, embedding similarity, and LLM scoring.

    Attributes:
        unieval_scorer: Optional UniEval scorer instance for semantic evaluation.
        embedding_scorer: Optional embedding scorer instance for semantic similarity computation.
        llm_scorer: Optional LLM scorer instance for quality judgment.
        unieval_weight: Weight for UniEval score in weighted average.
        embedding_weight: Weight for embedding similarity score in weighted average.
        llm_weight: Weight for LLM score in weighted average.
    """

    def __init__(
        self,
        unieval_scorer: Optional[
            Callable[[str, str, Optional[str], Optional[list[str]]], float]
        ] = None,
        embedding_scorer: Optional[Callable[[str, str], float]] = None,
        llm_scorer: Optional[Callable[[str, str, str], float]] = None,
        unieval_weight: float = 1.0,
        embedding_weight: float = 1.0,
        llm_weight: float = 1.0,
    ):
        """Initialize the custom evaluator.

        Args:
            unieval_scorer: Optional UniEval scorer instance. If None, UniEval
                scores default to 0.0.
            embedding_scorer: Optional embedding scorer instance for semantic similarity.
                If None, similarity scores default to 0.0.
            llm_scorer: Optional LLM scorer instance for quality judgment.
                If None, LLM scores default to 0.0.
            unieval_weight: Weight for UniEval score in weighted average.
            embedding_weight: Weight for embedding similarity score in weighted average.
            llm_weight: Weight for LLM score in weighted average.
        """
        self.unieval_scorer = unieval_scorer
        self.embedding_scorer = embedding_scorer
        self.llm_scorer = llm_scorer
        self.unieval_weight = unieval_weight
        self.embedding_weight = embedding_weight
        self.llm_weight = llm_weight

        logger.info(
            "Initialized CustomEvaluator with weights: "
            "unieval=%.2f, embedding=%.2f, llm=%.2f",
            unieval_weight,
            embedding_weight,
            llm_weight,
        )

    def __call__(
        self,
        pred: str,
        gt: str,
        question: Optional[str] = None,
        reference_content: Optional[list[str]] = None,
    ) -> dict[str, float]:
        """Compute all evaluation metrics for a prediction vs ground truth pair.

        Args:
            pred: Predicted answer string.
            gt: Ground truth answer string.
            question: Optional question for UniEval and LLM scorer context.
            reference_content: Optional reference documents for UniEval context.

        Returns:
            Dictionary with metric scores:
                - 'unieval': UniEval score (0-1), or 0.0 if unieval_scorer is None
                - 'embedding': Embedding similarity score (0-1), or 0.0 if embedding_scorer is None
                - 'llm': LLM-based score (0-1), or 0.0 if llm_scorer is None
                - 'overall': Weighted average of all three scores
        """
        unieval_score = (
            self.unieval_scorer(pred, gt, question, reference_content)
            if self.unieval_scorer is not None
            else 0.0
        )
        embedding_score = (
            self.embedding_scorer(pred, gt)
            if self.embedding_scorer is not None
            else 0.0
        )
        llm_score = (
            self.llm_scorer(question or "", pred, gt)
            if self.llm_scorer is not None
            else 0.0
        )

        total_weight = (
            self.unieval_weight + self.embedding_weight + self.llm_weight
        )
        overall_score = (
            unieval_score * self.unieval_weight
            + embedding_score * self.embedding_weight
            + llm_score * self.llm_weight
        ) / total_weight

        return {
            "unieval": unieval_score,
            "embedding": embedding_score,
            "llm": llm_score,
            "overall": overall_score,
        }
