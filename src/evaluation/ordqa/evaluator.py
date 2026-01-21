"""Multi-metric evaluator for ORD-QA benchmark.

This module provides an evaluator that computes BLEU, ROUGE-L, and optionally
UniEval scores for model predictions on EDA tool-related question answering tasks.
"""

import logging
from typing import Optional

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge

# Setup logger (Hydra will configure file handlers automatically)
logger = logging.getLogger(__name__)


class Evaluator:
    """Multi-metric evaluator for ORD-QA benchmark.

    Computes BLEU, ROUGE-L, and optionally UniEval scores for evaluating
    model-generated answers against ground truth answers.

    Attributes:
        unieval_scorer: Optional UniEval scorer instance for semantic evaluation.
        bleu_smooth: BLEU smoothing function to handle edge cases.
        rouge: ROUGE scorer for n-gram overlap metrics.
        current_question (Optional[str]): Current question for UniEval context.
        current_reference_content (Optional[list[str]]): Current reference documents
            for UniEval context.
    """

    def __init__(self, unieval_scorer: Optional[object] = None):
        """Initialize the evaluator.

        Args:
            unieval_scorer: Optional UniEval scorer instance. If None, UniEval
                scores default to 0.0.
        """
        self.unieval_scorer = unieval_scorer
        self.bleu_smooth = SmoothingFunction().method1
        self.rouge = Rouge()
        self.current_question = None
        self.current_reference_content = None

    def __call__(self, pred: str, gt: str) -> dict[str, float]:
        """Compute all evaluation metrics for a prediction vs ground truth pair.

        Args:
            pred: Predicted answer string.
            gt: Ground truth answer string.

        Returns:
            Dictionary with metric scores:
                - 'bleu': BLEU score (0-1)
                - 'rouge_l': ROUGE-L F1 score (0-1)
                - 'unieval': UniEval score (0-1), or 0.0 if unieval_scorer is None
        """
        bleu_score = self.bleu_score(pred, gt)
        rouge_l_score = self.rouge_l_score(pred, gt)
        unieval_score = self.unieval_score(pred, gt) if self.unieval_scorer else 0.0

        return {
            "bleu": bleu_score,
            "rouge_l": rouge_l_score,
            "unieval": unieval_score,
        }

    def bleu_score(self, pred: str, gt: str) -> float:
        """Compute sentence-level BLEU score.

        Uses 4-gram BLEU with uniform weights and smoothing for edge cases.

        Args:
            pred: Predicted text.
            gt: Ground truth text.

        Returns:
            BLEU score in range [0, 1].
        """
        bleu = sentence_bleu(
            [gt.split()],
            pred.split(),
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=self.bleu_smooth,
        )
        return bleu

    def rouge_l_score(self, pred: str, gt: str) -> float:
        """Compute ROUGE-L F1 score.

        ROUGE-L measures longest common subsequence between predicted and
        ground truth text.

        Args:
            pred: Predicted text.
            gt: Ground truth text.

        Returns:
            ROUGE-L F1 score in range [0, 1]. Returns 0 if computation fails.
        """
        try:
            rouge_scores = self.rouge.get_scores(hyps=[pred], refs=[gt])
            rouge_l = rouge_scores[0]["rouge-l"]["f"]
            return rouge_l
        except RecursionError as e:
            logger.warning(f"ROUGE-L computation failed with RecursionError: {e}")
            return 0.0
        except ValueError as e:
            logger.warning(f"ROUGE-L computation failed with ValueError: {e}")
            return 0.0

    def set_context(
        self, question: str, reference_content: Optional[list[str]] = None
    ) -> None:
        """Set context for UniEval evaluation.

        This method stores the current question and reference documents to be
        passed to UniEval scorer for context-aware evaluation.

        Args:
            question: The question being answered.
            reference_content: Optional list of reference documents.
        """
        self.current_question = question
        self.current_reference_content = reference_content

    def unieval_score(self, pred: str, gt: str) -> float:
        """Compute UniEval score for semantic evaluation.

        UniEval is a learned metric that evaluates the quality of generated text
        across multiple dimensions (coherence, consistency, fluency, relevance).
        Uses context set via set_context() method if available.

        Args:
            pred: Predicted text.
            gt: Ground truth text.

        Returns:
            UniEval score in range [0, 1].

        Raises:
            AttributeError: If unieval_scorer is None.
        """
        return self.unieval_scorer(
            pred, gt, self.current_question, self.current_reference_content
        )
