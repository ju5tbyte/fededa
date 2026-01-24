"""Evaluator for ORD-QA."""

import logging
from typing import Callable, Optional

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge

# Setup logger (Hydra will configure file handlers automatically)
logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator for ORD-QA benchmark.

    Attributes:
        unieval_scorer: Optional UniEval scorer instance for semantic evaluation.
        bleu_smooth: BLEU smoothing function to handle edge cases.
        rouge: ROUGE scorer for n-gram overlap metrics.
    """

    def __init__(
        self,
        unieval_scorer: Optional[
            Callable[[str, str, Optional[str], Optional[list[str]]], float]
        ] = None,
    ):
        """Initialize the evaluator.

        Args:
            unieval_scorer: Optional UniEval scorer instance. If None, UniEval
                scores default to 0.0.
        """
        self.unieval_scorer = unieval_scorer
        self.bleu_smooth = SmoothingFunction().method1
        self.rouge = Rouge()

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
            question: Optional question for UniEval context.
            reference_content: Optional reference documents for UniEval context.

        Returns:
            Dictionary with metric scores:
                - 'bleu': BLEU score (0-1)
                - 'rouge_l': ROUGE-L F1 score (0-1)
                - 'unieval': UniEval score (0-1), or 0.0 if unieval_scorer is None
        """
        bleu_score = self.bleu_score(pred, gt)
        rouge_l_score = self.rouge_l_score(pred, gt)
        unieval_score = self.unieval_score(
            pred, gt, question, reference_content
        )

        return {
            "bleu": bleu_score,
            "rouge_l": rouge_l_score,
            "unieval": unieval_score,
        }

    def bleu_score(self, pred: str, gt: str) -> float:
        """Compute sentence-level BLEU score. (4-gram with uniform weights)

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
        """Compute ROUGE-L F1 score. (longest common subsequence between predicted and ground truth text)

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
            logger.warning(
                "ROUGE computation failed with RecursionError: %s", e
            )
            return 0.0

    def unieval_score(
        self,
        pred: str,
        gt: str,
        question: Optional[str] = None,
        reference_content: Optional[list[str]] = None,
    ) -> float:
        """Compute UniEval score for semantic evaluation. (the quality of generated text evaluated by unieval-sum)

        Args:
            pred: Predicted text.
            gt: Ground truth text.
            question: Question string for context.
            reference_content: List of reference content strings for context.

        Returns:
            UniEval score in range [0, 1].

        Raises:
            AttributeError: If unieval_scorer is None.
        """
        if self.unieval_scorer is None:
            return 0.0

        return self.unieval_scorer(pred, gt, question, reference_content)
