"""Evaluator for MMCircuitEval."""

import logging
from typing import Callable, Optional

import numpy as np
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator for MMCircuitEval benchmark.

    Attributes:
        llm_scorer: Optional LLM scorer instance for answer quality evaluation.
        embedder: Optional embedder instance for semantic similarity computation.
        bleu_smooth: BLEU smoothing function to handle edge cases.
        rouge: ROUGE scorer for n-gram overlap metrics.
    """

    def __init__(
        self,
        llm_scorer: Optional[Callable[[str, str], float]] = None,
        embedder: Optional[Callable[[str], torch.Tensor | list[float]]] = None,
    ):
        """Initialize the evaluator.

        Args:
            llm_scorer: Optional LLM scorer instance. If None, LLM scores default to 0.0.
            embedder: Optional embedder instance. If None, embedding scores default to 0.0.
        """
        self.llm_scorer = llm_scorer
        self.embedder = embedder
        self.bleu_smooth = SmoothingFunction().method1
        self.rouge = Rouge()

    def __call__(self, pred: str, gt: str) -> dict[str, float]:
        """Compute all evaluation metrics for a prediction vs ground truth pair.

        Args:
            pred: Predicted answer string.
            gt: Ground truth answer string.

        Returns:
            Dictionary with metric scores:
                - 'bleu': BLEU score (0-1)
                - 'rouge': Average ROUGE score (0-1)
                - 'emb': Embedding similarity score (0-1), or 0.0 if embedder is None
                - 'llm': LLM-based score (0-1), or 0.0 if llm_scorer is None
        """
        bleu_score = self.BLEUScore(pred, gt)
        rouge_score = self.RougeScore(pred, gt)
        emb_score = self.embScore(pred, gt) if self.embedder else 0.0
        llm_score = self.llmScore(pred, gt) if self.llm_scorer else 0.0

        return {
            "bleu": bleu_score,
            "rouge": rouge_score,
            "emb": emb_score,
            "llm": llm_score,
        }

    def BLEUScore(self, pred: str, gt: str) -> float:
        """Compute sentence-level BLEU score. (4-gram BLEU with uniform weights)

        Args:
            pred: Predicted text.
            gt: Ground truth text.

        Returns:
            BLEU score in range [0, 1].
        """
        bleu_score = sentence_bleu(
            [gt.split()],
            pred.split(),
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=self.bleu_smooth,
        )
        return bleu_score

    def RougeScore(self, pred: str, gt: str) -> float:
        """Compute average ROUGE score. (ROUGE-1, ROUGE-2, and ROUGE-L averaged)

        Args:
            pred: Predicted text.
            gt: Ground truth text.

        Returns:
            Average ROUGE score in range [0, 1]. Returns 0 if computation fails.
        """
        try:
            rouge_score = self.rouge.get_scores(hyps=[pred], refs=[gt])
            rouge1_score = rouge_score[0]["rouge-1"]["f"]
            rouge2_score = rouge_score[0]["rouge-2"]["f"]
            rougel_score = rouge_score[0]["rouge-l"]["f"]
            return (rouge1_score + rouge2_score + rougel_score) / 3
        except RecursionError as e:
            logger.warning(
                "ROUGE computation failed with RecursionError: %s", e
            )
            return 0.0

    def embScore(self, pred: str, gt: str) -> float:
        """Compute embedding-based semantic similarity score.

        Args:
            pred: Predicted text.
            gt: Ground truth text.

        Returns:
            Cosine similarity score in range [-1, 1] (typically [0, 1] for text).

        Raises:
            AttributeError: If embedder is None.
        """
        if self.embedder is None:
            raise AttributeError(
                "Embedder is not set for embedding score computation."
            )

        pred_emb = self.embedder(pred)
        gt_emb = self.embedder(gt)
        return self._cosSim(pred_emb, gt_emb)

    def llmScore(self, pred: str, gt: str) -> float:
        """Compute LLM-based answer quality score.

        Args:
            pred: Predicted text.
            gt: Ground truth text.

        Returns:
            LLM-based quality score in range [0, 1].

        Raises:
            AttributeError: If llm_scorer is None.
        """
        if self.llm_scorer is None:
            raise AttributeError(
                "LLM scorer is not set for LLM-based score computation."
            )
        return self.llm_scorer(pred, gt)

    def _cosSim(
        self,
        vector1: torch.Tensor | list[float],
        vector2: torch.Tensor | list[float],
    ) -> float:
        """Compute cosine similarity between two vectors.

        Supports both PyTorch tensors and lists as input.

        Args:
            vector1: First vector (tensor or list).
            vector2: Second vector (tensor or list).

        Returns:
            Cosine similarity in range [-1, 1]. Returns 0.0 if either vector has zero magnitude.
        """
        if isinstance(vector1, torch.Tensor):
            vector1 = vector1.cpu().tolist()
        if isinstance(vector2, torch.Tensor):
            vector2 = vector2.cpu().tolist()

        vector1 = np.array(vector1).flatten()
        vector2 = np.array(vector2).flatten()

        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)

        if magnitude1 * magnitude2 == 0:
            return 0.0

        return float(dot_product / (magnitude1 * magnitude2))
