"""Multi-metric evaluator for MMCircuitEval benchmark.

This module provides an evaluator that computes BLEU, ROUGE, embedding similarity,
and LLM-based scores for model predictions. API-based metrics (embedding, LLM)
are optional and can be disabled via configuration.
"""

import logging
from typing import Optional

import numpy as np
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge

# Setup logger (Hydra will configure file handlers automatically)
logger = logging.getLogger(__name__)


class Evaluator:
    """Multi-metric evaluator for MMCircuitEval benchmark.

    Computes BLEU, ROUGE, embedding similarity, and LLM-based scores.
    API-based metrics (embedding, LLM) are optional and return 0.0 when disabled.

    Attributes:
        llm_scorer: Optional LLM scorer instance for answer quality evaluation.
        embedder: Optional embedder instance for semantic similarity computation.
        bleu_smooth: BLEU smoothing function to handle edge cases.
        rouge: ROUGE scorer for n-gram overlap metrics.
    """

    def __init__(
        self, llm_scorer: Optional[object] = None, embedder: Optional[object] = None
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
        """Compute sentence-level BLEU score.

        Uses 4-gram BLEU with uniform weights and smoothing for edge cases.

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
        """Compute average ROUGE score.

        Computes ROUGE-1, ROUGE-2, and ROUGE-L F1 scores and returns their average.

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
            logger.warning(f"ROUGE computation failed with RecursionError: {e}")
            return 0.0

    def embScore(self, pred: str, gt: str) -> float:
        """Compute embedding-based semantic similarity score.

        Uses the embedder to convert texts to embeddings and computes
        cosine similarity between them.

        Args:
            pred: Predicted text.
            gt: Ground truth text.

        Returns:
            Cosine similarity score in range [-1, 1] (typically [0, 1] for text).

        Raises:
            AttributeError: If embedder is None.
        """
        pred_emb = self.embedder(pred)
        gt_emb = self.embedder(gt)
        return self.cosSim(pred_emb, gt_emb)

    def llmScore(self, pred: str, gt: str) -> float:
        """Compute LLM-based answer quality score.

        Uses an LLM to evaluate the quality of the predicted answer
        against the ground truth.

        Args:
            pred: Predicted text.
            gt: Ground truth text.

        Returns:
            LLM-based quality score in range [0, 1].

        Raises:
            AttributeError: If llm_scorer is None.
        """
        return self.llm_scorer(pred, gt)

    def cosSim(
        self, vector1: torch.Tensor | list[float], vector2: torch.Tensor | list[float]
    ) -> float:
        """Compute cosine similarity between two vectors.

        Supports both PyTorch tensors and lists as input.

        Args:
            vector1: First vector (tensor or list).
            vector2: Second vector (tensor or list).

        Returns:
            Cosine similarity in range [-1, 1]. Returns 0.0 if either vector has zero magnitude.
        """
        # Convert tensors to lists if needed
        if isinstance(vector1, torch.Tensor):
            vector1 = vector1.cpu().tolist()
        if isinstance(vector2, torch.Tensor):
            vector2 = vector2.cpu().tolist()

        # Convert to numpy arrays and flatten to 1D
        vector1 = np.array(vector1).flatten()
        vector2 = np.array(vector2).flatten()

        # Compute cosine similarity
        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)

        if magnitude1 * magnitude2 == 0:
            return 0.0

        return float(dot_product / (magnitude1 * magnitude2))
