"""Embedding-based scorer using OpenAI Embedding API."""

import logging

import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)


class EmbeddingScorer:
    """Embedding-based semantic similarity scorer.

    Attributes:
        client: OpenAI client instance.
        model_id: OpenAI embedding model identifier.
    """

    def __init__(self, api_key: str, base_url: str, model_id: str):
        """Initialize the embedding scorer.

        Args:
            api_key: OpenAI API key for authentication.
            base_url: Base URL for OpenAI API endpoint.
            model_id: OpenAI embedding model identifier (e.g., 'text-embedding-3-large').
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_id = model_id
        logger.info("Initialized EmbeddingScorer with model: %s", model_id)

    def _get_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for a single text.

        Args:
            text: Input text string to embed.

        Returns:
            Embedding as numpy array.
        """
        response = self.client.embeddings.create(
            input=[text], model=self.model_id
        )
        embedding = response.data[0].embedding
        return np.array(embedding).flatten()

    def __call__(self, prediction: str, ground_truth: str) -> float:
        """Compute semantic similarity score between prediction and ground truth.

        Args:
            prediction: Predicted answer string.
            ground_truth: Ground truth answer string.

        Returns:
            Cosine similarity score in range [-1, 1].
            Returns 0.0 if scoring fails.
        """
        try:
            pred_embedding = self._get_embedding(prediction)
            gt_embedding = self._get_embedding(ground_truth)

            # Compute cosine similarity
            dot_product = np.dot(pred_embedding, gt_embedding)
            pred_norm = np.linalg.norm(pred_embedding)
            gt_norm = np.linalg.norm(gt_embedding)

            if pred_norm == 0 or gt_norm == 0:
                logger.warning("Zero norm encountered in embeddings")
                return 0.0

            cosine_sim = dot_product / (pred_norm * gt_norm)

            return float(cosine_sim)

        except Exception as e:
            logger.error("Embedding scoring failed: %s", e)
            return 0.0
