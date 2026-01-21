"""UniEval scorer wrapper for ORD-QA benchmark.

This module provides a wrapper around the UniEval metric for evaluating
model-generated answers on the ORD-QA benchmark. It uses UniEval's
summarization evaluator to assess answer quality across multiple dimensions.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import nltk

# Add UniEval to Python path
UNIEVAL_PATH = Path(__file__).resolve().parents[3] / "external" / "UniEval"
if str(UNIEVAL_PATH) not in sys.path:
    sys.path.insert(0, str(UNIEVAL_PATH))

from metric.evaluator import get_evaluator
from utils import convert_to_json

# Setup logger
logger = logging.getLogger(__name__)


def _ensure_nltk_resources() -> None:
    """Download required NLTK resources if not already present.

    UniEval internally uses NLTK's punkt tokenizer for sentence segmentation.
    This function ensures the required resources are available.
    """
    required_resources = ["punkt", "punkt_tab"]

    for resource in required_resources:
        try:
            nltk.data.find(f"tokenizers/{resource}")
            logger.debug(f"NLTK resource '{resource}' already downloaded")
        except LookupError:
            logger.info(f"Downloading NLTK resource '{resource}'...")
            try:
                nltk.download(resource, quiet=True)
                logger.info(f"Successfully downloaded NLTK resource '{resource}'")
            except Exception as e:
                logger.warning(f"Failed to download NLTK resource '{resource}': {e}")
                logger.warning("Evaluation may fail if this resource is required")


class UniEvalScorer:
    """UniEval scorer for question answering evaluation.

    Uses UniEval's summarization evaluator to assess the quality of
    model-generated answers across multiple dimensions: coherence,
    consistency, fluency, and relevance.

    Attributes:
        evaluator: UniEval evaluator instance.
        dimensions (list[str]): Evaluation dimensions to compute.
        use_reference (bool): Whether to use reference documents in evaluation.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        max_length: int = 1024,
        dimensions: Optional[list[str]] = None,
        use_reference: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """Initialize UniEval scorer.

        Args:
            device: Device to run the model on (e.g., 'cuda:0', 'cpu').
            max_length: Maximum sequence length for UniEval model.
            dimensions: List of dimensions to evaluate. If None, uses all
                available dimensions: ['coherence', 'consistency', 'fluency', 'relevance'].
            use_reference: Whether to use reference documents in evaluation.
                Set to False for knowledge-only evaluation mode.
            cache_dir: Optional directory to cache the downloaded model.
        """
        logger.info("Initializing UniEval scorer for summarization task")
        logger.info(f"Device: {device}")
        logger.info(f"Max length: {max_length}")

        # Ensure NLTK resources are available
        _ensure_nltk_resources()

        # Get evaluator for summarization task
        # Note: UniEval automatically uses MingZhong/unieval-sum for summarization
        self.evaluator = get_evaluator(
            task="summarization",
            max_length=max_length,
            device=device,
            cache_dir=cache_dir,
        )

        # Default dimensions for ORD-QA
        self.dimensions = dimensions or ["coherence", "consistency", "fluency", "relevance"]
        self.use_reference = use_reference

        logger.info(f"UniEval initialized with dimensions: {self.dimensions}")
        logger.info(f"Using reference documents: {self.use_reference}")

    def __call__(
        self,
        pred: str,
        gt: str,
        question: Optional[str] = None,
        reference_content: Optional[list[str]] = None,
    ) -> float:
        """Compute UniEval score for a prediction.

        Args:
            pred: Model-generated answer.
            gt: Ground truth answer (reference).
            question: The question being answered (optional, used as context).
            reference_content: List of reference documents (optional).

        Returns:
            Overall UniEval score in range [0, 1].
        """
        # Prepare source document (question + references)
        if self.use_reference and reference_content:
            # Combine question with reference documents
            src = f"Question: {question}\n\nReferences:\n"
            for i, ref in enumerate(reference_content, 1):
                src += f"Doc{i}: {ref}\n"
        else:
            # Use only question as source
            src = f"Question: {question}" if question else ""

        # Convert to UniEval format
        data = convert_to_json(
            output_list=[pred],   # Model-generated answer
            src_list=[src],       # Question + references
            ref_list=[gt],        # Ground truth answer
        )

        # Evaluate
        try:
            scores = self.evaluator.evaluate(
                data,
                dims=self.dimensions,
                overall=True,
                print_result=False,
            )

            # Extract overall score
            overall_score = scores[0].get("overall", 0.0)

            # Log dimension-wise scores for debugging
            logger.debug(f"UniEval scores: {scores[0]}")

            return overall_score

        except Exception as e:
            logger.error(f"UniEval evaluation failed: {e}")
            return 0.0
