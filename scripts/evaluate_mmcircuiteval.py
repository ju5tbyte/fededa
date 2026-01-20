"""MMCircuitEval benchmark evaluation script.

This script runs end-to-end evaluation of Vision-Language Models on the
MMCircuitEval benchmark. It uses Hydra for configuration management and
supports optional API-based metrics (embedding similarity, LLM scoring).

Usage:
    python scripts/evaluate_mmcircuiteval.py
    python scripts/evaluate_mmcircuiteval.py evaluation.field=spec
    python scripts/evaluate_mmcircuiteval.py evaluation.use_embedder=true
"""

import json
import logging
import sys
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from src.evaluation.mmcircuiteval.evaluator import Evaluator
from src.evaluation.mmcircuiteval.runner import Runner
from src.models.builder import build_model
from src.utils.set_seed import set_seed

# Load environment variables from .env file
load_dotenv()

# Add external MMCircuitEval to path for embedder/llm_scorer imports
external_path = Path(__file__).parent.parent / "external" / "MMCircuitEval"
sys.path.insert(0, str(external_path))

# Setup logger
logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../configs",
    config_name="evaluate_mmcircuiteval",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Run MMCircuitEval benchmark evaluation.

    This function orchestrates the three-phase evaluation process:
    1. Inference: Generate model predictions for all questions
    2. Evaluation: Compute metrics (BLEU, ROUGE, optional embedding/LLM)
    3. Results: Display aggregated scores by ability and modality

    Args:
        cfg: Hydra configuration containing model and evaluation settings.
    """
    # Setup reproducibility
    set_seed(cfg.evaluation.seed)

    logger.info("=" * 80)
    logger.info("MMCircuitEval Benchmark Evaluation")
    logger.info("=" * 80)

    # GPU configuration: Override device if gpu_ids is specified
    if cfg.evaluation.get("gpu_ids") is not None:
        if isinstance(cfg.evaluation.gpu_ids, int):
            device = f"cuda:{cfg.evaluation.gpu_ids}"
            logger.info(
                f"GPU configuration: Using GPU {cfg.evaluation.gpu_ids}"
            )
        else:
            # For future multi-GPU support
            logger.warning(
                f"Multi-GPU configuration detected but not yet supported. "
                f"Using first GPU: {cfg.evaluation.gpu_ids[0]}"
            )
            device = f"cuda:{cfg.evaluation.gpu_ids[0]}"
        # Override model device configuration
        cfg.model.params.device = device
    else:
        device = cfg.model.params.device
        logger.info(
            f"GPU configuration: Using default device from model config"
        )

    # Build model from registry
    logger.info(f"Building model: {cfg.model.name}")
    logger.info(f"  Model: {cfg.model.params.model_name}")
    logger.info(f"  Device: {device}")
    model = build_model(cfg.model)

    # Conditionally create embedder and LLM scorer
    embedder = None
    llm_scorer = None

    if cfg.evaluation.use_embedder:
        logger.info("Initializing OpenAI embedder")
        from evaluation.modules.embedder import Embedder

        embedder = Embedder(
            api_key=cfg.evaluation.api.openai_api_key,
            base_url=cfg.evaluation.api.openai_base_url,
            model_id=cfg.evaluation.api.embedding_model,
        )
        logger.info(f"  Embedding model: {cfg.evaluation.api.embedding_model}")

    if cfg.evaluation.use_llm_scorer:
        logger.info("Initializing LLM scorer")
        from evaluation.modules.llm_scorer import LLMScorer

        llm_scorer = LLMScorer(
            api_key=cfg.evaluation.api.openai_api_key,
            base_url=cfg.evaluation.api.openai_base_url,
            model_id=cfg.evaluation.api.llm_model,
        )
        logger.info(f"  LLM scorer model: {cfg.evaluation.api.llm_model}")

    # Create evaluator
    evaluator = Evaluator(llm_scorer=llm_scorer, embedder=embedder)

    # Create runner
    runner = Runner(
        version=cfg.evaluation.version,
        bleu_weight=cfg.evaluation.weights.bleu,
        rouge_weight=cfg.evaluation.weights.rouge,
        emb_weight=cfg.evaluation.weights.emb,
        llm_weight=cfg.evaluation.weights.llm,
    )

    # Define output paths
    # Use hydra.core.hydra_config to get Hydra's runtime output directory
    from hydra.core.hydra_config import HydraConfig

    hydra_cfg = HydraConfig.get()

    # Allow overriding output directory for reusing existing outputs
    if cfg.evaluation.get("output_dir") is not None:
        output_dir = Path(cfg.evaluation.output_dir)
        logger.info(f"Using specified output directory: {output_dir}")
        # Create directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(hydra_cfg.runtime.output_dir)
        logger.info(f"Using Hydra output directory: {output_dir}")

    preds_path = output_dir / cfg.evaluation.output.predictions_file
    results_path = output_dir / cfg.evaluation.output.results_file

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Predictions file: {preds_path}")
    logger.info(f"Results file: {results_path}")

    # Phase 1: Inference (conditional)
    if cfg.evaluation.execution.run_inference:
        logger.info("=" * 80)
        logger.info(f"Phase 1: Running inference on field '{cfg.evaluation.field}'")
        if cfg.evaluation.max_samples is not None:
            logger.info(
                f"Testing mode: Evaluating only {cfg.evaluation.max_samples} samples"
            )
        else:
            logger.info("Full evaluation mode: Evaluating all samples")
        logger.info("=" * 80)
        preds = runner.runInference(
            model,
            cfg.evaluation.field,
            str(preds_path),
            cot=cfg.evaluation.cot,
            max_samples=cfg.evaluation.max_samples,
        )
        logger.info(f"Inference complete. Predictions saved to {preds_path}")
    else:
        logger.info("=" * 80)
        logger.info("Phase 1: Skipped (execution.run_inference=false)")
        logger.info("=" * 80)
        # Validate that predictions file exists
        if not preds_path.exists():
            error_msg = (
                f"Predictions file not found: {preds_path}\n"
                f"Cannot run evaluation without predictions. "
                f"Please run inference first or provide a valid predictions file."
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Load existing predictions
        logger.info(f"Loading existing predictions from {preds_path}")
        with open(preds_path, "r") as f:
            preds = json.load(f)
        logger.info(f"Loaded {len(preds)} predictions")

    # Phase 2: Evaluation (conditional)
    if cfg.evaluation.execution.run_evaluation:
        logger.info("=" * 80)
        logger.info("Phase 2: Computing evaluation metrics")
        logger.info("=" * 80)
        logger.info(
            f"Metrics: BLEU (weight={cfg.evaluation.weights.bleu}), "
            f"ROUGE (weight={cfg.evaluation.weights.rouge}), "
            f"Embedding (weight={cfg.evaluation.weights.emb}, enabled={cfg.evaluation.use_embedder}), "
            f"LLM (weight={cfg.evaluation.weights.llm}, enabled={cfg.evaluation.use_llm_scorer})"
        )
        results = runner.runEvaluation(
            preds, cfg.evaluation.field, evaluator, str(results_path)
        )
        logger.info(f"Evaluation complete. Results saved to {results_path}")
    else:
        logger.info("=" * 80)
        logger.info("Phase 2: Skipped (execution.run_evaluation=false)")
        logger.info("=" * 80)
        # Validate that results file exists
        if not results_path.exists():
            error_msg = (
                f"Results file not found: {results_path}\n"
                f"Cannot display results without evaluation. "
                f"Please run evaluation first or provide a valid results file."
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Load existing results
        logger.info(f"Loading existing results from {results_path}")
        with open(results_path, "r") as f:
            results = json.load(f)
        logger.info(f"Loaded results for {len(results)} questions")

    # Phase 3: Display Results (conditional)
    if cfg.evaluation.execution.run_results:
        logger.info("=" * 80)
        logger.info("Phase 3: Displaying results")
        logger.info("=" * 80)
        runner.showResults(results, cfg.evaluation.field)
    else:
        logger.info("=" * 80)
        logger.info("Phase 3: Skipped (execution.run_results=false)")
        logger.info("=" * 80)

    logger.info("=" * 80)
    logger.info(f"Evaluation workflow complete. All outputs in {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
