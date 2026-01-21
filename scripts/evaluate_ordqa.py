"""ORD-QA benchmark evaluation script.

This script runs end-to-end evaluation of language models on the ORD-QA
benchmark. It uses Hydra for configuration management and supports optional
UniEval metric.

Usage:
    python scripts/evaluate_ordqa.py
    python scripts/evaluate_ordqa.py evaluation.use_reference=false
    python scripts/evaluate_ordqa.py evaluation.use_unieval=true
    python scripts/evaluate_ordqa.py model=qwen3_model
"""

import json
import logging
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from src.evaluation.ordqa.evaluator import Evaluator
from src.evaluation.ordqa.runner import Runner
from src.models.builder import build_model
from src.utils.set_seed import set_seed

# Load environment variables from .env file
load_dotenv()

# Setup logger
logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../configs",
    config_name="evaluate_ordqa",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Run ORD-QA benchmark evaluation.

    This function orchestrates the three-phase evaluation process:
    1. Inference: Generate model predictions for all questions
    2. Evaluation: Compute metrics (BLEU, ROUGE-L, optional UniEval)
    3. Results: Display aggregated scores by question type

    Args:
        cfg: Hydra configuration containing model and evaluation settings.
    """
    # Setup reproducibility
    set_seed(cfg.evaluation.seed)

    logger.info("=" * 80)
    logger.info("ORD-QA Benchmark Evaluation")
    logger.info("=" * 80)

    # GPU configuration: Override device if gpu_ids is specified
    if cfg.evaluation.get("gpu_ids") is not None:
        if isinstance(cfg.evaluation.gpu_ids, int):
            device = f"cuda:{cfg.evaluation.gpu_ids}"
            logger.info(f"GPU configuration: Using GPU {cfg.evaluation.gpu_ids}")
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
        logger.info(f"GPU configuration: Using default device from model config")

    # Build model from registry
    logger.info(f"Building model: {cfg.model.name}")
    logger.info(f"  Model: {cfg.model.params.model_name}")
    logger.info(f"  Device: {device}")
    model = build_model(cfg.model)

    # Conditionally create UniEval scorer
    unieval_scorer = None

    if cfg.evaluation.use_unieval:
        logger.info("=" * 80)
        logger.info("Initializing UniEval scorer")
        logger.info("=" * 80)

        try:
            from src.evaluation.ordqa.unieval_scorer import UniEvalScorer

            unieval_scorer = UniEvalScorer(
                device=device,
                max_length=cfg.evaluation.unieval.max_length,
                dimensions=list(cfg.evaluation.unieval.dimensions),
                use_reference=cfg.evaluation.use_reference,
                cache_dir=cfg.evaluation.unieval.get("cache_dir"),
            )
            logger.info("UniEval scorer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize UniEval scorer: {e}")
            logger.warning(
                "Continuing without UniEval. UniEval scores will default to 0.0"
            )
            unieval_scorer = None

    # Create evaluator
    evaluator = Evaluator(unieval_scorer=unieval_scorer)

    # Create runner
    runner = Runner(
        bleu_weight=cfg.evaluation.weights.bleu,
        rouge_l_weight=cfg.evaluation.weights.rouge_l,
        unieval_weight=cfg.evaluation.weights.unieval,
    )

    # Define output paths
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

    # Dataset path (relative to project root)
    dataset_path = Path(cfg.evaluation.dataset_path)
    if not dataset_path.is_absolute():
        # Convert to absolute path relative to project root
        project_root = Path(__file__).parent.parent
        dataset_path = project_root / dataset_path

    logger.info(f"Dataset path: {dataset_path}")

    # Phase 1: Inference (conditional)
    if cfg.evaluation.execution.run_inference:
        logger.info("=" * 80)
        logger.info("Phase 1: Running inference on ORD-QA dataset")
        logger.info(f"Using reference documents: {cfg.evaluation.use_reference}")
        if cfg.evaluation.max_samples is not None:
            logger.info(
                f"Testing mode: Evaluating only {cfg.evaluation.max_samples} samples"
            )
        else:
            logger.info("Full evaluation mode: Evaluating all samples")
        logger.info("=" * 80)

        preds = runner.run_inference(
            model,
            str(dataset_path),
            str(preds_path),
            use_reference=cfg.evaluation.use_reference,
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
        with open(preds_path, "r", encoding="utf-8") as f:
            preds = json.load(f)
        logger.info(f"Loaded {len(preds)} predictions")

    # Phase 2: Evaluation (conditional)
    if cfg.evaluation.execution.run_evaluation:
        logger.info("=" * 80)
        logger.info("Phase 2: Computing evaluation metrics")
        logger.info("=" * 80)
        logger.info(
            f"Metrics: BLEU (weight={cfg.evaluation.weights.bleu}), "
            f"ROUGE-L (weight={cfg.evaluation.weights.rouge_l}), "
            f"UniEval (weight={cfg.evaluation.weights.unieval}, "
            f"enabled={cfg.evaluation.use_unieval})"
        )

        results = runner.run_evaluation(
            preds, str(dataset_path), evaluator, str(results_path)
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
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        logger.info(f"Loaded results for {len(results)} questions")

    # Phase 3: Display Results (conditional)
    if cfg.evaluation.execution.run_results:
        logger.info("=" * 80)
        logger.info("Phase 3: Displaying results")
        logger.info("=" * 80)
        aggregated_scores = runner.show_results(results, str(dataset_path))
    else:
        logger.info("=" * 80)
        logger.info("Phase 3: Skipped (execution.run_results=false)")
        logger.info("=" * 80)

    logger.info("=" * 80)
    logger.info(f"Evaluation workflow complete. All outputs in {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
