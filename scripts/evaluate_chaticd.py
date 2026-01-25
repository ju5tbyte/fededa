"""ChatICD-Bench benchmark evaluation script.

Usage:
    python scripts/evaluate_chaticd.py
    python scripts/evaluate_chaticd.py model=qwen3_model model.params.model_name=Qwen/Qwen3-8B evaluation.gpu_id=4
"""

import json
import logging
from pathlib import Path

import hydra
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.evaluation.chaticd.runner import Runner
from src.evaluation.custom_evaluator import CustomEvaluator
from src.evaluation.embedding_scorer import EmbeddingScorer
from src.evaluation.llm_scorer import LLMScorer
from src.evaluation.unieval_scorer import UniEvalScorer
from src.models.builder import build_model
from src.utils.set_seed import set_seed

load_dotenv()

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../configs",
    config_name="evaluate_chaticd",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Run ChatICD-Bench benchmark evaluation.

    1. Inference: Generate model predictions for all questions
    2. Evaluation: Compute metrics (UniEval, Embedding Similarity, LLM Judge)
    3. Results: Display aggregated scores

    Args:
        cfg: Hydra configuration containing model and evaluation settings.
    """
    set_seed(cfg.evaluation.seed)

    logger.info("=" * 80)
    logger.info("ChatICD-Bench Benchmark Evaluation")
    logger.info("=" * 80)

    gpu_id = cfg.evaluation.get("gpu_id")

    if gpu_id is not None:
        if not isinstance(gpu_id, int):
            raise TypeError(f"gpu_id must be int, got {type(gpu_id).__name__}")
        device = f"cuda:{gpu_id}"
        logger.info("GPU configuration: Using GPU %d", gpu_id)
        cfg.model.params.device = device
    else:
        device = cfg.model.params.get("device", "cpu")
        logger.info("GPU configuration: Using default device '%s'", device)

    logger.info("Building model: %s", cfg.model.name)
    logger.info("  Model: %s", cfg.model.params.model_name)
    logger.info("  Device: %s", device)
    model = build_model(cfg.model)

    unieval_scorer = None
    embedding_scorer = None
    llm_scorer = None

    # (Optional) Create UniEval scorer
    if cfg.evaluation.use_unieval:
        logger.info("Initializing UniEval scorer")
        try:
            unieval_scorer = UniEvalScorer(
                device=device,
                max_length=cfg.evaluation.unieval.max_length,
                dimensions=list(cfg.evaluation.unieval.dimensions),
                use_reference=False,  # No reference in ChatICD-Bench
                cache_dir=cfg.evaluation.unieval.get("cache_dir"),
            )
            logger.info(
                "  UniEval dimensions: %s", cfg.evaluation.unieval.dimensions
            )
        except Exception as e:
            logger.error("Failed to initialize UniEval scorer: %s", e)
            logger.warning(
                "Continuing without UniEval. UniEval scores will default to 0.0"
            )
            unieval_scorer = None

    # (Optional) Create Embedding scorer
    if cfg.evaluation.use_embedding:
        logger.info("Initializing Embedding scorer")
        try:
            embedding_scorer = EmbeddingScorer(
                api_key=cfg.evaluation.embedding.api_key,
                base_url=cfg.evaluation.embedding.base_url,
                model_id=cfg.evaluation.embedding.model_id,
            )
            logger.info(
                "  Embedding model: %s", cfg.evaluation.embedding.model_id
            )
        except Exception as e:
            logger.error("Failed to initialize Embedding scorer: %s", e)
            logger.warning(
                "Continuing without Embedding scorer. Embedding scores will default to 0.0"
            )
            embedding_scorer = None

    # (Optional) Create LLM scorer
    if cfg.evaluation.use_llm:
        logger.info("Initializing LLM scorer")
        try:
            llm_scorer = LLMScorer(
                api_key=cfg.evaluation.llm.api_key,
                base_url=cfg.evaluation.llm.base_url,
                model_id=cfg.evaluation.llm.model_id,
                evaluation_prompt_template=cfg.evaluation.llm.get(
                    "evaluation_prompt_template"
                ),
            )
            logger.info("  LLM model: %s", cfg.evaluation.llm.model_id)
        except Exception as e:
            logger.error("Failed to initialize LLM scorer: %s", e)
            logger.warning(
                "Continuing without LLM scorer. LLM scores will default to 0.0"
            )
            llm_scorer = None

    evaluator = CustomEvaluator(
        unieval_scorer=unieval_scorer,
        embedding_scorer=embedding_scorer,
        llm_scorer=llm_scorer,
        unieval_weight=cfg.evaluation.weights.unieval,
        embedding_weight=cfg.evaluation.weights.embedding,
        llm_weight=cfg.evaluation.weights.llm,
    )

    runner = Runner(
        unieval_weight=cfg.evaluation.weights.unieval,
        embedding_weight=cfg.evaluation.weights.embedding,
        llm_weight=cfg.evaluation.weights.llm,
    )

    hydra_cfg = HydraConfig.get()

    # Allow overriding output directory for reusing existing outputs
    if cfg.evaluation.get("output_dir") is not None:
        output_dir = Path(cfg.evaluation.output_dir)
        logger.info("Using specified output directory: %s", output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(hydra_cfg.runtime.output_dir)
        logger.info("Using Hydra output directory: %s", output_dir)

    preds_path = output_dir / cfg.evaluation.output.predictions_file
    results_path = output_dir / cfg.evaluation.output.results_file

    dataset_name = cfg.evaluation.dataset_name

    logger.info("Output directory: %s", output_dir)
    logger.info("Predictions file: %s", preds_path)
    logger.info("Results file: %s", results_path)
    logger.info("Dataset: %s", dataset_name)

    # Phase 1: Inference
    if cfg.evaluation.execution.run_inference:
        logger.info("=" * 80)
        logger.info("Phase 1: Running inference on ChatICD-Bench dataset")
        # max_samples=None means evaluate all samples, else limit to specified number (for testing)
        if cfg.evaluation.max_samples is not None:
            logger.info(
                "Testing mode: Evaluating only %d samples",
                cfg.evaluation.max_samples,
            )
        else:
            logger.info("Full evaluation mode: Evaluating all samples")
        logger.info("=" * 80)
        preds = runner.run_inference(
            model,
            dataset_name,
            str(preds_path),
            max_samples=cfg.evaluation.max_samples,
        )
        logger.info("Inference complete. Predictions saved to %s", preds_path)
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
        logger.info("Loading existing predictions from %s", preds_path)
        with open(preds_path, "r", encoding="utf-8") as f:
            preds = json.load(f)
        logger.info("Loaded %d predictions", len(preds))

    # Phase 2: Evaluation
    if cfg.evaluation.execution.run_evaluation:
        logger.info("=" * 80)
        logger.info("Phase 2: Computing evaluation metrics")
        logger.info("=" * 80)
        logger.info(
            f"Metrics: UniEval (weight={cfg.evaluation.weights.unieval}, enabled={cfg.evaluation.use_unieval}), "
            f"Embedding (weight={cfg.evaluation.weights.embedding}, enabled={cfg.evaluation.use_embedding}), "
            f"LLM (weight={cfg.evaluation.weights.llm}, enabled={cfg.evaluation.use_llm})"
        )
        results = runner.run_evaluation(
            preds, dataset_name, evaluator, str(results_path)
        )
        logger.info("Evaluation complete. Results saved to %s", results_path)
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
        logger.info("Loading existing results from %s", results_path)
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        logger.info("Loaded results for %d questions", len(results))

    # Phase 3: Display Results
    if cfg.evaluation.execution.run_results:
        logger.info("=" * 80)
        logger.info("Phase 3: Displaying results")
        logger.info("=" * 80)
        runner.show_results(results, dataset_name)
    else:
        logger.info("=" * 80)
        logger.info("Phase 3: Skipped (execution.run_results=false)")
        logger.info("=" * 80)

    logger.info("=" * 80)
    logger.info("Evaluation workflow complete. All outputs in %s", output_dir)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
