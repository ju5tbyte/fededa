"""MMCircuitEval benchmark evaluation script.

Usage:
    python scripts/evaluate_mmcircuiteval.py
    python scripts/evaluate_mmcircuiteval.py evaluation.field=spec model=qwen_vl_model model.params.model_name=Qwen/Qwen3-VL-8B-Instruct evaluation.gpu_id=4
"""

import json
import logging
import sys
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from src.evaluation.mmcircuiteval.evaluator import Evaluator
from src.evaluation.mmcircuiteval.runner import Runner
from src.models.builder import build_model
from src.utils.set_seed import set_seed

load_dotenv()

# Path for importing modules from original MMCircuitEval repo
external_path = Path(__file__).parent.parent / "external" / "MMCircuitEval"
sys.path.insert(0, str(external_path))
from evaluation.modules.embedder import Embedder
from evaluation.modules.llm_scorer import LLMScorer

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../configs",
    config_name="evaluate_mmcircuiteval",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Run MMCircuitEval benchmark evaluation.

    1. Inference: Generate model predictions for all questions
    2. Evaluation: Compute metrics (BLEU, ROUGE, optional embedding/LLM)
    3. Results: Display aggregated scores by ability and modality

    Args:
        cfg: Hydra configuration containing model and evaluation settings.
    """
    set_seed(cfg.evaluation.seed)

    logger.info("=" * 80)
    logger.info("MMCircuitEval Benchmark Evaluation")
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

    # (Optional) Create embedder and LLM scorer
    embedder = None
    llm_scorer = None
    if cfg.evaluation.use_embedder:
        logger.info("Initializing OpenAI embedder")
        embedder = Embedder(
            api_key=cfg.evaluation.api.openai_api_key,
            base_url=cfg.evaluation.api.openai_base_url,
            model_id=cfg.evaluation.api.embedding_model,
        )
        logger.info("  Embedding model: %s", cfg.evaluation.api.embedding_model)
    if cfg.evaluation.use_llm_scorer:
        logger.info("Initializing LLM scorer")
        llm_scorer = LLMScorer(
            api_key=cfg.evaluation.api.openai_api_key,
            base_url=cfg.evaluation.api.openai_base_url,
            model_id=cfg.evaluation.api.llm_model,
        )
        logger.info("  LLM scorer model: %s", cfg.evaluation.api.llm_model)

    evaluator = Evaluator(llm_scorer=llm_scorer, embedder=embedder)
    runner = Runner(
        version=cfg.evaluation.version,
        bleu_weight=cfg.evaluation.weights.bleu,
        rouge_weight=cfg.evaluation.weights.rouge,
        emb_weight=cfg.evaluation.weights.emb,
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

    logger.info("Output directory: %s", output_dir)
    logger.info("Predictions file: %s", preds_path)
    logger.info("Results file: %s", results_path)

    # Phase 1: Inference
    if cfg.evaluation.execution.run_inference:
        logger.info("=" * 80)
        logger.info(
            "Phase 1: Running inference on field '%s'", cfg.evaluation.field
        )
        # max_samples=None means evaluate all samples, else limit to specified number (for testing)
        if cfg.evaluation.max_samples is not None:
            logger.info(
                "Testing mode: Evaluating only %d samples",
                cfg.evaluation.max_samples,
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
        with open(preds_path, "r") as f:
            preds = json.load(f)
        logger.info("Loaded %d predictions", len(preds))

    # Phase 2: Evaluation
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
        with open(results_path, "r") as f:
            results = json.load(f)
        logger.info("Loaded results for %d questions", len(results))

    # Phase 3: Display Results
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
    logger.info("Evaluation workflow complete. All outputs in %s", output_dir)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
