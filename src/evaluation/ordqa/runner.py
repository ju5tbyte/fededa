"""Runner for ORD-QA benchmark evaluation orchestration.

This module provides the Runner class that orchestrates the ORD-QA evaluation
process including data loading, inference, evaluation, and result reporting.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from tqdm import tqdm

# Setup logger (Hydra will configure file handlers automatically)
logger = logging.getLogger(__name__)


class Runner:
    """Runner for ORD-QA benchmark evaluation.

    This class orchestrates the evaluation workflow: loading data, running
    inference, computing metrics, and displaying results.

    Attributes:
        bleu_weight (float): Weight for BLEU metric in final score.
        rouge_l_weight (float): Weight for ROUGE-L metric in final score.
        unieval_weight (float): Weight for UniEval metric in final score.
    """

    def __init__(
        self,
        bleu_weight: float = 1.0,
        rouge_l_weight: float = 1.0,
        unieval_weight: float = 1.0,
    ):
        """Initialize the runner.

        Args:
            bleu_weight: Weight for BLEU score in final aggregation.
            rouge_l_weight: Weight for ROUGE-L score in final aggregation.
            unieval_weight: Weight for UniEval score in final aggregation.
        """
        self.bleu_weight = bleu_weight
        self.rouge_l_weight = rouge_l_weight
        self.unieval_weight = unieval_weight

    def load_dataset(self, dataset_path: str) -> list[dict]:
        """Load ORD-QA dataset from JSONL file.

        Args:
            dataset_path: Path to the ORD-QA.jsonl file.

        Returns:
            List of dataset samples, where each sample is a dictionary with
            keys: id, type, question, ref_num, reference, reference_content, answer.

        Raises:
            FileNotFoundError: If dataset file does not exist.
            json.JSONDecodeError: If JSONL file is malformed.
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        data = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                try:
                    sample = json.loads(line.strip())
                    data.append(sample)
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Failed to parse line {line_num} in {dataset_path}: {e}"
                    )
                    raise

        logger.info(f"Loaded {len(data)} samples from {dataset_path}")
        return data

    def build_prompt(
        self, question: str, reference_content: Optional[list[str]] = None
    ) -> str:
        """Build prompt for model inference.

        This method constructs the prompt following the format shown in Figure 7
        of the ORD-QA paper. The prompt includes a system instruction, the question,
        and optionally the related reference documents.

        Args:
            question: The question to answer.
            reference_content: Optional list of reference documents. If None,
                the model answers without reference (tests domain knowledge).

        Returns:
            Formatted prompt string ready for model inference.
        """
        system_prompt = (
            "You are the product consultant of a Electronic Design Automation (EDA) "
            "tool OpenROAD.\n"
        )

        if reference_content is not None:
            # RAG-style prompt with references
            system_prompt += (
                "Now given the user question and the related reference, you are "
                "required to answer the question referring to the provided reference.\n"
                "During answering the question, you have to follow these instructions:\n"
                "1. Make your answer as rigorous as possible, do not fabricate the fact "
                "that does not mentioned in the provided reference.\n"
                "2. Your answer should be strongly related to the provided reference, "
                "provide concrete solution for the answer, and do not ignore the "
                "precondition in the query.\n\n"
            )
            system_prompt += "Instruction:\n"
            system_prompt += f"Question: {question}\n"
            system_prompt += "Related reference:\n"
            for i, ref in enumerate(reference_content, start=1):
                system_prompt += f"Doc{i}: {ref}\n"
            system_prompt += "Answer:"
        else:
            # Knowledge-only prompt without references
            system_prompt += (
                "Now given the user question, you are required to answer the question "
                "based on your knowledge of EDA tools.\n\n"
            )
            system_prompt += "Instruction:\n"
            system_prompt += f"Question: {question}\n"
            system_prompt += "Answer:"

        return system_prompt

    def run_inference(
        self,
        model,
        dataset_path: str,
        out_path: str,
        use_reference: bool = True,
        max_samples: Optional[int] = None,
    ) -> dict[str, dict]:
        """Run inference on ORD-QA dataset.

        Args:
            model: The model to evaluate. Must implement __call__(prompt, images=None).
            dataset_path: Path to ORD-QA.jsonl file.
            out_path: Path to save predictions JSON file.
            use_reference: Whether to include reference documents in prompts.
            max_samples: Maximum number of samples to process. If None, process
                all samples.

        Returns:
            Dictionary mapping sample IDs (as strings) to prediction dictionaries
            with keys: 'prediction', 'raw_output'.

        Raises:
            FileNotFoundError: If dataset file does not exist.
        """
        # Load dataset
        data = self.load_dataset(dataset_path)

        # Limit samples if max_samples is specified
        if max_samples is not None and max_samples > 0:
            data = data[: min(max_samples, len(data))]
            logger.info(f"Limited to {len(data)} samples for testing")

        # Load existing predictions if available
        preds = {}
        out_path = Path(out_path)
        if out_path.exists():
            with open(out_path, "r", encoding="utf-8") as f:
                preds = json.load(f)
            logger.info(
                f"Loaded {len(preds)} existing predictions from {out_path}"
            )

        # Run inference
        logger.info(f"Starting inference on {len(data)} samples")
        logger.info(f"Using reference: {use_reference}")

        for sample in tqdm(data, desc="Running inference"):
            sample_id = str(sample["id"])

            # Skip if already processed
            if (
                sample_id in preds
                and preds[sample_id].get("prediction") is not None
            ):
                logger.debug(f"Skipping sample {sample_id} (already processed)")
                continue

            # Build prompt
            question = sample["question"]
            reference_content = (
                sample["reference_content"] if use_reference else None
            )
            prompt = self.build_prompt(question, reference_content)

            # Run model inference
            try:
                # For text-only models, pass empty images list
                raw_output = model(prompt, imgs=[])
                prediction = raw_output
            except (TypeError, IndexError, ValueError) as e:
                logger.warning(
                    f"Model inference failed for sample {sample_id}: "
                    f"{type(e).__name__}: {e}"
                )
                raw_output = None
                prediction = None

            # Save prediction
            preds[sample_id] = {
                "prediction": prediction,
                "raw_output": raw_output,
            }

            # Save incrementally to avoid data loss
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(preds, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Inference complete: {len(preds)} predictions saved to {out_path}"
        )
        return preds

    def run_evaluation(
        self,
        preds: dict[str, dict],
        dataset_path: str,
        evaluator,
        out_path: str,
    ) -> dict[str, dict[str, float]]:
        """Run evaluation on predictions.

        Args:
            preds: Dictionary mapping sample IDs to predictions.
            dataset_path: Path to ORD-QA.jsonl file.
            evaluator: Evaluator instance.
            out_path: Path to save evaluation results.

        Returns:
            Dictionary mapping sample IDs to score dictionaries with keys:
            'bleu', 'rouge_l', 'unieval'.

        Raises:
            FileNotFoundError: If dataset file does not exist.
            KeyError: If prediction is missing for any sample.
        """
        # Load dataset
        data = self.load_dataset(dataset_path)
        data_dict = {str(sample["id"]): sample for sample in data}

        # Run evaluation
        results = {}
        logger.info(f"Starting evaluation on {len(preds)} predictions")

        for sample_id, pred_dict in tqdm(
            preds.items(), desc="Running evaluation"
        ):
            if sample_id not in data_dict:
                logger.warning(
                    f"Sample {sample_id} not found in dataset, skipping"
                )
                continue

            sample = data_dict[sample_id]
            prediction = pred_dict.get("prediction")
            ground_truth = sample["answer"]
            question = sample["question"]
            reference_content = sample.get("reference_content")

            # Skip if prediction is None (inference failed)
            if prediction is None:
                logger.warning(
                    f"Skipping evaluation for sample {sample_id} (prediction is None)"
                )
                results[sample_id] = {
                    "bleu": 0.0,
                    "rouge_l": 0.0,
                    "unieval": 0.0,
                }
                continue

            # Set context for UniEval (if evaluator supports it)
            if hasattr(evaluator, "set_context"):
                evaluator.set_context(question, reference_content)

            # Compute metrics
            scores = evaluator(prediction, ground_truth)
            results[sample_id] = scores

        # Save results
        out_path = Path(out_path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation complete: Results saved to {out_path}")
        return results

    def show_results(
        self, results: dict[str, dict[str, float]], dataset_path: str
    ) -> dict[str, float]:
        """Display evaluation results with aggregated metrics.

        Args:
            results: Dictionary mapping sample IDs to score dictionaries.
            dataset_path: Path to ORD-QA.jsonl file.

        Returns:
            Dictionary with aggregated scores:
                - 'overall': Weighted average of all metrics
                - 'bleu': Average BLEU score
                - 'rouge_l': Average ROUGE-L score
                - 'unieval': Average UniEval score
                - 'by_type': Scores grouped by question type (functionality, gui&installation&test)
        """
        # Load dataset for metadata
        data = self.load_dataset(dataset_path)
        data_dict = {str(sample["id"]): sample for sample in data}

        # Aggregate scores
        all_bleu = []
        all_rouge_l = []
        all_unieval = []
        all_weighted = []

        # Group by question type
        type_scores = {}

        for sample_id, scores in results.items():
            if sample_id not in data_dict:
                continue

            sample = data_dict[sample_id]
            question_type = sample["type"]

            bleu = scores["bleu"]
            rouge_l = scores["rouge_l"]
            unieval = scores["unieval"]

            # Compute weighted score
            total_weight = (
                self.bleu_weight + self.rouge_l_weight + self.unieval_weight
            )
            weighted = (
                bleu * self.bleu_weight
                + rouge_l * self.rouge_l_weight
                + unieval * self.unieval_weight
            ) / total_weight

            all_bleu.append(bleu)
            all_rouge_l.append(rouge_l)
            all_unieval.append(unieval)
            all_weighted.append(weighted)

            # Group by type
            if question_type not in type_scores:
                type_scores[question_type] = []
            type_scores[question_type].append(weighted)

        # Calculate averages
        avg_bleu = sum(all_bleu) / len(all_bleu) if all_bleu else 0.0
        avg_rouge_l = (
            sum(all_rouge_l) / len(all_rouge_l) if all_rouge_l else 0.0
        )
        avg_unieval = (
            sum(all_unieval) / len(all_unieval) if all_unieval else 0.0
        )
        avg_weighted = (
            sum(all_weighted) / len(all_weighted) if all_weighted else 0.0
        )

        # Calculate type-wise averages
        type_averages = {}
        for qtype, scores in type_scores.items():
            type_averages[qtype] = sum(scores) / len(scores) if scores else 0.0

        # Log results
        logger.info("=" * 80)
        logger.info("ORD-QA EVALUATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"Number of samples: {len(results)}")
        logger.info(f"Overall Score: {avg_weighted * 100:.2f}")
        logger.info(f"BLEU Score: {avg_bleu * 100:.2f}")
        logger.info(f"ROUGE-L Score: {avg_rouge_l * 100:.2f}")
        logger.info(f"UniEval Score: {avg_unieval * 100:.2f}")
        logger.info("-" * 80)
        logger.info("Scores by Question Type:")
        for qtype, score in sorted(type_averages.items()):
            logger.info(f"  {qtype}: {score * 100:.2f}")
        logger.info("=" * 80)

        return {
            "overall": avg_weighted,
            "bleu": avg_bleu,
            "rouge_l": avg_rouge_l,
            "unieval": avg_unieval,
            "by_type": type_averages,
        }
