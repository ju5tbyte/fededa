"""Runner wrapper for MMCircuitEval evaluation orchestration.

This module imports the external MMCircuitEval runner and applies a monkey-patch
to fix the formatScore bug. It also extends the Runner class to support limiting
the number of samples for testing purposes.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm

# Setup logger (Hydra will configure file handlers automatically)
logger = logging.getLogger(__name__)

# Add external MMCircuitEval to Python path
external_path = Path(__file__).parents[3] / "external" / "MMCircuitEval"
sys.path.insert(0, str(external_path))

# Import external runner module and its dependencies
from evaluation.modules import runner as external_runner
from evaluation.modules.captioner import Captioner
from evaluation.utils.formatter import formatAnswer, formatModelOutput
from evaluation.utils.prompts import (
    caption_prompt,
    getQuestionPrompt,
    image_prompt,
)

# Import our fixed formatter
from src.evaluation.mmcircuiteval.formatter import formatScore

# Monkey-patch: Replace the formatScore function in the runner module
# This fixes the bug where formatScore expects 5 params but is called with 1
# Our fixed formatter has default parameter values, so it works with 1 argument
external_runner.formatScore = formatScore


class Runner(external_runner.Runner):
    """Extended Runner class with max_samples support.

    This class extends the external Runner to support limiting the number of
    samples processed during inference, which is useful for quick testing.

    Attributes:
        version (str): Prompt version.
        bleu_weight (float): Weight for BLEU metric.
        rouge_weight (float): Weight for ROUGE metric.
        emb_weight (float): Weight for embedding similarity metric.
        llm_weight (float): Weight for LLM scorer metric.
        captioner: Image captioner for text-only models.
    """

    def runInference(
        self,
        model,
        field: str,
        out_path: str,
        cot: bool = False,
        max_samples: Optional[int] = None,
    ):
        """Run inference on MMCircuitEval dataset with optional sample limiting.

        This method extends the parent runInference to support max_samples
        parameter for testing purposes. When max_samples is set, only the
        first N samples will be processed.

        Args:
            model: The model to evaluate.
            field: Dataset field to evaluate (general, spec, frontend, backend).
            out_path: Path to save predictions JSON file.
            cot: Whether to use Chain-of-Thought prompting.
            max_samples: Maximum number of samples to process. If None, process
                all samples.

        Returns:
            dict: Predictions dictionary mapping question indices to answers.

        Raises:
            AssertionError: If field or model modality is not supported.
        """
        assert field in [
            "general",
            "spec",
            "frontend",
            "backend",
        ], f"Unsupported field: {field}"
        assert model.modality in [
            "text",
            "multimodal",
        ], f"Unsupported model modality: {model.modality}"
        if model.modality == "text" and self.captioner is None:
            self.captioner = Captioner(model.device)
        # load data
        data = load_dataset("charlie314159/MMCircuitEval", split=field)
        logger.info(f"Loaded dataset: {len(data)} total samples")

        # Limit samples if max_samples is specified (for testing)
        if max_samples is not None and max_samples > 0:
            data = data.select(range(min(max_samples, len(data))))
            logger.info(f"Limited to {len(data)} samples for testing")

        preds = {}
        if os.path.exists(out_path):
            with open(out_path, "r") as f:
                preds = json.load(f)
            logger.info(f"Loaded {len(preds)} existing predictions from {out_path}")
        # run inference
        logger.info(f"Starting inference on {len(data)} samples")
        for question_idx, Q in tqdm(enumerate(data)):
            if (
                str(question_idx) in preds.keys()
                and None not in preds[str(question_idx)]["answers"]
                and None not in preds[str(question_idx)]["explanations"]
                and None not in preds[str(question_idx)]["raw_preds"]
            ):
                logger.debug(f"Skipping question {question_idx} (already processed)")
                continue
            statement = Q["statement"]
            questions = Q["questions"]
            question_types = Q["question_types"]
            images = Q["images"]
            answers = []
            explanations = []
            raw_preds = []
            for i, question in enumerate(questions):
                prompt = getQuestionPrompt(
                    statement, question, question_types[i], self.version, cot
                )
                if len(images) > 0:
                    if model.modality == "multimodal":
                        prompt += " " + image_prompt
                    else:
                        captions = ". ".join(self.captioner(images))
                        prompt += " " + caption_prompt + " " + captions
                try:
                    out = model(prompt, images)
                except (TypeError, IndexError, ValueError) as e:
                    logger.warning(
                        f"Model inference failed for question {question_idx}, "
                        f"sub-question {i}: {type(e).__name__}"
                    )
                    out = None
                answer, explanation = formatModelOutput(out, self.version)
                answers.append(answer)
                explanations.append(explanation)
                raw_preds.append(out)
            preds[str(question_idx)] = {
                "answers": answers,
                "explanations": explanations,
                "raw_preds": raw_preds,
            }
            with open(out_path, "w") as f:
                json.dump(preds, f, indent=4)

        logger.info(f"Inference complete: {len(preds)} predictions saved to {out_path}")
        return preds

    def runEvaluation(self, preds: dict, field: str, evaluator, out_path: str):
        """Run evaluation on predictions with fair explanation handling.

        This method overrides the parent's runEvaluation to ensure fair comparison
        between predictions and ground truth. The dataset contains 'explanations'
        fields but they are empty strings (""), while model outputs may include
        actual explanations.

        **Key Fix**: Only include explanations if BOTH pred and gt have non-empty
        explanations. If either side has empty/None explanation, compare answers
        only. This prevents unfair BLEU/ROUGE/embedding score penalties when
        model outputs "Answer: X. Explanation: Y." but GT only has "Answer: X.".

        Args:
            preds: Dictionary mapping question indices to predictions.
            field: Dataset field (general, spec, frontend, backend).
            evaluator: Evaluator instance.
            out_path: Path to save evaluation results.

        Returns:
            Dictionary mapping question indices to score lists.
        """
        data = load_dataset("charlie314159/MMCircuitEval", split=field)
        # run evaluation
        results = {}
        for question_idx, pred in tqdm(preds.items()):
            gt = data[int(question_idx)]
            assert (
                "answers" in pred and "answers" in gt
            ), f"Missing answers for question {question_idx}"
            assert (
                "explanations" in pred and "explanations" in gt
            ), f"Missing explanations for question {question_idx}"
            assert "raw_preds" in pred, f"Missing raw predictions for question {question_idx}"
            assert (
                len(pred["answers"])
                == len(gt["answers"])
                == len(pred["explanations"])
                == len(gt["explanations"])
                == len(pred["raw_preds"])
            ), f"Answer length mismatch for question {question_idx}"
            scores = []
            for i in range(len(pred["answers"])):
                # Check if BOTH have non-empty explanations (None or "" = empty)
                pred_exp = pred["explanations"][i]
                gt_exp = gt["explanations"][i]
                has_both_explanations = (
                    pred_exp is not None
                    and pred_exp != ""
                    and gt_exp is not None
                    and gt_exp != ""
                )

                # Conservative comparison: use explanations only if both have them
                if has_both_explanations:
                    answer_pred = formatAnswer(
                        pred["answers"][i], pred_exp, pred["raw_preds"][i]
                    )
                    answer_gt = formatAnswer(gt["answers"][i], gt_exp)
                else:
                    # At least one lacks explanation: compare answers only
                    answer_pred = formatAnswer(
                        pred["answers"][i], None, pred["raw_preds"][i]
                    )
                    answer_gt = formatAnswer(gt["answers"][i], None)

                scores.append(evaluator(answer_pred, answer_gt))
            results[question_idx] = scores
        with open(out_path, "w") as f:
            json.dump(results, f, indent=4)
        return results

    def showResults(self, results: dict, field: str):
        """Display evaluation results with various metrics.

        This method overrides the parent's showResults to fix a bug where
        scores are duplicated (line 114: scores += scores). The fixed version
        correctly processes scores without duplication.

        Args:
            results: Dictionary mapping question indices to evaluation scores.
            field: Dataset field that was evaluated (general, spec, frontend, backend).

        Raises:
            ValueError: If field is not valid.
        """
        data = load_dataset("charlie314159/MMCircuitEval", split=field)
        all_scores = []
        knowledge_scores = []
        comprehension_scores = []
        reasoning_scores = []
        computation_scores = []
        text_scores = []
        multimodal_scores = []

        for question_idx, scores in tqdm(results.items()):
            # Format scores (convert to float/numeric format)
            # Use instance weights from config instead of default values
            scores = [
                formatScore(
                    s,
                    self.bleu_weight,
                    self.rouge_weight,
                    self.emb_weight,
                    self.llm_weight,
                )
                for s in scores
            ]

            # BUG FIX: Do NOT duplicate scores (original had: scores += scores)
            # This was causing IndexError when accessing abilities[i]

            # Categorize by modality (text-only vs multimodal)
            if data[int(question_idx)]["images"] != []:
                multimodal_scores += scores
            else:
                text_scores += scores

            # Categorize by ability type
            for i, score in enumerate(scores):
                ability = data[int(question_idx)]["abilities"][i]
                if ability == "knowledge":
                    knowledge_scores.append(score)
                elif ability == "comprehension":
                    comprehension_scores.append(score)
                elif ability == "reasoning":
                    reasoning_scores.append(score)
                elif ability == "computation":
                    computation_scores.append(score)

            # Add to overall scores
            all_scores += scores

        # Calculate average scores
        avg_score = (
            sum(all_scores) / len(all_scores) if len(all_scores) > 0 else 0
        )
        knowledge_score = (
            sum(knowledge_scores) / len(knowledge_scores)
            if len(knowledge_scores) > 0
            else 0
        )
        comprehension_score = (
            sum(comprehension_scores) / len(comprehension_scores)
            if len(comprehension_scores) > 0
            else 0
        )
        reasoning_score = (
            sum(reasoning_scores) / len(reasoning_scores)
            if len(reasoning_scores) > 0
            else 0
        )
        computation_score = (
            sum(computation_scores) / len(computation_scores)
            if len(computation_scores) > 0
            else 0
        )
        text_score = (
            sum(text_scores) / len(text_scores) if len(text_scores) > 0 else 0
        )
        multimodal_score = (
            sum(multimodal_scores) / len(multimodal_scores)
            if len(multimodal_scores) > 0
            else 0
        )

        # Log and print results
        logger.info("=" * 80)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"Number of problems: {len(results)}")
        logger.info(f"Number of sub-questions: {len(all_scores)}")
        logger.info(f"Overall Score: {avg_score * 100:.2f}")
        logger.info(f"Knowledge Score: {knowledge_score * 100:.2f}")
        logger.info(f"Comprehension Score: {comprehension_score * 100:.2f}")
        logger.info(f"Reasoning Score: {reasoning_score * 100:.2f}")
        logger.info(f"Computation Score: {computation_score * 100:.2f}")
        logger.info(f"Text-only Score: {text_score * 100:.2f}")
        logger.info(f"Multimodal Score: {multimodal_score * 100:.2f}")
        logger.info("=" * 80)


__all__ = ["Runner"]
