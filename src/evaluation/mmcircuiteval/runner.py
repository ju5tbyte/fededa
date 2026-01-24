"""Runner wrapper for MMCircuitEval evaluation.

Most of functions are basically same as those in original MMCircuitEval repo,
with some bug fixes and improvements, and logging added.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Import modules from original MMCircuitEval repo
external_path = Path(__file__).parents[3] / "external" / "MMCircuitEval"
sys.path.insert(0, str(external_path))
from evaluation.modules import runner as external_runner
from evaluation.modules.captioner import Captioner
from evaluation.utils.formatter import formatAnswer, formatModelOutput
from evaluation.utils.prompts import (
    caption_prompt,
    getQuestionPrompt,
    image_prompt,
)

# Monkey-patch: Replace the formatScore function in the runner module
from src.evaluation.mmcircuiteval.formatter import formatScore

external_runner.formatScore = formatScore


class Runner(external_runner.Runner):
    """Extended Runner class.

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

        data = load_dataset("charlie314159/MMCircuitEval", split=field)
        logger.info("Loaded dataset: %d total samples", len(data))

        # Select data if max_samples is specified
        if max_samples is not None and max_samples > 0:
            data = data.select(range(min(max_samples, len(data))))
            logger.info("Limited to %d samples for testing", len(data))

        preds = {}
        if os.path.exists(out_path):
            with open(out_path, "r") as f:
                preds = json.load(f)
            logger.info(
                "Loaded %d existing predictions from %s", len(preds), out_path
            )

        logger.info("Starting inference on %d samples", len(data))
        for question_idx, Q in tqdm(enumerate(data)):
            if (
                str(question_idx) in preds.keys()
                and None not in preds[str(question_idx)]["answers"]
                and None not in preds[str(question_idx)]["explanations"]
                and None not in preds[str(question_idx)]["raw_preds"]
            ):
                logger.debug(
                    "Skipping question %d (already processed)", question_idx
                )
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
                        "Model inference failed for question %d, sub-question %d: %s",
                        question_idx,
                        i,
                        type(e).__name__,
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

        logger.info(
            "Inference complete: %d predictions saved to %s", len(preds), out_path
        )
        return preds

    def runEvaluation(self, preds: dict, field: str, evaluator, out_path: str):
        """Run evaluation on predictions with fair explanation handling.

        FIXED: Here explanations are only used for comparison if BOTH the prediction
        and ground truth have non-empty explanations.

        Args:
            preds: Dictionary mapping question indices to predictions.
            field: Dataset field (general, spec, frontend, backend).
            evaluator: Evaluator instance.
            out_path: Path to save evaluation results.

        Returns:
            Dictionary mapping question indices to score lists.

        Raises:
            AssertionError: If required fields are missing or lengths mismatch.
        """
        data = load_dataset("charlie314159/MMCircuitEval", split=field)

        results = {}
        for question_idx, pred in tqdm(preds.items()):
            gt = data[int(question_idx)]

            assert (
                "answers" in pred and "answers" in gt
            ), f"Missing answers for question {question_idx}"
            assert (
                "explanations" in pred and "explanations" in gt
            ), f"Missing explanations for question {question_idx}"
            assert (
                "raw_preds" in pred
            ), f"Missing raw predictions for question {question_idx}"
            assert (
                len(pred["answers"])
                == len(gt["answers"])
                == len(pred["explanations"])
                == len(gt["explanations"])
                == len(pred["raw_preds"])
            ), f"Answer length mismatch for question {question_idx}"

            scores = []
            for i in range(len(pred["answers"])):
                pred_exp = pred["explanations"][i]
                gt_exp = gt["explanations"][i]

                # Check if BOTH have non-empty explanations (None or "" = empty)
                has_both_explanations = (
                    pred_exp is not None
                    and pred_exp != ""
                    and gt_exp is not None
                    and gt_exp != ""
                )
                # Use explanations only if both have them
                if has_both_explanations:
                    answer_pred = formatAnswer(
                        pred["answers"][i], pred_exp, pred["raw_preds"][i]
                    )
                    answer_gt = formatAnswer(gt["answers"][i], gt_exp)
                else:  # If either is missing, do not use explanations for comparison
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
        """Display evaluation results.

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

            if data[int(question_idx)]["images"] != []:
                multimodal_scores += scores
            else:
                text_scores += scores

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

            all_scores += scores

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

        logger.info("=" * 80)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 80)
        logger.info("Number of problems: %d", len(results))
        logger.info("Number of sub-questions: %d", len(all_scores))
        logger.info("Overall Score: %.2f", avg_score * 100)
        logger.info("Knowledge Score: %.2f", knowledge_score * 100)
        logger.info("Comprehension Score: %.2f", comprehension_score * 100)
        logger.info("Reasoning Score: %.2f", reasoning_score * 100)
        logger.info("Computation Score: %.2f", computation_score * 100)
        logger.info("Text-only Score: %.2f", text_score * 100)
        logger.info("Multimodal Score: %.2f", multimodal_score * 100)
        logger.info("=" * 80)


__all__ = ["Runner"]
