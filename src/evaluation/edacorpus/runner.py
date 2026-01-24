"""Runner for EDA-Corpus benchmark evaluation."""

import csv
import json
import logging
from pathlib import Path
from typing import Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)


class Runner:
    """Runner for EDA-Corpus benchmark evaluation.

    Attributes:
        unieval_weight: Weight for UniEval metric in final score.
        embedding_weight: Weight for embedding similarity metric in final score.
        llm_weight: Weight for LLM-based metric in final score.
    """

    def __init__(
        self,
        unieval_weight: float = 1.0,
        embedding_weight: float = 1.0,
        llm_weight: float = 1.0,
    ):
        """Initialize the runner.

        Args:
            unieval_weight: Weight for UniEval score in final aggregation.
            embedding_weight: Weight for embedding similarity score in final aggregation.
            llm_weight: Weight for LLM score in final aggregation.
        """
        self.unieval_weight = unieval_weight
        self.embedding_weight = embedding_weight
        self.llm_weight = llm_weight

    def load_dataset(self, dataset_path: str) -> list[dict]:
        """Load EDA-Corpus dataset from CSV file.

        The CSV file should have two columns: 'Prompts' and 'Answers'.

        Args:
            dataset_path: Path to the Question-Answer_Dataset.csv file.

        Returns:
            List of dataset samples, where each sample is a dictionary with
            keys: id, question, answer.

        Raises:
            FileNotFoundError: If dataset file does not exist.
            ValueError: If CSV file is malformed or missing required columns.
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        data = []
        with open(dataset_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            # Validate required columns
            if reader.fieldnames is None:
                raise ValueError(f"CSV file {dataset_path} has no header row")

            required_cols = {"Prompts", "Answers"}
            if not required_cols.issubset(set(reader.fieldnames)):
                raise ValueError(
                    f"CSV file must contain columns {required_cols}, "
                    f"but found {set(reader.fieldnames)}"
                )

            for row in reader:
                # Skip empty rows where both Prompts and Answers are empty
                question = row["Prompts"].strip()
                answer = row["Answers"].strip()

                if not question and not answer:
                    continue

                sample = {
                    "id": len(
                        data
                    ),  # Use length of data list for sequential IDs
                    "question": question,
                    "answer": answer,
                }
                data.append(sample)

        logger.info("Loaded %d samples from %s", len(data), dataset_path)
        return data

    def build_prompt(self, question: str) -> str:
        """Build prompt for model inference.

        Args:
            question: The question to answer.

        Returns:
            Formatted prompt string ready for model inference.
        """
        system_prompt = (
            "You are an expert in Electronic Design Automation (EDA), specifically specializing "
            "in the OpenROAD tool and OpenROAD-flow-scripts.\n"
            "Answer the following question "
            "accurately based on your knowledge of physical design automation workflows.\n\n"
        )
        system_prompt += f"Question: {question}\n"
        system_prompt += "Answer:"

        return system_prompt

    def run_inference(
        self,
        model,
        dataset_path: str,
        out_path: str,
        max_samples: Optional[int] = None,
    ) -> dict[str, dict]:
        """Run inference on EDA-Corpus dataset.

        Args:
            model: The model to evaluate. Must implement __call__(prompt, images=None).
            dataset_path: Path to Question-Answer_Dataset.csv file.
            out_path: Path to save predictions JSON file.
            max_samples: Maximum number of samples to process. If None, process
                all samples.

        Returns:
            Dictionary mapping sample IDs (as strings) to prediction dictionaries
            with keys: 'prediction', 'raw_output'.

        Raises:
            FileNotFoundError: If dataset file does not exist.
        """
        data = self.load_dataset(dataset_path)

        # Select data if max_samples is specified
        if max_samples is not None and max_samples > 0:
            data = data[: min(max_samples, len(data))]
            logger.info("Limited to %d samples for testing", len(data))

        preds = {}
        out_path = Path(out_path)
        if out_path.exists():
            with open(out_path, "r", encoding="utf-8") as f:
                preds = json.load(f)
            logger.info(
                "Loaded %d existing predictions from %s", len(preds), out_path
            )

        logger.info("Starting inference on %d samples", len(data))

        for sample in tqdm(data, desc="Running inference"):
            sample_id = str(sample["id"])
            if (
                sample_id in preds
                and preds[sample_id].get("prediction") is not None
            ):
                logger.debug(
                    "Skipping sample %s (already processed)", sample_id
                )
                continue

            # Build prompt
            question = sample["question"]
            prompt = self.build_prompt(question)

            try:
                raw_output = model(prompt)
                prediction = raw_output
            except (TypeError, IndexError, ValueError) as e:
                logger.warning(
                    "Model inference failed for sample %s: %s: %s",
                    sample_id,
                    type(e).__name__,
                    e,
                )
                raw_output = None
                prediction = None

            preds[sample_id] = {
                "prediction": prediction,
                "raw_output": raw_output,
            }

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(preds, f, indent=2, ensure_ascii=False)

        logger.info(
            "Inference complete: %d predictions saved to %s",
            len(preds),
            out_path,
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
            dataset_path: Path to Question-Answer_Dataset.csv file.
            evaluator: CustomEvaluator instance.
            out_path: Path to save evaluation results.

        Returns:
            Dictionary mapping sample IDs to score dictionaries with keys:
            'unieval', 'embedding', 'llm', 'overall'.

        Raises:
            FileNotFoundError: If dataset file does not exist.
            KeyError: If prediction is missing for any sample.
        """
        data = self.load_dataset(dataset_path)
        data_dict = {str(sample["id"]): sample for sample in data}

        results = {}
        logger.info("Starting evaluation on %d predictions", len(preds))

        for sample_id, pred_dict in tqdm(
            preds.items(), desc="Running evaluation"
        ):
            if sample_id not in data_dict:
                logger.warning(
                    "Sample %s not found in dataset, skipping", sample_id
                )
                continue

            sample = data_dict[sample_id]
            prediction = pred_dict.get("prediction")
            ground_truth = sample["answer"]
            question = sample["question"]

            if prediction is None:
                logger.warning(
                    "Skipping evaluation for sample %s (prediction is None)",
                    sample_id,
                )
                results[sample_id] = {
                    "unieval": 0.0,
                    "embedding": 0.0,
                    "llm": 0.0,
                    "overall": 0.0,
                }
                continue

            scores = evaluator(
                pred=prediction,
                gt=ground_truth,
                question=question,
                reference_content=None,
            )
            results[sample_id] = scores

        out_path = Path(out_path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info("Evaluation complete: Results saved to %s", out_path)
        return results

    def show_results(
        self, results: dict[str, dict[str, float]], dataset_path: str
    ) -> dict[str, float]:
        """Display evaluation results with aggregated metrics.

        Args:
            results: Dictionary mapping sample IDs to score dictionaries.
            dataset_path: Path to Question-Answer_Dataset.csv file.

        Returns:
            Dictionary with aggregated scores:
                - 'overall': Weighted average of all metrics
                - 'unieval': Average UniEval score
                - 'embedding': Average embedding similarity score
                - 'llm': Average LLM score
        """
        data = self.load_dataset(dataset_path)
        data_dict = {str(sample["id"]): sample for sample in data}

        all_unieval = []
        all_embedding = []
        all_llm = []
        all_overall = []

        for sample_id, scores in results.items():
            if sample_id not in data_dict:
                continue

            unieval = scores["unieval"]
            embedding = scores["embedding"]
            llm = scores["llm"]
            overall = scores["overall"]

            all_unieval.append(unieval)
            all_embedding.append(embedding)
            all_llm.append(llm)
            all_overall.append(overall)

        avg_unieval = (
            sum(all_unieval) / len(all_unieval) if all_unieval else 0.0
        )
        avg_embedding = (
            sum(all_embedding) / len(all_embedding) if all_embedding else 0.0
        )
        avg_llm = sum(all_llm) / len(all_llm) if all_llm else 0.0
        avg_overall = (
            sum(all_overall) / len(all_overall) if all_overall else 0.0
        )

        logger.info("=" * 80)
        logger.info("EDA-CORPUS EVALUATION RESULTS")
        logger.info("=" * 80)
        logger.info("Number of samples: %d", len(results))
        logger.info("Overall Score: %.2f", avg_overall * 100)
        logger.info("UniEval Score: %.2f", avg_unieval * 100)
        logger.info("Embedding Similarity Score: %.2f", avg_embedding * 100)
        logger.info("LLM Score: %.2f", avg_llm * 100)
        logger.info("=" * 80)

        return {
            "overall": avg_overall,
            "unieval": avg_unieval,
            "embedding": avg_embedding,
            "llm": avg_llm,
        }
