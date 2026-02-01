"""LLM-based answer scorer using OpenAI API."""

import logging
import re
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_EVALUATION_PROMPT = """Compare the ground truth and prediction from AI models, to give a correctness score for the prediction. The correctness score is 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right). Just complete the last space of the correctness score. Strictly output only a single float number.
| Question | Ground truth | Prediction | Correctness |
| --- | --- | --- | --- |
| What is x in the equation?<image> | -1 <AND> -5 | x = 3 | 0.0 |
| What is x in the equation?<image> | -1 <AND> -5 | x = -1 | 0.5 |
| What is x in the equation?<image> | -1 <AND> -5 | x = -5 | 0.5 |
| What is x in the equation?<image> | -1 <AND> -5 | x = -5 or 5 | 0.5 |
| What is x in the equation?<image> | -1 <AND> -5 | x = -1 or x = -5 | 1.0 |
| Can you explain this meme?<image> | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme talks about Iceland and Greenland. It's pointing out that despite their names, Iceland is not very icy and Greenland isn't very green. | 0.4 |
| Can you explain this meme?<image> | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme is using humor to point out the misleading nature of Iceland's and Greenland's names. Iceland, despite its name, has lush green landscapes while Greenland is mostly covered in ice and snow. The text 'This is why I have trust issues' is a playful way to suggest that these contradictions can lead to distrust or confusion. The humor in this meme is derived from the unexpected contrast between the names of the countries and their actual physical characteristics. | 1.0 |
| {question} | {ground_truth} | {prediction} |
"""


class LLMScorer:
    """LLM-based answer quality scorer.

    Uses OpenAI GPT models to evaluate the similarity and correctness
    of predicted answers compared to ground truth answers. Supports both
    direct scoring and Chain-of-Thought (CoT) reasoning with score extraction.

    Attributes:
        client: OpenAI client instance.
        evaluation_prompt_template: Evaluation prompt template with placeholders.
        model_id: OpenAI model identifier.
        use_cot: Whether to use Chain-of-Thought evaluation.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_id: str,
        evaluation_prompt_template: Optional[str] = None,
        use_cot: bool = False,
    ):
        """Initialize the LLM scorer.

        Args:
            api_key: OpenAI API key for authentication.
            base_url: Base URL for OpenAI API endpoint.
            model_id: OpenAI model identifier (e.g., 'gpt-4o').
            evaluation_prompt_template: Custom evaluation prompt template.
                Should contain {question}, {ground_truth} and {prediction} placeholders.
                If None, uses DEFAULT_EVALUATION_PROMPT.
                When use_cot=True, the prompt should instruct the model to output
                the score in [[score]] format at the end of its response.
            use_cot: Whether to use Chain-of-Thought evaluation mode.
                If True, the model is expected to provide reasoning before outputting
                the score in [[score]] format. The score will be extracted via regex.
                If False, the model should output only a single float number.

        Raises:
            ValueError: If use_cot=True but evaluation_prompt_template is None.
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.use_cot = use_cot

        # Validate CoT configuration
        if self.use_cot and evaluation_prompt_template is None:
            raise ValueError(
                "evaluation_prompt_template must be provided when use_cot=True. "
                "CoT mode requires a custom prompt template in the configuration."
            )

        self.evaluation_prompt_template = (
            evaluation_prompt_template
            if evaluation_prompt_template is not None
            else DEFAULT_EVALUATION_PROMPT
        )
        self.model_id = model_id

        logger.info("Initialized LLMScorer with model: %s", model_id)
        if self.use_cot:
            logger.info(
                "CoT mode enabled: Score will be extracted from [[score]] format"
            )

    def __call__(
        self, question: str, prediction: str, ground_truth: str
    ) -> float:
        """Evaluate answer quality using LLM judgment.

        Args:
            question: Question text.
            prediction: Predicted answer string.
            ground_truth: Ground truth answer string.

        Returns:
            LLM-as-a-judge score in range [0, 1].
            Returns 0.5 if scoring fails.
        """
        formatted_prompt = self.evaluation_prompt_template.format(
            question=question, ground_truth=ground_truth, prediction=prediction
        )
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": formatted_prompt}],
                model=self.model_id,
            )
            response_content = chat_completion.choices[0].message.content

            if self.use_cot:
                # Extract score from [[score]] format using regex
                score = self._extract_score_from_cot(response_content)
            else:
                # Direct score parsing
                score = float(response_content)

            return score
        except (TypeError, KeyError, ValueError) as e:
            logger.warning(
                "LLM scoring failed: %s. Returning default score 0.5", e
            )
            return 0.5

    def _extract_score_from_cot(self, response: str) -> float:
        """Extract score from CoT response with [[score]] format.

        Args:
            response: LLM response containing reasoning and [[score]].

        Returns:
            Extracted score as float in range [0, 1].

        Raises:
            ValueError: If score cannot be extracted or is invalid.
        """
        # Regex pattern to match [[score]] format (handles both integer and float)
        pattern = r"\[\[(\d+(?:\.\d+)?)\]\]"
        matches = re.findall(pattern, response)

        if not matches:
            logger.error(
                "Failed to extract score from CoT response. "
                "Expected [[score]] format. Response: %s",
                response[:200],  # Log first 200 chars for debugging
            )
            raise ValueError(
                "No score found in [[score]] format in LLM response"
            )

        # Use the last match (in case there are multiple)
        score_str = matches[-1]
        score = float(score_str)

        logger.debug("Extracted score %f from CoT response", score)
        return score
