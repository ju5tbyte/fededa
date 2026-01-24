"""LLM-based answer scorer using OpenAI API."""

import logging
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
    of predicted answers compared to ground truth answers.

    Attributes:
        client: OpenAI client instance.
        evaluation_prompt_template: Evaluation prompt template with placeholders.
        model_id: OpenAI model identifier.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_id: str,
        evaluation_prompt_template: Optional[str] = None,
    ):
        """Initialize the LLM scorer.

        Args:
            api_key: OpenAI API key for authentication.
            base_url: Base URL for OpenAI API endpoint.
            model_id: OpenAI model identifier (e.g., 'gpt-4o').
            evaluation_prompt_template: Custom evaluation prompt template.
                Should contain {ground_truth} and {prediction} placeholders.
                If None, uses DEFAULT_EVALUATION_PROMPT.
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.evaluation_prompt_template = (
            evaluation_prompt_template
            if evaluation_prompt_template is not None
            else DEFAULT_EVALUATION_PROMPT
        )
        self.model_id = model_id
        logger.info("Initialized LLMScorer with model: %s", model_id)

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
            score = float(chat_completion.choices[0].message.content)
            return score
        except (TypeError, KeyError, ValueError) as e:
            logger.warning(
                "LLM scoring failed: %s. Returning default score 0.5", e
            )
            return 0.5
