"""Augment circuit dataset with Chain of Thought (CoT) reasoning.

This script processes circuit understanding datasets and augments them with
step-by-step reasoning using either OpenAI API (GPT-4o) or local VLLM models
(Qwen3-VL, etc.).

Example usage:
    # Using OpenAI API
    python scripts/augment_dataset_circuit_understanding.py data/finetune.json --model gpt-4o

    # Using OpenAI API with higher concurrency (faster processing)
    python scripts/augment_dataset_circuit_understanding.py data/finetune.json --model gpt-4o --max-concurrent 20

    # Using local VLLM with Qwen3-VL
    python scripts/augment_dataset_circuit_understanding.py data/finetune.json --model Qwen/Qwen3-VL-30B-A3B-Instruct

    # Using Qwen3-VL Thinking model
    python scripts/augment_dataset_circuit_understanding.py data/finetune.json --model Qwen/Qwen3-VL-30B-A3B-Thinking

    # Using VLLM with 4-bit quantization
    python scripts/augment_dataset_circuit_understanding.py data/finetune.json --model Qwen/Qwen3-VL-30B-A3B-Instruct --quantization 4bit

    # Augment only 50%% of the data
    python scripts/augment_dataset_circuit_understanding.py data/finetune.json --model gpt-4o --ratio 0.5

    # Custom output suffix
    python scripts/augment_dataset_circuit_understanding.py data/finetune.json --model gpt-4o --suffix custom
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Optional

import mimetypes
from openai import AsyncOpenAI

from vllm import LLM, SamplingParams
from vllm.model_executor.layers.quantization.bitsandbytes import (
    BitsAndBytesConfig,
)

from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_async

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.7
DEFAULT_SEED = 42
SUPPORTED_OPENAI_MODELS = ("gpt-5.2", "gpt-5.1", "gpt-4o", "gpt-4o-mini")


def encode_image(image_path: str) -> tuple[str, str]:
    """Encode an image file to base64 string and detect MIME type.

    Args:
        image_path: Path to the image file.

    Returns:
        Tuple of (base64-encoded string, MIME type).

    Raises:
        FileNotFoundError: If the image file does not exist.
        IOError: If the file cannot be read.
    """
    with open(image_path, "rb") as image_file:
        base64_data = base64.b64encode(image_file.read()).decode("utf-8")

    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or "image/jpeg"

    return base64_data, mime_type


def resolve_image_path(base_dir: str, rel_path: str) -> str:
    """Resolve the absolute path of an image.

    Args:
        base_dir: Base directory for relative path resolution.
        rel_path: Relative path to the image.

    Returns:
        Absolute path to the image.
    """
    return os.path.join(base_dir, rel_path)


def generate_augment_prompt(question: str, answer: str) -> str:
    """Generate the prompt for the augmentation model.

    Creates a detailed prompt instructing the model to provide step-by-step
    Chain of Thought reasoning for circuit analysis questions.

    Args:
        question: The circuit analysis question.
        answer: The correct answer to guide the reasoning.

    Returns:
        Formatted prompt string.
    """
    return f"""You are an expert in digital circuit and electrical engineering.
Analyze the provided image and Question to derive the Correct Answer.
Write out the logical reasoning process naturally as human expert would do.

Constraint:
- Ensure your explanation concludes with the Correct Answer provided below.
- Output plain text only (no Markdown).

Question: {question}
Correct Answer: {answer}"""


def extract_thinking_content(text: str) -> str:
    """Extract content from <think> tags for Qwen3-VL Thinking models.

    Qwen3-VL Thinking models wrap their reasoning process in <think> tags.
    This function extracts the content inside these tags, or returns the
    original text if no tags are found.

    Args:
        text: Raw model output that may contain <think> tags.

    Returns:
        Extracted reasoning content without <think> tags.
    """
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()
    return text.strip()


def is_thinking_model(model_name: str) -> bool:
    """Check if the model is a Thinking variant.

    Args:
        model_name: Name or path of the model.

    Returns:
        True if the model name contains 'thinking', False otherwise.
    """
    return "thinking" in model_name.lower()


def is_openai_model(model_name: str) -> bool:
    """Check if the model is an OpenAI model.

    Args:
        model_name: Name or path of the model.

    Returns:
        True if the model is an OpenAI model, False otherwise.
    """
    return model_name.lower().startswith(SUPPORTED_OPENAI_MODELS)


def parse_conversation_item(
    item: dict[str, Any],
) -> tuple[str | None, str, str | None]:
    """Parse a conversation item to extract question, answer, and image path.

    Args:
        item: Dataset item containing conversations.

    Returns:
        Tuple of (question_text, original_answer, image_rel_path).
        Returns None for question_text and image_rel_path if parsing fails.
        original_answer defaults to empty string.
    """
    if "conversations" not in item:
        return None, "", None

    user_turn = None
    assistant_turn = None

    for turn in item["conversations"]:
        if turn.get("from") == "human":
            user_turn = turn
        elif turn.get("from") == "gpt":
            assistant_turn = turn

    if not user_turn or not assistant_turn:
        return None, "", None

    question_text = user_turn.get("value", "").replace("<image>", "").strip()
    original_answer = assistant_turn.get("value", "")
    image_rel_path = item.get("image")

    return question_text, original_answer, image_rel_path


async def process_item_with_openai(
    client: Any,
    model_name: str,
    item: dict[str, Any],
    image_base_dir: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> None:
    """Process a single item using OpenAI API asynchronously.

    Args:
        client: AsyncOpenAI client instance.
        model_name: OpenAI model name.
        item: Dataset item to process.
        image_base_dir: Base directory for image path resolution.
        max_tokens: Maximum tokens to generate.
        semaphore: Semaphore to limit concurrent requests.
    """
    question_text, original_answer, image_rel_path = parse_conversation_item(
        item
    )

    if not question_text or not image_rel_path:
        return

    assistant_turn = next(
        (t for t in item.get("conversations", []) if t.get("from") == "gpt"),
        None,
    )
    if not assistant_turn:
        return

    abs_image_path = resolve_image_path(image_base_dir, image_rel_path)
    if not os.path.exists(abs_image_path):
        logger.warning(f"Image not found: {abs_image_path}")
        return

    base64_image, mime_type = encode_image(abs_image_path)
    prompt = generate_augment_prompt(question_text, original_answer)

    async with semaphore:
        try:
            # Determine the correct parameter name for max tokens based on model
            if model_name.startswith("gpt-5"):
                max_tokens_param = {"max_completion_tokens": max_tokens}
            else:
                max_tokens_param = {"max_tokens": max_tokens}

            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                **max_tokens_param,
            )
            reasoning = response.choices[0].message.content

            # Use the model's generated output directly without manual prefix
            assistant_turn["value"] = reasoning

        except Exception as e:
            logger.error(f"Error processing item with OpenAI: {e}")


async def process_with_openai_async(
    model_name: str,
    items: list[dict[str, Any]],
    image_base_dir: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_concurrent: int = 10,
) -> list[dict[str, Any]]:
    """Process items using OpenAI API asynchronously.

    Args:
        model_name: OpenAI model name (e.g., 'gpt-4o', 'gpt-4o-mini').
        items: List of dataset items to process.
        image_base_dir: Base directory for image path resolution.
        max_tokens: Maximum tokens to generate.
        max_concurrent: Maximum number of concurrent API calls.

    Returns:
        List of processed items with augmented reasoning.
    """
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [
        process_item_with_openai(
            client, model_name, item, image_base_dir, max_tokens, semaphore
        )
        for item in items
    ]

    for task in tqdm_async.as_completed(tasks, desc="Augmenting with OpenAI"):
        await task

    return items


def process_with_openai(
    model_name: str,
    items: list[dict[str, Any]],
    image_base_dir: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_concurrent: int = 10,
) -> list[dict[str, Any]]:
    """Process items using OpenAI API (synchronous wrapper).

    Args:
        model_name: OpenAI model name (e.g., 'gpt-4o', 'gpt-4o-mini').
        items: List of dataset items to process.
        image_base_dir: Base directory for image path resolution.
        max_tokens: Maximum tokens to generate.
        max_concurrent: Maximum number of concurrent API calls.

    Returns:
        List of processed items with augmented reasoning.
    """
    return asyncio.run(
        process_with_openai_async(
            model_name, items, image_base_dir, max_tokens, max_concurrent
        )
    )


def chunk_list(items: list, batch_size: int) -> list:
    """Split a list into chunks of specified size.

    Args:
        items: List to split.
        batch_size: Size of each chunk.

    Returns:
        List of chunks.
    """
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def process_with_vllm(
    model_name: str,
    items: list[dict[str, Any]],
    image_base_dir: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    batch_size: int = 1,
    quantization: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Process items using local VLLM.

    Supports multimodal models like Qwen3-VL. Automatically handles
    <think> tags for Thinking model variants. Supports batch processing
    for improved throughput and quantization for memory efficiency.

    Args:
        model_name: Local model path or HuggingFace model ID.
        items: List of dataset items to process.
        image_base_dir: Base directory for image path resolution.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        batch_size: Number of items to process in a batch. Default: 1.
        quantization: Quantization method ('4bit' or '8bit' using BitsAndBytes). Default: None.

    Returns:
        List of processed items with augmented reasoning.
    """
    logger.info(f"Initializing VLLM with model: {model_name}")
    quantization_config = None
    if quantization == "4bit":
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        logger.info("Using 4-bit quantization with BitsAndBytes")
    elif quantization == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        logger.info("Using 8-bit quantization with BitsAndBytes")

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        quantization="bitsandbytes" if quantization_config else None,
        quantization_config=quantization_config,
    )
    sampling_params = SamplingParams(
        temperature=temperature, max_tokens=max_tokens
    )

    is_thinking = is_thinking_model(model_name)
    if is_thinking:
        logger.info("Detected Thinking model - will parse <think> tags")

    if batch_size > 1:
        logger.info(f"Using batch processing with batch_size={batch_size}")

    # Process items in batches
    chunks = chunk_list(items, batch_size)

    for chunk in tqdm(chunks, desc="Augmenting with VLLM"):
        conversations = []
        chunk_metadata = (
            []
        )  # Store (item, assistant_turn, original_answer) for each valid item

        # Prepare batch
        for item in chunk:
            question_text, original_answer, image_rel_path = (
                parse_conversation_item(item)
            )

            if not question_text or not image_rel_path:
                continue

            assistant_turn = next(
                (
                    t
                    for t in item.get("conversations", [])
                    if t.get("from") == "gpt"
                ),
                None,
            )
            if not assistant_turn:
                continue

            abs_image_path = resolve_image_path(image_base_dir, image_rel_path)
            if not os.path.exists(abs_image_path):
                logger.warning(f"Image not found: {abs_image_path}")
                continue

            prompt = generate_augment_prompt(question_text, original_answer)

            # Construct conversation for VLLM
            # Note: VLLM uses "image_url" type for local file paths
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": abs_image_path},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            conversations.append(conversation)
            chunk_metadata.append((item, assistant_turn, original_answer))

        if not conversations:
            continue

        # Process batch
        try:
            if len(conversations) == 1:
                # Single item - use simple chat
                outputs = llm.chat(
                    messages=conversations[0],
                    sampling_params=sampling_params,
                )
            else:
                # Batch processing
                outputs = llm.chat(
                    messages=conversations,
                    sampling_params=sampling_params,
                )

            # Update items with generated outputs
            for i, output in enumerate(outputs):
                item, assistant_turn, original_answer = chunk_metadata[i]
                generated_text = output.outputs[0].text

                # Extract thinking content if using Thinking model
                if is_thinking:
                    generated_text = extract_thinking_content(generated_text)

                # Use the model's generated output directly without manual prefix
                assistant_turn["value"] = generated_text

        except Exception as e:
            logger.error(f"Error processing batch with VLLM: {e}")
            continue

    return items


def load_dataset(input_file: str) -> list[dict[str, Any]]:
    """Load dataset from JSON file.

    Args:
        input_file: Path to the input JSON file.

    Returns:
        Loaded dataset as a list of dictionaries.

    Raises:
        FileNotFoundError: If the input file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset(data: list[dict[str, Any]], output_path: str) -> None:
    """Save dataset to JSON file.

    Args:
        data: Dataset to save.
        output_path: Path to the output file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def sample_items(
    data: list[dict[str, Any]],
    ratio: float,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Sample items from dataset based on ratio.

    Args:
        data: Full dataset.
        ratio: Sampling ratio (0.0 to 1.0).
        seed: Random seed for reproducibility.

    Returns:
        Sampled subset of items.
    """
    if ratio >= 1.0:
        return data[:]

    random.seed(seed)
    sample_size = int(len(data) * ratio)
    indices_to_augment = set(random.sample(range(len(data)), sample_size))
    return [data[i] for i in indices_to_augment]


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Augment circuit dataset with CoT reasoning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using OpenAI API
  python %(prog)s data/finetune.json --model gpt-4o

  # Using OpenAI API with higher concurrency (faster processing)
  python %(prog)s data/finetune.json --model gpt-4o --max-concurrent 20

  # Using local VLLM with Qwen3-VL
  python %(prog)s data/finetune.json --model Qwen/Qwen3-VL-30B-A3B-Instruct

  # Using Qwen3-VL Thinking model
  python %(prog)s data/finetune.json --model Qwen/Qwen3-VL-30B-A3B-Thinking

  # Using VLLM with 4-bit quantization
  python %(prog)s data/finetune.json --model Qwen/Qwen3-VL-30B-A3B-Instruct --quantization 4bit

  # Augment only 50%% of the data
  python %(prog)s data/finetune.json --model gpt-4o --ratio 0.5

  # Custom output suffix
  python %(prog)s data/finetune.json --model gpt-4o --suffix custom
        """,
    )
    parser.add_argument(
        "input_file",
        help="Path to the input *_finetune.json file",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to use: gpt-4o, gpt-4o-mini, or local model path/ID",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        help="Ratio of data to augment (0.0 to 1.0). Default: 1.0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for sampling. Default: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum tokens to generate. Default: {DEFAULT_MAX_TOKENS}",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature for VLLM. Default: {DEFAULT_TEMPERATURE}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for VLLM processing. Higher values improve throughput but require more memory. Default: 1",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["4bit", "8bit"],
        help="Quantization method for VLLM. Options: 4bit, 8bit. Default: None",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum number of concurrent API calls (OpenAI/Gemini). Default: 10",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="augmented",
        help="Suffix to append to the output filename. Default: augmented",
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    # Determine image base directory
    image_base_dir = str(input_path.parent)

    # Load dataset
    logger.info(f"Loading data from {args.input_file}...")
    data = load_dataset(args.input_file)
    logger.info(f"Total items: {len(data)}")

    # Sample items
    items_to_process = sample_items(data, args.ratio, args.seed)
    if args.ratio < 1.0:
        logger.info(
            f"Sampled {len(items_to_process)} items (ratio={args.ratio})"
        )

    # Process with appropriate backend
    if is_openai_model(args.model):
        process_with_openai(
            model_name=args.model,
            items=items_to_process,
            image_base_dir=image_base_dir,
            max_tokens=args.max_tokens,
            max_concurrent=args.max_concurrent,
        )
    else:
        process_with_vllm(
            model_name=args.model,
            items=items_to_process,
            image_base_dir=image_base_dir,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
            quantization=args.quantization,
        )

    # Save output
    # Create output filename: input_{suffix}.json
    output_path = (
        input_path.parent
        / f"{input_path.stem}_{args.suffix}{input_path.suffix}"
    )

    logger.info(f"Saving augmented dataset to {output_path}...")
    save_dataset(items_to_process, str(output_path))
    logger.info("Done.")


if __name__ == "__main__":
    main()
