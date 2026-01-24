"""Qwen3 text-only Model wrapper."""

from typing import Optional, Union

import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.quantization import prepare_qwen_quantization_config


class Qwen3Model:
    """Qwen3 text-only Model for question answering.

    Attributes:
        modality (str): The modality type, always 'text'.
        model (AutoModelForCausalLM): The loaded Qwen3 model.
        tokenizer (AutoTokenizer): The tokenizer for input formatting.
        device (str): The device to run the model on (e.g., 'cuda:0', 'cpu').
        max_new_tokens (int): Maximum number of tokens to generate.
        enable_thinking (bool): Whether to enable thinking mode for reasoning.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        torch_dtype: str = "float16",
        attn_implementation: str = "sdpa",
        max_new_tokens: int = 512,
        enable_thinking: bool = False,
        quantization: Optional[Union[DictConfig, dict]] = None,
    ):
        """Initialize the Qwen3 model with optional quantization.

        Args:
            model_name: HuggingFace model identifier (e.g., 'Qwen/Qwen3-8B').
            device: Device to run the model on (default: 'cuda:0').
            torch_dtype: Data type for model weights (default: 'float16').
            attn_implementation: Attention implementation (default: 'sdpa').
            max_new_tokens: Maximum number of tokens to generate (default: 512).
            enable_thinking: Enable thinking mode for reasoning tasks (default: False).
            quantization: Optional quantization configuration dict or DictConfig.

        Raises:
            ValueError: If model_name is not provided or quantization config is invalid.
        """
        if not model_name:
            raise ValueError("model_name must be provided")

        self.modality = "text"
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(torch_dtype, torch.float16)

        quantization_config = prepare_qwen_quantization_config(
            quantization, dtype
        )

        model_kwargs = {
            "torch_dtype": dtype,
            "attn_implementation": attn_implementation,
        }

        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs
        )

        if quantization_config is None:
            self.model = self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, question: str) -> str:
        """Generate an answer to a text question.

        Processes the input question through the model to generate a text response.

        When enable_thinking is True, the model outputs thinking content wrapped
        in <think>...</think> tags followed by the final answer. This method
        automatically parses and returns only the final answer.

        Args:
            question: The text question to answer.

        Returns:
            The generated text answer as a string. If thinking mode is enabled,
            returns only the final answer (thinking content is parsed out).

        Raises:
            RuntimeError: If model inference fails.
        """
        with torch.no_grad():
            messages = [{"role": "user", "content": question}]

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )

            inputs = self.tokenizer([text], return_tensors="pt")

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Follow the recommended options from Qwen Official Docs
            if self.enable_thinking:
                generation_config = {
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                    "do_sample": True,
                }
            else:
                generation_config = {
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "do_sample": True,
                }

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                **generation_config,
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]

            output_ids = generated_ids_trimmed[0].tolist()

            # Trim ...</think> for thinking mode
            if self.enable_thinking:
                try:
                    index = len(output_ids) - output_ids[::-1].index(151668)
                except ValueError:
                    index = 0

                content = self.tokenizer.decode(
                    output_ids[index:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip("\n")

                return content

            output_text = self.tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip("\n")

            return output_text
