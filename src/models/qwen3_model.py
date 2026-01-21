"""Qwen3 text-only Model wrapper for text-based benchmarks.

This module provides a unified interface for Qwen3 text-only models with support
for quantization using bitsandbytes, AWQ, or GPTQ.
"""

from typing import Optional, Union

import torch
from omegaconf import DictConfig
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


class Qwen3Model:
    """Qwen3 text-only Model for question answering.

    This class wraps the Qwen3 model series for use with text-based benchmarks
    like ORD-QA. It provides a unified interface for processing text inputs to
    generate answers, with optional quantization support.

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

        # Convert torch_dtype string to torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(torch_dtype, torch.float16)

        # Prepare quantization configuration
        quantization_config = self._prepare_quantization_config(
            quantization, dtype
        )

        # Prepare model loading arguments
        model_kwargs = {
            "torch_dtype": dtype,
            "attn_implementation": attn_implementation,
        }

        # Add quantization config and device_map if enabled
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            # Use device_map="auto" for quantized models (required for quantization)
            model_kwargs["device_map"] = "auto"

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs
        )

        # Explicitly move model to device if not using quantization
        # (quantization uses device_map="auto" which already handles device placement)
        if quantization_config is None:
            self.model = self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _prepare_quantization_config(
        self,
        quantization: Optional[Union[DictConfig, dict]],
        dtype: torch.dtype,
    ) -> Optional[BitsAndBytesConfig]:
        """Prepare quantization configuration from config dict.

        Args:
            quantization: Quantization configuration dict or DictConfig.
            dtype: Default torch dtype for compute.

        Returns:
            BitsAndBytesConfig if quantization is enabled, None otherwise.

        Raises:
            ValueError: If quantization method is unsupported or config is invalid.
        """
        if quantization is None or not quantization.get("enabled", False):
            return None

        method = quantization.get("method", "bitsandbytes")

        if method != "bitsandbytes":
            raise ValueError(
                f"Unsupported quantization method: {method}. "
                "Currently only 'bitsandbytes' is supported."
            )

        bits = quantization.get("bits", 8)
        load_in_8bit = quantization.get("load_in_8bit", False)
        load_in_4bit = quantization.get("load_in_4bit", False)

        # Determine quantization mode
        if bits == 8 or load_in_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)
        elif bits == 4 or load_in_4bit:
            # Get 4-bit quantization parameters
            compute_dtype_str = quantization.get(
                "bnb_4bit_compute_dtype", "float16"
            )
            compute_dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            compute_dtype = compute_dtype_map.get(compute_dtype_str, dtype)

            use_double_quant = quantization.get(
                "bnb_4bit_use_double_quant", True
            )
            quant_type = quantization.get("bnb_4bit_quant_type", "nf4")

            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=use_double_quant,
                bnb_4bit_quant_type=quant_type,
            )
        else:
            raise ValueError(
                f"Unsupported quantization bits: {bits}. "
                "Supported values are 4 or 8."
            )

    def __call__(
        self, question: str, imgs: Optional[list[Image.Image]] = None
    ) -> str:
        """Generate an answer to a text question.

        Processes the input question through the model to generate a text response.
        The imgs parameter is ignored as this is a text-only model, but is kept
        for interface compatibility with vision-language models.

        When enable_thinking is True, the model outputs thinking content wrapped
        in <think>...</think> tags followed by the final answer. This method
        automatically parses and returns only the final answer.

        Args:
            question: The text question to answer.
            imgs: Ignored (for interface compatibility).

        Returns:
            The generated text answer as a string. If thinking mode is enabled,
            returns only the final answer (thinking content is parsed out).

        Raises:
            RuntimeError: If model inference fails.
        """
        with torch.no_grad():
            # Prepare message
            messages = [{"role": "user", "content": question}]

            # Apply chat template with thinking mode setting
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )

            # Tokenize
            inputs = self.tokenizer([text], return_tensors="pt")

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Prepare generation parameters based on thinking mode
            # According to Qwen3 documentation:
            # - Thinking mode: Temperature=0.6, TopP=0.95, TopK=20
            # - Non-thinking mode: Temperature=0.7, TopP=0.8, TopK=20
            if self.enable_thinking:
                generation_config = {
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                    "do_sample": True,  # Must use sampling, not greedy decoding
                }
            else:
                generation_config = {
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "do_sample": True,
                }

            # Generate response
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                **generation_config,
            )

            # Trim input tokens from generated output
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]

            # Extract output token IDs (first batch item)
            output_ids = generated_ids_trimmed[0].tolist()

            # Parse thinking content if thinking mode is enabled
            if self.enable_thinking:
                # 151668 is the token ID for </think>
                try:
                    # Find the index of </think> token from the end
                    index = len(output_ids) - output_ids[::-1].index(151668)
                except ValueError:
                    # No </think> token found, return everything
                    index = 0

                # Decode only the content after </think> (final answer)
                content = self.tokenizer.decode(
                    output_ids[index:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip("\n")

                return content
            else:
                # Non-thinking mode: decode everything
                output_text = self.tokenizer.decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip("\n")

                return output_text
