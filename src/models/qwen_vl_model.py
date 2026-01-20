"""Qwen Vision-Language Model wrapper for MMCircuitEval benchmark.

This module provides a unified interface for Qwen3-VL models with support
for quantization using bitsandbytes, AWQ, or GPTQ.
"""

import base64
from io import BytesIO
from typing import Optional, Union

import torch
from omegaconf import DictConfig
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)


class QwenVLModel:
    """Qwen Vision-Language Model for multimodal question answering.

    This class wraps the Qwen3-VL model series for use with the MMCircuitEval
    benchmark. It provides a unified interface for processing text and image
    inputs to generate answers, with optional quantization support.

    Attributes:
        modality (str): The modality type, always 'multimodal'.
        model (Qwen3VLForConditionalGeneration): The loaded Qwen3-VL model.
        processor (AutoProcessor): The processor for input formatting.
        device (str): The device to run the model on (e.g., 'cuda:0', 'cpu').
        max_new_tokens (int): Maximum number of tokens to generate.
        min_pixels (int): Minimum number of pixels for image processing.
        max_pixels (int): Maximum number of pixels for image processing.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        torch_dtype: str = "float16",
        attn_implementation: str = "sdpa",
        max_new_tokens: int = 512,
        quantization: Optional[Union[DictConfig, dict]] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
    ):
        """Initialize the Qwen3-VL model with optional quantization.

        Args:
            model_name: HuggingFace model identifier (e.g., 'Qwen/Qwen3-VL-8B-Instruct').
            device: Device to run the model on (default: 'cuda:0').
            torch_dtype: Data type for model weights (default: 'float16').
            attn_implementation: Attention implementation (default: 'sdpa').
            max_new_tokens: Maximum number of tokens to generate (default: 512).
            quantization: Optional quantization configuration dict or DictConfig.
            min_pixels: Minimum number of pixels for image processing (default: 256*28*28).
            max_pixels: Maximum number of pixels for image processing (default: 1280*28*28).

        Raises:
            ValueError: If model_name is not provided or quantization config is invalid.
        """
        if not model_name:
            raise ValueError("model_name must be provided")

        self.modality = "multimodal"
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

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

        # Load model and processor
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, **model_kwargs
        )

        # Explicitly move model to device if not using quantization
        # (quantization uses device_map="auto" which already handles device placement)
        if quantization_config is None:
            self.model = self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_name)

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
        """Generate an answer to a question with optional images.

        Processes the input question and images (if provided) through the model
        to generate a text response. Images are directly passed as PIL Image objects.

        Args:
            question: The text question to answer.
            imgs: Optional list of PIL Images to use as visual context.

        Returns:
            The generated text answer as a string.

        Raises:
            RuntimeError: If model inference fails.
        """
        with torch.no_grad():
            if imgs is None:
                imgs = []

            # Prepare content with images and text
            content = []
            for img in imgs:
                # For Qwen3-VL, images can be passed directly as PIL Images
                content.append({"type": "image", "image": img})

            content.append({"type": "text", "text": question})

            # Create message with multimodal content
            messages = [{"role": "user", "content": content}]

            # Apply chat template with return_dict=True for Qwen3-VL
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )

            # Remove token_type_ids if present (not needed for generation)
            inputs.pop("token_type_ids", None)

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate response
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )

            # Trim input tokens from generated output
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]

            # Decode output
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            return output_text[0]
