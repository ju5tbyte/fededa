"""Qwen Vision-Language Finetuned Model Wrapper.

This module provides an interface for Qwen3-VL model with LoRA finetuning.
"""

from typing import Optional, Union

import torch
from omegaconf import DictConfig
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from src.utils.quantization import prepare_qwen_quantization_config


class QwenVLFinetunedModel:
    """Qwen Vision-Language Finetuned Model for multimodal question answering.

    Attributes:
        modality (str): The modality type, always 'multimodal'.
        model (PeftModel): The loaded Qwen3-VL model with LoRA adapter.
        processor (AutoProcessor): The processor for input formatting.
        device (str): The device to run the model on (e.g., 'cuda:0', 'cpu').
        max_new_tokens (int): Maximum number of tokens to generate.
        min_pixels (int): Minimum number of pixels for image processing.
        max_pixels (int): Maximum number of pixels for image processing.
    """

    def __init__(
        self,
        model_name: str,
        finetune_path: str,
        device: str = "cuda:0",
        torch_dtype: str = "float16",
        attn_implementation: str = "sdpa",
        max_new_tokens: int = 512,
        quantization: Optional[Union[DictConfig, dict]] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
    ):
        """Initialize the Qwen3-VL finetuned model with LoRA adapter and optional quantization.

        Args:
            model_name: HuggingFace model identifier (e.g., 'Qwen/Qwen3-VL-8B-Instruct').
            finetune_path: Path to the LoRA finetune checkpoint directory.
            device: Device to run the model on (default: 'cuda:0').
            torch_dtype: Data type for model weights (default: 'float16').
            attn_implementation: Attention implementation (default: 'sdpa').
            max_new_tokens: Maximum number of tokens to generate (default: 512).
            quantization: Optional quantization configuration dict or DictConfig.
            min_pixels: Minimum number of pixels for image processing (default: 256*28*28).
            max_pixels: Maximum number of pixels for image processing (default: 1280*28*28).

        Raises:
            ValueError: If model_name or finetune_path is not provided or quantization config is invalid.
        """
        if not model_name:
            raise ValueError("model_name must be provided")
        if not finetune_path:
            raise ValueError("finetune_path must be provided")

        self.modality = "multimodal"
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

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

        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, **model_kwargs
        )

        self.model = PeftModel.from_pretrained(base_model, finetune_path)

        if quantization_config is None:
            self.model = self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_name)

    def __call__(
        self, question: str, imgs: Optional[list[Image.Image]] = None
    ) -> str:
        """Generate an answer to a question with optional images.

        Args:
            question: The text question to answer.
            imgs: Optional list of PIL Images to use as visual context.

        Returns:
            The generated text answer as a string.
        """
        with torch.no_grad():
            if imgs is None:
                imgs = []

            content = []
            for img in imgs:
                content.append({"type": "image", "image": img})

            content.append({"type": "text", "text": question})

            messages = [{"role": "user", "content": content}]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )

            inputs.pop("token_type_ids", None)

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            return output_text[0]
