"""Quantization configuration utilities.
"""

from typing import Optional, Union

import torch
from omegaconf import DictConfig
from transformers import BitsAndBytesConfig


def prepare_qwen_quantization_config(
    quantization: Optional[Union[DictConfig, dict]],
    dtype: torch.dtype,
) -> Optional[BitsAndBytesConfig]:
    """Prepare quantization configuration for Qwen family models.

    Supports 4-bit and 8-bit quantization using bitsandbytes for Qwen models.
    This function is specifically designed for Qwen family models (Qwen3, Qwen3-VL, etc.)
    and may not be compatible with other model families.

    Args:
        quantization: Quantization configuration dict or DictConfig containing:
            - enabled (bool): Whether to enable quantization.
            - method (str): Quantization method (currently only 'bitsandbytes').
            - bits (int): Number of bits (4 or 8).
            - load_in_8bit (bool): Alternative way to specify 8-bit quantization.
            - load_in_4bit (bool): Alternative way to specify 4-bit quantization.
            - bnb_4bit_compute_dtype (str): Compute dtype for 4-bit ('float16', 'bfloat16', 'float32').
            - bnb_4bit_use_double_quant (bool): Use double quantization (default: True).
            - bnb_4bit_quant_type (str): Quantization type ('nf4' or 'fp4', default: 'nf4').
        dtype: Default torch dtype for compute.

    Returns:
        BitsAndBytesConfig if quantization is enabled, None otherwise.

    Raises:
        ValueError: If quantization method is unsupported or config is invalid.

    Example:
        >>> quantization_cfg = {
        ...     "enabled": True,
        ...     "method": "bitsandbytes",
        ...     "bits": 4,
        ...     "bnb_4bit_compute_dtype": "float16",
        ...     "bnb_4bit_use_double_quant": True,
        ...     "bnb_4bit_quant_type": "nf4"
        ... }
        >>> config = prepare_qwen_quantization_config(quantization_cfg, torch.float16)
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

    if bits == 8 or load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    elif bits == 4 or load_in_4bit:
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