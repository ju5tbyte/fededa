"""Model factory for building model instances from configuration.

This module provides a registry pattern for instantiating models based on
configuration. New models should be registered in MODEL_REGISTRY.
"""

from typing import Union

from omegaconf import DictConfig

from src.models.qwen3_model import Qwen3Model
from src.models.qwen_vl_model import QwenVLModel
from src.models.qwen_vl_finetuned_model import QwenVLFinetunedModel

MODEL_REGISTRY = {
    "QwenVLModel": QwenVLModel,
    "Qwen3Model": Qwen3Model,
    "QwenVLFinetunedModel": QwenVLFinetunedModel,
}


def build_model(model_cfg: DictConfig) -> Union[QwenVLModel, Qwen3Model]:
    """Build and return a model instance from configuration.

    Args:
        model_cfg: Configuration containing model name and parameters.

    Returns:
        Instantiated model object.

    Raises:
        ValueError: If the specified model name is not found in the registry.
    """
    model_name = model_cfg.name
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in registry")

    return MODEL_REGISTRY[model_name](**model_cfg.params)
