from src.models.qwen_vl_model import QwenVLModel
from omegaconf import DictConfig

# Registry mapping model names to their corresponding classes.
MODEL_REGISTRY = {
    "QwenVLModel": QwenVLModel,
}


def build_model(model_cfg: DictConfig):
    """Builds and returns a model instance based on the provided configuration.

    Args:
        model_cfg (DictConfig): Configuration containing model name and parameters.

    Returns: Instantiated model object.

    Raises:
        ValueError: If the specified model name is not found in the registry.
    """
    model_name = model_cfg.name
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in registry")

    return MODEL_REGISTRY[model_name](**model_cfg.params)
