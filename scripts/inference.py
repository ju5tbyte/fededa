#!/usr/bin/env python3
"""Inference script for models in src/models.

This script loads a model configuration, builds the model, and performs inference
on sample inputs. Supports both text-only and multimodal models.
"""

import argparse
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from PIL import Image

from src.models import build_model


def main():
    parser = argparse.ArgumentParser(description="Run inference with a model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the model configuration YAML file.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="What is the capital of France?",
        help="Question to ask the model.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an image file (for multimodal models).",
    )
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    model_cfg = OmegaConf.load(config_path)
    if not isinstance(model_cfg, OmegaConf.DictConfig):
        raise ValueError("Configuration must be a dictionary.")

    # Build model
    model = build_model(model_cfg)

    # Prepare inputs
    question = args.question
    imgs = None
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        imgs = [Image.open(image_path)]

    # Perform inference
    if model.modality == "text":
        answer = model(question)  # type: ignore
    elif model.modality == "multimodal":
        answer = model(question, imgs)  # type: ignore
    else:
        raise ValueError(f"Unsupported modality: {model.modality}")

    # Print result
    print("Question:", question)
    if imgs:
        print("Image:", args.image)
    print("Answer:", answer)


if __name__ == "__main__":
    main()
