"""Reproducibility utilities for fixing random seeds."""

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for all libraries to ensure reproducibility.

    Args:
        seed: The seed value to set (default: 42).

    Raises:
        ValueError: If seed is not a positive integer.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a positive integer")

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("Random seed set to %d.", seed)
