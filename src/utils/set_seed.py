import random
import os
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Fix random seeds for all libraries to ensure reproducibility

    Args:
        seed (int): The seed value to set (default: 42)

    Raises:
        ValueError: If seed is not a positive integer
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a positive integer")

    # Set seeds for all libraries
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # PyTorch specific settings
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU

    # CuDNN settings (may slightly reduce speed but ensures reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed}.")
