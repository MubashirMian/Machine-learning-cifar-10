"""Reproducibility utilities: setting random seeds across libraries."""
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True):
    """Sets random seeds for Python, NumPy, and PyTorch (CPU, CUDA, MPS).

    Args:
        seed: Integer seed value.
        deterministic: If True, sets cudnn flags for deterministic behavior.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS uses torch.manual_seed directly
        pass
