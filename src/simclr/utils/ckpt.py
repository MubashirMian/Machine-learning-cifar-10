"""Checkpoint saving and loading utilities for reproducible training and resume."""
import os
from pathlib import Path
import random
from typing import Any, Dict, Optional
import numpy as np
import torch


def get_rng_states() -> Dict[str, Any]:
    """Captures current RNG states across all generators."""
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: Dict[str, Any]):
    """Restores RNG states from captured dictionary."""
    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        np.random.set_state(states["numpy"])
    if "torch" in states:
        torch_state = states["torch"]
        if isinstance(torch_state, torch.Tensor):
            torch_state = torch_state.cpu().to(torch.uint8)
        torch.set_rng_state(torch_state)
    if "cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])


def save_checkpoint(
    path: str,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    best_metric: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
):
    """Saves complete checkpoint dictionary to disk."""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": best_metric,
        "rng_states": get_rng_states(),
        "config": config,
    }
    torch.save(state, save_path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Loads state from checkpoint and updates model, optimizer, scheduler, and RNGs."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No checkpoint found at '{path}'")

    map_location = device if device is not None else "cpu"
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if "rng_states" in checkpoint:
        set_rng_states(checkpoint["rng_states"])

    return checkpoint
