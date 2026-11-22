"""Dataset loaders, stratified splitting, and subset sampling for SimCLR on CIFAR-10."""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
from torchvision import datasets

from src.simclr.augment import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    SimCLRPairTransform,
    get_eval_transform,
    get_linear_train_transform,
    get_simclr_pretrain_transform,
)


class SimCLRDatasetWrapper(Dataset):
    """Wraps an underlying dataset (like torchvision CIFAR10) to yield two augmented views.

    Labels are explicitly omitted from the return signature during pretraining
    to guarantee unsupervised contrastive learning.
    """

    def __init__(self, base_dataset: Dataset, transform: SimCLRPairTransform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, _ = self.base_dataset[idx]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        view1, view2 = self.transform(img)
        return view1, view2


class TransformedSubset(Dataset):
    """Wraps a Subset of a dataset with a specific torchvision transform."""

    def __init__(self, base_dataset: Dataset, indices: List[int], transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        actual_idx = self.indices[idx]
        img, label = self.base_dataset[actual_idx]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(label)


def create_or_load_splits(
    labels: np.ndarray,
    split_dir: Union[str, Path] = "data/splits",
    val_size: int = 5000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates or loads reproducible stratified train/validation index splits.

    Args:
        labels: Array of integer class labels (50,000 for CIFAR-10).
        split_dir: Directory where split indices are saved.
        val_size: Total validation samples (default 5,000 = 500/class).
        seed: Random seed for reproducibility.

    Returns:
        (train_indices, val_indices) where len(train_indices) == 45,000 and len(val_indices) == 5,000.
    """
    split_path = Path(split_dir)
    split_path.mkdir(parents=True, exist_ok=True)
    val_file = split_path / "val_idx.npy"
    train_file = split_path / "train_idx.npy"

    if val_file.exists() and train_file.exists():
        val_idx = np.load(val_file)
        train_idx = np.load(train_file)
        return train_idx, val_idx

    rng = np.random.RandomState(seed)
    num_classes = len(np.unique(labels))
    samples_per_class = val_size // num_classes

    val_indices = []
    train_indices = []

    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        shuffled = rng.permutation(class_indices)
        val_indices.extend(shuffled[:samples_per_class])
        train_indices.extend(shuffled[samples_per_class:])

    val_idx = np.array(sorted(val_indices), dtype=np.int64)
    train_idx = np.array(sorted(train_indices), dtype=np.int64)

    np.save(val_file, val_idx)
    np.save(train_file, train_idx)
    return train_idx, val_idx


def get_stratified_subsets(
    train_indices: np.ndarray,
    labels: np.ndarray,
    fractions: List[float],
    seed: int = 42,
) -> Dict[float, np.ndarray]:
    """Generates nested stratified subsets of training indices for data-fraction experiments.

    Ensures that for fractions f1 < f2, subset(f1) is a strict subset of subset(f2).
    None of the returned indices overlap with validation indices.

    Args:
        train_indices: Pool of available training indices (e.g. 45,000).
        labels: Full labels array.
        fractions: List of fractions (e.g. [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]).
        seed: Random seed.

    Returns:
        Dictionary mapping fraction -> numpy array of selected indices.
    """
    rng = np.random.RandomState(seed)
    num_classes = len(np.unique(labels))
    pool_labels = labels[train_indices]

    sorted_fractions = sorted(fractions)
    per_class_shuffled = {}
    for c in range(num_classes):
        class_pos = np.where(pool_labels == c)[0]
        per_class_shuffled[c] = rng.permutation(class_pos)

    subset_dict = {}
    total_class_size = len(per_class_shuffled[0])  # 4500

    for frac in sorted_fractions:
        count_per_class = max(1, int(round(total_class_size * frac)))
        chosen_positions = []
        for c in range(num_classes):
            chosen_positions.extend(per_class_shuffled[c][:count_per_class])
        chosen_train_indices = train_indices[chosen_positions]
        subset_dict[frac] = np.array(sorted(chosen_train_indices), dtype=np.int64)

    return subset_dict


def get_cifar10_raw_datasets(data_dir: str = "data"):
    """Loads CIFAR-10 datasets from raw batches without transforms."""
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=False)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=False)
    return train_dataset, test_dataset


def get_pretrain_dataloader(
    data_dir: str = "data",
    split_dir: str = "data/splits",
    batch_size: int = 256,
    subset_fraction: float = 1.0,
    seed: int = 42,
    normalize: bool = True,
    num_workers: int = 2,
    drop_last: bool = True,
) -> DataLoader:
    """Builds DataLoader for SimCLR pretraining."""
    raw_train, _ = get_cifar10_raw_datasets(data_dir)
    labels = np.array(raw_train.targets)
    train_idx, _ = create_or_load_splits(labels, split_dir=split_dir, seed=seed)

    if subset_fraction < 1.0:
        subsets = get_stratified_subsets(train_idx, labels, [subset_fraction], seed=seed)
        active_indices = subsets[subset_fraction]
    else:
        active_indices = train_idx

    pretrain_transform = get_simclr_pretrain_transform(normalize=normalize)
    pair_transform = SimCLRPairTransform(pretrain_transform)

    base_subset = Subset(raw_train, active_indices.tolist())
    simclr_dataset = SimCLRDatasetWrapper(base_subset, pair_transform)

    loader = DataLoader(
        simclr_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )
    return loader


def get_linear_dataloaders(
    data_dir: str = "data",
    split_dir: str = "data/splits",
    batch_size: int = 256,
    seed: int = 42,
    normalize: bool = True,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Builds DataLoaders for linear evaluation: (train_loader, val_loader, test_loader)."""
    raw_train, raw_test = get_cifar10_raw_datasets(data_dir)
    labels = np.array(raw_train.targets)
    train_idx, val_idx = create_or_load_splits(labels, split_dir=split_dir, seed=seed)

    linear_train_transform = get_linear_train_transform(normalize=normalize)
    eval_transform = get_eval_transform(normalize=normalize)

    train_ds = TransformedSubset(raw_train, train_idx.tolist(), linear_train_transform)
    val_ds = TransformedSubset(raw_train, val_idx.tolist(), eval_transform)
    test_ds = TransformedSubset(raw_test, list(range(len(raw_test))), eval_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader
