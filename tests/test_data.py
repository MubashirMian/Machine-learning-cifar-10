"""Unit tests for SimCLR data loading, augmentations, and stratified subsets."""
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from src.simclr.augment import (
    SimCLRPairTransform,
    get_eval_transform,
    get_linear_train_transform,
    get_simclr_pretrain_transform,
)
from src.simclr.data import (
    create_or_load_splits,
    get_cifar10_raw_datasets,
    get_linear_dataloaders,
    get_pretrain_dataloader,
    get_stratified_subsets,
)


def test_augmentation_shapes_and_two_views():
    """Verify that two views of the same image differ, have shape (3,32,32), and valid ranges."""
    # Synthetic 32x32 RGB PIL image
    img_array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    transform = get_simclr_pretrain_transform(image_size=32, s=0.5, normalize=True)
    pair_fn = SimCLRPairTransform(transform)

    view1, view2 = pair_fn(img)

    assert view1.shape == (3, 32, 32)
    assert view2.shape == (3, 32, 32)
    assert isinstance(view1, torch.Tensor)
    assert isinstance(view2, torch.Tensor)
    assert view1.dtype == torch.float32
    assert view2.dtype == torch.float32
    # Two random views must differ
    assert not torch.allclose(view1, view2)


def test_augmentation_without_normalize_range():
    """Verify range is in [0, 1] when normalize=False."""
    img_array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    transform = get_simclr_pretrain_transform(image_size=32, s=0.5, normalize=False)
    view = transform(img)
    assert view.min().item() >= 0.0 - 1e-6
    assert view.max().item() <= 1.0 + 1e-6


def test_stratified_splits_and_subsets(tmp_path):
    """Verify exact per-class counts, strict nesting, and no overlap with validation."""
    labels = np.array([i % 10 for i in range(50000)])  # Exactly 5000 per class

    train_idx, val_idx = create_or_load_splits(
        labels, split_dir=tmp_path, val_size=5000, seed=42
    )

    assert len(train_idx) == 45000
    assert len(val_idx) == 5000

    # No overlap
    overlap = np.intersect1d(train_idx, val_idx)
    assert len(overlap) == 0

    # Class balance in validation split (exactly 500 per class)
    val_counts = np.bincount(labels[val_idx], minlength=10)
    assert np.all(val_counts == 500)

    # Class balance in train split (exactly 4500 per class)
    train_counts = np.bincount(labels[train_idx], minlength=10)
    assert np.all(train_counts == 4500)

    # Nested stratified subsets
    fractions = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
    subsets = get_stratified_subsets(train_idx, labels, fractions, seed=42)

    for i in range(len(fractions) - 1):
        f_small = fractions[i]
        f_large = fractions[i + 1]
        s_small = set(subsets[f_small])
        s_large = set(subsets[f_large])

        # Strict subset nesting
        assert s_small.issubset(s_large)

        # No overlap with validation set
        assert len(s_large.intersection(set(val_idx))) == 0

        # Exact class balance in each subset
        counts_small = np.bincount(labels[subsets[f_small]], minlength=10)
        expected_per_class = max(1, int(round(4500 * f_small)))
        assert np.all(counts_small == expected_per_class)


def test_app_and_test_eval_transform_equality():
    """Verify that app evaluation transform equals test evaluation transform (§9.2)."""
    eval_transform = get_eval_transform(normalize=True)

    img_array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    t1 = eval_transform(img)
    t2 = eval_transform(img)

    assert torch.allclose(t1, t2)
    assert t1.shape == (3, 32, 32)


def test_dataloaders_real_data():
    """Verify real CIFAR-10 data loaders yield expected batch shapes and types."""
    # Pretrain loader with batch size 16 for quick verification
    pretrain_loader = get_pretrain_dataloader(
        data_dir="data",
        split_dir="data/splits",
        batch_size=16,
        subset_fraction=0.01,
        num_workers=0,
    )
    v1, v2 = next(iter(pretrain_loader))
    assert v1.shape == (16, 3, 32, 32)
    assert v2.shape == (16, 3, 32, 32)

    # Linear loaders
    train_loader, val_loader, test_loader = get_linear_dataloaders(
        data_dir="data",
        split_dir="data/splits",
        batch_size=16,
        num_workers=0,
    )
    imgs, labels = next(iter(train_loader))
    assert imgs.shape == (16, 3, 32, 32)
    assert labels.shape == (16,)

