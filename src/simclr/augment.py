"""Data augmentations for SimCLR pretraining, linear evaluation, and inference.

Follows official SimCLR specification (Chen et al., ICML 2020 Appendix A & B.9,
and google-research/simclr commit 383d414):
- Random resized crop to 32x32 (scale 0.08-1.0, ratio 3/4-4/3, bicubic)
- Random horizontal flip (p=0.5)
- Color jitter (p=0.8, strength s=0.5: brightness 0.4, contrast 0.4, saturation 0.4, hue 0.1)
- Random grayscale (p=0.2)
- Gaussian blur disabled for CIFAR-10
"""
from typing import Optional, Tuple
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# CIFAR-10 dataset channel statistics (computed on 50,000 training images)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_simclr_pretrain_transform(
    image_size: int = 32,
    s: float = 0.5,
    normalize: bool = True,
    mean: Tuple[float, float, float] = CIFAR10_MEAN,
    std: Tuple[float, float, float] = CIFAR10_STD,
) -> transforms.Compose:
    """SimCLR stochastic data augmentation for self-supervised pretraining.

    Args:
        image_size: Target image size (32 for CIFAR-10).
        s: Color jitter strength (0.5 for CIFAR-10 per paper Appendix B.9).
        normalize: Whether to apply standard channel normalization.
        mean: Normalization mean tuple.
        std: Normalization std tuple.

    Returns:
        torchvision transforms.Compose pipeline.
    """
    transform_list = [
        transforms.RandomResizedCrop(
            size=image_size,
            scale=(0.08, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.8 * s,
                    contrast=0.8 * s,
                    saturation=0.8 * s,
                    hue=0.2 * s,
                )
            ],
            p=0.8,
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ]
    if normalize:
        transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)


def get_linear_train_transform(
    image_size: int = 32,
    normalize: bool = True,
    mean: Tuple[float, float, float] = CIFAR10_MEAN,
    std: Tuple[float, float, float] = CIFAR10_STD,
) -> transforms.Compose:
    """Data augmentation for downstream linear classifier training.

    Paper & official repo protocol: crop + flip only (no color jitter).
    """
    transform_list = [
        transforms.RandomResizedCrop(
            size=image_size,
            scale=(0.08, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ]
    if normalize:
        transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)


def get_eval_transform(
    normalize: bool = True,
    mean: Tuple[float, float, float] = CIFAR10_MEAN,
    std: Tuple[float, float, float] = CIFAR10_STD,
) -> transforms.Compose:
    """Deterministic preprocessing for validation, testing, and deployment.

    For 32x32 CIFAR-10 images: no cropping, only conversion to tensor
    and optional normalization (repo: test crop disabled when image_size <= 32).
    """
    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(transform_list)


class SimCLRPairTransform:
    """Wraps a transform to produce two independent stochastic views of an input image."""

    def __init__(self, base_transform: transforms.Compose):
        self.base_transform = base_transform

    def __call__(self, img: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        view1 = self.base_transform(img)
        view2 = self.base_transform(img)
        return view1, view2
