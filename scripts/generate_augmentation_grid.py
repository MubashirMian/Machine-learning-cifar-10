"""Generate and save visual grid of SimCLR augmentations on CIFAR-10 sample images."""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import datasets

from src.simclr.augment import CIFAR10_MEAN, CIFAR10_STD, get_simclr_pretrain_transform


def unnormalize(tensor, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    """Reverses channel normalization for visualization."""
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    t = (tensor * s + m).clamp(0.0, 1.0)
    return t.permute(1, 2, 0).numpy()


def main():
    # Load raw CIFAR-10 without transforms
    dataset = datasets.CIFAR10(root="data", train=True, download=False)
    classes = dataset.classes

    # Pick 4 representative classes: airplane, bird, deer, ship
    sample_indices = [3, 6, 12, 1]  # cat, frog, horse, truck / diverse images
    num_samples = len(sample_indices)
    num_views = 5

    transform = get_simclr_pretrain_transform(image_size=32, s=0.5, normalize=True)

    fig, axes = plt.subplots(num_samples, num_views + 1, figsize=(12, 8))
    fig.suptitle(
        "SimCLR Data Augmentations (Random Crop & Resize, Flip, Color Jitter s=0.5, Grayscale p=0.2)",
        fontsize=13,
        fontweight="bold",
    )

    for row, idx in enumerate(sample_indices):
        raw_img, label = dataset[idx]
        class_name = classes[label]

        # Original image
        axes[row, 0].imshow(raw_img)
        axes[row, 0].set_title(f"Original ({class_name})", fontsize=10)
        axes[row, 0].axis("off")

        # 5 random augmented views
        for col in range(1, num_views + 1):
            aug_tensor = transform(raw_img)
            vis_img = unnormalize(aug_tensor)
            axes[row, col].imshow(vis_img)
            axes[row, col].set_title(f"Aug View {col}", fontsize=9)
            axes[row, col].axis("off")

    plt.tight_layout()
    output_path = "data/sample_augmentations.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Successfully saved augmentation grid to {output_path}")


if __name__ == "__main__":
    main()
