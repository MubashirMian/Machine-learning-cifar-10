"""Downstream linear evaluation protocol for frozen SimCLR representations (§4.6)."""
import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Tuple
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.simclr.data import get_linear_dataloaders
from src.simclr.model import LinearClassifier, ResNet18CIFAR
from src.simclr.observability.wandb_logger import WandbLogger
from src.simclr.utils.ckpt import load_checkpoint, save_checkpoint
from src.simclr.utils.logging import CSVLogger, save_config
from src.simclr.utils.seed import set_seed


def evaluate(
    encoder: nn.Module,
    classifier: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluates top-1 and top-5 accuracy on a given dataloader."""
    encoder.eval()
    classifier.eval()

    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = labels.size(0)

            features = encoder(images)
            logits = classifier(features)
            loss = F.cross_entropy(logits, labels)

            total_loss += loss.item() * batch_size

            # Top-1 accuracy
            preds_top1 = logits.argmax(dim=1)
            correct_top1 += (preds_top1 == labels).sum().item()

            # Top-5 accuracy
            _, preds_top5 = logits.topk(min(5, logits.size(1)), dim=1, largest=True, sorted=True)
            correct_top5 += preds_top5.eq(labels.view(-1, 1).expand_as(preds_top5)).sum().item()

            total_samples += batch_size

    avg_loss = total_loss / max(1, total_samples)
    top1_acc = (correct_top1 / max(1, total_samples)) * 100.0
    top5_acc = (correct_top5 / max(1, total_samples)) * 100.0
    return avg_loss, top1_acc, top5_acc


def run_linear_eval(config: Dict[str, Any]) -> Dict[str, Any]:
    run_id = config.get("experiment_id", f"linear_{int(time.time())}")
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    save_config(config, str(run_dir / "config.yaml"))

    seed = config.get("seed", 42)
    set_seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    batch_size = config.get("batch_size", 256)
    data_dir = config.get("data_dir", "data")
    split_dir = config.get("split_dir", "data/splits")
    num_workers = config.get("num_workers", 2)
    normalize = config.get("normalize", True)

    train_loader, val_loader, test_loader = get_linear_dataloaders(
        data_dir=data_dir,
        split_dir=split_dir,
        batch_size=batch_size,
        seed=seed,
        normalize=normalize,
        num_workers=num_workers,
    )

    # Base encoder
    encoder = ResNet18CIFAR().to(device)

    # Load pretrained weights into encoder
    encoder_ckpt_path = config.get("encoder_checkpoint", "")
    if encoder_ckpt_path and os.path.isfile(encoder_ckpt_path):
        print(f"Loading encoder checkpoint: {encoder_ckpt_path}")
        ckpt = torch.load(encoder_ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"]
        # Filter state_dict in case it contains projection head
        encoder_state = {}
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                encoder_state[k[len("encoder.") :]] = v
            elif not k.startswith("projector."):
                encoder_state[k] = v
        encoder.load_state_dict(encoder_state, strict=False)
    else:
        print("Notice: No pretrained encoder checkpoint provided; evaluating random initialization baseline.")

    # Freeze encoder
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # Linear classifier
    classifier = LinearClassifier(feature_dim=512, num_classes=10).to(device)

    epochs = config.get("epochs", 40)
    lr = config.get("lr", 0.1)
    momentum = config.get("momentum", 0.9)
    weight_decay = config.get("weight_decay", 0.0)

    optimizer = torch.optim.SGD(
        classifier.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    # Cosine decay schedule without restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    csv_logger = CSVLogger(
        str(run_dir / "metrics.csv"),
        fieldnames=[
            "epoch",
            "train_loss",
            "val_loss",
            "val_top1",
            "val_top5",
            "lr",
            "epoch_time_sec",
        ],
    )
    wandb_enabled = config.get("wandb", False)
    wandb_logger = WandbLogger(
        enabled=wandb_enabled,
        name=run_id,
        config=config,
        tags=["linear_eval"],
    )

    best_val_acc = -1.0
    total_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        classifier.train()
        train_loss = 0.0
        total_train = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                features = encoder(images)

            logits = classifier(features)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            total_train += labels.size(0)

        scheduler.step()
        epoch_time = time.time() - t0
        avg_train_loss = train_loss / max(1, total_train)

        val_loss, val_top1, val_top5 = evaluate(encoder, classifier, val_loader, device)

        current_lr = scheduler.get_last_lr()[0]
        row = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_top1": round(val_top1, 2),
            "val_top5": round(val_top5, 2),
            "lr": round(current_lr, 6),
            "epoch_time_sec": round(epoch_time, 2),
        }
        csv_logger.log(row)
        wandb_logger.log(row, step=epoch)

        print(
            f"Epoch [{epoch:3d}/{epochs:3d}] Train Loss: {avg_train_loss:.4f} | "
            f"Val Top-1: {val_top1:.2f}% | Val Top-5: {val_top5:.2f}% | Time: {epoch_time:.1f}s"
        )

        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            save_checkpoint(
                path=str(run_dir / "best_classifier.pt"),
                epoch=epoch,
                global_step=epoch * len(train_loader),
                model=classifier,
                optimizer=optimizer,
                best_metric=best_val_acc,
                config=config,
            )

    # Load best classifier for final test set evaluation
    best_ckpt = torch.load(str(run_dir / "best_classifier.pt"), map_location=device, weights_only=False)
    classifier.load_state_dict(best_ckpt["model_state_dict"])

    test_loss, test_top1, test_top5 = evaluate(encoder, classifier, test_loader, device)
    total_time = time.time() - total_start

    print("=" * 60)
    print(f"Final Linear Evaluation on Test Set: Top-1: {test_top1:.2f}% | Top-5: {test_top5:.2f}%")
    print("=" * 60)

    result_data = {
        "run_id": run_id,
        "encoder_checkpoint": encoder_ckpt_path,
        "best_val_top1": best_val_acc,
        "test_top1": test_top1,
        "test_top5": test_top5,
        "test_loss": test_loss,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "optimizer": "sgd",
        "total_time_sec": total_time,
        "device": str(device),
    }

    with open(run_dir / "result.json", "w") as f:
        json.dump(result_data, f, indent=2)

    wandb_logger.log({"test_top1": test_top1, "test_top5": test_top5})
    wandb_logger.finish()

    return result_data


def main():
    parser = argparse.ArgumentParser(description="SimCLR Linear Evaluation on CIFAR-10")
    parser.add_argument("--config", type=str, default="", help="Path to config YAML")
    parser.add_argument("--encoder-checkpoint", type=str, default="", help="Path to pretrained encoder")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    config = {}
    if args.config and os.path.isfile(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    if args.encoder_checkpoint:
        config["encoder_checkpoint"] = args.encoder_checkpoint
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.lr is not None:
        config["lr"] = args.lr
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.no_wandb:
        config["wandb"] = False
    elif args.wandb:
        config["wandb"] = True

    run_linear_eval(config)


if __name__ == "__main__":
    main()
