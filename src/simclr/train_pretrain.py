"""SimCLR pretraining script with cosine decay, warmup, and checkpoint recovery."""
import argparse
import os
from pathlib import Path
import time
from typing import Any, Dict, Tuple
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.simclr.data import get_pretrain_dataloader
from src.simclr.loss import NTXentLoss
from src.simclr.model import SimCLRModel
from src.simclr.observability.wandb_logger import WandbLogger
from src.simclr.utils.ckpt import load_checkpoint, save_checkpoint
from src.simclr.utils.logging import CSVLogger, save_config
from src.simclr.utils.seed import set_seed


def split_weights_for_weight_decay(
    model: nn.Module, weight_decay: float
) -> list:
    """Separates parameters so that BatchNorm weights and all biases are excluded from weight decay (§4.5)."""
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    param_groups = split_weights_for_weight_decay(model, weight_decay)
    opt_name = optimizer_name.lower()
    if opt_name == "sgd":
        return torch.optim.SGD(param_groups, lr=lr, momentum=momentum)
    elif opt_name == "adam":
        return torch.optim.Adam(param_groups, lr=lr)
    elif opt_name == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing to 0 (§4.5)."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = max(total_steps, warmup_steps + 1)
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            alpha = self.current_step / max(1, self.warmup_steps)
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group["lr"] = base_lr * alpha
        else:
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)).item())
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group["lr"] = self.min_lr + (base_lr - self.min_lr) * cosine

    def get_last_lr(self) -> list:
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> Dict[str, Any]:
        return {"current_step": self.current_step}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.current_step = state_dict.get("current_step", 0)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = len(dataloader)

    for view1, view2 in dataloader:
        view1 = view1.to(device)
        view2 = view2.to(device)

        _, z1 = model(view1)
        _, z2 = model(view2)

        loss, acc = criterion(z1, z2, return_acc=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_acc += acc

    avg_loss = total_loss / max(1, num_batches)
    avg_acc = total_acc / max(1, num_batches)
    return avg_loss, avg_acc


def run_pretrain(config: Dict[str, Any], resume_ckpt: str = "") -> Dict[str, Any]:
    run_id = config.get("experiment_id", f"simclr_{int(time.time())}")
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    save_config(config, str(run_dir / "config.yaml"))

    # Seed
    seed = config.get("seed", 42)
    set_seed(seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Dataloader
    batch_size = config.get("batch_size", 256)
    subset_fraction = config.get("subset_fraction", 1.0)
    data_dir = config.get("data_dir", "data")
    split_dir = config.get("split_dir", "data/splits")
    num_workers = config.get("num_workers", 2)
    normalize = config.get("normalize", True)

    dataloader = get_pretrain_dataloader(
        data_dir=data_dir,
        split_dir=split_dir,
        batch_size=batch_size,
        subset_fraction=subset_fraction,
        seed=seed,
        normalize=normalize,
        num_workers=num_workers,
        drop_last=True,
    )

    # Model & Loss
    model = SimCLRModel().to(device)
    temperature = config.get("temperature", 0.5)
    criterion = NTXentLoss(temperature=temperature)

    # Optimizer & Scheduler
    epochs = config.get("epochs", 100)
    warmup_epochs = config.get("warmup_epochs", 10)
    lr = config.get("lr", 0.3 * (batch_size / 256.0))
    weight_decay = config.get("weight_decay", 1e-4)
    optimizer_name = config.get("optimizer", "sgd")
    momentum = config.get("momentum", 0.9)

    optimizer = build_optimizer(model, optimizer_name, lr, weight_decay, momentum)

    steps_per_epoch = len(dataloader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    # Logging
    csv_logger = CSVLogger(
        str(run_dir / "metrics.csv"),
        fieldnames=["epoch", "loss", "contrastive_acc", "lr", "epoch_time_sec"],
    )
    wandb_enabled = config.get("wandb", False)
    wandb_logger = WandbLogger(
        enabled=wandb_enabled,
        name=run_id,
        config=config,
        tags=["pretrain", f"frac_{subset_fraction}"],
    )

    start_epoch = 1
    best_loss = float("inf")

    # Resume if provided
    if resume_ckpt and os.path.isfile(resume_ckpt):
        print(f"Resuming pretraining from checkpoint: {resume_ckpt}")
        ckpt = load_checkpoint(resume_ckpt, model, optimizer, scheduler, device=device)
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_metric", float("inf"))

    total_start_time = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        avg_loss, avg_acc = train_epoch(model, dataloader, criterion, optimizer, scheduler, device)
        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        row = {
            "epoch": epoch,
            "loss": round(avg_loss, 4),
            "contrastive_acc": round(avg_acc, 4),
            "lr": round(current_lr, 6),
            "epoch_time_sec": round(epoch_time, 2),
        }
        csv_logger.log(row)
        wandb_logger.log(row, step=epoch)

        print(
            f"Epoch [{epoch:3d}/{epochs:3d}] Loss: {avg_loss:.4f} | Contrastive Acc: {avg_acc*100:.2f}% | "
            f"LR: {current_lr:.5f} | Time: {epoch_time:.1f}s"
        )

        # Save last checkpoint
        save_checkpoint(
            path=str(run_dir / "last.pt"),
            epoch=epoch,
            global_step=epoch * steps_per_epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_metric=best_loss,
            config=config,
        )

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                path=str(run_dir / "best.pt"),
                epoch=epoch,
                global_step=epoch * steps_per_epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_metric=best_loss,
                config=config,
            )

    total_duration = time.time() - total_start_time
    wandb_logger.finish()

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "epochs": epochs,
        "best_loss": best_loss,
        "total_duration_sec": total_duration,
        "device": str(device),
    }


def main():
    parser = argparse.ArgumentParser(description="SimCLR Pretraining on CIFAR-10")
    parser.add_argument("--config", type=str, default="", help="Path to config YAML")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    config = {}
    if args.config and os.path.isfile(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["lr"] = args.lr
    if args.no_wandb:
        config["wandb"] = False
    elif args.wandb:
        config["wandb"] = True

    run_pretrain(config, resume_ckpt=args.resume)


if __name__ == "__main__":
    main()
