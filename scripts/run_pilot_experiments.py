"""Pilot experiments for hyperparameter and optimizer selection on validation split (Stage S5)."""
import json
from pathlib import Path
import time
import pandas as pd
import torch

from src.simclr.train_linear import run_linear_eval
from src.simclr.train_pretrain import run_pretrain


def run_pilots():
    pilot_configs = [
        {
            "name": "P1_SGD_lr0.1_tau0.5",
            "optimizer": "sgd",
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "temperature": 0.5,
        },
        {
            "name": "P2_SGD_lr0.3_tau0.5",
            "optimizer": "sgd",
            "lr": 0.3,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "temperature": 0.5,
        },
        {
            "name": "P3_Adam_lr1e-3_tau0.5",
            "optimizer": "adam",
            "lr": 1e-3,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "temperature": 0.5,
        },
        {
            "name": "P4_AdamW_lr1e-3_tau0.5",
            "optimizer": "adamw",
            "lr": 1e-3,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "temperature": 0.5,
        },
        {
            "name": "P5_SGD_lr0.3_tau0.1",
            "optimizer": "sgd",
            "lr": 0.3,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "temperature": 0.1,
        },
    ]

    results = []

    for cfg in pilot_configs:
        exp_id = f"pilot_{cfg['name']}"
        print(f"\n================ Running {exp_id} ================")

        pretrain_cfg = {
            "experiment_id": exp_id,
            "data_dir": "data",
            "split_dir": "data/splits",
            "subset_fraction": 0.10,  # 10% pretraining pool for efficient pilot
            "batch_size": 256,
            "epochs": 5,
            "warmup_epochs": 1,
            "lr": cfg["lr"],
            "optimizer": cfg["optimizer"],
            "momentum": cfg.get("momentum", 0.9),
            "weight_decay": cfg.get("weight_decay", 1e-4),
            "temperature": cfg["temperature"],
            "normalize": True,
            "num_workers": 2,
            "wandb": False,
            "seed": 42,
        }

        t0 = time.time()
        pretrain_res = run_pretrain(pretrain_cfg)
        pretrain_time = time.time() - t0

        # Read last metrics from CSV
        metrics_df = pd.read_csv(f"runs/{exp_id}/metrics.csv")
        final_loss = metrics_df["loss"].iloc[-1]
        final_cont_acc = metrics_df["contrastive_acc"].iloc[-1]

        # Fast 5-epoch linear probe on validation split
        linear_exp_id = f"linear_{exp_id}"
        linear_cfg = {
            "experiment_id": linear_exp_id,
            "data_dir": "data",
            "split_dir": "data/splits",
            "encoder_checkpoint": f"runs/{exp_id}/best.pt",
            "batch_size": 256,
            "epochs": 5,
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 0.0,
            "normalize": True,
            "num_workers": 2,
            "wandb": False,
            "seed": 42,
        }

        linear_res = run_linear_eval(linear_cfg)
        val_top1 = linear_res["best_val_top1"]

        record = {
            "Pilot": cfg["name"],
            "Optimizer": cfg["optimizer"].upper(),
            "LR": cfg["lr"],
            "Tau": cfg["temperature"],
            "Pretrain Loss": final_loss,
            "Contrastive Acc (%)": round(final_cont_acc * 100, 2),
            "Val Linear Top-1 (%)": round(val_top1, 2),
            "Pretrain Time (s)": round(pretrain_time, 1),
        }
        results.append(record)
        print(f"--> Result: {record}")

    df_results = pd.DataFrame(results)
    print("\n================ PILOT RESULTS SUMMARY ================")
    print(df_results.to_markdown(index=False))

    summary_path = Path("runs/pilot_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    df_results.to_csv("runs/pilot_summary.csv", index=False)


if __name__ == "__main__":
    run_pilots()
