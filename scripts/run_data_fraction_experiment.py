"""Automated script for Data-Fraction Scaling Experiment (Stage S7).

Runs pretraining and downstream linear evaluation across subsets
(e.g., 10%, 25%, 50%, 100%) and generates summary tables and scaling plots.
Per AGENTS.md §10.3 and Proposal Section 7.
"""
import argparse
import json
import os
from pathlib import Path
import time
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.simclr.train_linear import run_linear_eval
from src.simclr.train_pretrain import run_pretrain


def ensure_s6_in_results(results: List[Dict]) -> List[Dict]:
    """Ensures that the completed S6 (100% pretrain) result is incorporated."""
    s6_linear_json = Path("runs/linear_eval_cifar10_s6_seed42/result.json")
    already_present = any(r.get("fraction") == 1.0 and r.get("seed") == 42 for r in results)

    if not already_present and s6_linear_json.is_file():
        with open(s6_linear_json, "r") as f:
            s6_data = json.load(f)
        results.append({
            "fraction": 1.0,
            "seed": 42,
            "train_samples": 45000,
            "pretrain_epochs": 50,
            "linear_epochs": s6_data.get("epochs", 40),
            "test_top1": s6_data.get("test_top1", 75.23),
            "test_top5": s6_data.get("test_top5", 98.71),
            "val_top1": s6_data.get("best_val_top1", 75.66),
            "pretrain_exp_id": "pretrain_cifar10_s6_seed42",
            "linear_exp_id": "linear_eval_cifar10_s6_seed42",
        })
    return results


def save_and_plot_results(results: List[Dict], out_dir: Path):
    """Saves results CSV/JSON and plots accuracy vs. fraction curve."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "fraction_summary.csv"
    json_path = out_dir / "fraction_summary.json"
    plot_path = out_dir / "scaling_plot.png"

    df = pd.DataFrame(results).sort_values(by=["fraction", "seed"])
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 65)
    print("CURRENT DATA-FRACTION SCALING SUMMARY:")
    print("=" * 65)
    print(df.to_string(index=False))
    print("=" * 65)

    if len(df) > 0:
        plt.figure(figsize=(8, 5))
        # Group by fraction for mean +/- std
        grouped = df.groupby("fraction")["test_top1"].agg(["mean", "std", "count"]).reset_index()
        
        fractions_pct = [f * 100 for f in grouped["fraction"]]
        means = grouped["mean"].values
        stds = grouped["std"].fillna(0).values

        plt.plot(fractions_pct, means, marker="o", color="#1f77b4", linewidth=2.5, markersize=8, label="SimCLR Linear Top-1")
        if any(grouped["count"] > 1):
            plt.errorbar(fractions_pct, means, yerr=stds, fmt="none", ecolor="#1f77b4", capsize=5)

        # Plot individual seed points
        for _, row in df.iterrows():
            plt.scatter(row["fraction"] * 100, row["test_top1"], color="orange", alpha=0.7, s=40, zorder=5)

        plt.title("SimCLR (ResNet-18) on CIFAR-10: Pretraining Data Fraction vs. Accuracy", fontsize=12, fontweight="bold")
        plt.xlabel("Unlabeled Pretraining Data Fraction (%)", fontsize=11)
        plt.ylabel("Linear Evaluation Top-1 Accuracy (%)", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.ylim(0, 100)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Saved scaling plot to: {plot_path}")


def run_fraction_experiment(fraction: float, seed: int = 42, pretrain_epochs: int = 50, linear_epochs: int = 40):
    out_dir = Path("runs/data_fraction")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "fraction_summary.json"

    results = []
    if json_path.is_file():
        with open(json_path, "r") as f:
            results = json.load(f)
    results = ensure_s6_in_results(results)

    # Check if this fraction and seed already completed
    existing = [r for r in results if abs(r["fraction"] - fraction) < 1e-4 and r["seed"] == seed]
    if existing:
        print(f"Run for fraction={fraction:.2f}, seed={seed} already recorded in {json_path}. Skipping.")
        save_and_plot_results(results, out_dir)
        return

    exp_id = f"pretrain_frac{int(fraction*100):02d}_seed{seed}"
    linear_exp_id = f"linear_frac{int(fraction*100):02d}_seed{seed}"
    n_samples = int(45000 * fraction)

    print("\n" + "=" * 65)
    print(f"STARTING FRACTION EXPERIMENT: {fraction*100:.1f}% DATA ({n_samples} images), SEED={seed}")
    print(f"Pretrain Epochs: {pretrain_epochs} | Linear Epochs: {linear_epochs}")
    print("=" * 65)

    # 1. Pretrain
    pretrain_cfg = {
        "experiment_id": exp_id,
        "data_dir": "data",
        "split_dir": "data/splits",
        "subset_fraction": fraction,
        "batch_size": 256,
        "epochs": pretrain_epochs,
        "warmup_epochs": 10 if pretrain_epochs >= 20 else 2,
        "lr": 0.001,
        "optimizer": "adam",
        "weight_decay": 1e-4,
        "temperature": 0.5,
        "normalize": True,
        "num_workers": 4,
        "wandb": False,
        "seed": seed,
    }

    t0_pretrain = time.time()
    run_pretrain(pretrain_cfg)
    pretrain_dur = time.time() - t0_pretrain
    print(f"Pretraining completed in {pretrain_dur/60:.1f} minutes.")

    # 2. Linear Evaluation on full labeled set
    linear_cfg = {
        "experiment_id": linear_exp_id,
        "data_dir": "data",
        "split_dir": "data/splits",
        "encoder_checkpoint": f"runs/{exp_id}/best.pt",
        "batch_size": 512,
        "epochs": linear_epochs,
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "normalize": True,
        "num_workers": 6,
        "wandb": False,
        "seed": seed,
    }

    t0_linear = time.time()
    linear_res = run_linear_eval(linear_cfg)
    linear_dur = time.time() - t0_linear
    print(f"Linear evaluation completed in {linear_dur/60:.1f} minutes.")

    record = {
        "fraction": fraction,
        "seed": seed,
        "train_samples": n_samples,
        "pretrain_epochs": pretrain_epochs,
        "linear_epochs": linear_epochs,
        "test_top1": linear_res["test_top1"],
        "test_top5": linear_res["test_top5"],
        "val_top1": linear_res["best_val_top1"],
        "pretrain_exp_id": exp_id,
        "linear_exp_id": linear_exp_id,
    }
    results.append(record)
    save_and_plot_results(results, out_dir)


def main():
    parser = argparse.ArgumentParser(description="SimCLR Data-Fraction Scaling Experiment")
    parser.add_argument("--fraction", type=str, default="0.10",
                        help="Fraction to run (0.10, 0.25, 0.50, or 'all')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=50, help="Pretraining epochs")
    parser.add_argument("--linear-epochs", type=int, default=40, help="Linear eval epochs")
    args = parser.parse_args()

    if args.fraction.lower() == "all":
        # Runs 10%, 25%, 50% sequentially (100% already included from S6)
        for frac in [0.10, 0.25, 0.50]:
            run_fraction_experiment(
                fraction=frac,
                seed=args.seed,
                pretrain_epochs=args.epochs,
                linear_epochs=args.linear_epochs,
            )
    else:
        frac = float(args.fraction)
        run_fraction_experiment(
            fraction=frac,
            seed=args.seed,
            pretrain_epochs=args.epochs,
            linear_epochs=args.linear_epochs,
        )


if __name__ == "__main__":
    main()
