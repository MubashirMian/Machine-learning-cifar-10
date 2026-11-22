"""Integration smoke test for end-to-end pretraining and linear evaluation loops."""
import json
from pathlib import Path
import pytest
import torch

from src.simclr.train_linear import run_linear_eval
from src.simclr.train_pretrain import run_pretrain


def test_smoke_pretraining_and_resume(tmp_path):
    """Test 2-epoch pretraining, verify artifacts, and test resuming."""
    config = {
        "experiment_id": "test_smoke_pretrain",
        "data_dir": "data",
        "split_dir": "data/splits",
        "subset_fraction": 0.01,
        "batch_size": 32,
        "epochs": 2,
        "warmup_epochs": 1,
        "lr": 0.1,
        "optimizer": "sgd",
        "temperature": 0.5,
        "normalize": True,
        "num_workers": 0,
        "wandb": False,
        "seed": 42,
    }

    result = run_pretrain(config)
    assert result["epochs"] == 2
    assert Path("runs/test_smoke_pretrain/best.pt").is_file()
    assert Path("runs/test_smoke_pretrain/last.pt").is_file()
    assert Path("runs/test_smoke_pretrain/metrics.csv").is_file()

    # Test resume to epoch 3
    config["epochs"] = 3
    resume_result = run_pretrain(config, resume_ckpt="runs/test_smoke_pretrain/last.pt")
    assert resume_result["epochs"] == 3


def test_smoke_linear_eval():
    """Test 1-epoch linear evaluation on frozen encoder representations."""
    config = {
        "experiment_id": "test_smoke_linear",
        "data_dir": "data",
        "split_dir": "data/splits",
        "encoder_checkpoint": "runs/test_smoke_pretrain/best.pt",
        "batch_size": 64,
        "epochs": 1,
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "normalize": True,
        "num_workers": 0,
        "wandb": False,
        "seed": 42,
    }

    eval_result = run_linear_eval(config)
    assert "test_top1" in eval_result
    assert "test_top5" in eval_result
    assert eval_result["test_top1"] >= 0.0
    assert Path("runs/test_smoke_linear/result.json").is_file()
    assert Path("runs/test_smoke_linear/best_classifier.pt").is_file()
