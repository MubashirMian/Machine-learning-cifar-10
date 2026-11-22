"""Environment and workspace integrity tests."""
import os
from pathlib import Path


def test_governance_files_exist():
    """Verify that all core governance files required by AGENTS.md exist."""
    required_files = [
        "AGENTS.md",
        "PROJECT_PLAN.md",
        "SOURCES.md",
        "DECISIONS.md",
        "README.md",
        "requirements.txt",
        ".gitignore",
        "checkpoints/INDEX.md",
        "pipeline_original/eda_pipeline.ipynb",
    ]
    for rel_path in required_files:
        assert Path(rel_path).is_file(), f"Required file missing: {rel_path}"


def test_directory_structure_exists():
    """Verify standard repository layout defined in AGENTS.md §5."""
    required_dirs = [
        "configs",
        "src/simclr",
        "src/simclr/utils",
        "src/simclr/observability",
        "app",
        "tests",
        "pipeline_original",
        "checkpoints",
        "checkpoints/S0_setup",
        "runs",
        "data",
    ]
    for rel_dir in required_dirs:
        assert Path(rel_dir).is_dir(), f"Required directory missing: {rel_dir}"


def test_core_dependencies_importable():
    """Verify that PyTorch, torchvision, numpy and pytest are properly installed."""
    import torch
    import torchvision
    import numpy as np

    assert torch.__version__ is not None
    assert torchvision.__version__ is not None
    assert np.__version__ is not None
