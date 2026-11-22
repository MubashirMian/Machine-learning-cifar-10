"""Local CSV logging and experiment metrics tracking."""
import csv
from pathlib import Path
from typing import Any, Dict, List
import yaml


class CSVLogger:
    """Logs metrics per epoch/step to a CSV file."""

    def __init__(self, filepath: str, fieldnames: List[str]):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        self.file_exists = self.filepath.exists()

        if not self.file_exists:
            with open(self.filepath, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    def log(self, row: Dict[str, Any]):
        with open(self.filepath, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            # Only write known fields
            filtered = {k: row.get(k, "") for k in self.fieldnames}
            writer.writerow(filtered)


def save_config(config: Dict[str, Any], filepath: str):
    """Saves run configuration dictionary to YAML."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def load_config(filepath: str) -> Dict[str, Any]:
    """Loads configuration dictionary from YAML."""
    with open(filepath, "r") as f:
        return yaml.safe_load(f)
