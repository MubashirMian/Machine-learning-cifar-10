"""Isolated Weights & Biases (W&B) experiment tracking integration."""
import os
from typing import Any, Dict, Optional


class WandbLogger:
    """Safe wrapper around wandb that cleanly falls back to offline/disabled mode."""

    def __init__(
        self,
        enabled: bool = True,
        project: str = "simclr-cifar10",
        entity: Optional[str] = "mubashirmian199-iba-institute-of-business-administration",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
    ):
        self.enabled = enabled
        self.run = None

        if not self.enabled:
            return

        try:
            import wandb

            self.wandb = wandb
            # Check if login credentials exist in environment or ~/.netrc
            is_logged_in = False
            try:
                is_logged_in = self.wandb.login(silent=True)
            except Exception:
                pass

            mode = os.environ.get("WANDB_MODE")
            if not mode:
                mode = "online" if is_logged_in else "offline"
            os.environ["WANDB_MODE"] = mode

            self.run = self.wandb.init(
                project=project,
                entity=entity,
                name=name,
                config=config,
                tags=tags,
                mode=mode,
            )
            print(f"[W&B] Initialized run '{name}' (mode: {mode}, project: {project})")
        except Exception as e:
            print(f"[W&B Warning] Could not initialize wandb ({e}). Continuing with W&B disabled.")
            self.enabled = False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        if self.enabled and self.run is not None:
            try:
                self.wandb.log(metrics, step=step)
            except Exception as e:
                print(f"[W&B Warning] Failed to log metrics: {e}")

    def finish(self):
        if self.enabled and self.run is not None:
            try:
                self.wandb.finish()
            except Exception:
                pass
