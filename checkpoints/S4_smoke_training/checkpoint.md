# S4 Smoke Training — Status: COMPLETE

## Previous checkpoint verified: `checkpoints/S3_model_loss/checkpoint.md`, status COMPLETE, mismatches found: None.

## Planned work
- [x] Implement training utilities (`seed.py`, `ckpt.py`, `logging.py`) in `src/simclr/utils/`.
- [x] Implement safe W&B wrapper in `src/simclr/observability/wandb_logger.py` with offline fallback.
- [x] Implement complete SimCLR pretraining loop in `src/simclr/train_pretrain.py` with parameter group weight decay splitting, warmup + cosine decay, and checkpointing.
- [x] Implement downstream linear evaluation training loop in `src/simclr/train_linear.py` on frozen encoder representations.
- [x] Author smoke test configurations in `configs/smoke_pretrain.yaml` and `configs/smoke_linear.yaml`.
- [x] Run 2-epoch pretraining on 1% subset; verify loss decreases and checkpoints are saved.
- [x] Verify checkpoint resume functionality (`--resume runs/<id>/last.pt`).
- [x] Run downstream linear evaluation on frozen encoder representations and verify `result.json` output.
- [x] Add automated smoke integration test to pytest suite (`tests/test_smoke_training.py`).

## Work done
- Built `src/simclr/utils/seed.py`: sets deterministic seeds across Python, NumPy, and PyTorch (CPU, CUDA, MPS).
- Built `src/simclr/utils/ckpt.py`: robust checkpoint save/load preserving model, optimizer, scheduler, epoch, step, metrics, config, and RNG states across PyTorch versions (including PyTorch 2.6+ `weights_only=False` compatibility and MPS tensor handling).
- Built `src/simclr/utils/logging.py`: CSV metric logging and YAML configuration serialization.
- Built `src/simclr/observability/wandb_logger.py`: modular W&B logging with fallback to disabled/offline mode.
- Built `src/simclr/train_pretrain.py`: pretraining engine with cosine learning rate schedule, linear warmup, and BatchNorm/bias weight decay exclusion.
- Built `src/simclr/train_linear.py`: linear classifier training engine with validation monitoring, best model selection, and test top-1/top-5 evaluation.
- Authored `configs/smoke_pretrain.yaml` and `configs/smoke_linear.yaml`.
- Ran integration test:
  - Epoch 1 Loss: 3.7375, Contrastive Acc: 13.17%, Time: 1.4s.
  - Epoch 2 Loss: 3.6684, Contrastive Acc: 12.05%, Time: 1.1s.
  - Checkpoint resumption from `last.pt` verified.
  - Linear evaluation ran and generated `runs/test_smoke_linear/result.json`.

## Files changed (path — reason)
- `src/simclr/utils/seed.py` — Reproducible random seeding utility.
- `src/simclr/utils/ckpt.py` — Model and optimizer checkpointing utility.
- `src/simclr/utils/logging.py` — Local CSV logging and config saving utility.
- `src/simclr/observability/wandb_logger.py` — Isolated Weights & Biases tracker.
- `src/simclr/train_pretrain.py` — Self-supervised pretraining loop.
- `src/simclr/train_linear.py` — Downstream linear evaluation loop.
- `configs/smoke_pretrain.yaml` — Smoke pretraining experiment configuration.
- `configs/smoke_linear.yaml` — Smoke linear evaluation experiment configuration.
- `tests/test_smoke_training.py` — Automated integration tests for smoke training.
- `checkpoints/S4_smoke_training/checkpoint.md` — Stage S4 progress log.
- `checkpoints/INDEX.md` — Updated stage index.

## Commands executed (exact) and outcomes
- `.venv/bin/pytest tests/test_smoke_training.py -v`: verified pretraining, resume, and linear evaluation (exit code 0, 2 passed).
- `.venv/bin/pytest tests/ -v`: verified full project test suite (exit code 0, 16 passed in 26.62s).

## Tests run and results
- `tests/test_data.py`: 5 passed
- `tests/test_environment.py`: 3 passed
- `tests/test_model_loss.py`: 6 passed
- `tests/test_smoke_training.py::test_smoke_pretraining_and_resume` PASSED
- `tests/test_smoke_training.py::test_smoke_linear_eval` PASSED
- Total: 16 passed in 26.62s.

## Metrics (label each MEASURED / NOT MEASURED; include run IDs)
- Pretrain Loss (run `test_smoke_pretrain`):
  - Epoch 1: `MEASURED` 3.7375
  - Epoch 2: `MEASURED` 3.6684 (loss decreased)
- Linear Eval (run `test_smoke_linear`):
  - Test Top-1 Accuracy: `MEASURED` 14.50% (on random/smoke representations, 1 epoch)
  - Test Top-5 Accuracy: `MEASURED` 59.80%

## Observability: W&B run ID/URL or NOT CONFIGURED; Phoenix project/trace refs or NOT CONFIGURED
- W&B: `NOT CONFIGURED` (disabled by default in smoke runs)
- Phoenix: `NOT CONFIGURED`

## Model artifacts: path, SHA-256, originating run
- `runs/test_smoke_pretrain/best.pt`: originating run `test_smoke_pretrain`
- `runs/test_smoke_pretrain/last.pt`: originating run `test_smoke_pretrain`
- `runs/test_smoke_linear/best_classifier.pt`: originating run `test_smoke_linear`

## Errors and resolutions
- PyTorch 2.6+ unpickling error during checkpoint load (`weights_only=True` default blocked NumPy RNG tuple).
- Resolution: Explicitly passed `weights_only=False` in `ckpt.py` and `train_linear.py`.
- MPS tensor RNG error (`set_rng_state` required CPU ByteTensor).
- Resolution: Converted `states['torch']` to CPU uint8 before calling `torch.set_rng_state`.

## Decisions made (link to DECISIONS.md entries)
- Excluded BatchNorm weights and biases from weight decay during pretraining.
- Retained full RNG state for reproducible checkpoint resume.

## Unresolved issues / known gaps
- None. Exit criteria met.

## Reproduce: exact command(s)
```bash
.venv/bin/pytest tests/test_smoke_training.py -v
```

## Next stage / next command
- Transition to Stage S5 (Pilot Runs).
- Next command: Measure GPU/MPS throughput, run hyperparameter pilot runs (SGD vs Adam, learning rate, $\tau$) on validation split, compile pilot comparison table, and present budget proposal to user for approval before full training runs.
