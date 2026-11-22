# S3 Model + Loss — Status: COMPLETE

## Previous checkpoint verified: `checkpoints/S2_data_module/checkpoint.md`, status COMPLETE, mismatches found: None.

## Planned work
- [x] Implement ResNet-18 base encoder with CIFAR stem (3x3 conv stride 1, no max pooling, BasicBlocks [2,2,2,2], 512-d representation) in `src/simclr/model.py` (§4.2).
- [x] Implement 2-layer MLP projection head (Linear 512->512 + BN + ReLU + Linear 512->128 + BN) in `src/simclr/model.py` (§4.3).
- [x] Implement full `SimCLRModel` and `LinearClassifier` modules in `src/simclr/model.py`.
- [x] Implement vectorized NT-Xent loss with cosine similarity, temperature scaling, diagonal masking, and contrastive accuracy in `src/simclr/loss.py` (§4.4).
- [x] Implement explicit-loop unvectorized reference loss for numerical verification (§4.4, §9.2).
- [x] Author and run unit tests in `tests/test_model_loss.py`.

## Work done
- Created `src/simclr/model.py`:
  - `BasicBlock`: convs with `bias=False`, BN with eps=1e-5, momentum=0.1.
  - `ResNet18CIFAR`: Conv1 (3x3 stride 1, 64 channels), BN, ReLU, no max pooling. Residual stages [2,2,2,2] with channels [64, 128, 256, 512]. Adaptive average pooling output $h \in \mathbb{R}^{512}$. Kaiming normal init, last BN in each block initialized with $\gamma=0$.
  - `ProjectionHead`: 2-layer MLP (512 $\to$ 512 $\to$ 128). Hidden layer has BN + ReLU; output layer has BN with no ReLU, matching SimCLR v1.
  - `SimCLRModel`: integrates encoder and projection head; provides `forward(x)` returning $(h, z)$ and `get_representation(x)` returning only $h$.
  - `LinearClassifier`: single linear layer $512 \to 10$ with zero-initialization on weights and bias.
- Created `src/simclr/loss.py`:
  - `NTXentLoss`: vectorized cross-entropy with temperature scaling, diagonal masking ($-10^9$), and targets corresponding to augmented pairs. Optional contrastive top-1 accuracy computation.
  - `nt_xent_loss_explicit_loop`: slow reference implementation for exact numerical comparison.
- Created `tests/test_model_loss.py` with 6 unit tests covering shapes, CIFAR stem integrity, vectorized vs reference loop equality, loss properties, and linear eval parameter/BN freezing.

## Files changed (path — reason)
- `src/simclr/model.py` — ResNet-18 CIFAR encoder, projection head, SimCLR model, and linear classifier (§4.2, §4.3, §4.6).
- `src/simclr/loss.py` — NT-Xent loss and reference loop (§4.4).
- `tests/test_model_loss.py` — Unit tests for model architecture and loss properties (§9.2).
- `checkpoints/S3_model_loss/checkpoint.md` — Stage S3 progress log.
- `checkpoints/INDEX.md` — Updated stage index.

## Commands executed (exact) and outcomes
- `.venv/bin/pytest tests/test_model_loss.py -v`: initial run caught attribute error in test snapshot (clone of module vs tensor).
- `.venv/bin/pytest tests/test_model_loss.py -v`: passed 6 of 6 tests (exit code 0).
- `.venv/bin/pytest tests/ -v`: ran full suite across environment, data, and model/loss modules (exit code 0, 14 passed).

## Tests run and results
- `tests/test_model_loss.py::test_encoder_cifar_stem_and_output_shape` PASSED
- `tests/test_model_loss.py::test_projection_head_shape` PASSED
- `tests/test_model_loss.py::test_full_simclr_model_forward` PASSED
- `tests/test_model_loss.py::test_nt_xent_loss_vs_reference_loop` PASSED
- `tests/test_model_loss.py::test_nt_xent_loss_properties` PASSED
- `tests/test_model_loss.py::test_linear_eval_encoder_frozen_invariance` PASSED
- Overall test suite: 14 passed in 2.06s.

## Metrics (label each MEASURED / NOT MEASURED; include run IDs)
- Vectorized loss vs reference loop absolute difference: `MEASURED` $< 10^{-6}$ (tolerated $10^{-5}$).

## Observability: W&B run ID/URL or NOT CONFIGURED; Phoenix project/trace refs or NOT CONFIGURED
- W&B: `NOT CONFIGURED`
- Phoenix: `NOT CONFIGURED`

## Model artifacts: path, SHA-256, originating run
- None (Model definition stage).

## Errors and resolutions
- In `test_model_loss.py`, initial test code attempted to call `.clone()` on BatchNorm module rather than `module.running_mean.clone()`. Corrected to inspect tensor buffers directly.

## Decisions made (link to DECISIONS.md entries)
- Loss scaling: NT-Xent computes mean over all $2N$ views, matching paper Algorithm 1 (C1).
- CIFAR stem: $3\times 3$ stride 1 conv without max-pool confirmed (D01).

## Unresolved issues / known gaps
- None. Exit criteria met.

## Reproduce: exact command(s)
```bash
.venv/bin/pytest tests/test_model_loss.py -v
```

## Next stage / next command
- Transition to Stage S4 (Smoke Training).
- Next command: Implement training utilities (seed, checkpointing, logging) in `src/simclr/utils/`, pretraining loop in `src/simclr/train_pretrain.py`, and linear eval loop in `src/simclr/train_linear.py`. Run 1-2 epoch smoke test on small subset.
