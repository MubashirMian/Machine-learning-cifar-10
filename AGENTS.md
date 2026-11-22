# AGENTS.md — SimCLR on CIFAR-10: Research & Implementation Agent

> Read this whole file at the start of every session. Sections are numbered; cite them as `§N.M`.
> Where this file and a user instruction in the current session conflict, the user instruction wins — record the conflict in the active checkpoint.

---

## 0. Session Start (do this first, every session)

1. Read `AGENTS.md` (this file).
2. Read `PROJECT_PLAN.md` and `checkpoints/INDEX.md`.
3. Read the latest checkpoint whose status is `COMPLETE`, then any newer `IN_PROGRESS` / `INCOMPLETE` checkpoint.
4. Inspect the repository state (files, last modified code, configs, `runs/` directory) and confirm it matches the latest checkpoint.
5. If anything disagrees (missing files, code changed after the checkpoint, metrics without artifacts), **stop and report the mismatch** before doing new work.
6. Continue only from a verified state.

**First session only (bootstrap):** if `PROJECT_PLAN.md`, `SOURCES.md`, `DECISIONS.md` or `checkpoints/INDEX.md` do not exist, create them from the templates in §18, record that in `checkpoints/S0_setup/checkpoint.md`, and ask the user to confirm the stage plan in §7.

---

## 1. Role and Scope

You are the research and implementation agent for a course project that **re-implements SimCLR (v1) from scratch in PyTorch on CIFAR-10**, runs an additional data-scaling experiment, and deploys a small inference web app.

Responsibilities:
- Research the paper and the official implementation; explain technical and mathematical decisions to the user.
- Review (and, only with approval, adapt) the team's existing CIFAR-10 data pipeline.
- Implement the four SimCLR modules, training, linear evaluation, the experiment and the app.
- Run experiments, record real results, maintain documentation and checkpoints.

You must not invent results, sources or code behaviour, and must not claim work is complete without evidence (§3, §17).

Out of scope unless the user asks: ImageNet-scale training, SimCLRv2 features (SK-ResNets, 3-layer head fine-tuning, distillation), multi-GPU training.

---

## 2. Project Summary (from the approved proposal)

| Item | Decision | Status |
|---|---|---|
| Paper | Chen, Kornblith, Norouzi, Hinton. *A Simple Framework for Contrastive Learning of Visual Representations*. ICML 2020 (PMLR 119). arXiv:2002.05709 | FIXED |
| Official code | https://github.com/google-research/simclr (TensorFlow; reference only) | FIXED |
| Framework | PyTorch, code written by the team | FIXED |
| Dataset | CIFAR-10: 50,000 train / 10,000 test, 32×32 RGB, 10 classes | FIXED |
| Encoder | ResNet-18 with CIFAR stem (deviation: paper's CIFAR results use ResNet-50) | FIXED |
| Projection head | 2-layer MLP → 128-d (as in SimCLR v1) | FIXED |
| Loss | NT-Xent (normalized temperature-scaled cross-entropy) | FIXED |
| Optimizer | SGD (momentum) or Adam — chosen after pilot runs; **not** LARS (deviation) | OPEN (§19) |
| Batch size | 256 (deviation: paper uses up to 4096–8192 on ImageNet) | FIXED |
| Epochs | < 1000; set after measuring throughput on the available GPU | OPEN (§19) |
| Evaluation | Linear evaluation on frozen encoder; top-1 test accuracy | FIXED |
| Paper comparison | CIFAR-10 results in Appendix B.9 / Figure B.7 (exact row to be identified and recorded in `SOURCES.md`) | TO VERIFY |
| Extra experiment | Pretraining-data fraction vs. linear-eval accuracy (stratified subsets) | FIXED (design details in §10.3) |
| Optional extension | Tiny-ImageNet, only if compute allows and the user approves | OPTIONAL |
| Deployment | Streamlit app: upload image → preprocess → frozen encoder + linear classifier → label + confidence; hosted on Hugging Face Spaces | FIXED (observability scope OPEN, §19) |
| Compute | One GPU: Google Colab and one team member's machine, same codebase | FIXED |

---

## 3. Evidence Rules

### 3.1 Source-of-truth hierarchy
1. Assignment instructions and the approved proposal.
2. The original paper (arXiv:2002.05709, including appendices).
3. The official repository (google-research/simclr; `tf2/` is the most readable version).
4. Official framework documentation (PyTorch, torchvision, W&B, Phoenix, Streamlit).
5. Peer-reviewed or reputable supporting research.
6. Other sources (community PyTorch re-implementations, blogs). Use only to cross-check.

When sources conflict, **do not silently pick one**: record the conflict in `DECISIONS.md` (§18.4) with both sources, your choice and why.

### 3.2 Status vocabulary (the only labels to use)
| Label | Meaning |
|---|---|
| `VERIFIED` | Directly supported by an inspected source (with reference) or an executed command/test (with output) |
| `INFERRED` | Reasoned interpretation, not explicitly established |
| `UNKNOWN` | Evidence unavailable |
| `PROPOSED` | Suggested, not implemented / not approved |
| `PAPER-REPORTED` | A number taken from the paper, never produced by this project |
| `MEASURED` | A number produced by this project's code, with run reference |
| `NOT MEASURED` | Metric not yet produced |
| `NOT CONFIGURED` | Integration (W&B, Phoenix, …) not set up |
| Stage status: `PLANNED` / `IN_PROGRESS` / `INCOMPLETE` / `COMPLETE` / `BLOCKED` | See §7, §17 |

### 3.3 Anti-hallucination rules
- Never invent paper sections, equations, table rows, citations, repository files, functions, terminal output, metrics, timings, W&B run IDs or Phoenix trace IDs.
- Open and inspect a source before describing it. Give URL + section/figure/table, or file path + line range.
- If a source cannot be accessed, say so and mark the claim `UNKNOWN`.
- Never claim a test passed unless it was executed in this session or recorded with output in a checkpoint.
- Always label numbers as `PAPER-REPORTED` or `MEASURED`.
- If uncertain, ask the user or mark `UNKNOWN`.

---

## 4. Technical Reference Specification

This section records how the official implementation does things, so the PyTorch version can match it deliberately. Items marked *(repo)* were verified in google-research/simclr @ commit `383d414`; items marked *(paper — verify)* must be checked against the PDF before relying on them. Any deviation from this spec must be recorded in `DECISIONS.md`.

### 4.1 Data augmentation (pretraining only)
Applied independently twice per image to produce a positive pair. Inputs are float in [0, 1]. *(repo: `tf2/data_util.py` `preprocess_for_train`, `random_color_jitter`, `crop_and_resize`)*
1. **Random resized crop** → 32×32: area scale ∈ [0.08, 1.0], aspect ratio ∈ [3/4, 4/3], bicubic resize. (torchvision: `RandomResizedCrop(32, scale=(0.08, 1.0), ratio=(3/4, 4/3), interpolation=BICUBIC)`.)
2. **Horizontal flip**, p = 0.5.
3. **Color jitter** with p = 0.8, strength `s`: brightness 0.8s, contrast 0.8s, saturation 0.8s, hue 0.2s, applied in random order. Official CIFAR-10 command uses **s = 0.5**.
4. **Random grayscale**, p = 0.2.
5. **No Gaussian blur on CIFAR-10** (official CIFAR command sets `--use_blur=False`).
6. No mean/std normalization in the official code. If the team adds normalization, it must be identical in pretraining, linear evaluation, testing and the deployed app (§13).

Fine-tuning / linear-eval training: crop + flip only (no color jitter). Test: no crop for 32×32 images, just conversion to float tensor (repo: test crop disabled when `image_size <= 32`).

Known repo detail: brightness in the repo's default `simclrv2` mode is multiplicative; torchvision's `ColorJitter` is also multiplicative. The SimCLR v1 TF1 code used an additive variant. Record which is used.

### 4.2 Encoder (ResNet-18, CIFAR variant)
*(repo: `tf2/resnet.py`, `cifar_stem=True` when `image_size <= 32`)*
- Stem: 3×3 conv, stride 1, 64 channels → BN → ReLU, **no max-pool**.
- Stages: basic blocks `[2, 2, 2, 2]`, channels 64/128/256/512, strides 1/2/2/2.
- Global average pool → representation **h ∈ ℝ⁵¹²** (this is the vector used for linear evaluation and in the app).
- Repo details worth matching: conv layers have no bias; BN momentum 0.9 (TF convention = PyTorch `momentum=0.1`), eps 1e-5; last BN in each residual block initialized with γ = 0.
- Do **not** use torchvision `resnet18()` unmodified: its stem (7×7 stride 2 + max-pool) is for ImageNet.

### 4.3 Projection head g(·)
- SimCLR v1: 2-layer MLP. *(paper — verify Section 2.1)*
- Repo (`tf2/model.py` `ProjectionHead`): each layer is Linear + BN; hidden layer width = input width (512 here) with ReLU; output layer 128-d with BN and no ReLU. Set `num_proj_layers=2` to match v1.
- The head is used only for the loss and is **discarded** for evaluation and deployment.

### 4.4 NT-Xent loss
For a batch of N images → 2N views; `z = normalize(g(f(x)))`; cosine similarity `sim(u,v) = uᵀv / (‖u‖‖v‖)`; temperature τ.
For anchor i with positive j: `ℓ(i,j) = −log( exp(sim(z_i,z_j)/τ) / Σ_{k≠i} exp(sim(z_i,z_k)/τ) )`, averaged over all 2N anchors. *(paper — verify equation number in Section 2.1)*
- Each anchor has 1 positive and 2N−2 negatives.
- Repo implementation (`tf2/objective.py` `add_contrastive_loss`): masks self-similarity with −1e9, computes the loss for both directions and returns `mean(loss_a + loss_b)` over N anchors, which is **2×** the per-anchor mean above. Choose one scaling, document it (it changes the effective learning rate).
- Official CIFAR-10 command uses **τ = 0.5**.
- Required unit test: compare the vectorized loss against a slow, explicit loop on a small random batch (§9.2).

### 4.5 Optimization reference (for context; the project deviates)
*(repo: `tf2/model.py` `WarmUpAndCosineDecay`, `tf2/lars_optimizer.py`)*
- Official CIFAR-10 pretraining command: batch 512, 1000 epochs, LARS, learning rate 1.0 (linear scaling: `lr × batch/256`), weight decay 1e-4, τ = 0.5, ResNet-18, `color_jitter_strength=0.5`, no blur.
- Schedule: linear warmup (default 10 epochs) followed by cosine decay to 0.
- LARS excludes BN parameters and biases from weight decay.
- Project: SGD+momentum or Adam (§2). Keep warmup + cosine unless a pilot run justifies otherwise; exclude BN/bias from weight decay; tune lr and τ for batch 256 with pilot runs on a **validation split** (§8.3).

### 4.6 Linear evaluation protocol
*(repo README CIFAR command: `--train_mode=finetune --fine_tune_after_block=4 --zero_init_logits_layer=True --optimizer=momentum --learning_rate=0.1 --weight_decay=0 --train_epochs=100 --train_batch_size=512 --warmup_epochs=0`)*
- Freeze the encoder (`requires_grad=False` **and** `model.eval()` so BN statistics do not update).
- Train one linear layer 512 → 10 on h (projection head removed), cross-entropy, inputs with crop + flip only.
- Report top-1 (and optionally top-5) accuracy on the **full 10,000-image test set**, once, after all choices are made on the validation split.
- Repo's reference result for its CIFAR-10 command: ~91% top-1 — `PAPER-REPORTED`-equivalent (README-reported), not a project result.

### 4.7 Conflicts register (start of `DECISIONS.md`)
| # | Topic | Paper | Official repo | Project choice |
|---|---|---|---|---|
| C1 | Loss scaling | mean over 2N anchors *(verify)* | sum of both directions / N | TBD |
| C2 | Brightness jitter | *(verify)* | multiplicative (v2) vs additive (v1) | TBD |
| C3 | Encoder for CIFAR | ResNet-50, CIFAR stem *(verify B.9)* | ResNet-18 in README command | ResNet-18 |
| C4 | Optimizer | LARS | LARS default | SGD or Adam |

---

## 5. Repository Layout (proposed — confirm with user in S0)

```
AGENTS.md  PROJECT_PLAN.md  SOURCES.md  DECISIONS.md  README.md
requirements.txt
configs/                 # one YAML per run (pretrain_*.yaml, linear_*.yaml)
src/simclr/
  data.py                # datasets, splits, stratified subsets, two-view wrapper
  augment.py             # §4.1
  model.py               # encoder §4.2, projection head §4.3
  loss.py                # NT-Xent §4.4
  train_pretrain.py      # pretraining loop
  train_linear.py        # linear evaluation §4.6
  utils/{seed.py, ckpt.py, logging.py}
  observability/{wandb_logger.py, phoenix_tracing.py}   # kept separate from core logic (§12.4)
app/app.py               # Streamlit app §13
tests/                   # pytest §9.2
pipeline_original/       # untouched copy of the team's cleaning pipeline (§8.1)
checkpoints/INDEX.md
checkpoints/S<k>_<name>/checkpoint.md
runs/<experiment_id>/    # config.yaml, metrics.csv, model weights, result.json  (gitignored weights)
data/                    # CIFAR-10 (gitignored, never uploaded)
```

**Terminology:** "stage checkpoint" = `checkpoints/S*/checkpoint.md` (progress log). "Model checkpoint" = weights file in `runs/<id>/` (e.g. `last.pt`, `best.pt`). Always say which one.

---

## 6. Research Rules

Before implementing a major component:
1. Quote the relevant proposal requirement.
2. Inspect the relevant paper section and the official implementation (§4 gives starting points).
3. List similarities and differences; add any conflict to `DECISIONS.md`.
4. Explain the planned implementation to the user.
5. Get approval for significant design changes (§11).

`SOURCES.md` entry per source: title, authors, venue & year, URL, relevant sections/files, what it supports, date accessed.

---

## 7. Stage Plan (each stage has its own checkpoint)

| Stage | Goal | Exit criteria (evidence required) |
|---|---|---|
| S0 Setup | Repo layout, environment, governance files | `requirements.txt` installs; `pytest` runs; files from §18 exist |
| S1 Pipeline review | Review existing CIFAR-10 pipeline | `checkpoints/S1_*/pipeline_review.md` per §8.2; original preserved |
| S2 Data module | Splits, two-view augmentation, stratified subsets | Tests in §9.2 (data) pass; augmentation grid image saved |
| S3 Model + loss | Encoder, head, NT-Xent | Shape/loss tests pass; loss equals reference loop |
| S4 Smoke training | 1–2 epoch pretrain + linear eval on small subset | Runs end-to-end; loss decreases; checkpoint resume works |
| S5 Pilot runs | Choose optimizer, lr, τ, epochs on validation split | Pilot table in checkpoint; decision in `DECISIONS.md`; user approves full budget |
| S6 Main reproduction | Full CIFAR-10 pretrain + linear eval | `MEASURED` test accuracy; comparison with `PAPER-REPORTED` row and gap analysis |
| S7 Data-fraction experiment | §10.3 | Result table + plot with all seeds; hypothesis evaluated |
| S8 App | Streamlit app with frozen encoder + linear head | Local run; test requests; Phoenix traces if configured |
| S9 Deployment | Hosted app + README | Public URL works; README items from §13.3 present |
| S10 Report | Final report material | All numbers traceable to runs; attribution section complete |

Optional S7b: Tiny-ImageNet extension (only with user approval).

---

## 8. Data Rules

### 8.1 Existing pipeline protection
Before modifying the team's existing data-cleaning pipeline:
1. Copy it untouched into `pipeline_original/` (or record the git commit hash as baseline).
2. Inspect all relevant files; document inputs/outputs of each stage.
3. Document every transformation and assumption.
4. Check for bugs and leakage (§8.2).
5. Produce a pipeline review report and pfropose changes with reasons.
Never overwrite or delete the original pipeline. Replacing it requires user approval.

### 8.2 CIFAR-10–specific review checklist
CIFAR-10 is a curated benchmark; comparisons with the paper assume the **standard, unmodified** splits. The review must answer:
- Does the pipeline remove, relabel, deduplicate or reweight any images? If yes, results are not directly comparable to the paper — flag it and keep a standard-split path.
- Is the **test set** touched by any fitting step (statistics, filtering, dedup against train)? It must not be.
- Are normalization statistics (if any) computed on training data only?
- Are augmentations done **on the fly** per epoch? Pre-computed/static augmented copies are incompatible with SimCLR (two fresh random views per step).
- Is image dtype/range consistent with §4.1 (float [0,1] before optional normalization)?
- Are labels ignored during pretraining (no label-dependent filtering of pretraining data except stratified subset sampling in §10.3)?

### 8.3 Splits
- Hold out a **stratified validation split** from the 50,000 training images (default 5,000 = 500/class; fixed seed; save index file `data/splits/val_idx.npy`).
- Use validation for all pilot runs, hyperparameter choices and early stopping of the linear classifier.
- Use the 10,000-image test set only for final reported numbers.
- Record whether the final pretraining used 45k or 50k images (paper comparison note).

---

## 9. Code and Testing Rules

### 9.1 Code modification
Before changing code: inspect existing implementation, explain the change, list affected files, avoid unnecessary rewrites.
After changing code: run relevant tests, inspect output, report pass/fail, record in the stage checkpoint.
Never silently modify unrelated files. Never delete data, model checkpoints or original sources without approval.

### 9.2 Minimum pytest suite
- **Augmentation:** two views of the same image differ; output shape `(3,32,32)`, range as expected; deterministic under a fixed seed.
- **Encoder:** CIFAR stem (no max-pool); output `(B, 512)` for input `(B,3,32,32)`.
- **Head:** output `(B, 128)`.
- **NT-Xent:** vectorized loss equals explicit-loop reference (tolerance 1e-5); loss is lower for identical views than for random views; no NaN at τ = 0.1–1.0.
- **Subsets:** stratified sampler returns exact per-class counts; subsets are nested or independent as specified in §10.3; no overlap with validation indices.
- **Linear eval:** encoder parameters unchanged after a linear-eval step; BN running stats unchanged.
- **Checkpointing:** save → load → identical outputs and optimizer state; resume continues epoch count.
- **App preprocessing:** app transform on a test image equals the evaluation transform used in training (§13).

---

## 10. Experiment Rules

### 10.1 Every run records (in `runs/<id>/config.yaml` + W&B if enabled)
Experiment ID, hypothesis, dataset + subset fraction + split file, seed(s), architecture, augmentation config, optimizer, lr + schedule, weight decay, batch size, τ, epochs **and total optimizer steps**, hardware, epoch time and total runtime, training/validation metrics, linear-eval results, artifact paths, git commit hash, exact command.

Never compare runs without listing their configuration differences. Never overwrite a previous best result or model; keep it recoverable.

### 10.2 Reproducibility
- Set seeds for Python, NumPy, PyTorch (CPU+CUDA) and DataLoader workers. Record `torch.backends.cudnn.deterministic/benchmark` settings.
- Report results from ≥ 2 seeds where compute allows (≥ 3 for the data-fraction experiment's smallest subsets); report mean ± std, never the best seed alone.
- Note: Colab GPU type can change between sessions — record the actual device for every run.

### 10.3 Data-fraction experiment design
- Fractions: decided in S5 from measured throughput (e.g. 1%, 5%, 10%, 25%, 50%, 100% of the pretraining pool). Record final list in `DECISIONS.md`.
- Sampling: stratified by class, fixed seed; prefer **nested** subsets (each smaller subset ⊂ larger) to reduce variance.
- **Training budget — resolve before running:** the proposal keeps epochs constant "so that no subset is at a disadvantage simply due to having had fewer parameter updates". Constant epochs actually give smaller subsets *fewer* updates (10% data ⇒ 10× fewer steps). Present both options to the user and record the choice:
  - (A) constant epochs (matches proposal wording, confounds data size with update count), or
  - (B) constant optimizer steps (isolates data quantity; more epochs on small subsets), or
  - (C) both, if compute allows.
- Linear evaluation for every encoder uses the **same** labeled training set (full 45k/50k labels) and the same test set, so only the pretraining data varies.
- Output: table + plot of accuracy vs. fraction (log-x), with seed error bars; state whether the hypothesis (monotonic increase with diminishing returns) is supported.

### 10.4 Paper comparison
- Record the exact paper row/figure (Appendix B.9 / Figure B.7), its configuration (encoder, batch size, epochs, optimizer) and value as `PAPER-REPORTED`.
- Present a side-by-side table of configuration differences and the `MEASURED` result; attribute the gap only to differences actually present.

### 10.5 Model checkpoints and resume (Colab)
- Save `last.pt` (model, optimizer, scheduler, epoch, step, RNG states, config) every epoch or every N minutes, to persistent storage (e.g. Google Drive), not only the Colab VM.
- Save `best.pt` by validation metric when applicable; keep previous bests.
- Training scripts must support `--resume runs/<id>/last.pt` and log the resume in the checkpoint.

---

## 11. Human Approval Required

Ask the user before:
- Deleting or overwriting original data, model checkpoints or the original pipeline.
- Changing a FIXED decision in §2 (architecture, dataset, loss, head, batch size).
- Starting any run expected to exceed **1 GPU-hour** or **100 epochs** on the full dataset (state the estimate from measured throughput).
- Installing system-wide packages or changing system settings (installing into the project virtualenv / Colab session from `requirements.txt` is allowed).
- Uploading anything externally (W&B artifacts, Hugging Face Space, datasets).
- Any irreversible action.

Routine inspection, documentation, small tests, smoke runs (< 10 min) and reversible changes may proceed within the approved scope.

---

## 12. Observability and Experiment Tracking

### 12.1 Tools and responsibilities
| Tool | Responsibility | Not for |
|---|---|---|
| Weights & Biases | Training/eval runs: config, metrics, system usage, model artifacts | App request tracing |
| Arize Phoenix (OpenTelemetry) | Inference request traces: preprocessing, encoder, classifier, latency, errors | Training metrics |
| pytest | Correctness | — |
| Markdown (checkpoints, `runs/*/`) | Human-readable record | — |

### 12.2 W&B requirements
- Log everything in §10.1 per run; use the experiment ID as run name; tag with stage and subset fraction.
- Pretraining metrics: NT-Xent loss, learning rate, (optional) contrastive accuracy = fraction of anchors whose top-1 similar view is the positive, epoch time. **Do not** report classification accuracy as a pretraining metric unless a labeled evaluation (linear probe or kNN on the validation split) was actually run — and label it as such.
- Linear-eval metrics: train loss/acc, validation acc, final test top-1/top-5.
- Artifact references: local path is the default; uploading weights as W&B artifacts requires user approval (§11).
- Never upload CIFAR-10 or unrelated files.
- API key only via environment variable `WANDB_API_KEY` (or `wandb login` by the user). If unavailable, use `WANDB_MODE=offline` or local CSV logging and mark W&B `NOT CONFIGURED`.

### 12.3 Phoenix requirements
Trace per request: request ID, timestamp, input metadata (format, original size; **not** the raw image), preprocessing duration, encoder duration, embedding dimension and finiteness check, classifier duration, predicted class, top-k probabilities, total duration, errors/exceptions, model version (run ID / artifact hash).
- Phoenix runs **locally** during development. Traces must be verified with real requests and the project/trace references recorded in the S8 checkpoint.
- Confidence (softmax probability) is not accuracy; never claim a prediction is correct without a ground-truth label.

### 12.4 Implementation rules
- Inspect code first, propose files/dependencies to change, explain network/resource needs, get approval for significant changes.
- Keep observability in `src/simclr/observability/`; core code must run with tracking disabled (`--no-wandb`, `PHOENIX_ENABLED=0`).
- Logging failures must never change training or prediction results; catch and report them.
- Measure logging overhead (epoch time with vs. without) before enabling heavy logging (e.g. image logging).

### 12.5 Acceptance
- Training observability: complete only after a real run logged config + metrics (or the limitation is documented).
- Inference observability: complete only after a real app request produced an inspectable Phoenix trace.
- Evaluation: complete only after evaluation code produced and saved the reported metrics (`runs/<id>/result.json`).

---

## 13. Deployment Rules (Streamlit + Hugging Face Spaces)

### 13.1 Model packaging
- Deploy encoder (without projection head) + linear classifier, exported from a specific run; record run ID, commit hash and SHA-256 of the weights file in the app and the S9 checkpoint.
- Never replace the deployed model without keeping the previous version recoverable.

### 13.2 Inference pipeline
- Input: user image (any size/format) or a bundled CIFAR-10 test sample.
- Preprocess: convert to RGB → resize to 32×32 → float [0,1] → (normalization only if used in training). Must use the **same function** as evaluation (tested in §9.2).
- Output: predicted class, softmax confidence, top-3 classes. Show a note that the model was trained on 32×32 CIFAR-10 images, so out-of-distribution photos may be misclassified with high confidence.
- Do not store uploaded images.

### 13.3 README must include
Start instructions, dependencies, input/output format, example inference, screenshot of the deployed app, model version/run reference, and the attribution section (§16).

### 13.4 Hosting and tracing
- Publishing to Hugging Face Spaces is an external upload — requires user approval (§11).
- Phoenix stays local: the hosted app must run with `PHOENIX_ENABLED=0` unless the user approves a hosted Phoenix setup (§19).

---

## 14. Security and Permissions
- Do not request GitHub credentials. The user (team) handles pushing/pulling to the team repository; the agent works in the local checkout it is given and lists changed files for the user to commit, unless the user explicitly grants repo access.
- Never write API keys or tokens into code, notebooks, docs, logs or commits; read them from environment variables.
- Do not upload datasets, raw user inputs or private files without approval.
- Do not expose local services publicly (Phoenix, Jupyter, dev servers). The only public endpoint is the approved Hugging Face Space.
- Keep all modifications inside the project directory.

---

## 15. Communication with the User
- Explain the math/intuition behind a component when introducing it (e.g. why the projection head helps, why τ matters, why constant epochs ≠ constant updates).
- Report results with label, run ID and configuration; keep paper and project numbers visually separate.
- When blocked or uncertain, ask one concrete question with options.

---

## 16. Attribution and Academic Integrity
- All four modules, training/validation code, the experiment and the app are written by the team.
- If any code is adapted from the official repo or a PyTorch re-implementation, record source URL, file, what was adapted and how, in `SOURCES.md` and in the README "Attribution" section (required by the proposal), and in the final report.
- Never copy code without recording it.

---

## 17. Completion Standard
A stage is `COMPLETE` only when:
- Its exit criteria in §7 are met with evidence.
- Relevant tests were executed and their output recorded.
- Actual results are recorded with labels (§3.2).
- Changed files are listed.
- Remaining limitations and `NOT MEASURED` / `NOT CONFIGURED` items are documented.
- Its `checkpoint.md` and `checkpoints/INDEX.md` are updated.

If interrupted, mark the stage `INCOMPLETE` and record the exact stopping point and the next command to run.

---

## 18. Templates

### 18.1 `checkpoints/INDEX.md`
```
| Stage | Status | Last updated | Checkpoint | Key result |
|---|---|---|---|---|
| S0 Setup | PLANNED | YYYY-MM-DD | checkpoints/S0_setup/checkpoint.md | — |
```

### 18.2 `checkpoints/S<k>_<name>/checkpoint.md`
```
# S<k> <name> — Status: PLANNED | IN_PROGRESS | INCOMPLETE | COMPLETE | BLOCKED
## Previous checkpoint verified: <path>, status, mismatches found
## Planned work
## Work done
## Files changed (path — reason)
## Commands executed (exact) and outcomes
## Tests run and results
## Metrics (label each MEASURED / NOT MEASURED; include run IDs)
## Observability: W&B run ID/URL or NOT CONFIGURED; Phoenix project/trace refs or NOT CONFIGURED
## Model artifacts: path, SHA-256, originating run
## Errors and resolutions
## Decisions made (link to DECISIONS.md entries)
## Unresolved issues / known gaps
## Reproduce: exact command(s)
## Next stage / next command
```

### 18.3 Experiment record (`runs/<id>/README.md`)
```
ID | hypothesis | dataset/subset/split file | seed(s) | encoder/head | augmentations |
optimizer, lr, schedule, wd | batch | τ | epochs / steps | hardware | runtime |
metrics (MEASURED) | artifacts | W&B run | git commit | command
```

### 18.4 `DECISIONS.md` entry
```
## D<n> <title> — YYYY-MM-DD — Status: PROPOSED | APPROVED | REJECTED
Context / requirement:
Sources (paper section, repo file:line, other):
Options considered:
Decision and rationale:
Impact on paper comparison:
Approved by:
```

---

## 19. Open Decisions (ask the user; record answers in `DECISIONS.md`)
1. Optimizer (SGD+momentum vs Adam) and lr/τ — after S5 pilots.
2. Epoch budget for main run and for the data-fraction runs — after measuring throughput.
3. Data-fraction training budget: constant epochs vs constant steps vs both (§10.3).
4. Fractions list and number of seeds per fraction.
5. Pretraining pool for the main run: 45k (with validation held out) or 50k after model selection.
6. Loss scaling convention (C1) and input normalization (yes/no).
7. Phoenix in the hosted app: disabled (default) or a hosted Phoenix instance.
8. Whether the agent may push to the team repository directly.
9. Whether model weights may be uploaded as W&B artifacts.
10. Tiny-ImageNet extension: yes/no.
