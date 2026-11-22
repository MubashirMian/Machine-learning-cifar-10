# Project Plan — SimCLR (v1) on CIFAR-10: Reproduction & Scaling

## 1. Project Overview & Objectives
This project is an empirical reproduction and extension of SimCLR (v1) (*A Simple Framework for Contrastive Learning of Visual Representations*, Chen et al., ICML 2020) using PyTorch on the CIFAR-10 dataset, carried out for the Machine Learning Paper Reproduction & Deployment Assignment.

### Core Objectives:
1. **From-Scratch Implementation:** Implement all four modules of SimCLR (stochastic data augmentation, base encoder ResNet-18 with CIFAR stem, 2-layer MLP projection head, and NT-Xent loss) cleanly in PyTorch.
2. **Review & Adapt Existing Pipeline:** Preserve and rigorously audit the team's existing data cleaning/EDA pipeline (`existing_project/eda_pipeline.ipynb`) without data leakage or non-standard modifications.
3. **Main Paper Reproduction:** Pretrain SimCLR on CIFAR-10 and evaluate learned representations using the frozen linear evaluation protocol (top-1 accuracy on the standard 10,000-image test set), comparing directly against the paper's Appendix B.9 / Figure B.7 baselines.
4. **Data-Scaling Experiment:** Explore an under-studied ablation: pretraining-data fraction (e.g. 1%, 5%, 10%, 25%, 50%, 100%) vs. representation quality under fixed linear evaluation.
5. **Interactive Deployment & Observability:** Build and deploy a Streamlit inference web application (hosted on Hugging Face Spaces) with optional local Arize Phoenix OpenTelemetry tracing, and track training runs via Weights & Biases (W&B).

---

## 2. Methodology & Architectural Specifications

### 2.1 Four SimCLR Modules
1. **Data Augmentation (Pretraining):**
   - Applied stochastically twice per input image to create positive pairs $(x_i, x_j)$.
   - Operations (per paper Appendix B.9 and repo `tf2/data_util.py`):
     - Random resized crop to 32×32: area scale $\in [0.08, 1.0]$, aspect ratio $\in [3/4, 4/3]$, bicubic interpolation.
     - Random horizontal flip ($p = 0.5$).
     - Random color jitter ($p = 0.8$, strength $s = 0.5$ for CIFAR-10: brightness 0.4, contrast 0.4, saturation 0.4, hue 0.1).
     - Random grayscale ($p = 0.2$).
     - **No Gaussian blur on CIFAR-10** (as per paper Appendix B.9 and official repo `--use_blur=False`).
   - Inputs are float in $[0, 1]$.
2. **Base Encoder $f(\cdot)$:**
   - Modified ResNet-18 with CIFAR stem (paper Appendix B.9):
     - Replace standard $7\times 7$ stride 2 conv + max-pool with a single $3\times 3$ stride 1 conv (64 channels) $\to$ BatchNorm $\to$ ReLU (**no max pooling**).
     - Standard residual stages: $[2, 2, 2, 2]$ basic blocks with channels 64, 128, 256, 512.
     - Global average pooling output: representation vector $h \in \mathbb{R}^{512}$.
3. **Projection Head $g(\cdot)$:**
   - 2-layer MLP (SimCLR v1): Linear(512, 512) $\to$ BatchNorm $\to$ ReLU $\to$ Linear(512, 128) $\to$ BatchNorm (no ReLU on final output).
   - $z = g(h) \in \mathbb{R}^{128}$.
   - Discarded after pretraining; only $h$ is used for downstream linear evaluation and inference.
4. **NT-Xent Loss:**
   - For batch of $N$ images $\to 2N$ augmented views. Cosine similarity $\text{sim}(u, v) = \frac{u^T v}{\|u\|_2 \|v\|_2}$ with temperature $\tau = 0.5$.
   - Anchor $i$ with positive $j$:
     $$\ell(i, j) = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k \neq i}^{2N} \exp(\text{sim}(z_i, z_k)/\tau)}$$
   - Averaged across all $2N$ views.

### 2.2 Linear Evaluation Protocol
- Encoder frozen: `requires_grad = False` and `model.eval()` (freezing BatchNorm running statistics).
- A single linear layer ($512 \to 10$) trained with cross-entropy on representations $h$.
- Augmentation for linear training: random resized crop + horizontal flip only (no color distortion).
- Test evaluation: direct conversion to tensor / center evaluation on the standard 10,000-image test set.

---

## 3. Stages and Milestones

| Stage | Milestone | Primary Deliverable | Exit Criteria |
|---|---|---|---|
| **S0** | Environment & Setup | Governance files, repo layout, `requirements.txt` | Layout created; `requirements.txt` ready; pytest working |
| **S1** | Pipeline Review | `checkpoints/S1_pipeline_review/pipeline_review.md` | Audit of `existing_project/eda_pipeline.ipynb`; verify no data leakage |
| **S2** | Data Module | `src/simclr/data.py`, `src/simclr/augment.py` | Pytest passing on augmentations & stratified splits; sample grid saved |
| **S3** | Model & Loss | `src/simclr/model.py`, `src/simclr/loss.py` | Shape tests passing; NT-Xent loss matches explicit reference loop |
| **S4** | Smoke Training | End-to-end pipeline run (1-2 epochs) | Pretraining + linear eval completes on subset; checkpoint resume verified |
| **S5** | Pilot Runs | Hyperparameter tuning on validation split | Optimizer choice (SGD vs Adam), lr, $\tau$ validated; throughput measured |
| **S6** | Main Reproduction | Full pretrain & linear eval run | Measured top-1 accuracy on CIFAR-10 test set; paper gap analysis |
| **S7** | Data-Fraction Exp | Pretraining data scaling (1%, 5%, 10%, 25%, 50%, 100%) | Results table & plot across seeds; hypothesis evaluated |
| **S8** | App Development | `app/app.py` Streamlit web application | Working local inference app with top-3 predictions and confidence score |
| **S9** | Deployment | Hosted on Hugging Face Spaces | Public working URL; README documentation complete |
| **S10** | Final Report | Summary report & documentation | All metrics traceable to runs; full attribution & reproducibility guide |

---

## 4. Hardware and Execution Plan
- **Local Machine:** Development, unit testing, smoke tests, Streamlit app local development.
- **Single GPU (Google Colab / Local GPU):** Pretraining pilot runs, full CIFAR-10 pretraining, data-fraction experiment runs.
- **Checkpoint Resilience:** Save model checkpoints (`last.pt`, `best.pt`) with full RNG states and optimizer status every epoch / period to enable preemption recovery.
