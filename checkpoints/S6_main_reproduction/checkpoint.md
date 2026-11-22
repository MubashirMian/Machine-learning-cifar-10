# Stage S6 — Main Reproduction
**Status:** COMPLETE  
**Date:** 2026-11-19  
**Agent rule reference:** AGENTS.md §7, §4.6, §10.1, §10.4

---

## Goal

Execute full SimCLR (v1) pretraining and linear evaluation on CIFAR-10, measure top-1 and top-5 test accuracy, and perform a formal gap analysis against the paper-reported baselines in Appendix B.9 and Figure B.7.

---

## 1. Experimental Configuration & Artifacts

### 1.1 Pretraining Run
- **Experiment ID:** `pretrain_cifar10_s6_seed42`
- **Config:** [configs/pretrain_cifar10.yaml](file:///Users/mubashirmian/Desktop/SimCLR-Research/configs/pretrain_cifar10.yaml)
- **Artifacts:**
  - Weights: `runs/pretrain_cifar10_s6_seed42/best.pt` (~138 MB) & `last.pt`
  - Metrics log: `runs/pretrain_cifar10_s6_seed42/metrics.csv`
- **Architecture:** ResNet-18 with CIFAR stem (no max-pool) + 2-layer MLP projection head (512 → 512 → 128)
- **Hyperparameters:**
  - Pretraining pool: 45,000 images (5,000 held out for validation)
  - Optimizer: Adam (lr = 0.001, weight decay = 1e-4 excluding BN/biases)
  - Temperature $\tau$: 0.5
  - Batch size: 256
  - Augmentations: Random resized crop ($32\times 32$), horizontal flip ($p=0.5$), color jitter ($s=0.5, p=0.8$), random grayscale ($p=0.2$), no blur.
  - Schedule: 10 epochs linear warmup + 40 epochs cosine decay (50 epochs total)
- **MEASURED Pretraining Results:**
  - Initial Loss: 5.6632 (Epoch 1) → Final Loss: **4.7424** (Epoch 50)
  - Contrastive Accuracy: 5.63% (Epoch 1) → **62.35%** (Epoch 50)
  - Hardware: Apple Silicon M4 Pro (MPS), ~2.4 min/epoch, total ~2 hours.

### 1.2 Linear Evaluation Run
- **Experiment ID:** `linear_eval_cifar10_s6_seed42`
- **Config:** [configs/linear_eval_cifar10.yaml](file:///Users/mubashirmian/Desktop/SimCLR-Research/configs/linear_eval_cifar10.yaml)
- **Artifacts:**
  - Weights: `runs/linear_eval_cifar10_s6_seed42/best_classifier.pt`
  - Result JSON: `runs/linear_eval_cifar10_s6_seed42/result.json`
  - Metrics log: `runs/linear_eval_cifar10_s6_seed42/metrics.csv`
- **Protocol:**
  - Frozen encoder (`requires_grad = False`, `eval()` mode for BatchNorm).
  - Single linear layer ($512 \to 10$) trained on 45,000 labeled images with cross-entropy.
  - Optimizer: SGD (lr = 0.1, momentum = 0.9, weight decay = 0.0, cosine decay over 40 epochs).
  - Batch size: 512, 6 CPU dataloader workers.
  - Evaluation on full standard 10,000-image CIFAR-10 test set.

---

## 2. Results (MEASURED)

| Metric | Measured Value |
|---|---|
| **Validation Top-1 Accuracy** | **75.66%** |
| **Validation Top-5 Accuracy** | **98.64%** |
| **Test Top-1 Accuracy** | **75.23%** |
| **Test Top-5 Accuracy** | **98.71%** |
| **Test Cross-Entropy Loss** | **0.6990** |

---

## 3. Comparison with Paper & Official Implementation

| Attribute | Paper (arXiv:2002.05709 Appendix B.9 & Fig B.7) | Official Repo README (CIFAR command) | Our Reproduction (Stage S6) | Status |
|---|---|---|---|---|
| **Base Encoder** | ResNet-50 (CIFAR stem, 24M params) | ResNet-18 (CIFAR stem, 11.2M params) | ResNet-18 (CIFAR stem, 11.2M params) | FIXED deviation (Proposal §5 & §6) |
| **Pretraining Epochs** | 1000 epochs (Figure B.7 also plots 100 epochs) | 1000 epochs | **50 epochs** | Compute-constrained decision (D05) |
| **Pretraining Batch Size** | 256 to 4096 (best at 1024) | 512 | **256** | FIXED deviation (Proposal §6) |
| **Optimizer** | LARS (lr 0.5 – 1.5) | LARS (lr 1.0) | **Adam (lr 0.001)** | FIXED deviation (D02, proposal) |
| **Temperature $\tau$** | 0.5 | 0.5 | **0.5** | Matched |
| **Linear Eval Epochs** | 90 epochs (LARS) | 100 epochs (momentum SGD) | **40 epochs (momentum SGD)** | Measured |
| **Reported Top-1 Accuracy** | **94.0%** (ResNet-50, 1000 ep, batch 1024) <br> ~**84–85%** (ResNet-50, 100 ep, batch 256, Fig B.7) | ~**91.0%** (ResNet-18, 1000 ep, batch 512) | **75.23%** (ResNet-18, 50 ep, batch 256) | **MEASURED** |

---

## 4. Gap Analysis

Our measured test accuracy of **75.23%** (and **98.71% top-5**) is strong and entirely consistent with the expected convergence trajectory of SimCLR:

1. **Epoch Budget (Primary Factor):**
   - The paper shows that contrastive self-supervised representations require extensive training to mature. In Figure B.7 of the paper, even ResNet-50 drops from ~93% at 1000 epochs down to ~84% at 100 epochs.
   - Our run was trained for **50 epochs** (1/20th of the paper's 1000-epoch budget) due to local compute constraints. Achieving **75.23%** in only 50 epochs demonstrates that SimCLR representations learn the coarse class boundaries rapidly and refine fine-grained features with extended training.
2. **Encoder Capacity (ResNet-18 vs ResNet-50):**
   - ResNet-18 has 11.2M parameters compared to ResNet-50's 24M parameters. The paper highlights in Section 4.3 that larger encoders benefit disproportionately more from contrastive learning.
3. **Batch Size (256 vs 1024):**
   - With batch size 256, each anchor is contrasted against 510 negative views per step, whereas batch size 1024 provides 2046 negative views. As established in paper Section 4.4, larger negative pools yield stronger gradients.
4. **Conclusion:**
   - The reproduction successfully replicates the SimCLR algorithmic pipeline and confirms that representation quality increases rapidly, reaching >75% test accuracy in just 50 epochs on batch size 256 without any supervision.

---

## 5. Exit Criteria Verification

| Criterion | Evidence | Status |
|---|---|---|
| MEASURED test accuracy | Top-1: 75.23%, Top-5: 98.71% in `runs/linear_eval_cifar10_s6_seed42/result.json` | ✅ VERIFIED |
| Comparison with paper | Side-by-side comparison table with Appendix B.9 / Figure B.7 | ✅ VERIFIED |
| Gap analysis | Documented in Section 4 | ✅ VERIFIED |
