# Stage S7 — Data-Fraction Scaling Experiment
**Status:** COMPLETE  
**Date:** 2026-11-19  
**Agent rule reference:** AGENTS.md §7, §10.3, Proposal Section 7, DECISIONS.md D03

---

## 1. Goal & Hypothesis

### 1.1 Research Question
How does the volume of unlabeled pretraining data affect the quality of learned contrastive representations when evaluated on a fixed downstream linear classification benchmark?

### 1.2 Hypothesis
As the unlabeled pretraining data fraction increases, linear evaluation top-1 accuracy will exhibit **monotonic growth with diminishing marginal returns** (logarithmic scaling behavior).

---

## 2. Experimental Protocol & Setup

- **Dataset:** CIFAR-10 training pool (45,000 images, 5,000 held out for validation).
- **Subsets:** Nested, stratified subsets sampled by class with fixed seed (42):
  - **10%:** 4,500 images (450 per class)
  - **25%:** 11,250 images (1,125 per class)
  - **50%:** 22,500 images (2,250 per class)
  - **100%:** 45,000 images (4,500 per class) — completed in Stage S6.
- **Pretraining Budget:** Constant 50 epochs across all fractions (Decision D03, strictly adhering to proposal text).
  - Adam optimizer, lr = 0.001, $\tau = 0.5$, batch size = 256.
- **Linear Evaluation Protocol:**
  - Evaluated on the **identical labeled training pool** (45,000 images) and tested on the **identical standard 10,000-image CIFAR-10 test set**.
  - Only the pretraining encoder varies; classifier architecture ($512 \to 10$) and linear training settings (40 epochs, batch size 512, SGD lr = 0.1) are held strictly constant.

---

## 3. Results (MEASURED)

All metrics below are **MEASURED** from the executed runs and recorded in [runs/data_fraction/fraction_summary.csv](file:///Users/mubashirmian/Desktop/SimCLR-Research/runs/data_fraction/fraction_summary.csv):

| Fraction | Pretrain Samples | Pretrain Epochs | Linear Epochs | Val Top-1 (%) | **Test Top-1 (%)** | **Test Top-5 (%)** | Marginal Gain (Top-1) |
|---|---|---|---|---|---|---|---|
| **10%** | 4,500 | 50 | 40 | 56.20% | **54.17%** | 94.52% | — (baseline) |
| **25%** | 11,250 | 50 | 40 | 65.64% | **65.37%** | 97.14% | **+11.20%** (for 2.5× data) |
| **50%** | 22,500 | 50 | 40 | 72.08% | **71.04%** | 98.07% | **+5.67%** (for 2.0× data) |
| **100%** | 45,000 | 50 | 40 | 75.66% | **75.23%** | 98.71% | **+4.19%** (for 2.0× data) |

---

## 4. Scaling Curve & Hypothesis Evaluation

### 4.1 Generated Artifact
- **Plot:** [runs/data_fraction/scaling_plot.png](file:///Users/mubashirmian/Desktop/SimCLR-Research/runs/data_fraction/scaling_plot.png)

### 4.2 Key Findings:
1. **Monotonic Improvement:** Top-1 accuracy strictly increases with data volume: $54.17\% \to 65.37\% \to 71.04\% \to 75.23\%$.
2. **Diminishing Marginal Returns:** 
   - Moving from 10% to 25% data (+6,750 images) yields a massive **+11.20%** accuracy boost (+1.66% per 1,000 images).
   - Moving from 25% to 50% data (+11,250 images) yields **+5.67%** (+0.50% per 1,000 images).
   - Moving from 50% to 100% data (+22,500 images) yields **+4.19%** (+0.19% per 1,000 images).
3. **Data Efficiency of Contrastive Learning:**
   - With only 10% unlabeled data (4,500 images), SimCLR achieves 54.17% test accuracy and 94.52% top-5 accuracy.
   - With 25% unlabeled data, accuracy crosses 65%, reaching over 86% of the full-dataset performance.
4. **Hypothesis Status:** **SUPPORTED**. The empirical scaling trajectory matches the classic power-law / logarithmic return characteristic of self-supervised visual representation learning.

---

## 5. Exit Criteria Verification

| Criterion | Evidence | Status |
|---|---|---|
| Result table across fractions | `runs/data_fraction/fraction_summary.csv` | ✅ VERIFIED |
| Scaling plot generated | `runs/data_fraction/scaling_plot.png` | ✅ VERIFIED |
| Hypothesis evaluated | Section 4: Monotonic growth with diminishing returns confirmed | ✅ VERIFIED |
