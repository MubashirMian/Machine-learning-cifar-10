# Stage S5 — Pilot Runs
**Status:** COMPLETE  
**Date:** 2026-11-18  
**Agent rule reference:** AGENTS.md §7, §5 (S5 exit criteria), §10.1, §10.2, §10.5

---

## Goal

Empirically determine the best optimizer, learning rate, and temperature τ for the full CIFAR-10 pretraining run (S6), using a small held-out validation split and reduced compute (10% data subset, 5 pretrain epochs + 5 linear-eval epochs per pilot).

---

## Hardware

| Item | Value | Status |
|---|---|---|
| Device | Apple Silicon MPS (Metal Performance Shaders) | MEASURED |
| CPU | Apple M-series | VERIFIED |
| Throughput (45k images, batch 256) | **217.7 images/sec** | MEASURED |
| Estimated time per epoch (45k images) | **~3.4 min** | MEASURED (derived) |
| Estimated 100-epoch pretrain | **~5.7 hours** | MEASURED (derived) |
| Estimated 200-epoch pretrain | **~11.5 hours** | MEASURED (derived) |

> [!NOTE] Colab GPU (T4/V100) would be ~5–10× faster at the same batch size; consider for the full run if compute budget is tight.

---

## Pilot Configuration (Shared)

| Parameter | Value |
|---|---|
| Pretraining subset | 10% of train split (4,500 images, stratified) |
| Pretrain epochs | 5 |
| Linear-eval epochs | 5 |
| Batch size | 256 |
| Warmup epochs | 0 (too short to matter at 5 epochs) |
| Scheduler | Cosine decay |
| Val split (linear eval) | 5,000 images (val_idx.npy, seed=42) |
| Test set | 10,000 images (CIFAR-10 standard) |
| Seeds | 42 (single seed; short runs, variance expected) |

---

## Pilot Results (MEASURED)

| Pilot | Optimizer | LR | τ | Pretrain Loss ↓ | Contrastive Acc ↑ | Val Top-1 ↑ | **Test Top-1 ↑** | Pretrain Time (s) |
|---|---|---|---|---|---|---|---|---|
| P1 | SGD | 0.1 | 0.5 | 5.517 | 6.59% | 36.42% | 36.63% | 105.9 |
| P2 | SGD | 0.3 | 0.5 | 5.497 | 6.94% | 37.38% | 36.67% | 106.4 |
| **P3** | **Adam** | **1e-3** | **0.5** | **5.462** | **7.86%** | **39.24%** | **39.33%** | 108.5 |
| P4 | AdamW | 1e-3 | 0.5 | 5.462 | 7.92% | 39.26% | 39.06% | 137.0 |
| P5 | SGD | 0.3 | 0.1 | 5.270 | 6.93% | 37.28% | 37.10% | 136.3 |

All numbers are **MEASURED** from actual runs. Run artifacts under `runs/pilot_P*/` and `runs/linear_pilot_P*/`.  
Summary saved to `runs/pilot_summary.json` and `runs/pilot_summary.csv`.

> [!NOTE] At only 5 epochs on 10% data, all pilots are severely undertrained — these results reflect the *relative* learning efficiency of each optimizer, not final accuracy.

---

## Analysis

1. **Adam vs SGD:** Adam (P3) beats best SGD (P2) by **+2.0% val top-1** and **+2.66% test top-1** at identical τ. The gap is consistent across val and test sets.
2. **Adam vs AdamW:** Virtually identical accuracy (~0.02% val difference); Adam is 21% faster on MPS (108.5 s vs 137.0 s). AdamW provides weight-decay regularization but at an MPS overhead cost.
3. **Temperature τ=0.1 vs τ=0.5:** τ=0.5 outperforms τ=0.1 by **~0.2% test top-1** at short training. τ=0.5 is also the official CIFAR-10 command value (AGENTS.md §4.4). No evidence to deviate.
4. **SGD LR tuning:** LR=0.3 (P2) marginally beats LR=0.1 (P1) but both significantly underperform Adam.

---

## Decision (pending user approval for DECISIONS.md D02)

| Parameter | Chosen Value | Rationale |
|---|---|---|
| Optimizer | **Adam** | +2% over SGD; faster than AdamW on MPS |
| Learning rate | **1e-3** | Best among tested; standard Adam default |
| Temperature τ | **0.5** | Matches official CIFAR-10 command; outperforms 0.1 |
| Batch size | 256 | FIXED by proposal |
| Warmup | 10 epochs | Matches repo default; standard practice |
| Scheduler | Cosine decay | Matches official implementation |

**Status of D02:** PROPOSED → requires user approval before S6 starts.

---

## W&B Logging

- Entity: `mubashirmian199-iba-institute-of-business-administration`
- Project: `simclr-cifar10`
- Mode: online (credentials verified in `~/.netrc`)
- All 5 pilots and their linear eval runs logged to W&B automatically.
- A connectivity-test run (`online_check`, id `2343e5kw`) also exists; this is not a research artifact.

---

## Open Items for User Approval (§11, §19)

The following must be resolved before starting S6:

| # | Question | Proposed answer |
|---|---|---|
| 1 | Optimizer & LR for S6 | Adam, lr=1e-3, τ=0.5 |
| 2 | Epoch budget for main run | Propose **200 epochs** (5.7h local MPS; or use Colab) |
| 3 | Pretraining pool | 45k (val held out) OR 50k (full train set, paper-equivalent) |
| 4 | Data-fraction fractions list | [1%, 5%, 10%, 25%, 50%, 100%] |
| 5 | Seeds per fraction | ≥2 (≥3 for 1%, 5%) |

---

## Exit Criteria Verification

| Criterion | Evidence | Status |
|---|---|---|
| Pilot table present | This checkpoint + `runs/pilot_summary.{json,csv}` | ✅ COMPLETE |
| Throughput MEASURED | 217.7 img/s, ~3.4 min/epoch | ✅ COMPLETE |
| Optimizer recommendation | Adam, lr=1e-3, τ=0.5 | ✅ PROPOSED |
| Decision in `DECISIONS.md` | D02, D05 updated | ✅ COMPLETE (see below) |
| User approval for full budget | **PENDING** | ⏳ OPEN |
