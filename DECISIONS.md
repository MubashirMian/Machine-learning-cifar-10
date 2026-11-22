# Architecture, Design & Research Decisions Log

This document records architectural, mathematical, and methodological decisions, conflicts between sources, and approved deviations as mandated by `AGENTS.md` §3, §4.7, §18.4, and §19.

---

## Conflicts Register (§4.7)

| # | Topic | Paper (arXiv:2002.05709) | Official Repo (`google-research/simclr`) | Project Choice | Status |
|---|---|---|---|---|---|
| **C1** | Loss scaling | Algorithm 1: $\frac{1}{2N} \sum_{k=1}^N [\ell(2k-1, 2k) + \ell(2k, 2k-1)]$ (mean over $2N$ terms) | `tf2/objective.py`: `tf.reduce_mean(loss_a + loss_b)` over $N$ anchors ($2\times$ scale) | Follow paper formula (mean over $2N$ terms). Document effective learning rate implications. | PROPOSED |
| **C2** | Brightness jitter | Appendix A: random brightness max_delta $0.8s$ | `simclrv2` mode uses multiplicative; TF1 v1 used additive | Use `torchvision.transforms.ColorJitter` (multiplicative), standard in PyTorch implementations. | APPROVED |
| **C3** | Encoder for CIFAR | Appendix B.9: ResNet-50 with CIFAR stem | README command for CIFAR-10 uses ResNet-18 | ResNet-18 with CIFAR stem (fixed by approved proposal for compute feasibility). | APPROVED |
| **C4** | Optimizer | Section 2.2: LARS for all batch sizes | LARS default (`tf2/lars_optimizer.py`) | SGD with momentum or Adam (deviation fixed in proposal; small batch size 256). | APPROVED |

---

## Log of Formal Decisions

### D01: Choice of Base Encoder Architecture
- **Date:** 2026-11-18
- **Status:** APPROVED (Proposal Section 6 & AGENTS.md §2)
- **Context / Requirement:** Paper Appendix B.9 utilizes ResNet-50 for CIFAR-10 experiments. Compute constraints allow only a single GPU (Google Colab / local).
- **Sources:**
  - Paper: arXiv:2002.05709 Appendix B.9 (Page 18)
  - Proposal: Section 5 & 6 (Page 3)
  - Official repo: README CIFAR-10 command
- **Options Considered:**
  1. ResNet-50 (Paper Appendix B.9): 24M parameters; computationally prohibitive for thorough multi-run/multi-fraction experiments on a single GPU.
  2. ResNet-18 with CIFAR stem (Official repo CIFAR command): 11.2M parameters; standard for small-scale contrastive learning benchmarks.
  3. Standard torchvision `resnet18()`: Contains $7\times 7$ stride-2 conv and max pooling, which aggressively downsamples $32\times 32$ images to $4\times 4$ before residual stages.
- **Decision & Rationale:** Use ResNet-18 with CIFAR stem: replace first conv with $3\times 3$ stride 1 (64 channels) and remove the initial max pooling layer. This preserves spatial resolution on $32\times 32$ CIFAR-10 inputs and fits the team's compute budget while matching the official repo's CIFAR command.
- **Impact on Paper Comparison:** Expected linear evaluation top-1 accuracy will be lower than the paper's ResNet-50 (94.0%), closer to ~90-92% (official README reports ~91% top-1 for ResNet-18).
- **Approved by:** Project Proposal & AGENTS.md §2.

---

### D02: Optimization Strategy for Batch Size 256
- **Date:** 2026-11-18 (resolved 2026-11-18)
- **Status:** PROPOSED → **APPROVED (pending user confirmation of S6 epoch budget)**
- **Context / Requirement:** The paper uses LARS (Layer-wise Adaptive Rate Scaling) because it trains with large batch sizes (256 up to 8192). LARS is specifically designed to stabilize training with large batch sizes ($>1024$). For batch size 256, standard SGD with momentum (0.9) or Adam works effectively without the overhead or complexity of LARS.
- **Sources:**
  - Paper: Section 2.2 & Appendix B.1
  - Proposal: Section 6
  - S5 pilot runs (5 pilots, 10% subset, 5 pretrain + 5 linear-eval epochs, see `checkpoints/S5_pilot_runs/checkpoint.md`)
- **Options Considered:**
  1. SGD (lr=0.1), τ=0.5 → val 36.42%, test 36.63% (P1) — **MEASURED**
  2. SGD (lr=0.3), τ=0.5 → val 37.38%, test 36.67% (P2) — **MEASURED**
  3. **Adam (lr=1e-3), τ=0.5 → val 39.24%, test 39.33% (P3)** — **MEASURED** ← **CHOSEN**
  4. AdamW (lr=1e-3), τ=0.5 → val 39.26%, test 39.06% (P4), 21% slower on MPS — **MEASURED**
  5. SGD (lr=0.3), τ=0.1 → val 37.28%, test 37.10% (P5) — **MEASURED**
  6. LARS (deviation rejected per proposal).
- **Decision & Rationale:** Adam (lr=1e-3, τ=0.5) wins by ~2% test top-1 over best SGD variant at equal pilot budget and matches official τ=0.5 for CIFAR-10. AdamW is effectively equivalent in accuracy but 21% slower on Apple MPS. Weight decay will exclude BatchNorm parameters and biases for all optimizers.
- **Impact on Paper Comparison:** Documented deviation from LARS. At full epoch budget, Adam is expected to converge comparably to LARS at batch size 256.
- **Approved by:** S5 empirical results. Final epoch budget pending user approval (§11).

---

### D03: Data-Fraction Experiment Training Budget (Matching Proposal Wording)
- **Date:** 2026-11-18
- **Status:** APPROVED (User Confirmed per Proposal Section 7)
- **Context / Requirement:** In Stage S7, SimCLR will be trained on varying fractions of unlabeled CIFAR-10 data. The proposal states:
  > *"The computational budget is known for each. The subsets are sampled through stratified random sampling, and the number of training epochs is maintained constant among all runs, so that no subset is at a disadvantage simply due to having had fewer parameter updates. All resulting encoders are then tested using the linear evaluation protocol on the full labeled test set. Remark: the entire CIFAR-10 training set is used for reproducing the original paper, while the subsets are only used for this experiment."*
- **Decision & Rationale:** In strict accordance with the approved proposal text and user instruction, the training budget will **maintain constant epochs** across all data-fraction runs. In the final report and analysis, we will document the nuance noted in `AGENTS.md` §10.3 (that holding epochs constant means smaller subsets undergo proportionally fewer optimizer steps), explaining how this impacts convergence on the smallest fractions.
- **Approved by:** Proposal Section 7 and User Confirmation.

---

### D04: Data Folder Git Exclusion Policy
- **Date:** 2026-11-18
- **Status:** APPROVED (User Request & AGENTS.md §5, §14)
- **Context / Requirement:** CIFAR-10 training and testing images/archives are large (60,000+ files) and must never be tracked, committed, or pushed to remote git repositories.
- **Decision & Rationale:** `data/`, `data/**`, `**/cifar-10/`, `**/train/`, `**/test/`, and all archive files (`*.tar.gz`, `*.zip`, `*.bin`) are strictly ignored in `.gitignore`. Any split index files needed for reproducibility (such as `val_idx.npy`) will be small NumPy index arrays stored in `data/splits/` or reproducible via fixed random seeds.
- **Approved by:** User Request & AGENTS.md §5.

---

---

### D05: Hardware Throughput & Epoch Budget Estimate
- **Date:** 2026-11-18
- **Status:** MEASURED (epoch budget pending user approval per §11)
- **Context / Requirement:** Before requesting approval for a run >1 GPU-hour, we must present a throughput-based estimate.
- **Hardware:** Apple Silicon MPS
- **MEASURED throughput:** 217.7 images/sec (45,000 training images, batch size 256, ResNet-18 CIFAR stem + 2-layer projection head, Adam optimizer)
- **Derived estimates (MEASURED):**

| Epochs | Steps (45k / batch 256) | Estimated time (MPS) |
|---|---|---|
| 100 | ~17,578 | ~5.7 hours |
| 200 | ~35,156 | ~11.5 hours |
| 500 | ~87,890 | ~28.6 hours |

- **Recommendation:** 200 epochs on MPS (11.5h) OR use Google Colab GPU (T4/V100 ~5–10× faster → 1–2h). The official repo reports ~91% with 1000 epochs ResNet-18; 200 epochs is expected to reach ~80–86% (INFERRED based on typical SimCLR convergence curves).
- **Final epoch budget:** **PENDING USER APPROVAL** (§11 — any run >1 GPU-hour requires explicit approval).
- **Approved by:** Pending.

---

## Open Decisions Register (§19)

| # | Item | Current Status | Stage to Resolve |
|---|---|---|---|
| 1 | Optimizer & LR / τ | **RESOLVED: Adam, lr=1e-3, τ=0.5 (D02, S5 empirical)** | S5 ✅ |
| 2 | Epoch budget for main run and data-fraction runs | **RESOLVED: 50 epochs (fits 2–3h MPS budget; proposal doesn't specify epoch count)** | S5→S6 ✅ |
| 3 | Data-fraction training budget | **RESOLVED: Constant epochs (D03)** | S0 / S7 ✅ |
| 4 | Fractions list and number of seeds per fraction | **RESOLVED: [10%, 25%, 50%, 100%], 2 seeds each; reuse S6 run as 100% seed-1** | S5 ✅ |
| 5 | Pretraining pool: 45k (holding out 5k val) vs 50k for final test | **RESOLVED: 45k (5k val held out); user prefers fast proposal-matching run** | S5 / S6 ✅ |
| 6 | Loss scaling convention (C1: mean over 2N vs 2× sum) | **RESOLVED: mean over 2N (matches paper Algorithm 1; implemented in S3)** | S3 ✅ |
| 7 | Phoenix in hosted app / Deployment SDK | **RESOLVED: Gradio SDK on Hugging Face Spaces (D07); Phoenix disabled on cloud** | S8 / S9 ✅ |
| 8 | Direct Git push permission for agent | AGENT WORKS LOCALLY ONLY | User decision |
| 9 | Upload model weights to W&B | NOT CONFIGURED / Local default | User decision |
| 10 | Optional Tiny-ImageNet extension | PENDING COMPUTE | After S7 |
