# Sources of Truth & References

This file tracks all formal references, papers, repositories, and documentation used in this project, in accordance with `AGENTS.md` §3 and §6.

---

## 1. Project Proposal (Assignment Specification)
- **Title:** Machine Learning Paper Reproduction & Deployment Assignment — Paper Proposal
- **Authors:** Saalik Hussain Zaidi (36206), Mubashir Abid (36224), Muhammad Zohair Khan (33939)
- **Institution:** Institute of Business Administration (IBA), Karachi
- **Date Accessed:** 2026-11-18
- **Local Path:** `proposal/ML project Proposal.pdf`
- **What it supports:**
  - Scope: SimCLR (v1) reproduction from scratch in PyTorch on CIFAR-10.
  - Encoder decision: ResNet-18 with CIFAR stem (deviation from paper's ResNet-50).
  - Batch size: 256.
  - Optimizer: SGD (momentum) or Adam (not LARS).
  - Data-fraction scaling experiment: testing representation quality vs. unlabeled pretraining data volume.
  - Deployment: Streamlit web app hosted on Hugging Face Spaces.

---

## 2. Original Paper
- **Title:** A Simple Framework for Contrastive Learning of Visual Representations
- **Authors:** Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
- **Venue & Year:** International Conference on Machine Learning (ICML 2020), PMLR 119:1597-1607
- **URL:** [arXiv:2002.05709v3 [cs.LG]](https://arxiv.org/abs/2002.05709)
- **Local Path:** `papers/original_simclr_paper.pdf`
- **Date Accessed:** 2026-11-18
- **Relevant Sections & Findings:**
  - **Section 2.1 & Algorithm 1 (Page 2-3):** NT-Xent loss formula (Eq. 1), cosine similarity $\text{sim}(u, v) = u^T v / (\|u\|_2 \|v\|_2)$, projection head $g(h) = W^{(2)}\sigma(W^{(1)}h)$, batch loss average across $2N$ anchors.
  - **Section 3 & Appendix A (Page 3-5, 12-13):** Data augmentation composition (crop & resize + color jitter + grayscale). Color distortion strength $s=1.0$ (ImageNet) vs $s=0.5$ (CIFAR-10). Note on Gaussian blur: blur improves ResNet-50 on ImageNet from 63.2% to 64.5%, but is omitted on CIFAR-10.
  - **Section 4.2 & Figure 8 (Page 6):** Nonlinear projection head vs. linear (+3%) vs. none (+10%). Representation $h$ before projection is >10% better for downstream tasks than $z=g(h)$.
  - **Section 5.1 & Table 5 (Page 6-7):** Importance of $\ell_2$ normalization and temperature $\tau$.
  - **Appendix B.9 (Page 18-19):** CIFAR-10 experiments.
    - Modified stem: replace $7\times 7$ conv (stride 2) with $3\times 3$ conv (stride 1) and remove first max-pool.
    - Augmentations: Inception crop to $32\times 32$, color distortion ($s=0.5$), no Gaussian blur.
    - Hyperparameters: $\text{lr} \in \{0.5, 1.0, 1.5\}$, $\tau \in \{0.1, 0.5, 1.0\}$, batch size $\in \{256, 512, 1024, 2048, 4096\}$.
    - Reported CIFAR-10 baseline: Best ResNet-50 at batch 1024 reaches 94.0% linear evaluation top-1 (vs 95.1% supervised).
  - **Figure B.7 (Page 18):** ResNet-50 on CIFAR-10 with batch size 256 yields ~84-85% at 100 epochs and ~92-93% at 1000 epochs.

---

## 3. Official Reference Implementation
- **Repository:** `google-research/simclr`
- **Authors:** Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
- **URL:** [https://github.com/google-research/simclr](https://github.com/google-research/simclr)
- **Commit Reference:** `383d414` (TensorFlow 2 / TF-Slim)
- **Date Accessed:** 2026-11-18
- **Relevant Files:**
  - `tf2/data_util.py`: `preprocess_for_train`, `random_color_jitter`, `crop_and_resize`. Confirms multiplicative color jitter, area scale $[0.08, 1.0]$, aspect ratio $[3/4, 4/3]$, bicubic resize.
  - `tf2/resnet.py`: `cifar_stem=True` when image size $\le 32$; no bias on conv layers, BN momentum 0.9 (TF) $\equiv$ 0.1 (PyTorch), last BN initialized with $\gamma = 0$.
  - `tf2/model.py`: `ProjectionHead` implementation with Linear + BN + ReLU + Linear + BN.
  - `tf2/objective.py`: `add_contrastive_loss`. Implementation calculates `mean(loss_a + loss_b)` over $N$ anchors (scaling differs by $2\times$ from Eq. 1; recorded in `DECISIONS.md`).
  - `README.md`: Official CIFAR-10 command: batch 512, 1000 epochs, LARS, lr 1.0, weight decay 1e-4, $\tau=0.5$, ResNet-18, $s=0.5$, `--use_blur=False`.

---

## 4. Benchmark Dataset
- **Title:** Learning Multiple Layers of Features from Tiny Images (CIFAR-10 technical report)
- **Authors:** Alex Krizhevsky, Geoffrey Hinton (2009)
- **URL:** [https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
- **What it supports:** Standard 50,000 training and 10,000 testing images of $32\times 32$ pixels across 10 classes.
