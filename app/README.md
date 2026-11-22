---
title: SimCLR CIFAR-10 Visual Representation Classifier
emoji: 🖼️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
---

# SimCLR (v1) on CIFAR-10: Self-Supervised Learning & Deployment

This Space hosts the interactive inference demonstration for the reproduction of **SimCLR (v1)** (*A Simple Framework for Contrastive Learning of Visual Representations*, Chen et al., ICML 2020) trained on CIFAR-10 from scratch in PyTorch.

### Key Milestones:
- **Base Encoder:** Modified ResNet-18 with CIFAR stem (no max-pool, $h \in \mathbb{R}^{512}$).
- **Pretraining NT-Xent Loss:** Contrastive loss on $2N$ augmented views.
- **Linear Evaluation Test Top-1 Accuracy:** **75.23%** (Top-5: **98.71%**).
- **Project Authors:** Saalik Hussain Zaidi (36206), Mubashir Abid (36224), Muhammad Zohair Khan (33939) — *Institute of Business Administration (IBA), Karachi*.
