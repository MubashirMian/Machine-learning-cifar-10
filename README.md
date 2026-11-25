#  A Simple Framework for Contrastive Learning of Visual Representations

A clean, modular PyTorch reproduction and data-efficiency study of **SimCLR** (*A Simple Framework for Contrastive Learning of Visual Representations*, Chen et al., ICML 2020) on the CIFAR-10 dataset, featuring an interactive inference web app deployed via Streamlit.

---

## 1. Project Overview
- **Paper:** Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations*, ICML 2020. [arXiv:2002.05709](https://arxiv.org/abs/2002.05709).
- **Core Goal:** Re-implement SimCLR from scratch in PyTorch, reproduce representation learning on CIFAR-10, measure linear evaluation top-1 accuracy on the frozen representations, evaluate the impact of unlabeled pretraining dataset size (data scaling), and deploy a real-time classifier.
- **Key Modifications:**
  - Base Encoder: ResNet-18 with modified CIFAR stem (removes max pooling, $3\times 3$ stride 1 conv) instead of ResNet-50.
  - Optimization: Standard SGD with momentum or Adam for batch size 256 (instead of LARS designed for massive batch sizes).
  - Data Augmentation: Random crop & resize ($32\times 32$), random horizontal flip, random color jitter ($s=0.5$), random grayscale; no Gaussian blur (following paper Appendix B.9 for CIFAR-10).

---

## 2. Repository Layout

```
├── AGENTS.md                  # Project constitution & agent operational rules
├── PROJECT_PLAN.md            # Comprehensive roadmap and milestone tracker
├── SOURCES.md                 # Primary and secondary citations & sources of truth
├── DECISIONS.md               # Architecture and experiment decisions log
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── configs/                   # Experiment configuration files (YAML)
├── src/
│   └── simclr/
│       ├── data.py            # Datasets, stratified splits, two-view data loader
│       ├── augment.py         # SimCLR augmentation pipeline
│       ├── model.py           # ResNet-18 CIFAR encoder & projection head
│       ├── loss.py            # NT-Xent contrastive loss function
│       ├── train_pretrain.py  # Self-supervised contrastive pretraining loop
│       ├── train_linear.py    # Downstream linear evaluation training loop
│       ├── utils/             # Seeding, checkpointing, logging utilities
│       └── observability/     # W&B experiment tracking & Phoenix OpenTelemetry
├── app/
│   └── app.py                 # Streamlit web application
├── tests/                     # Unit tests (pytest suite)
├── pipeline_original/         # Untouched archive of the team's initial pipeline
├── checkpoints/               # Progress logs per stage (INDEX.md & S0-S10)
├── runs/                      # Local experiment metrics, configs, and models
└── data/                      # CIFAR-10 data directory
```

---

## 3. Environment & Setup

### Prerequisites
- Python 3.9+
- Recommended: PyTorch with CUDA or Apple Silicon MPS support

### Installation
```bash
# Clone the repository
git clone <repo_url>
cd SimCLR-Research

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Tests
```bash
pytest tests/ -v
```

---

## 4. Usage

### Self-Supervised Pretraining
```bash
python -m src.simclr.train_pretrain --config configs/pretrain_cifar10.yaml
```

### Downstream Linear Evaluation
```bash
python -m src.simclr.train_linear --config configs/linear_eval.yaml --encoder-checkpoint runs/<run_id>/best.pt
```

### Running the Streamlit App Locally
```bash
streamlit run app/app.py
```

---

## 5. Attribution & Academic Integrity
- **Authors:** Saalik Hussain Zaidi, Mubashir Abid, Muhammad Zohair Khan (IBA)
- **Reference Code:** [google-research/simclr](https://github.com/google-research/simclr) (TensorFlow reference by Chen et al.)
- All four SimCLR modules, training pipelines, linear evaluation protocols, experiments, and the application are written by the project team in PyTorch. Any external references, mathematical formulations, and adaptations are thoroughly documented in `SOURCES.md` and `DECISIONS.md`.
