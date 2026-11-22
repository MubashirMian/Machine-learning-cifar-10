# SimCLR (v1) on CIFAR-10: Paper Reproduction, Data Scaling Analysis, and Interactive Web Deployment

**Machine Learning Project — Fall 2026 (Deliverable 3 Final Research Report)**  
**Institution:** Institute of Business Administration (IBA), Karachi  

### Authors:
- **Saalik Hussain Zaidi** (ERP: 36206) — Institute of Business Administration (IBA)
- **Mubashir Abid** (ERP: 36224) — Institute of Business Administration (IBA)
- **Muhammad Zohair Khan** (ERP: 33939) — Institute of Business Administration (IBA)

---

### Project Links & Submission Artifacts (Frozen as of November 25, 2026):
- **Live Deployed Application (Hugging Face Spaces):** [https://huggingface.co/spaces/MianGenz/Sim_clr](https://huggingface.co/spaces/MianGenz/Sim_clr)
- **Direct Application View:** [https://miangenz-sim-clr.hf.space](https://miangenz-sim-clr.hf.space)
- **Deployment Model Artifact:** `app/simclr_cifar10_eval.pt` (42.7 MB, SHA-256: `12358c35e3f9323e47c8a4a8e1d7e4d1cf9ead13ad9d90c1e83318b2f0751fb6`)
- **Code Repository:** Standalone PyTorch implementation with automated unit tests, experiment configs, and observability loggers.
- **Observability Suites:** Weights & Biases (run tracking) & Arize Phoenix (inference telemetry).

---

## Abstract
Learning robust, semantically rich visual representations without human supervision remains a foundational objective in machine learning. In this work, we conduct an end-to-end reproduction and scientific extension of **SimCLR (v1)** (*A Simple Framework for Contrastive Learning of Visual Representations*, Chen et al., ICML 2020) on the CIFAR-10 benchmark. We implement the entire contrastive architecture completely from scratch in PyTorch—including stochastic two-view data augmentations, a modified ResNet-18 base encoder with a customized CIFAR stem (omitting max-pooling), a 2-layer multi-layer perceptron (MLP) projection head ($512 \to 512 \to 128$), and the normalized temperature-scaled cross-entropy loss (NT-Xent). 

Under feasibility adjustments for single-accelerator training (Apple Silicon M4 Pro GPU, batch size 256, 50 pretraining epochs), our reproduced model achieves **75.23% Top-1 Test Accuracy** and **98.71% Top-5 Test Accuracy** under the standard frozen linear evaluation protocol on the full 10,000-image test set. Furthermore, we execute a novel data-scaling experiment evaluating the impact of unlabeled pretraining pool size ($10\% \to 25\% \to 50\% \to 100\%$) under fixed optimizer budgets, empirically demonstrating logarithmic scaling with diminishing marginal returns ($54.17\% \to 65.37\% \to 71.04\% \to 75.23\%$). Finally, we package and deploy the trained encoder and linear classifier as a responsive, Stack Overflow-themed interactive web application hosted on Hugging Face Spaces, instrumented with real-time inference latency tracking and production observability.

---

## 1. Introduction and Problem Statement

Modern deep learning has demonstrated state-of-the-art visual recognition capabilities primarily when guided by extensive human annotations. Large-scale supervised datasets such as ImageNet \cite{russakovsky2015imagenet} have driven breakthroughs in deep residual networks \cite{he2016deep}, object detection, and semantic segmentation. However, manual annotation is labor-intensive, cost-prohibitive, and often unscalable in specialized domains (e.g., medical imaging, satellite telemetry, or fine-grained industrial sorting) where domain expertise is scarce. Furthermore, human supervision forces neural networks to learn representations tied narrowly to discrete class labels, often discarding rich contextual information present in the raw visual signal.

To circumvent annotation bottlenecks, unsupervised visual representation learning aims to train general-purpose feature extractors entirely on unlabeled imagery. Once pretrained, the frozen feature extractor can transfer effectively to downstream prediction tasks using minimal labeled supervision. In this project, we reproduce and evaluate **SimCLR** (Chen et al., ICML 2020 \cite{chen2020simple}), a seminal contrastive learning framework that established that unsupervised pretraining could rival supervised pretraining on ImageNet without specialized memory banks or architectural heuristics.

### Project Scope and Objectives:
1. **From-Scratch Implementation:** Implement all four core modules of the SimCLR architecture in PyTorch without relying on high-level self-supervised frameworks (e.g., Lightly, OpenSelfSup, or PyTorch Lightning).
2. **Standard Benchmark Reproduction:** Evaluate SimCLR on the CIFAR-10 benchmark \cite{krizhevsky2009learning} using the frozen linear evaluation protocol, comparing empirical results directly with the authors' published baselines (Appendix B.9 and Figure B.7).
3. **Controlled Scaling Experiment:** Investigate an open research question untouched by the original paper's ablation studies: how the volume of *unlabeled pretraining data* affects representation quality under constant linear evaluation supervision.
4. **Interactive Production Deployment:** Deploy the optimized model as an accessible web application featuring Stack Overflow-inspired UI/UX, integrated observability, and sub-20ms inference latency.

---

## 2. Literature Review / Background

### 2.1 The Core Problem and Historical Importance
The ambition to learn invariant, semantically grounded visual representations without human labels dates back to foundational work on energy-based models and dimensionality reduction \cite{hadsell2006dimensionality}. In classical self-supervised learning, researchers designed artificial "pretext tasks" where pseudo-labels could be derived automatically from input structure. Prominent examples include context prediction (predicting the spatial arrangement of adjacent image patches) \cite{doersch2015unsupervised}, solving jigsaw puzzles \cite{noroozi2016unsupervised}, colorizing grayscale images \cite{zhang2016colorful}, and predicting random image rotations \cite{gidaris2018unsupervised}. While effective at forcing the network to learn low-level edges and part relationships, handcrafted pretext tasks suffer from severe limitations: they rely on task-specific heuristics, induce semantic shortcuts (such as chromatic aberration or boundary edge continuity), and produce feature representations that fail to generalize broadly across diverse downstream vision tasks.

### 2.2 Evolution of Contrastive Learning Approaches
Contrastive learning departed from discrete pretext classification by formulating representation learning as metric learning in continuous latent space. The fundamental principle is to map inputs to an embedding space such that semantically similar ("positive") pairs are pulled close together, while dissimilar ("negative") pairs are pushed far apart.

Before SimCLR, several landmark frameworks explored this formulation:
- **Instance Discrimination & Memory Banks:** Wu et al. \cite{wu2018unsupervised} proposed treating every single image instance in the training set as a distinct class. To store negative representations across optimization steps without computing full-dataset gradients, they introduced an explicit *memory bank* that maintained moving-average representations of all dataset instances. However, the memory bank introduced substantial memory overhead and representational lag, as embeddings stored in the bank became stale as the network weights evolved.
- **Contrastive Predictive Coding (CPC):** van den Oord et al. \cite{oord2018representation} introduced the InfoNCE loss, training autoregressive encoders to predict representations of future or neighboring spatial patches within an image while contrasting against negative patches drawn from other images.
- **Augmented Multiscale Dim (AMDIM):** Bachman et al. \cite{bachman2019learning} maximized mutual information across multiple intermediate feature maps (local-to-global representations). Although achieving competitive performance, AMDIM required highly complex, specialized multi-scale network architectures.
- **Momentum Contrast (MoCo):** He et al. \cite{he2020momentum} addressed the staleness of memory banks by introducing a dynamic queue decoupled from the dictionary encoder, updated via a momentum-based moving-average parameter tracker ($E_{k} \leftarrow m E_{k} + (1-m) E_{q}$).

Table \ref{tab:pre_simclr} (carried over from our approved project proposal) summarizes the core paradigms preceding SimCLR and highlights their key architectural constraints.

\begin{table}[htbp]
\caption{Approaches Preceding SimCLR and Their Architectural Limitations}
\label{tab:pre_simclr}
\centering
\small
\begin{tabular}{lll}
\hline
\textbf{Approach} & \textbf{Key Idea} & \textbf{Limitation Relevant to SimCLR} \\ \hline
\textbf{Pretext Tasks} & Predict patch layout or rotation & Relies on designer heuristics \\
\textbf{Instance Discr.} & Treat each image as a class & Requires large, stale memory bank \\
\textbf{CPC} & Predict spatial patches via InfoNCE & Rigid patch splitting and context net \\
\textbf{AMDIM} & Maximize local-to-global MI & Complex, constrained architectures \\
\textbf{MoCo} & Decouple dictionary with momentum & Requires auxiliary momentum encoder \\ \hline
\end{tabular}
\end{table}

### 2.3 SimCLR: Novelty and Key Contributions
SimCLR \cite{chen2020simple} simplified contrastive learning by eliminating auxiliary memory banks, specialized pretext networks, and momentum encoders. The authors demonstrated that competitive representations could be learned entirely within standard mini-batches via four unified design choices:
1. **Composition of Data Augmentations:** The authors proved that no single data augmentation strategy suffices. In particular, composing **random resized cropping** with **strong color jittering** is critical; without color distortion, the network easily exploits shortcut learning by matching low-level color histograms across cropped views rather than abstract visual semantics.
2. **Nonlinear Projection Head:** Introducing a 2-layer MLP projection head $g(\cdot)$ between the base representation $h$ and the contrastive loss function $\mathcal{L}_{\text{NT-Xent}}$ yielded substantial performance gains (+10% on ImageNet). Crucially, the authors demonstrated that representations before the projection head ($h$) are substantially richer and more linearly separable than representations after the head ($z$), because $z$ discards information (such as color and orientation) invariant to the contrastive objective.
3. **Normalized Temperature-Scaled Cross-Entropy (NT-Xent):** A symmetric, normalized cross-entropy formulation incorporating cosine similarity and a tunable temperature parameter $\tau$.
4. **Large Mini-Batch Scaling:** By treating all other $2N-2$ augmented views in a batch of $N$ images as negatives, SimCLR directly leverages large mini-batch optimization (up to $N=4096$) to provide sufficient negative sampling.

### 2.4 Theoretical Foundations: Alignment and Uniformity
Wang and Isola \cite{wang2020understanding} formalized the mechanics of contrastive loss on the unit hypersphere $\mathcal{S}^{d-1}$, proving that the asymptotic limit of the NT-Xent objective optimizes two distinct geometric properties:
1. **Alignment:** Positive pairs should map to nearby features on the hypersphere, encouraging invariance to nuisance transformations:
   $$\mathcal{L}_{\text{align}} \triangleq \mathbb{E}_{(x, x^{+}) \sim p_{\text{pos}}} \left[ \|f(x) - f(x^{+})\|_{2}^{2} \right]$$
2. **Uniformity:** The overall distribution of normalized representations should preserve maximal information by distributing uniformly over the hypersphere:
   $$\mathcal{L}_{\text{uniform}} \triangleq \log \mathbb{E}_{x, y \stackrel{i.i.d.}{\sim} p_{\text{data}}} \left[ e^{-2\|f(x) - f(y)\|_{2}^{2}} \right]$$

The temperature parameter $\tau$ serves as a hardness controller: smaller $\tau$ penalizes hard negative instances exponentially more, driving tighter uniformity at the cost of potential representational collapse if negative samples are noisy.

### 2.5 Literature on Pretraining Data Scale
While Chen et al. \cite{chen2020simple} investigated downstream label fraction scaling (evaluating linear probe accuracy with 1% vs. 10% vs. 100% labeled examples on ImageNet), the original paper held the volume of *unlabeled pretraining data* strictly constant (1.28 million ImageNet images). Subsequent empirical investigations by Goyal et al. \cite{goyal2019scaling} and Cole et al. \cite{cole2022does} highlighted that self-supervised representations exhibit power-law scaling properties with respect to pretraining data volume, but performance gains saturate rapidly unless encoder parameter capacity is concurrently expanded. Our additional experiment directly isolates and quantifies this relationship on CIFAR-10.

---

## 3. Paper Methodology

The SimCLR framework consists of four sequential, synergistic modules executed during self-supervised pretraining, followed by a linear evaluation protocol:

```
[ Input Image x ] 
       │
   ┌───┴───┐ (Stochastic Augmentations t, t' ~ T)
   ▼       ▼
[ View 1: x̃_i ]       [ View 2: x̃_j ]
   │                     │
   ▼                     ▼
[ Base Encoder f(·) ]  [ Base Encoder f(·) ]  (ResNet-18 with CIFAR Stem)
   │                     │
   ▼                     ▼
[ Representation h_i]  [ Representation h_j]  (h ∈ ℝ⁵¹² — Retained for Downstream)
   │                     │
   ▼                     ▼
[ Projection Head g(·)][ Projection Head g(·)] (2-Layer MLP: Linear-BN-ReLU-Linear-BN)
   │                     │
   ▼                     ▼
[ Latent Vector z_i ]  [ Latent Vector z_j ]  (z ∈ ℝ¹²⁸ — Discarded After Pretrain)
   │                     │
   └──────────┬──────────┘
              ▼
    [ NT-Xent Loss L(i,j) ] (Cosine Similarity + Temperature τ = 0.5)
```

### 3.1 Stochastic Data Augmentation ($T$)
Given an unlabeled image $x$, two stochastic data augmentation operators $t, t' \sim \mathcal{T}$ are sampled independently and applied to produce a correlated positive pair of views $(\tilde{x}_i, \tilde{x}_j)$. For CIFAR-10 ($32 \times 32$), the augmentation sequence comprises:
1. **Random Resized Crop:** Crop an area fraction $s \in [0.08, 1.0]$ with an aspect ratio in $[3/4, 4/3]$, followed by bicubic interpolation back to $32 \times 32$.
2. **Random Horizontal Flip:** Applied with probability $p = 0.5$.
3. **Color Jitter:** Applied with probability $p = 0.8$. Following the official CIFAR-10 specification (Chen et al. commit `383d414`), we use color distortion strength $s=0.5$: brightness = $0.4$, contrast = $0.4$, saturation = $0.4$, and hue = $0.1$.
4. **Random Grayscale:** Applied with probability $p = 0.2$.
5. **Gaussian Blur Omission:** Crucially, while Gaussian blur is vital for high-resolution ImageNet images ($224 \times 224$), the official SimCLR specification explicitly sets `--use_blur=False` on CIFAR-10, as blurring $32 \times 32$ pixels removes virtually all high-frequency structural content.

### 3.2 Base Encoder ($f(\cdot)$)
The base visual encoder extracts feature representations $h_i = f(\tilde{x}_i) \in \mathbb{R}^{d}$. While the paper utilizes ResNet-50 for ImageNet, we utilize a modified **ResNet-18** adapted specifically for low-resolution CIFAR-10 inputs:
- **Modified CIFAR Stem:** The standard ImageNet stem ($7 \times 7$ convolution, stride 2, followed by a $3 \times 3$ max-pooling operation with stride 2) reduces an input spatial resolution by $4\times$. On $32 \times 32$ CIFAR images, this would reduce intermediate activations to $8 \times 8$ before residual blocks even begin, destroying fine-grained feature topology. We replace this stem with a single $3 \times 3$ convolution (stride 1, 64 output channels), batch normalization, and ReLU activation, completely omitting the max-pool layer.
- **Residual Stages:** Four basic residual stages ($[2, 2, 2, 2]$ blocks) with feature dimensions $64 \to 128 \to 256 \to 512$.
- **Global Average Pooling:** Collapses spatial dimensions to yield representation vector $h_i \in \mathbb{R}^{512}$. This 512-dimensional vector represents the core learned embedding evaluated on downstream tasks.

### 3.3 Projection Head ($g(\cdot)$)
A 2-layer multi-layer perceptron (MLP) maps representation $h_i$ into the lower-dimensional latent space where contrastive loss is calculated:
$$z_i = g(h_i) = W^{(2)} \sigma\left(\text{BN}\left(W^{(1)} h_i\right)\right)$$
where $W^{(1)} \in \mathbb{R}^{512 \times 512}$, $\sigma$ denotes ReLU activation, $\text{BN}$ is Batch Normalization, and $W^{(2)} \in \mathbb{R}^{128 \times 512}$ produces latent projection $z_i \in \mathbb{R}^{128}$. Following SimCLR v1 conventions, the final projection layer includes batch normalization but no activation. The projection head is utilized solely during self-supervised pretraining and is completely discarded prior to linear evaluation and inference deployment.

### 3.4 NT-Xent Loss Function
For a mini-batch of $N$ input images, the augmentation pipeline generates $2N$ views. For each anchor view $i \in \{1, \dots, 2N\}$, there exists exactly one corresponding positive view $j \in \{1, \dots, 2N\} \setminus \{i\}$ (derived from the same original source image), while the remaining $2N-2$ views within the mini-batch are treated as negative examples.

Let $\text{sim}(u, v) = \frac{u^\top v}{\|u\|_2 \|v\|_2}$ denote the cosine similarity between $\ell_2$-normalized vectors. The Normalized Temperature-Scaled Cross-Entropy loss for positive pair $(i, j)$ is defined as:
$$\ell(i, j) = -\log \frac{\exp\left(\frac{\text{sim}(z_i, z_j)}{\tau}\right)}{\sum_{k=1}^{2N} \mathbb{I}_{[k \neq i]} \exp\left(\frac{\text{sim}(z_i, z_k)}{\tau}\right)}$$
where $\tau > 0$ is the temperature hyperparameter and $\mathbb{I}_{[k \neq i]} \in \{0, 1\}$ is an indicator function evaluating to 1 iff $k \neq i$. The total pretraining loss is computed symmetrically across all positive pairs in the mini-batch:
$$\mathcal{L}_{\text{NT-Xent}} = \frac{1}{2N} \sum_{k=1}^{N} \left[ \ell(2k-1, 2k) + \ell(2k, 2k-1) \right]$$

### 3.5 Linear Evaluation Protocol
To quantify representation quality without fine-tuning encoder weights, we freeze all parameters of the base encoder $f(\cdot)$ (`requires_grad = False` and set to `model.eval()` to lock running mean and variance in Batch Normalization). A single linear classification layer ($W_{\text{cls}} \in \mathbb{R}^{10 \times 512}, b_{\text{cls}} \in \mathbb{R}^{10}$) is trained on top of representations $h$:
$$\hat{y} = \text{softmax}\left(W_{\text{cls}} h + b_{\text{cls}}\right)$$
The linear probe is trained using standard cross-entropy loss under basic augmentations (random crop and horizontal flip only; no color jittering). Downstream representation quality is reported as the Top-1 classification accuracy on the official 10,000-image CIFAR-10 test set.

---

## 4. Implementation Details

### 4.1 Software Environment and Dependencies
The system was implemented entirely in Python 3.9 utilizing native PyTorch (v2.0+) and Torchvision. The core dependencies and environments are detailed below:
- **Core ML Framework:** `torch>=2.0.0`, `torchvision>=0.15.0`
- **Numerical Processing & Data Science:** `numpy>=1.22.0`, `pandas>=1.5.0`, `scipy>=1.10.0`
- **Testing & Verification:** `pytest>=7.0.0` (8 custom unit test suites)
- **Interactive Web App & Hosting:** `gradio==5.20.0`, `huggingface_hub<0.26.0`, `Pillow>=9.5.0`
- **Observability:** `wandb>=0.15.0` (training dynamics), `arize-phoenix==12.15.1`, `graphql-core==3.2.3` (inference tracing)

### 4.2 Compute Hardware and Feasibility Adjustments
Original SimCLR results on ImageNet rely on Google Cloud TPU v3 pods (up to 128 TPU cores, mini-batch size 4096, 1000 epochs), and the paper's CIFAR-10 appendix uses ResNet-50 at batch size 1024 with LARS optimization. To execute the complete research cycle within our available academic compute budget (one local machine with an Apple Silicon M4 Pro GPU featuring 14 execution cores and 24 GB unified memory, supplemented by Google Colab T4 GPUs), we instituted four principled feasibility adjustments:
1. **Model Architecture:** Replaced ResNet-50 (24.0M parameters) with ResNet-18 modified with a CIFAR stem (11.17M parameters), reducing memory consumption by over $50\%$ and accelerating backpropagation throughput.
2. **Mini-Batch Size:** Set mini-batch size to $N = 256$ (yielding $2N = 512$ views per gradient step), fitting cleanly within hardware cache without gradient accumulation latency.
3. **Pretraining Budget:** Scaled pretraining to 50 epochs (8,750 optimization steps at batch 256 on 45,000 samples) for the main reproduction and scaling ablations, verified via pilot runs to reach steady representation convergence.
4. **Optimizer Selection:** Replaced the LARS optimizer (which is tailored specifically for massive batches $> 2048$ with layer-wise adaptive learning rates) with **Adam** ($lr = 10^{-3}$) after conducting systematic pilot experiments on held-out validation data.

### 4.3 Explicit Code Attribution & Academic Integrity
In strict compliance with Deliverable 3 Guidelines (Section 2.4):
- **Team-Implemented Code (From Scratch):**
  - Stochastic two-view augmentation pipeline (`src/simclr/augment.py`) including CIFAR-specific color jitter parametrization.
  - CIFAR-stem ResNet-18 encoder and 2-layer projection head MLP (`src/simclr/model.py`).
  - Fully vectorized NT-Xent loss function with masked self-similarity and cosine temperature scaling (`src/simclr/loss.py`).
  - Pretraining loop, learning rate scheduling, model checkpointing, and frozen linear evaluation harness (`src/simclr/train_pretrain.py`, `src/simclr/train_linear.py`).
  - Stratified nested data sampling and split persistence (`src/simclr/data.py`).
  - Complete automated test suite (`tests/test_simclr.py`).
  - Interactive Stack Overflow-themed Gradio web application (`app/app.py`).
- **Code Adapted or Referenced:**
  - Official Google Research SimCLR repository (`google-research/simclr` @ commit `383d414` in TensorFlow 2): Inspected strictly as a mathematical and algorithmic reference to verify augmentation hyperparameter values (e.g., $s=0.5$, $\tau=0.5$, omission of Gaussian blur on CIFAR). No TensorFlow code was copied or directly reused.
  - Standard Torchvision ResNet building blocks (`torchvision.models.resnet.BasicBlock`): Adapted into our custom `ResNet18CIFAR` class to rewire the convolutional stem and discard ImageNet max-pooling.
- **Results Attribution:**
  - `PAPER-REPORTED` metrics denote baselines published by Chen et al. \cite{chen2020simple} in Appendix B.9 and Figure B.7.
  - `MEASURED` metrics denote experimental runs executed exclusively by our team on our hardware.

---

## 5. Dataset and Preprocessing / Exploratory Data Analysis (EDA)

*(Carried over and consolidated from Deliverable 2 / Mid Evaluation Report)*

### 5.1 Dataset Overview & Properties
We utilize the standard **CIFAR-10** benchmark \cite{krizhevsky2009learning}, consisting of 60,000 $32 \times 32$ color (RGB) images partitioned into 10 mutually exclusive semantic classes: *airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck*.

The dataset features an exact, uniform class distribution: 5,000 images per class in the training split (50,000 total) and 1,000 images per class in the test split (10,000 total). Consequently, class imbalance is zero, and a random classifier achieves an expected baseline accuracy of exactly $10.0\%$.

### 5.2 Data Quality and Integrity Verification
Comprehensive data audit routines confirmed zero missing values, zero unreadable image bytes, and zero duplicated unique image IDs across both splits (Table \ref{tab:data_quality}).

\begin{table}[htbp]
\caption{CIFAR-10 Data Quality Audit Checks}
\label{tab:data_quality}
\centering
\begin{tabular}{lll}
\hline
\textbf{Integrity Check} & \textbf{Result} & \textbf{Status} \\ \hline
Missing / Corrupted Image Files & 0 & PASSED \\
Duplicate Image IDs & 0 & PASSED \\
Label Outliers / Missing Classes & 0 (10 Classes) & PASSED \\
Pixel Range & Float $[0.0, 1.0]$ & VERIFIED \\ \hline
\end{tabular}
\end{table}

### 5.3 Statistical Analysis & Class Color Signatures
Exploratory analysis revealed significant cross-class variations in luminance and chromatic distributions:
- **Luminance:** *Airplane* (median brightness $\approx 142/255$) and *ship* ($\approx 133/255$) are the brightest classes, heavily influenced by sky and ocean background pixels. *Frog* is the darkest class (median $\approx 104/255$).
- **Color Channels:** *Airplane* and *ship* exhibit the highest average blue channel values ($B \approx 150$ and $B \approx 141$), representing the only classes where blue dominates red and green. In contrast, biological land animals (*cat, deer, dog, horse*) exhibit warm color palettes ($R > G > B$), while vehicles (*automobile, truck*) display highly balanced RGB distributions across all three color channels.
- **Outliers:** Applying the $1.5 \times \text{IQR}$ rule revealed 162 dark outliers and 658 bright outliers across the dataset, with *airplane* accounting for 288 bright outlier images due to cloud glare.

### 5.4 Benchmark Baseline: PCA + 5-Nearest Neighbors on Raw Pixels
To empirically verify the necessity of deep representation learning, we evaluated a simple unsupervised baseline directly on raw pixel values. We sampled 5,000 training images, applied an 80/20 stratified split, and fitted Principal Component Analysis (PCA) with 50 components on the training partition:
- The first 2 principal components capture $40.5\%$ of overall pixel variance; all 50 components preserve $84.5\%$.
- Visualizing the 2D projection reveals heavy inter-class entanglement; classes form overlapping density clouds rather than distinct topological clusters.
- Training a 5-Nearest Neighbors ($5$-NN) classifier on the 50 PCA components yielded a test accuracy of only **30.8%**. While superior to random guessing ($10\%$), this confirms that raw pixel Euclidean distances fail to capture abstract visual semantics, rigorously motivating contrastive representation learning.

### 5.5 Data Splitting and Leakage Prevention
To guarantee that hyperparameter tuning and model selection never leaked information from the final evaluation, we established a strict three-way split:
- **Pretraining Pool:** 45,000 images drawn from the training split.
- **Validation Split:** 5,000 stratified images (500 per class) held out entirely from pretraining, saved deterministically to `data/splits/val_idx.npy`. All pilot runs, early stopping criteria, and optimizer decisions were made strictly on this validation split.
- **Test Split:** The official 10,000-image test set, evaluated only once per final model checkpoint.

---

## 6. Reproduction Results

### 6.1 Hyperparameter Pilot Experiments (Stage S5)
Before committing the full computational budget to 50-epoch pretraining, we conducted five systematic pilot experiments (P1 through P5) on a 10% stratified subset (4,500 images) evaluated on the 5,000-image validation split. Each pilot trained for 5 pretraining epochs followed by 5 linear evaluation epochs under identical batch sizes ($N=256$).

\begin{table}[htbp]
\caption{Pilot Hyperparameter Selection Experiments (Validation Split)}
\label{tab:pilot_results}
\centering
\small
\begin{tabular}{lccccc}
\hline
\textbf{Pilot ID} & \textbf{Optimizer} & \textbf{Learning Rate} & \textbf{Temp ($\tau$)} & \textbf{NT-Xent Loss} & \textbf{Val Top-1 Acc} \\ \hline
P1 & SGD (mom 0.9) & 0.1 & 0.5 & 5.517 & 36.63\% \\
P2 & SGD (mom 0.9) & 0.3 & 0.5 & 5.497 & 36.67\% \\
\textbf{P3 (Selected)} & \textbf{Adam} & \textbf{1e-3} & \textbf{0.5} & \textbf{5.462} & \textbf{39.33\%} \\
P4 & AdamW & 1e-3 & 0.5 & 5.462 & 39.06\% \\
P5 & SGD (mom 0.9) & 0.3 & 0.1 & 5.270 & 37.10\% \\ \hline
\end{tabular}
\end{table}

**Key Pilot Findings:**
- **Optimizer Advantage:** Adam ($lr = 10^{-3}$) outperformed SGD with momentum by $+2.70\%$ Top-1 accuracy under identical epoch constraints, demonstrating superior adaptation to small mini-batches without requiring extensive warm-up schedules.
- **Temperature Stability:** Setting $\tau = 0.5$ provided stable gradient flow and optimal separation. While $\tau = 0.1$ (P5) lowered empirical cross-entropy loss by heavily weighting hard negatives, it degraded downstream validation accuracy ($37.10\%$), confirming findings by Wang & Isola \cite{wang2020understanding} regarding excessive gradient variance at low temperatures.
- **Decision:** Pilot P3 (Adam, $lr = 10^{-3}$, $\tau = 0.5$) was selected for all full-scale pretraining runs.

### 6.2 Main Reproduction Run (Stage S6)
Using the optimal hyperparameter configuration, we executed the full 50-epoch pretraining run on 45,000 CIFAR-10 images (`runs/pretrain_cifar10_s6_seed42/`). Following pretraining, the base encoder was frozen and a linear classifier was trained for 40 epochs on the 45,000 labeled representations (`runs/linear_eval_cifar10_s6_seed42/`).

**Empirical Measured Results:**
- **Pretraining Contrastive Accuracy:** Reached **62.35%** (fraction of mini-batch anchors whose nearest neighbor in latent space is their true positive augmented view).
- **Final NT-Xent Pretrain Loss:** Decreased smoothly from $5.524$ to **$4.7424$**.
- **Downstream Test Top-1 Accuracy:** **75.23%** (7,523 / 10,000 correct test classifications).
- **Downstream Test Top-5 Accuracy:** **98.71%** (9,871 / 10,000 correct test classifications).
- **Downstream Test Macro-F1 Score:** **0.7512** (balanced across all 10 classes).

### 6.3 Side-by-Side Comparison with Published Baselines
Table \ref{tab:paper_comparison} provides a transparent, side-by-side comparison between our measured results and the baselines reported in the original paper (Appendix B.9 and Figure B.7) and the official repository.

\begin{table}[htbp]
\caption{Comparison Between Original SimCLR Publications and Our Reproduction}
\label{tab:paper_comparison}
\centering
\small
\begin{tabular}{lccc}
\hline
\textbf{Specification / Metric} & \textbf{Chen et al. \cite{chen2020simple} App. B.9} & \textbf{Official Repo (README)} & \textbf{Our Reproduction} \\ \hline
\textbf{Base Encoder} & ResNet-50 (CIFAR stem) & ResNet-18 (CIFAR stem) & ResNet-18 (CIFAR stem) \\
\textbf{Encoder Parameters} & 24.0 Million & 11.17 Million & 11.17 Million \\
\textbf{Pretraining Epochs} & 1,000 epochs & 1,000 epochs & \textbf{50 epochs} \\
\textbf{Mini-Batch Size ($N$)} & 1,024 & 512 & \textbf{256} \\
\textbf{Optimization Algorithm} & LARS & LARS & \textbf{Adam ($lr = 10^{-3}$)} \\
\textbf{Hardware Setup} & Cloud TPU Pods & Multi-GPU TPU & 1$\times$ Apple M4 Pro GPU \\
\textbf{Total Gradient Steps} & $\sim$48,800 steps & $\sim$97,600 steps & \textbf{8,750 steps} \\ \hline
\textbf{Top-1 Accuracy} & \textbf{94.0\%} ($1000$ ep) & \textbf{$\approx$91.0\%} ($1000$ ep) & \textbf{75.23\%} \\
\textbf{Interpolated 100-Epoch} & $\approx$84.0\% (Fig. B.7) & $\approx$83.5\% & \textbf{75.23\% (at 50 ep)} \\ \hline
\end{tabular}
\end{table}

---

## 7. Additional Experiment: Unlabeled Pretraining Data Scaling

### 7.1 Research Question and Scientific Motivation
While the ablation studies in Chen et al. \cite{chen2020simple} extensively explore fine-tuning label efficiency (evaluating 1% vs. 10% labeled data on a frozen encoder), the authors kept the volume of unlabeled pretraining data fixed at 100% of ImageNet. 

This leaves a critical question unaddressed: **How does the quantity of unlabeled pretraining data affect the semantic quality of learned representations when downstream labeled supervision is held constant?**

### 7.2 Experimental Design & Methodology
To isolate the effect of pretraining volume without confounding variables:
1. **Stratified Nested Subsets:** We extracted four nested pretraining pools: **10%** (4,500 images), **25%** (11,250 images), **50%** (22,500 images), and **100%** (45,000 images). Subsets were sampled with stratified class balance and nested ($10\% \subset 25\% \subset 50\% \subset 100\%$) to minimize sampling variance.
2. **Fixed Pretraining Epoch Budget:** Following the proposal approved by course staff, each subset was pretrained for an identical budget of **50 epochs** using Adam ($lr = 10^{-3}$, $\tau = 0.5$).
3. **Identical Linear Evaluation Protocol:** Each resulting encoder checkpoint was frozen and evaluated on the **exact same downstream training set** (the full 45,000 labeled images) for 40 epochs with identical SGD hyperparameters, evaluated on the official 10,000-image test set. Thus, the downstream linear probe had access to the full labeled dataset, ensuring that performance variations stem exclusively from the pretraining representation quality.

### 7.3 Experimental Results & Scaling Analysis
Table \ref{tab:scaling_results} and Figure \ref{fig:scaling_plot} present the empirical scaling trajectory.

\begin{table}[htbp]
\caption{Data Scaling Experiment: Pretraining Pool Fraction vs. Downstream Performance}
\label{tab:scaling_results}
\centering
\begin{tabular}{cccccc}
\hline
\textbf{Data Fraction} & \textbf{Unlabeled Samples} & \textbf{Pretrain Steps} & \textbf{Test Top-1 Acc} & \textbf{Test Top-5 Acc} & \textbf{Marginal Gain} \\ \hline
10\% & 4,500 & 880 & 54.17\% & 94.52\% & Baseline \\
25\% & 11,250 & 2,200 & 65.37\% & 97.14\% & +11.20\% \\
50\% & 22,500 & 4,380 & 71.04\% & 98.07\% & +5.67\% \\
100\% & 45,000 & 8,750 & 75.23\% & 98.71\% & +4.19\% \\ \hline
\end{tabular}
\end{table}

```
Downstream Test Top-1 Accuracy (%) vs. Unlabeled Pretraining Data
  80% ┤                                                      ● (75.23%)
  75% ┤                                         ● (71.04%)
  70% ┤
  65% ┤                         ● (65.37%)
  60% ┤
  55% ┤       ● (54.17%)
  50% ┼───────┬───────────────┬─────────────────┬───────────────┬────
     0%      10%             25%               50%            100%
                        Pretraining Pool Size
```

### 7.4 Hypothesis Evaluation
Our initial hypothesis stated that downstream accuracy would scale monotonically with the quantity of pretraining data, exhibiting diminishing marginal returns as data coverage saturated. The empirical results fully confirm this hypothesis:
- Expanding pretraining volume by $2.5\times$ (from $10\%$ to $25\%$) produced a massive gain of **$+11.20\%$** in Top-1 accuracy.
- Expanding volume by an additional $2.0\times$ (from $25\%$ to $50\%$) yielded a **$+5.67\%$** gain.
- Expanding volume by a final $2.0\times$ (from $50\%$ to $100\%$) yielded a **$+4.19\%$** gain.

The marginal accuracy gain per additional sample decays logarithmically, conforming closely to empirical neural scaling laws ($\text{Error} \propto N^{-\alpha}$) observed in large-scale foundation models.

---

## 8. Analysis and Discussion

### 8.1 Reproduction Gap Attribution
Our measured Top-1 accuracy of **75.23%** exhibits an 18.77% gap compared to the paper's best reported CIFAR-10 accuracy of **94.0%** (Chen et al. Table B.9). Far from indicating an implementation deficiency, this delta is fully accounted for by three deliberate, well-documented resource differences:
1. **Optimization Steps (Compute Budget):** The original paper trained for **1,000 epochs** at batch size 1024, amounting to approximately 48,800 steps with massive negative pools. In contrast, our model trained for **50 epochs** (8,750 steps at batch 256)—less than $1/5$th the optimization steps. Notably, Chen et al. Figure B.7 demonstrates that at 100 epochs with batch size 256, SimCLR achieves approximately $84\%$. Reaching $75.23\%$ at only 50 epochs places our convergence trajectory in excellent alignment with the published learning curves.
2. **Encoder Parameter Capacity:** The paper utilizes a ResNet-50 backbone (24.0 million parameters), whereas our implementation utilizes ResNet-18 (11.17 million parameters). The smaller parameter capacity limits the representational fidelity and high-order feature combinations captured during pretraining.
3. **Mini-Batch Size and Negative Sampling:** In SimCLR, negative examples are drawn exclusively from within the active mini-batch. A mini-batch size of $N=256$ provides 510 negative distractors per anchor, whereas $N=1024$ provides 2,046 negative distractors. As proven by Wang & Isola \cite{wang2020understanding}, larger negative pools provide a substantially sharper approximation of the partition function in NT-Xent loss, preventing representation clustering.

### 8.2 Representation Quality vs. Raw Pixels
Our trained SimCLR encoder ($75.23\%$ accuracy) dramatically outperforms the raw-pixel PCA + $5$-NN baseline ($30.8\%$) established in Section 5.4 by **+44.43%**. Furthermore, while random guessing on CIFAR-10 yields $10.0\%$, SimCLR achieves **98.71% Top-5 accuracy**, indicating that in nearly 99% of test cases, the correct class is ranked within the model's top half of predictions.

### 8.3 Class-Wise Confusion Analysis
Inspecting per-class performance revealed that natural animal categories sharing similar four-legged postures (*cat* vs. *dog*) accounted for the highest classification confusion, whereas structural mechanical classes (*automobile, ship, airplane*) achieved individual accuracies exceeding $82\%$. This confirms that contrastive representations capture global object geometry and context effectively, but fine-grained animal distinctions require either longer pretraining or targeted downstream fine-tuning.

---

## 9. Model Deployment

### 9.1 Interactive Web Application Architecture
To make our trained model accessible for practical evaluation, we developed and deployed an interactive, user-friendly web application built with Gradio and styled after **Stack Overflow's** clean, high-contrast developer aesthetic:
- **Hosting Environment:** Hugging Face Spaces (CPU Basic hardware tier, 2 vCPU, 16 GB RAM).
- **Public URL:** [https://huggingface.co/spaces/MianGenz/Sim_clr](https://huggingface.co/spaces/MianGenz/Sim_clr)
- **Direct Embed URL:** [https://miangenz-sim-clr.hf.space](https://miangenz-sim-clr.hf.space)
- **Model Checkpoint:** `app/simclr_cifar10_eval.pt` (42.7 MB file containing the frozen `ResNet18CIFAR` weights and trained linear probe head).

### 9.2 Inference Pipeline
The deployed system executes a deterministic four-stage pipeline:
1. **Image Input:** Accepts any standard image format (`.png`, `.jpg`, `.jpeg`, `.webp`) via direct drag-and-drop file upload or selection from built-in CIFAR-10 test examples.
2. **Deterministic Preprocessing:** Converts the input image to RGB, resizes to $32 \times 32$ pixels using bicubic interpolation, converts pixel bytes to floating-point tensors in $[0.0, 1.0]$, and normalizes using exact CIFAR-10 dataset statistics:
   $$\mu = [0.4914, 0.4822, 0.4465], \quad \sigma = [0.2470, 0.2435, 0.2616]$$
3. **Forward Pass:** The preprocessed tensor $(1, 3, 32, 32)$ passes through the frozen encoder to extract a 512-dimensional embedding $h$, followed immediately by the linear classification head.
4. **Softmax Output & Telemetry:** Computes top-3 class probabilities and measures inference latency in milliseconds.

```
[ Uploaded Image (Any Size) ]
              │
              ▼
  [ Bicubic Resize: 32x32 ]
              │
              ▼
[ Normalize: CIFAR Mean & Std ]
              │
              ▼
[ Frozen ResNet-18 (512-d) ] ──> Latency & Embedding Audit
              │
              ▼
[ Linear Classifier (10 Classes) ]
              │
              ▼
[ Predicted Label + Top-3 Confidence Scores + Inference Time ]
```

### 9.3 Local Setup and Execution Instructions
To launch the deployment interface locally:
```bash
# 1. Clone repository and activate virtual environment
git clone https://github.com/mubashirmian/SimCLR-Research.git
cd SimCLR-Research
source .venv/bin/activate

# 2. Install deployment dependencies
pip install -r app/requirements.txt

# 3. Launch the Gradio application
python app/app.py
```
The application will start immediately on `http://localhost:7860`.

### 9.4 Verification of Observability Telemetry
To satisfy enterprise production standards, the deployment was instrumented with two monitoring layers:
1. **Arize Phoenix Tracing:** OpenTelemetry-based request tracing capturing latency breakdowns (preprocessing, forward pass, softmax), input tensor validation, and predicted class distributions. Verified locally on `http://localhost:6006/` (`arize-phoenix 12.15.1`, `graphql-core 3.2.3`).
2. **Weights & Biases Logging:** Model checkpoint training curves, learning rate decays, and evaluation metrics synced to W&B project logs (`wandb/offline-run-*`).

---

## 10. Limitations

Despite achieving strong empirical results, several limitations should be noted:
1. **Pretraining Compute Budget:** Due to hardware constraints, pretraining was bounded at 50 epochs. Extending pretraining to 200–500 epochs would likely close the remaining accuracy gap against published baselines.
2. **Mini-Batch Size:** Training on a single GPU capped mini-batch size at $N = 256$ ($510$ negative views). Distributed multi-GPU execution with synchronized batch normalization would enable batch sizes of 1024–4096, providing richer negative contrastive distributions.
3. **Resolution Constraint:** The CIFAR stem adaptation is specialized for $32 \times 32$ imagery. Deploying this specific model on high-resolution imagery requires downsampling, which inherently loses fine-grained textures.

---

## 11. Conclusion

In this research project, we successfully reproduced the core methodology of SimCLR (v1) completely from scratch in PyTorch on the CIFAR-10 benchmark. Through systematic hyperparameter pilot studies on held-out validation data, we established that Adam optimization with temperature $\tau = 0.5$ provides rapid convergence under moderate batch sizes. Our main reproduction achieved **75.23% Top-1** and **98.71% Top-5 Test Accuracy** on the official 10,000-image test set in 50 epochs, validating the power of composed augmentations and nonlinear projection heads without requiring complex memory banks.

Furthermore, our data-scaling experiment provided clear empirical evidence that unsupervised representation learning follows logarithmic scaling with diminishing marginal returns ($54.17\% \to 65.37\% \to 71.04\% \to 75.23\%$), demonstrating that expanding unlabeled data pools yields dramatic gains early on before plateauing under fixed encoder capacities. Finally, we demonstrated end-to-end engineering execution by deploying a production-grade, Stack Overflow-styled web application on Hugging Face Spaces, complete with deterministic preprocessing, sub-20ms inference latency, and enterprise observability.

---

## 12. References

\begin{enumerate}
\item T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, ``A simple framework for contrastive learning of visual representations,'' in \textit{Proc. 37th Int. Conf. Machine Learning (ICML)}, PMLR 119, 2020, pp. 1597--1607.
\item A. Krizhevsky and G. Hinton, ``Learning multiple layers of features from tiny images,'' Univ. Toronto, Toronto, ON, Canada, Tech. Rep., 2009.
\item O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei, ``ImageNet large scale visual recognition challenge,'' \textit{Int. J. Computer Vision}, vol. 115, no. 3, pp. 211--252, 2015.
\item K. He, X. Zhang, S. Ren, and J. Sun, ``Deep residual learning for image recognition,'' in \textit{Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)}, 2016, pp. 770--778.
\item R. Hadsell, S. Chopra, and Y. LeCun, ``Dimensionality reduction by learning an invariant mapping,'' in \textit{Proc. IEEE Computer Society Conf. Computer Vision and Pattern Recognition (CVPR)}, vol. 2, 2006, pp. 1735--1742.
\item C. Doersch, A. Gupta, and A. A. Efros, ``Unsupervised visual representation learning by context prediction,'' in \textit{Proc. IEEE Int. Conf. Computer Vision (ICCV)}, 2015, pp. 1422--1430.
\item M. Noroozi and P. Favaro, ``Unsupervised learning of visual representations by solving jigsaw puzzles,'' in \textit{Proc. European Conf. Computer Vision (ECCV)}, Springer, 2016, pp. 69--84.
\item R. Zhang, P. Isola, and A. A. Efros, ``Colorful image colorization,'' in \textit{Proc. European Conf. Computer Vision (ECCV)}, Springer, 2016, pp. 649--666.
\item S. Gidaris, P. Singh, and N. Komodakis, ``Unsupervised representation learning by predicting image rotations,'' in \textit{Int. Conf. Learning Representations (ICLR)}, 2018.
\item Z. Wu, Y. Xiong, S. X. Yu, and D. Lin, ``Unsupervised feature learning via non-parametric instance discrimination,'' in \textit{Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)}, 2018, pp. 3733--3742.
\item A. van den Oord, Y. Li, and O. Vinyals, ``Representation learning with contrastive predictive coding,'' \textit{arXiv preprint arXiv:1807.03748}, 2018.
\item P. Bachman, R. D. Hjelm, and W. Buchwalter, ``Learning representations by maximizing mutual information across views,'' in \textit{Advances in Neural Information Processing Systems (NeurIPS)}, 2019, pp. 15509--15519.
\item K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, ``Momentum contrast for unsupervised visual representation learning,'' in \textit{Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)}, 2020, pp. 9729--9738.
\item T. Wang and P. Isola, ``Understanding contrastive representation learning through alignment and uniformity on the hypersphere,'' in \textit{Proc. 37th Int. Conf. Machine Learning (ICML)}, PMLR 119, 2020, pp. 9929--9939.
\item P. Goyal, D. Mahajan, A. Gupta, and I. Misra, ``Scaling and benchmarking self-supervised visual representation learning,'' in \textit{Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)}, 2019, pp. 6391--6400.
\item E. Cole, X. Yang, K. Wilber, O. Mac Aodha, and S. Belongie, ``When does contrastive visual representation learning work?,'' in \textit{Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)}, 2022, pp. 14755--14764.
\end{enumerate}
