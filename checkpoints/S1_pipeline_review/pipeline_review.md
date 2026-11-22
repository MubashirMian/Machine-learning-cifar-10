# Pipeline Review Report — CIFAR-10 Data & Preprocessing

**Stage:** S1 Pipeline Review  
**Date:** 2026-11-18  
**Artifact Inspected:** `pipeline_original/eda_pipeline.ipynb` (38 cells)  
**Preserved Copy:** [`pipeline_original/eda_pipeline.ipynb`](file:///Users/mubashirmian/Desktop/SimCLR-Research/pipeline_original/eda_pipeline.ipynb)  
**Audit Checklist Reference:** `AGENTS.md` §8.1, §8.2, §8.3  

---

## 1. Executive Summary
The team's existing Jupyter notebook (`eda_pipeline.ipynb`) establishes an exploratory data analysis (EDA) and prototype PyTorch data pipeline for CIFAR-10. It correctly confirms dataset balance (5,000 images/class, 10 classes), checks data cleanliness (no missing values, no corrupt files), measures RGB channel statistics, and prototypes a two-view contrastive data loader and a stratified subset sampler.

However, several critical limitations, potential leakage risks, and deviations from the SimCLR protocol were discovered during this audit:
1. **Fallback Test Set Leakage Risk:** If the official torchvision test set fails to download, cell 34 carves a 10% test split from the same `X` array used for pretraining, violating test isolation.
2. **Missing Augmentations in Linear Evaluation Training:** Cell 34 uses a bare `eval_transform` (only normalization) for linear classifier training. The official SimCLR protocol requires `RandomResizedCrop` and `RandomHorizontalFlip` during linear evaluation training to prevent classifier overfitting.
3. **Validation Split Omission:** The original pipeline pretrains and trains the linear classifier on all 50,000 images without holding out a dedicated, fixed validation split for hyperparameter tuning.
4. **Subsets Drawn from Full Dataset:** The `stratified_subset` function samples subsets from the entire dataset rather than from a pretraining pool that excludes validation data.
5. **Channel Normalization Deviation:** SimCLR v1 in the official repo does not normalize by channel mean/std (operates directly on float $[0, 1]$). The notebook introduces channel normalization. While valid in PyTorch workflows, it must be strictly harmonized across all stages.

---

## 2. Line-by-Line / Cell-by-Cell Pipeline Audit

### 2.1 EDA & Data Ingestion (Cells 1–23)
- **Cell 2 & 5:** Reads `trainLabels.csv` and Kaggle train directory. Verifies 50,000 training images, 10 classes, perfectly balanced (5,000 per class).
- **Cell 8:** Iterates through images, converting to RGB numpy array $X \in \mathbb{R}^{50000 \times 32 \times 32 \times 3}$ (dtype uint8). Checks for corrupt or missing files (found 0).
- **Cell 14:** Computes channel statistics:
  - $\text{Mean} = (0.491, 0.482, 0.447)$
  - $\text{Std} = (0.247, 0.243, 0.262)$
  - Verified: Calculated strictly on training images $X$.
- **Cell 17–21:** Exploratory brightness, color histograms, PCA, and kNN baseline on raw pixels (31% accuracy). Confirms that raw pixel features are insufficient for linear separation and motivates learned representations.

### 2.2 Data Loading & SimCLR Pretraining Pipeline (Cells 24–32)
- **Cell 28 (`simclr_transform`):**
  - `RandomResizedCrop(32, scale=(0.08, 1.0))`
  - `RandomHorizontalFlip(p=0.5)`
  - `RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)`
  - `RandomGrayscale(p=0.2)`
  - `ToTensor()`
  - `Normalize(MEAN, STD)`
  - **Assessment:** Correctly omits Gaussian blur per Appendix B.9 for CIFAR-10. Jitter strength $s=0.5$ correctly matches paper $(0.8s=0.4, 0.2s=0.1)$. Augmentations are executed on the fly per `__getitem__` call.
- **Cell 30 (`ArrayDataset` & `pretrain_loader`):**
  - Implements two independent transforms `(self.transform(img), self.transform(img))` for positive pairs.
  - `batch_size=256`, `drop_last=True`, `shuffle=True`.
  - **Assessment:** Clean, correct implementation of two-view contrastive pair generation.

### 2.3 Linear Evaluation Pipeline (Cells 33–34)
- **Cell 34:**
  - `train_set = ArrayDataset(X, y, eval_transform)` where `eval_transform` has NO data augmentation.
  - **Issue:** Official SimCLR protocol (`tf2/data_util.py` `preprocess_for_eval`) and AGENTS.md §4.1 state that linear evaluation *training* must use random crop and flip. Only *test evaluation* is deterministic.
  - **Fallback Danger:** Cell 34 contains:
    ```python
    except Exception as error:
        train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=0.1, stratify=y, random_state=SEED)
        test_set = Subset(train_set, test_idx)
        train_set = Subset(train_set, train_idx)
    ```
    If torchvision cannot reach external servers to download CIFAR-10, it splits $X$. Because $X$ was already used in its entirety for pretraining, the fallback test set has already been seen by the encoder during contrastive learning. This would constitute severe data leakage.

### 2.4 Data-Fraction Subsets (Cells 35–36)
- **Cell 36 (`stratified_subset`):**
  - Samples $N \times \text{fraction}$ images per class using `np.random.RandomState(seed)`.
  - **Assessment:** Stratification is exact. However, subsets are sampled across all 50k indices of $X$. If a validation split is introduced, subsets must be strictly sampled from the pretraining pool (excluding validation indices).

---

## 3. Systematic Checklist Verification (AGENTS.md §8.2)

| Checklist Item | Finding | Status / Action Required |
|---|---|---|
| **1. Data alteration?** | Does not remove, relabel, deduplicate or reweight images. | **PASS** — Unmodified 50k benchmark images preserved. |
| **2. Test set isolation?** | Test set is not touched by fitting statistics. Fallback split has potential leakage risk. | **ACTION REQUIRED** — Ensure official test set is loaded from local or downloaded data; disable ad-hoc fallback split. |
| **3. Normalization on train only?** | Channel mean and std computed solely on training array $X$. | **PASS** — No data leakage in statistical computation. |
| **4. On-the-fly augmentation?** | Performed on the fly in PyTorch `__getitem__`. | **PASS** — Dynamic two-view generation is fully compliant. |
| **5. Dtype and range?** | Normalized float tensors with mean ~0, std ~1. | **PASS** — Standard PyTorch tensor format. |
| **6. Label ignorance in pretraining?** | `ArrayDataset` returns labels, but SimCLR training loop will drop them. | **PASS** — Unsupervised pretraining verified. |

---

## 4. Proposed Architectural Changes for `src/simclr/data.py`

To eliminate all identified risks and bring the pipeline into full compliance with `AGENTS.md` and the official paper:

1. **Establish Fixed Stratified Validation Split (§8.3):**
   - Partition the 50,000 training images into:
     - Pretraining / Linear Training Pool: 45,000 images (4,500 per class).
     - Validation Split: 5,000 images (500 per class), saved to `data/splits/val_idx.npy` with seed 42.
   - Use validation split for hyperparameter tuning, pilot runs, and linear probe early stopping.
2. **Strict Test Set Management:**
   - Use torchvision standard CIFAR-10 binary test dataset (10,000 images). If network download is restricted, load directly from standard local CIFAR-10 python batches.
   - Eliminate any fallback that carves test data from the training set.
3. **Separate Transforms for Linear Eval Training vs. Testing:**
   - `linear_train_transform`: `RandomResizedCrop(32, scale=(0.08, 1.0))`, `RandomHorizontalFlip(p=0.5)`, `ToTensor()`, `Normalize()`.
   - `eval_transform` (Testing & Deployment): `ToTensor()`, `Normalize()`.
4. **Nested Stratified Subsets for Pretraining Data-Fraction Experiment (§10.3):**
   - Sample subsets strictly from the 45,000 training indices (0% overlap with `val_idx.npy`).
   - Use nested sampling so that `subset(1%) ⊂ subset(5%) ⊂ subset(10%) ⊂ ...` to minimize sampling variance.
5. **Decouple Normalization as a Configurable Option:**
   - Support both normalized (`Normalize(MEAN, STD)`) and standard float $[0, 1]$ tensors, ensuring exact parity between pretraining, linear evaluation, testing, and deployment.
