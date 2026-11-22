# Stage S9 — Cloud Deployment (Hugging Face Spaces)
**Status:** COMPLETE  
**Date:** 2026-11-19  
**Agent rule reference:** AGENTS.md §7, §13.3, §11

---

## 1. Deployment Details

| Field | Detail |
|---|---|
| **Hosting Platform** | Hugging Face Spaces |
| **Space Name** | `MianGenz/Sim_clr` |
| **Public URL** | [https://huggingface.co/spaces/MianGenz/Sim_clr](https://huggingface.co/spaces/MianGenz/Sim_clr) |
| **Direct Web App View** | [https://miangenz-sim-clr.hf.space](https://miangenz-sim-clr.hf.space) |
| **SDK** | Gradio (ZeroGPU / CPU Basic, Python 3.9) |
| **Deployed Model Hash (SHA-256)** | `12358c35e3f9323e47c8a4a8e1d7e4d1cf9ead13ad9d90c1e83318b2f0751fb6` |
| **Pretrained Best Checkpoint Hash** | `c3188c25fef0eb2b8984666598f712c607b135909c82597b9eddf4e9cfd78fec` |
| **Git Commit Reference** | `38eeb15` |

---

## 2. Deployed Files & Manifest

1. `app.py`: Stack Overflow themed Gradio web application with real-time prediction and embedding representation inspection.
2. `model_arch.py`: PyTorch module definitions for `ResNet18CIFAR` and `LinearClassifier`.
3. `simclr_cifar10_eval.pt`: Lightweight weights bundle (42.7 MB).
4. `requirements.txt`: PyTorch, Torchvision, Pillow.
5. `README.md`: Space metadata YAML frontmatter.
6. `samples/`: 10 sample test images across all CIFAR-10 classes (`airplane.png`, `automobile.png`, `bird.png`, `cat.png`, `deer.png`, `dog.png`, `frog.png`, `horse.png`, `ship.png`, `truck.png`).

---

## 3. Exit Criteria Verification

| Criterion | Evidence | Status |
|---|---|---|
| Public URL active | `https://huggingface.co/spaces/MianGenz/Sim_clr` | ✅ VERIFIED |
| SHA-256 hash recorded | `12358c35e3f9323e47c8a4a8e1d7e4d1cf9ead13ad9d90c1e83318b2f0751fb6` | ✅ VERIFIED |
| Git commit pushed | Pushed to `main` branch on Hugging Face Spaces | ✅ VERIFIED |
| Attribution present | Saalik Hussain Zaidi, Mubashir Abid, Muhammad Zohair Khan | ✅ VERIFIED |
