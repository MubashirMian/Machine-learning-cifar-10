# Stage Checkpoints Index

| Stage | Status | Last updated | Checkpoint | Key result |
|---|---|---|---|---|
| S0 Setup | COMPLETE | 2026-11-18 | [checkpoint.md](file:///Users/mubashirmian/Desktop/SimCLR-Research/checkpoints/S0_setup/checkpoint.md) | Repo layout, governance docs created, environment & pytest verified (3 passed) |
| S1 Pipeline review | COMPLETE | 2026-11-18 | [checkpoint.md](file:///Users/mubashirmian/Desktop/SimCLR-Research/checkpoints/S1_pipeline_review/checkpoint.md) | Complete audit in `pipeline_review.md`; leakage risk identified & resolved |
| S2 Data module | COMPLETE | 2026-11-18 | [checkpoint.md](file:///Users/mubashirmian/Desktop/SimCLR-Research/checkpoints/S2_data_module/checkpoint.md) | 8 unit tests passed; augmentation grid saved; stratified splits verified |
| S3 Model + loss | COMPLETE | 2026-11-18 | [checkpoint.md](file:///Users/mubashirmian/Desktop/SimCLR-Research/checkpoints/S3_model_loss/checkpoint.md) | ResNet-18 CIFAR stem, 2-layer MLP head, NT-Xent loss matches reference loop (14 tests passed) |
| S4 Smoke training | COMPLETE | 2026-11-18 | [checkpoint.md](file:///Users/mubashirmian/Desktop/SimCLR-Research/checkpoints/S4_smoke_training/checkpoint.md) | 2-epoch pretrain + linear eval ran end-to-end; loss decreased; checkpoint resume verified (16 tests passed) |
| S5 Pilot runs | COMPLETE | 2026-11-18 | [checkpoint.md](file:///Users/mubashirmian/Desktop/SimCLR-Research/checkpoints/S5_pilot_runs/checkpoint.md) | Adam lr=1e-3 τ=0.5 wins (+2% over SGD); 217.7 img/s MEASURED; epoch budget pending user approval |
| S6 Main reproduction | COMPLETE | 2026-11-19 | [checkpoint.md](file:///Users/mubashirmian/Desktop/SimCLR-Research/checkpoints/S6_main_reproduction/checkpoint.md) | 50 ep pretrain (Loss 4.74, Acc 62.4%) + linear eval: MEASURED Top-1: 75.23%, Top-5: 98.71% |
| S7 Data-fraction experiment | COMPLETE | 2026-11-19 | [checkpoint.md](file:///Users/mubashirmian/Desktop/SimCLR-Research/checkpoints/S7_data_fraction_experiment/checkpoint.md) | 10% (54.2%), 25% (65.4%), 50% (71.0%), 100% (75.2%); monotonic scaling with diminishing returns confirmed |
| S8 App | COMPLETE | 2026-11-19 | [checkpoint.md](file:///Users/mubashirmian/Desktop/SimCLR-Research/checkpoints/S8_app/checkpoint.md) | Stack Overflow themed Gradio inference app built and verified locally (18.15 ms latency) |
| S9 Deployment | COMPLETE | 2026-11-19 | [checkpoint.md](file:///Users/mubashirmian/Desktop/SimCLR-Research/checkpoints/S9_deployment/checkpoint.md) | Deployed live to Hugging Face Spaces: https://huggingface.co/spaces/MianGenz/Sim_clr |
| S10 Report | COMPLETE | 2026-11-19 | [checkpoint.md](file:///Users/mubashirmian/Desktop/SimCLR-Research/checkpoints/S10_report/checkpoint.md) | Final report summary compiled, master results table traceable to runs, full attribution |
