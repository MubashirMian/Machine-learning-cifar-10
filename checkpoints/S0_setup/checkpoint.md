# S0 Setup — Status: COMPLETE

## Previous checkpoint verified: None (bootstrap session, first initialization)
- Mismatches found: None. Clean project bootstrap.

## Planned work
- [x] Inspect project directory and historical artifacts (`papers/`, `proposal/`, `existing_project/`, `data/`).
- [x] Archive team's original data cleaning pipeline into `pipeline_original/` untouched (§8.1).
- [x] Establish standard repository directory structure (§5).
- [x] Create project governance documents (`PROJECT_PLAN.md`, `SOURCES.md`, `DECISIONS.md`, `checkpoints/INDEX.md`, `README.md`, `.gitignore`).
- [x] Create `requirements.txt` with project dependencies.
- [x] Initialize local virtual environment (`.venv`) using Python 3.9.
- [x] Install dependencies from `requirements.txt` into `.venv`.
- [x] Run pytest test suite to verify baseline environment.
- [x] Confirm stage plan and data budget wording with user (§7, §10.3).
- [x] Configure `.gitignore` to ensure `data/` folder is never committed or pushed.

## Work done
- Inspected `proposal/ML project Proposal.pdf` and verified key requirements: SimCLR v1 on CIFAR-10, ResNet-18 CIFAR stem encoder, batch size 256, NT-Xent loss, SGD/Adam optimizer, Streamlit deployment, and unlabeled data-scaling experiment with constant epochs.
- Inspected `papers/original_simclr_paper.pdf`, specifically verifying Algorithm 1 (NT-Xent loss formulation), Appendix A (data augmentations), Appendix B.9 (CIFAR-10 experiments & architecture), Table 1, Table 6, and Figure B.7.
- Preserved existing data pipeline `existing_project/eda_pipeline.ipynb` by copying to `pipeline_original/eda_pipeline.ipynb` untouched (§8.1).
- Created directory structure: `configs/`, `src/simclr/utils/`, `src/simclr/observability/`, `app/`, `tests/`, `checkpoints/`, `runs/`, `data/splits/`.
- Created and configured `.gitignore` to strictly exclude `data/` and all dataset files/folders (`data/**`, `**/cifar-10/**`, `**/train/`, `**/test/`). Verified `git ls-files data` returns empty.
- Authored `requirements.txt` detailing vision, ML, testing, deployment, and observability requirements.
- Authored `PROJECT_PLAN.md`, `SOURCES.md`, `DECISIONS.md`, `README.md`, and `checkpoints/INDEX.md`.
- Created Python 3.9 virtual environment at `.venv/` and installed all requirements (PyTorch 2.8.0, torchvision 0.23.0, pytest 8.4.2, streamlit 1.50.0, etc.).
- Created `tests/test_environment.py` and executed pytest suite. All 3 tests passed.

## Files changed (path — reason)
- `pipeline_original/eda_pipeline.ipynb` — Preserved untouched copy of existing pipeline per §8.1.
- `.gitignore` — Ignore datasets (`data/**`), weights (`*.pt`), caches, and logs per §5 and D04.
- `requirements.txt` — Project dependencies per §7.
- `PROJECT_PLAN.md` — Project roadmap, architecture specs, and execution plan per §7 and §18.
- `SOURCES.md` — Verified citations from proposal, paper, and official repo per §3 and §6.
- `DECISIONS.md` — Conflicts register (C1-C4), formal decisions (D01-D04), and open decisions per §4.7, §18.4, and §19.
- `README.md` — Repository documentation and attribution per §13.3 and §16.
- `checkpoints/INDEX.md` — Stage progress tracker per §18.1.
- `checkpoints/S0_setup/checkpoint.md` — Stage S0 progress checkpoint per §18.2.
- `tests/test_environment.py` — Baseline environment and governance verification tests.

## Commands executed (exact) and outcomes
- `mkdir -p pipeline_original && cp existing_project/eda_pipeline.ipynb pipeline_original/`: copied original pipeline (exit code 0).
- `mkdir -p configs src/simclr/utils src/simclr/observability app tests checkpoints/S0_setup runs data/splits`: initialized directory structure (exit code 0).
- `/opt/homebrew/bin/python3.9 -m venv .venv`: created isolated Python 3.9 virtual environment (exit code 0).
- `.venv/bin/pip install -r requirements.txt`: successfully installed dependencies (exit code 0).
- `.venv/bin/pytest tests/ -v`: verified environment integrity (exit code 0, 3 passed).
- `git ls-files data`: verified no data files are tracked by git.

## Tests run and results
- `tests/test_environment.py::test_governance_files_exist` PASSED
- `tests/test_environment.py::test_directory_structure_exists` PASSED
- `tests/test_environment.py::test_core_dependencies_importable` PASSED
- Outcome: 3 passed in 16.16s.

## Metrics (label each MEASURED / NOT MEASURED; include run IDs)
- None (Setup stage).

## Observability: W&B run ID/URL or NOT CONFIGURED; Phoenix project/trace refs or NOT CONFIGURED
- W&B: `NOT CONFIGURED`
- Phoenix: `NOT CONFIGURED`

## Model artifacts: path, SHA-256, originating run
- None (Setup stage).

## Errors and resolutions
- Initial sandboxed pip install failed due to network restriction (`[Errno 8] nodename nor servname provided, or not known`).
- Resolution: Installed dependencies with unsandboxed pip run approved by user.
- User request: Ensure data folder containing test/training data is never pushed to git.
- Resolution: Updated `.gitignore` with strict rules (`data/`, `data/**`, `**/cifar-10/**`, `**/train/`, `**/test/`). Verified `git status` and `git ls-files data`.

## Decisions made (link to DECISIONS.md entries)
- D01: Base encoder set to ResNet-18 with CIFAR stem (approved in Proposal).
- D02: Optimizer choice set to SGD or Adam for batch size 256 (to be evaluated in S5).
- D03: Data-fraction training budget maintains constant epochs across all runs, strictly matching proposal text.
- D04: Data folder excluded from git version control.
- C1: Loss scaling convention set to paper mean over $2N$ views.
- C2: Brightness jitter set to standard multiplicative `ColorJitter`.

## Unresolved issues / known gaps
- None for S0.

## Reproduce: exact command(s)
```bash
/opt/homebrew/bin/python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```

## Next stage / next command
- Transition to Stage S1 (Pipeline Review).
- Next command: Inspect `pipeline_original/eda_pipeline.ipynb` in detail against checklist in AGENTS.md §8.2 and write `checkpoints/S1_pipeline_review/pipeline_review.md`.
