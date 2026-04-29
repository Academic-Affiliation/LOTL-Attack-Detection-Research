# LOTL Attack Detection Research

This repository contains code and notebooks for LOTL (Living-off-the-Land) attack detection research.

**Quick Start**

- **Prerequisites:** Python 3.8+ and `pip`.
- Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Run the notebooks**

- Launch Jupyter and open the notebooks in the repo root:

```bash
jupyter lab
# or
jupyter notebook
```
- Notebooks to try:
  - `LOTL Attack Detection.ipynb`
  - `LOTL Detection Analysis.ipynb`
  - `LOTL Detection Baselines .ipynb`

**Run the codebase (scripts)**

- Prepare data:

```bash
python scripts/prepare_data.py
```

- Train Halo-BERT model (example):

```bash
python scripts/train_halo_bert.py
```

- Run a full experiment (end-to-end):

```bash
python scripts/run_full_experiment.py
```

**Project Structure (key files)**

- Notebooks: `LOTL Attack Detection.ipynb`, `LOTL Detection Analysis.ipynb`, `LOTL Detection Baselines .ipynb`
- Main scripts: `scripts/prepare_data.py`, `scripts/train_halo_bert.py`, `scripts/run_full_experiment.py`
- Model code: `baselines/` and `halo_bert/`
- Datasets (CSV): `balanced_combined_lotl_dataset.csv`, `complete_volttyphoon_dataset.csv`, `enhanced_lotl_obfuscated_dataset.csv`

**Data**

Sample datasets (CSV) are included in the repository root. Adjust paths in scripts if you move data.

**Notes**

- Use the virtual environment for reproducible runs.
- If you encounter missing packages, install them with `pip install <package>` or update `requirements.txt`.

If you'd like, I can also: add a Makefile, add more detailed script examples, or create a minimal reproducible example for training.
