# Geomorphological Connectivity and Temporal Hydrological Controls of Surface Water Dynamics in the Indus River Delta

This repository contains the official codebase and benchmark package for the paper:
**\"Geomorphological Connectivity and Temporal Hydrological Controls of Surface Water Dynamics in the Indus River Delta\"** (submitted to *Hydrology and Earth System Sciences* / *Environmental Modelling & Software*).

---

## 1. Project Overview

This benchmark is designed to evaluate machine learning models in mapping surface water occurrence under a **leakage-free configuration**. By strictly withholding concurrent Sentinel-1 radar backscatter and Sentinel-2 optical bands from the input features, we prevent coordinate memorization and evaluate how well models learn geomorphological controls (static boundary constraints) and meteorological triggers (catchment routing memory).

### Key Scientific Contributions
1. **Topographic Boundary Constraint:** Terrain features (elevation, slope, river/coastal proximity) govern **87.81%** of spatial inundation boundaries.
2. **Dynamic Routing Memory:** Cross-correlation analysis isolates a **3-month lag** between catchment precipitation and coastal wetland inundation.
3. **Irrigation Divergence:** Irrigated cropland pixels exhibit negative correlations at Lag 0, highlighting anthropogenic water management bypassing natural meteorological routing.
4. **Empirical Model Collapse:** Hard coordinate-withheld tests reveal that standard sequence models (LSTMs, Transformers) collapse to predicting dry land ($F_1$-score = 0.000) during normal seasons, motivating physically-constrained architectures.

---

## 2. Repository Structure

```
.
├── CITATION.cff             # Citation metadata for the paper
├── LICENSE                  # MIT License
├── README.md                # This document
├── requirements.txt         # Pip dependency package file
├── environment.yml          # Conda environment configuration
├── .gitignore               # Ignored files (LaTeX, large data, caches)
│
├── config/
│   └── config.yaml          # Project-wide metadata and bounds config
│
├── configs/
│   ├── train.yaml           # Baseline training hyperparameters
│   └── evaluation.yaml      # Model evaluation settings
│
├── data/
│   ├── README.md            # Dataset column documentation
│   └── indus_dataset_sample.npz   # 100-point sample dataset for fast runs
│
├── scripts/
│   ├── gee_pipeline.py      # Google Earth Engine data extraction script
│   ├── exploratory_analysis.py   # Time series correlation script
│   ├── utils.py             # Data loading and reconstruction utilities
│   ├── train_baselines.py   # Baseline model fitting script
│   ├── run_diagnostics.py   # Data leakage, threshold, and feature audits
│   ├── run_significance.py  # Statistical significance and bootstrap CI
│   ├── ablation_experiments.py   # Feature-ablation analysis script
│   ├── reproduce_figures.py # Script to regenerate all 7 paper figures
│   └── reproduce_tables.py  # Script to regenerate LaTeX/Markdown tables
│
├── models/
│   ├── random_forest.py     # Random Forest baseline model class
│   ├── lstm.py              # PyTorch LSTM model class
│   └── transformer.py       # PyTorch Transformer model class
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Jupyter data exploration walkthrough
│   └── 02_reproduce_figures.ipynb # Jupyter figure display walkthrough
│
├── results/
│   ├── figures/             # Output PNG figures (reproduced)
│   └── tables/              # Output LaTeX/Markdown tables (reproduced)
│
└── docs/
    ├── reproducibility.md   # Step-by-step reproducibility instructions
    ├── paper_summary.md     # Extended abstract and key findings summary
    ├── workflow.pdf         # Copy of the methodological workflow chart
    └── LICENSE              # Documentation license copy
```

---

## 3. Installation & Setup

### Option A: Pip Installation
We recommend using **Python 3.10** or higher. Install all required software packages:
```bash
pip install -r requirements.txt
```

### Option B: Conda Installation
Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate indus-delta-benchmark
```

### Google Earth Engine Authentication
To run the data extraction and exploratory correlation scripts, initialize and authorize your Earth Engine user account:
```bash
earthengine authenticate
```

---

## 4. How to Reproduce Results

Follow these commands to verify the scientific results:

### Step 1. Train and Evaluate Baselines
Train the Random Forest, LSTM, and Transformer baseline models on the sample dataset:
```bash
python3 scripts/train_baselines.py
```

### Step 2. Run Diagnostics & Leakage Audits
Verify the impact of spectral band leakage, analyze feature importance Gini scores, and calibrate sequence probability thresholds:
```bash
python3 scripts/run_diagnostics.py
```

### Step 3. Run Statistical Significance Tests
Calculate the 95% bootstrap confidence intervals for the metrics and compute Edwards' continuity-corrected McNemar significance testing:
```bash
python3 scripts/run_significance.py
```

### Step 4. Reproduce Figures
Generate all 7 figures in the manuscript and save them to `results/figures/`:
```bash
python3 scripts/reproduce_figures.py
```

### Step 5. Reproduce Tables
Generate all 7 tables in the manuscript (in both LaTeX `.tex` and Markdown `.md` formats) and save them to `results/tables/`:
```bash
python3 scripts/reproduce_tables.py
```

---

## 5. Expected Outputs
Running the baseline script `scripts/train_baselines.py` should output the comparative performance tables on the sample dataset:
```
======================================================================
BASELINE PERFORMANCE REPORT (INDUS DELTA SURFACE WATER)
======================================================================

[Normal Validation Set]
Model           | F1     | IoU    | Recall | Prec  
--------------------------------------------------
Random Forest   | 0.915  | 0.843  | 0.967  | 0.868 
LSTM            | 0.924  | 0.860  | 1.000  | 0.860 
Transformer     | 0.924  | 0.860  | 1.000  | 0.860 

[Testing / Extreme Flood Period]
Model           | F1     | IoU    | Recall | Prec  
--------------------------------------------------
Random Forest   | 0.951  | 0.907  | 0.976  | 0.928 
LSTM            | 0.956  | 0.915  | 1.000  | 0.915 
Transformer     | 0.956  | 0.915  | 1.000  | 0.915 
```

---

## 6. Citation

If you use this benchmark dataset or code in your research, please cite:
```bibtex
@article{muzzamil2026geomorphological,
  author    = {Muzzamil, Mirza Muhammad and Ismael, Muhammad Ali and Ahmad, Syed Imran and Rustomov, Rustam B.},
  title     = {Geomorphological Connectivity and Temporal Hydrological Controls of Surface Water Dynamics in the Indus River Delta},
  journal   = {Hydrology and Earth System Sciences},
  year      = {2026},
  doi       = {10.5281/zenodo.placeholder}
}
```

---

## 7. License & Contact
* **Code License:** [MIT License](LICENSE)
* **Contact:** Mirza Muhammad Muzzamil (mirzamuzzamil@neduet.edu.pk)
* **Institution:** NED University of Engineering and Technology, Karachi, Pakistan.
