# Reproducibility Guide — Indus Delta Surface Water Benchmark

This document details the exact, step-by-step process required to reproduce the dataset extraction, baseline model training, diagnostics, statistical significance tests, and paper figures/tables.

---

### Step 1: Authenticate Google Earth Engine
First, verify that your Google Earth Engine (GEE) Python API is authenticated. If this is a fresh setup, authenticate your user account:
```bash
earthengine authenticate
```
*Note: Make sure your authenticated Google account has access to the GEE Code Editor and is registered for Earth Engine.*

---

### Step 2: Run Data Extraction
Extract raw Sentinel-1, Sentinel-2, climate, and topography composites for the Indus Delta region:
```bash
python3 scripts/gee_pipeline.py --start_date 2022-08-01 --end_date 2022-10-31
```
This script queries Copernicus Sentinel-1, Sentinel-2 collections, SRTM DEM, CHIRPS, and ERA5-Land monthly aggregation.

---

### Step 3: Generate Dataset
Compile the spatial-temporal coordinates and extract tabular pixel values into a numpy format. 
*Note: If you do not have GEE active, download the compiled paper dataset (`indus_dataset.npz` and `indus_dataset_v2.npz`) directly from the corresponding author, or place the sample dataset `indus_dataset_sample.npz` in the `data/` folder.*

---

### Step 4: Train Random Forest
Train and evaluate the Random Forest baseline model:
```bash
python3 scripts/train_baselines.py
```
This trains the static Random Forest classifier under the leakage-free climate & terrain configuration.

---

### Step 5: Train LSTM
Train and evaluate the PyTorch LSTM sequence model:
```bash
python3 scripts/train_baselines.py
```
*(Runs concurrently within the baseline training pipeline, fitting on the sequential training coordinates)*

---

### Step 6: Train Transformer
Train and evaluate the PyTorch Transformer model:
```bash
python3 scripts/train_baselines.py
```
*(Runs concurrently within the baseline training pipeline, executing self-attention checks across time steps)*

---

### Step 7: Run Diagnostics
Analyze potential data leakage from spectral water indices, and extract feature importance rankings:
```bash
python3 scripts/run_diagnostics.py
```
This outputs:
1. Classification accuracy comparing models with and without data leakage features.
2. Random Forest feature importance rankings.
3. Decision threshold sweeps for sequence models.

---

### Step 8: Generate All Figures
Regenerate Figures 1 through 7 in PNG format:
```bash
python3 scripts/reproduce_figures.py
```
This generates:
* `results/figures/fig1_study_area.png`
* `results/figures/fig2_workflow.png`
* `results/figures/fig3_feature_importance.png`
* `results/figures/fig4_routing_lag.png`
* `results/figures/fig5_baseline_comparison.png`
* `results/figures/fig6_ablation.png`
* `results/figures/fig7_motivation.png`

---

### Step 9: Generate All Tables
Regenerate Tables 1 through 7 in both LaTeX (`.tex`) and Markdown (`.md`) formats:
```bash
python3 scripts/reproduce_tables.py
```
All outputs are saved to `results/tables/`.

---

### Step 10: Verify Statistical Tests
Calculate the 95% bootstrap confidence intervals for the metrics and compute the Edwards' continuity-corrected McNemar statistical significance test:
```bash
python3 scripts/run_significance.py
```
This compares Terrain Only vs. Combined models and computes the corresponding Chi-Square p-values.
