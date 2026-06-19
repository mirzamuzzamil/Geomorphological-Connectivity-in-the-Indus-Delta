# Dataset Documentation — Indus Delta Surface Water Benchmark

This folder contains the benchmark datasets used to train and evaluate the baseline models. The datasets are compiled in NumPy compressed format (`.npz`).

---

## 1. Dataset Files

### `indus_dataset_sample.npz`
* **Purpose:** A small-scale sample dataset containing a subset of coordinates (100 points, 36 months) for fast dry-runs and continuous integration testing.
* **Format:**
  * `X`: Feature matrix of shape `(3600, 11)`. The columns are:
    * Columns 0-3: Climate variables (precipitation, temperature, evaporation, potential evaporation).
    * Columns 4-8: Topographic features (elevation, slope, flow accumulation, river distance, coastal distance).
    * Column 9: Year.
    * Column 10: Month.
  * `y`: Binary occurrence labels `(3600,)` (0: land, 1: water).

### `indus_dataset.npz`
* **Purpose:** The full scientific dataset containing 83,000 monthly observations across 1,000 sampling coordinates (2018, 2022, 2024).
* **Format:**
  * `X`: Feature matrix of shape `(33000, 18)`. The columns are:
    * Columns 0-3: Sentinel-2 optical bands/indices (NDVI, NDWI, MNDWI, LSWI, EVI).
    * Columns 4-6: Sentinel-1 SAR VV, VH backscatter, and VH/VV ratio.
    * Columns 7-10: CHIRPS and ERA5-Land climate metrics.
    * Columns 11-15: SRTM and HydroSHEDS elevation/routing features.
    * Column 16: Year.
    * Column 17: Month.
  * `y`: Ground-truth water occurrence labels derived from Sentinel-1 VV backscatter.

---

## 2. Dataset Reconstruction
To reconstruct the 3D spatial-temporal grid from these flat matrices, use the helper function `reconstruct_grid` in `scripts/utils.py`.
