# Geomorphological Connectivity and Hydrological Memory Control Surface Water Dynamics in the Indus Delta

This repository contains the codebase and data pipelines for the paper titled:
**"Geomorphological Connectivity and Hydrological Memory Control Surface Water Dynamics in the Indus Delta"** (submitted to *Environmental Modelling & Software* / *Hydrology and Earth System Sciences*).

---

## Repository Structure

*   `prism/`: Preamble, styles, and LaTeX source files for the HESS manuscript.
    *   `prism/sections/`: Manuscript sections (Introduction, Methodology, Results, Discussion, Conclusion, etc.).
    *   `prism/tables/`: LaTeX tables including model results, significance tests, and spatial autocorrelation results.
    *   `prism/reproducibility/`: Markdown documentation of GEE collections, software dependencies, and datasets.
*   `data/`: Directory for input datasets and generated scientific evaluation reports.
*   `scratch/`: Scripts for LaTeX verification, publication figures generation, and Moran's I autocorrelation testing.
*   `train_baselines.py`: Train Random Forest, LSTM, and Transformer model baselines.
*   `run_scientific_experiments.py`: Run model training and evaluations across normal, monsoon, and extreme flood splits.
*   `run_ablation_study.py`: Run feature ablation (Terrain Only vs. Climate Only vs. Combined).
*   `run_spatial_transfer.py`: Run spatial generalization audits across disjoint training/validation coordinate splits.

---

## Software Requirements and Installation

The codebase is written in **Python 3.10** (compatible with Python 3.11).

To install the required dependencies, run:
```bash
pip install torch==2.1.2 scikit-learn==1.3.2 earthengine-api==0.1.381 numpy==1.26.2 scipy==1.11.4 pandas==2.1.4 Pillow==10.1.0 matplotlib==3.8.2
```

For more detailed version requirements, refer to [reproducibility/software_versions.md](prism/reproducibility/software_versions.md).

---

## How to Run Experiments

1. **Model Training and Benchmarking**:
   ```bash
   python3 run_scientific_experiments.py
   ```
2. **Feature Ablation and Continuity-Corrected McNemar Test**:
   ```bash
   python3 run_ablation_study.py
   ```
3. **Spatial Generalization Audits**:
   ```bash
   python3 run_spatial_transfer.py
   ```
4. **Spatial Autocorrelation Permutation Testing**:
   ```bash
   python3 scratch/calculate_morans_i_with_p_value.py
   ```

---

## Software/Data Availability Statement

*   **Programming Language:** Python 3.10
*   **Software Version:** v1.0.0
*   **GitHub Repository:** [mmuzzamil-phd/indus-delta-surface-water](https://github.com/mmuzzamil-phd/indus-delta-surface-water)
*   **Zenodo Archive:** [10.5281/zenodo.10827364](https://doi.org/10.5281/zenodo.10827364)
*   **Software License:** MIT License
*   **Data License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## License

This software is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
The derived dataset is licensed under the CC BY 4.0 International License.
