import os

OUT_DIR = 'results/tables'

TABLES = {
    "model_results": {
        "title": "Table 1. Classification performance (F1-score) of baseline models across normal, monsoon, and extreme flood periods on the Indus Delta point dataset (2018-2024 test split).",
        "markdown": """| Model | Architecture | Normal F1 (Dry Season) | Monsoon F1 (Jul-Sep 2022) | Extreme F1 (Aug-Oct 2022) |
| :--- | :--- | :---: | :---: | :---: |
| Random Forest (Terrain Only) | Static classifier | 0.148 | 0.835 | 0.842 |
| Random Forest (Combined) | Static classifier | 0.151 | 0.838 | 0.845 |
| LSTM | Sequence model | 0.000 | 0.752 | 0.745 |
| Transformer | Sequence model | 0.000 | 0.767 | 0.751 |""",
        "latex": """\\begin{table*}[t]
\\caption{Classification performance (F1-score) of baseline models across normal, monsoon, and extreme flood periods on the Indus Delta point dataset (2018--2024 test split).}
\\label{tab:model_results}
\\centering
\\small
\\begin{tabular}{p{4.2cm} p{2.8cm} c c c}
\\hline
\\textbf{Model} & \\textbf{Architecture} & \\textbf{Normal F1} & \\textbf{Monsoon F1} & \\textbf{Extreme F1} \\
 & & \\textbf{(Dry Season)} & \\textbf{(Jul--Sep 2022)} & \\textbf{(Aug--Oct 2022)} \\
\\hline
Random Forest (Terrain Only) & Static classifier   & 0.148 & 0.835 & 0.842 \\
Random Forest (Combined)     & Static classifier   & 0.151 & 0.838 & 0.845 \\
LSTM                         & Sequence model      & 0.000 & 0.752 & 0.745 \\
Transformer                  & Sequence model      & 0.000 & 0.767 & 0.751 \\
\\hline
\\end{tabular}
\\end{table*}"""
    },
    "significance_tests": {
        "title": "Table 2. Feature split ablation: classification performance with 95% bootstrap confidence intervals (CI) on the 2022 extreme flood test period.",
        "markdown": """| Feature Split | F1 [95% CI] | IoU [95% CI] | Precision [95% CI] | Recall [95% CI] | AUC [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Terrain Only | 0.842 [0.833, 0.851] | 0.728 [0.714, 0.740] | 0.894 [0.884, 0.903] | 0.797 [0.784, 0.809] | 0.938 [0.934, 0.943] |
| Climate Only | 0.626 [0.615, 0.638] | 0.456 [0.444, 0.468] | 0.640 [0.626, 0.654] | 0.613 [0.598, 0.627] | 0.743 [0.733, 0.752] |
| Combined | 0.845 [0.837, 0.854] | 0.732 [0.719, 0.745] | 0.893 [0.884, 0.902] | 0.803 [0.790, 0.814] | 0.931 [0.926, 0.936] |""",
        "latex": """\\begin{table*}[t]
\\caption{Feature split ablation: classification performance with 95\\% bootstrap confidence intervals (CI) on the 2022 extreme flood test period.}
\\label{tab:significance_tests}
\\centering
\\small
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{l c c c c c}
\\hline
\\textbf{Feature Split} & \\textbf{F1 [95\\% CI]} & \\textbf{IoU [95\\% CI]} & \\textbf{Precision [95\\% CI]} & \\textbf{Recall [95\\% CI]} & \\textbf{AUC [95\\% CI]} \\
\\hline
Terrain Only  & 0.842 [0.833, 0.851] & 0.728 [0.714, 0.740] & 0.894 [0.884, 0.903] & 0.797 [0.784, 0.809] & 0.938 [0.934, 0.943] \\
Climate Only  & 0.626 [0.615, 0.638] & 0.456 [0.444, 0.468] & 0.640 [0.626, 0.654] & 0.613 [0.598, 0.627] & 0.743 [0.733, 0.752] \\
Combined      & 0.845 [0.837, 0.854] & 0.732 [0.719, 0.745] & 0.893 [0.884, 0.902] & 0.803 [0.790, 0.814] & 0.931 [0.926, 0.936] \\
\\hline
\\end{tabular}%
}
\\end{table*}"""
    },
    "spatial_autocorrelation": {
        "title": "Table 3. Global Moran's I spatial autocorrelation statistics and permutation-based significance testing results (999 permutations) for the stratified sampling points and Random Forest model spatial residuals on validation coordinates in the Indus Delta.",
        "markdown": """| Variable / Residual | Moran's I | p-value | Spatial Interpretation |
| :--- | :---: | :---: | :--- |
| Water Occurrence (All Points) | 0.209442 | 0.0010 | Significant positive spatial autocorrelation |
| Water Occurrence (Validation) | 0.209384 | 0.0010 | Significant positive spatial autocorrelation |
| Elevation (All Points) | 0.180902 | 0.0010 | Significant positive spatial autocorrelation |
| Random Forest Residuals (Validation) | 0.002022 | 0.8930 | No statistically significant spatial autocorrelation (spatially random) |""",
        "latex": """\\begin{table}[t]
\\caption{Global Moran's I spatial autocorrelation statistics and permutation-based significance testing results (999 permutations) for the stratified sampling points and Random Forest model spatial residuals on validation coordinates in the Indus Delta.}
\\label{tab:spatial_autocorrelation}
\\centering
\\small
\\begin{tabular}{p{4.2cm} c c p{6.5cm}}
\\hline
\\textbf{Variable / Residual} & \\textbf{Moran's I} & \\textbf{p-value} & \\textbf{Spatial Interpretation} \\
\\hline
Water Occurrence (All Points) & $0.209442$ & $0.0010$ & Significant positive spatial autocorrelation \\
Water Occurrence (Validation) & $0.209384$ & $0.0010$ & Significant positive spatial autocorrelation \\
Elevation (All Points) & $0.180902$ & $0.0010$ & Significant positive spatial autocorrelation \\
Random Forest Residuals (Validation) & $0.002022$ & $0.8930$ & No statistically significant spatial autocorrelation (spatially random) \\
\\hline
\\end{tabular}
\\end{table}"""
    },
    "leakage_audit": {
        "title": "Table 4. Leakage audit: data source roles and leakage mitigation strategies.",
        "markdown": """| Data Source | Input | Target | Leakage Mitigation |
| :--- | :---: | :---: | :--- |
| Sentinel-1 VV | No | Yes | Excluded from inputs; used only to derive ground truth labels. |
| Sentinel-1 VH | No | No | Excluded; used only for supplementary validation. |
| Sentinel-2 Optical | No | No | Excluded; used for independent manual target validation. |
| ERA5-Land | Yes | No | Coarse resolution (9 km) prevents coordinate memorization. |
| CHIRPS Rainfall | Yes | No | Coarse resolution (5.5 km) prevents coordinate memorization. |
| SRTM DEM | Yes | No | Static; temporal dynamics learned from climate inputs only. |""",
        "latex": """\\begin{table}[t]
\\caption{Leakage audit: data source roles and leakage mitigation strategies.}
\\label{tab:leakage_audit}
\\centering
\\small
\\begin{tabular}{p{2.6cm} c c p{6.8cm}}
\\hline
\\textbf{Data Source} & \\textbf{Input} & \\textbf{Target} & \\textbf{Leakage Mitigation} \\
\\hline
Sentinel-1 VV  & No  & Yes & Excluded from inputs; used only to derive ground truth labels. \\
Sentinel-1 VH  & No  & No  & Excluded; used only for supplementary validation. \\
Sentinel-2 Optical & No & No & Excluded; used for independent manual target validation. \\
ERA5-Land      & Yes & No  & Coarse resolution (9\\,km) prevents coordinate memorization. \\
CHIRPS Rainfall & Yes & No  & Coarse resolution (5.5\\,km) prevents coordinate memorization. \\
SRTM DEM       & Yes & No  & Static; temporal dynamics learned from climate inputs only. \\
\\hline
\\end{tabular}
\\end{table}"""
    },
    "contribution_summary": {
        "title": "Table 5. Summary of core scientific findings, empirical evidence, statistical validation, and corresponding manuscript contributions.",
        "markdown": """| Finding | Evidence | Statistical Support | Paper Contribution |
| :--- | :--- | :--- | :--- |
| Geomorphological Control | 87.81% Gini feature importance in Random Forest | McNemar test yields $p = 0.2584$ comparing Terrain-only and Combined models | Topographic control over spatial boundaries (noting statistical power limits for small differences) |
| Hydrological Memory | 3-month precipitation routing lag in coastal wetlands | Temporal cross-correlation analysis | Routing lag and cropland correlation analysis |
| Sequence Model Collapse | F1 = 0.000 during normal dry periods for LSTM and Transformer | Diagnostic threshold sweeps and class imbalance testing | Sub-pixel channel smoothing limits sequence models |""",
        "latex": """\\begin{table}[t]
\\caption{Summary of core scientific findings, empirical evidence, statistical validation, and corresponding manuscript contributions.}
\\label{tab:contribution_summary}
\\centering
\\small
\\begin{tabular}{p{2.8cm} p{2.8cm} p{4.0cm} p{2.8cm}}
\\hline
\\textbf{Finding} & \\textbf{Evidence} & \\textbf{Statistical Support} & \\textbf{Paper Contribution} \\
\\hline
Geomorphological Control & 87.81\\% Gini feature importance in Random Forest & McNemar test yields $p = 0.2584$ comparing Terrain-only and Combined models & Topographic control over spatial boundaries (noting statistical power limits for small differences) \\
\\hline
Hydrological Memory & 3-month precipitation routing lag in coastal wetlands & Temporal cross-correlation analysis & Routing lag and cropland correlation analysis \\
\\hline
Sequence Model Collapse & F1 = 0.000 during normal dry periods for LSTM and Transformer & Diagnostic threshold sweeps and class imbalance testing & Sub-pixel channel smoothing limits sequence models \\
\\hline
\\end{tabular}
\\end{table}"""
    },
    "dataset_summary": {
        "title": "Table 6. Remote sensing and climate datasets used to construct the Indus Delta spatial-temporal point dataset (2018-2024). All datasets were accessed via Google Earth Engine (June 2026).",
        "markdown": """| Dataset | Platform | Resolution | Features Extracted | Hydrological Role |
| :--- | :--- | :--- | :--- | :--- |
| Sentinel-1 GRD | ESA Copernicus | 10 m (SAR) | VV backscatter | Ground-truth water label (VV < -15 dB) |
| Sentinel-2 MSI | ESA Copernicus | 10-20 m (Optical) | NDWI, MNDWI, NDVI | Independent manual validation only |
| SRTM DEM | NASA/USGS | 30 m (Static) | Elevation, slope, flow acc. | Static topographic routing features |
| HydroSHEDS | WWF | 15 arcsec | River distance | Channel proximity feature |
| CHIRPS | Climate Hazards Group | 0.05° (5.5 km) | Monthly total rainfall | Dynamic precipitation forcing |
| ERA5-Land | ECMWF | 9 km (Reanalysis) | Temperature, ET, PET | Catchment energy and evaporation |
| JRC Global SW | European Commission | 30 m (Static) | Occurrence probability | Stratified point sampling frame |""",
        "latex": """\\begin{table*}[t]
\\caption{Remote sensing and climate datasets used to construct the Indus Delta spatial-temporal point dataset (2018--2024). All datasets were accessed via Google Earth Engine (June 2026). Note: August 2018 is excluded from the temporal records due to 100\\% cloud cover in the Sentinel-2 optical validation imagery.}
\\label{tab:dataset_summary}
\\centering
\\small
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{p{2.5cm} p{2.5cm} p{2.0cm} p{3.2cm} p{4.8cm}}
\\hline
\\textbf{Dataset} & \\textbf{Platform} & \\textbf{Resolution} & \\textbf{Features Extracted} & \\textbf{Hydrological Role} \\
\\hline
Sentinel-1 GRD      & ESA Copernicus         & 10\\,m (SAR)           & VV backscatter              & Ground-truth water label (VV $<$ $-15$\\,dB) \\
Sentinel-2 MSI      & ESA Copernicus         & 10--20\\,m (Optical)   & NDWI, MNDWI, NDVI           & Independent manual validation only \\
SRTM DEM            & NASA/USGS              & 30\\,m (Static)        & Elevation, slope, flow acc. & Static topographic routing features \\
HydroSHEDS          & WWF                    & 15\\,arcsec            & River distance              & Channel proximity feature \\
CHIRPS              & Climate Hazards Group  & 0.05$^\\circ$ (5.5\\,km) & Monthly total rainfall     & Dynamic precipitation forcing \\
ERA5-Land           & ECMWF                  & 9\\,km (Reanalysis)    & Temperature, ET, PET        & Catchment energy and evaporation \\
JRC Global SW       & European Commission    & 30\\,m (Static)        & Occurrence probability      & Stratified point sampling frame \\
\\hline
\\end{tabular}%
}
\\end{table*}"""
    },
    "literature_review": {
        "title": "Table 7. Literature review matrix summarizing objectives, methods, and limitations of key studies in remote sensing flood mapping and hydrological AI.",
        "markdown": """| Author | Year | Journal | Objective | Method | Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Pekel et al. | 2016 | Nature | Global surface water occurrence mapping | Random Forest classification of Landsat series | Vulnerable to cloud occlusion; neglects temporal catchment memory |
| Twele et al. | 2016 | NHESS | Automated Sentinel-1 flood mapping | Split-based thresholding and fuzzy logic | Relies on concurrent SAR overpasses; ignores temporal climate memory |
| Kratzert et al. | 2018 | HESS | Rainfall-runoff streamflow prediction | Long Short-Term Memory (LSTM) network | Operates on lumped basin discharge; cannot map 2D inundation boundaries |
| Kratzert et al. | 2019 | HESS | Regionalization in ungauged basins | Entity-Aware LSTM (EA-LSTM) with static features | Focuses on lumped discharge points; ignores spatial routing networks |
| Wu et al. | 2020 | WIRES Water | Overview of flood forecasting | Ensemble numerical weather and hydrologic models | Neglects sub-pixel spatial boundaries and local geomorphology |
| Nearing et al. | 2020 | WRR | Assessment of ML in streamflow forecasting | LSTM neural network compared to process models | Lacks spatial context; cannot map flood boundaries |
| Tejedor et al. | 2015 | WRR | Delta network connectivity and topology | Graph-theoretic transport and vulnerability metrics | Focuses on network geometry; does not integrate dynamic climate triggers |
| Muñoz-Sabater et al. | 2021 | ESSD | Global high-resolution reanalysis dataset | ERA5-Land reanalysis model integration | Coarse resolution (9 km) fails to resolve sub-pixel channels |
| Feng et al. | 2023 | HESS | Physics-informed hydrologic modeling | Process-based differentiable modeling | High computational cost; parameter equifinality in ungauged areas |""",
        "latex": """\\begin{table*}[t]
\\caption{Literature review matrix summarizing objectives, methods, and limitations of key studies in remote sensing flood mapping and hydrological AI.}
\\label{tab:literature_review}
\\centering
\\footnotesize
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{p{2.0cm} c p{2.5cm} p{3.0cm} p{2.8cm} p{4.0cm}}
\\hline
\\textbf{Author} & \\textbf{Year} & \\textbf{Journal} & \\textbf{Objective} & \\textbf{Method} & \\textbf{Limitation} \\
\\hline
Pekel et al. & 2016 & Nature & Global surface water occurrence mapping & Random Forest classification of Landsat series & Vulnerable to cloud occlusion; neglects temporal catchment memory \\
Twele et al. & 2016 & NHESS & Automated Sentinel-1 flood mapping & Split-based thresholding and fuzzy logic & Relies on concurrent SAR overpasses; ignores temporal climate memory \\
Kratzert et al. & 2018 & HESS & Rainfall--runoff streamflow prediction & Long Short-Term Memory (LSTM) network & Operates on lumped basin discharge; cannot map 2D inundation boundaries \\
Kratzert et al. & 2019 & HESS & Regionalization in ungauged basins & Entity-Aware LSTM (EA-LSTM) with static features & Focuses on lumped discharge points; ignores spatial routing networks \\
Wu et al. & 2020 & WIRES Water & Overview of flood forecasting & Ensemble numerical weather and hydrologic models & Neglects sub-pixel spatial boundaries and local geomorphology \\
Nearing et al. & 2020 & WRR & Assessment of ML in streamflow forecasting & LSTM neural network compared to process models & Lacks spatial context; cannot map flood boundaries \\
Tejedor et al. & 2015 & WRR & Delta network connectivity and topology & Graph-theoretic transport and vulnerability metrics & Focuses on network geometry; does not integrate dynamic climate triggers \\
Muñoz-Sabater et al. & 2021 & ESSD & Global high-resolution reanalysis dataset & ERA5-Land reanalysis model integration & Coarse resolution (9 km) fails to resolve sub-pixel channels \\
Feng et al. & 2023 & HESS & Physics-informed hydrologic modeling & Process-based differentiable modeling & High computational cost; parameter equifinality in ungauged areas \\
\\hline
\\end{tabular}%
}
\\end{table*}"""
    }
}

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    for key, data in TABLES.items():
        # Save LaTeX file
        tex_path = os.path.join(OUT_DIR, f"{key}.tex")
        with open(tex_path, "w") as f:
            f.write(data["latex"])
            
        # Save Markdown file
        md_path = os.path.join(OUT_DIR, f"{key}.md")
        with open(md_path, "w") as f:
            f.write(f"### {data['title']}\n\n{data['markdown']}\n")
            
        print(f"Generated Table files for '{key}':")
        print(f"  - {tex_path}")
        print(f"  - {md_path}")

if __name__ == "__main__":
    main()
