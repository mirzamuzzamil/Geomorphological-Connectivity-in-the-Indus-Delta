### Table 4. Leakage audit: data source roles and leakage mitigation strategies.

| Data Source | Input | Target | Leakage Mitigation |
| :--- | :---: | :---: | :--- |
| Sentinel-1 VV | No | Yes | Excluded from inputs; used only to derive ground truth labels. |
| Sentinel-1 VH | No | No | Excluded; used only for supplementary validation. |
| Sentinel-2 Optical | No | No | Excluded; used for independent manual target validation. |
| ERA5-Land | Yes | No | Coarse resolution (9 km) prevents coordinate memorization. |
| CHIRPS Rainfall | Yes | No | Coarse resolution (5.5 km) prevents coordinate memorization. |
| SRTM DEM | Yes | No | Static; temporal dynamics learned from climate inputs only. |
