# Dataset Description

This document details the dataset compiled for the Indus Delta surface water mapping benchmark.

## Stratified Spatial Sampling
The dataset covers the Indus River Delta, Pakistan ($23.0^\circ\text{N}$ to $24.8^\circ\text{N}$, $67.0^\circ\text{E}$ to $69.0^\circ\text{E}$) with 1,000 spatial point locations. The points are stratified based on the historical Joint Research Centre (JRC) Global Surface Water occurrence dataset:
- **300 points** in Permanent Water (JRC occurrence probability $> 50\%$)
- **300 points** in Seasonal Wetlands (JRC occurrence probability between $1\%$ and $50\%$)
- **400 points** in Dry Land (JRC occurrence probability of $0\%$)

## Temporal Scope
Data is compiled monthly for the years 2018 to 2024, representing 83 sequential months (excluding August 2018 due to 100% cloud cover). This yields a total of 83,000 monthly point-level records.

## Inputs and Targets
The database consists of the following features at each coordinate $i$ and month $t$:
- **Dynamic Climate Input Forcings:**
  - CHIRPS Monthly precipitation (mm)
  - ERA5-Land Temperature (K), Total Evaporation ($ET$, m of water equivalent), and Potential Evapotranspiration ($PET$, m of water equivalent).
- **Static Topographic Terrain Parameters (SRTM 30m):**
  - Elevation ($z$, m)
  - Slope (degrees)
  - Flow accumulation (dimensionless drainage area proxy)
  - Euclidean distance to the nearest river channel ($D_{\text{river}}$, m)
  - Euclidean distance to the coastline ($D_{\text{coast}}$, m)
- **Target Inundation Label ($y \in \{0, 1\}$):**
  - Derived from Sentinel-1 VV active microwave polarization backscatter:
    $$y_i(t) = \begin{cases} 1 & \text{if } VV_i(t) < -15\text{ dB} \\ 0 & \text{otherwise} \end{cases}$$
  - Validated against 100 cloud-free manually digitized Sentinel-2 optical masks with an overall accuracy of $92.4\%$.

## Train-Test Splitting and Validation
To evaluate model performance under extreme events, a temporal split is enforced:
- **Training and Validation:** Years 2018--2021 (normal years and moderate seasonal floods). Points are split at the point level ($80\%$ train, $20\%$ validation) to prevent spatial autocorrelation leakage.
- **Testing:** Years 2022--2024 (containing the historic 2022 Pakistan monsoon flood event).
