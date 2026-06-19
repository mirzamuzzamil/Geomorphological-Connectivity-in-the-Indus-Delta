# Google Earth Engine Data Collections

This document specifies the exact Google Earth Engine (GEE) collection identifiers and parameterizations used to extract our spatial-temporal point dataset.

## Data Collections Summary

| Data Layer | GEE Collection ID | Resolution | Key Bands Used | Temporal Filtering / Composite |
| --- | --- | --- | --- | --- |
| **Sentinel-1 SAR** | `COPERNICUS/S1_GRD` | 10 m | `VV`, `VH` | Monthly median composite, IW mode, descending orbit |
| **Sentinel-2 MSI** | `COPERNICUS/S2_SR_HARMONIZED` | 10--20 m | `B2`, `B3`, `B4`, `B8`, `B11`, `B12`, `QA60` | Monthly median composite, cloud-masked using QA60 |
| **SRTM DEM** | `USGS/SRTMGL1_003` | 30 m | `elevation` | Static elevation, slope, and flow accumulation calculations |
| **CHIRPS Rainfall** | `UCSB-CHG/CHIRPS/DAILY` | 0.05$^\circ$ | `precipitation` | Summed monthly total precipitation |
| **ERA5-Land** | `ECMWF/ERA5_LAND/MONTHLY_AGGR` | 9 km | `temperature_2m`, `total_evaporation_sum`, `potential_evapotranspiration_sum` | Monthly aggregated climate reanalysis |
| **JRC occurrence** | `JRC/GSW1_4/GlobalSurfaceWater` | 30 m | `occurrence` | Static JRC GSW occurrence probability |

## Preprocessing Details
1. **Sentinel-1 Filtering:** Filtered for descending orbit pass, Instrument Mode: 'IW', and polarizations: 'VV' and 'VH'. Refined Lee filter was applied to reduce speckle noise prior to thresholding.
2. **Sentinel-2 QA Cloud Masking:** Pixels with bit 10 and bit 11 set in the `QA60` band were masked out to remove clouds and cirrus.
3. **Topographic Feature Calculations:** Elevation, slope, and flow accumulation were derived using the `ee.Terrain` package. Spatial distance to river networks (derived from flow accumulation thresholds $> 5000$) and distance to coast (derived from JRC land mask) were calculated using Euclidean distance functions on the Earth Engine server.
