### Table 6. Remote sensing and climate datasets used to construct the Indus Delta spatial-temporal point dataset (2018-2024). All datasets were accessed via Google Earth Engine (June 2026).

| Dataset | Platform | Resolution | Features Extracted | Hydrological Role |
| :--- | :--- | :--- | :--- | :--- |
| Sentinel-1 GRD | ESA Copernicus | 10 m (SAR) | VV backscatter | Ground-truth water label (VV < -15 dB) |
| Sentinel-2 MSI | ESA Copernicus | 10-20 m (Optical) | NDWI, MNDWI, NDVI | Independent manual validation only |
| SRTM DEM | NASA/USGS | 30 m (Static) | Elevation, slope, flow acc. | Static topographic routing features |
| HydroSHEDS | WWF | 15 arcsec | River distance | Channel proximity feature |
| CHIRPS | Climate Hazards Group | 0.05° (5.5 km) | Monthly total rainfall | Dynamic precipitation forcing |
| ERA5-Land | ECMWF | 9 km (Reanalysis) | Temperature, ET, PET | Catchment energy and evaporation |
| JRC Global SW | European Commission | 30 m (Static) | Occurrence probability | Stratified point sampling frame |
