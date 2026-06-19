import ee
import json
import math

def init_gee():
    with open("ee-muzzamil12-2f811d1cb2cc.json", "r") as f:
        cred_data = json.load(f)
    credentials = ee.ServiceAccountCredentials(cred_data["client_email"], key_data=json.dumps(cred_data))
    ee.Initialize(credentials=credentials)

def mean(values):
    return sum(values) / len(values)

def std_dev(values, mu):
    variance = sum((x - mu) ** 2 for x in values) / len(values)
    return math.sqrt(variance)

def correlation(x, y):
    if len(x) != len(y) or len(x) == 0:
        return 0.0
    mu_x, mu_y = mean(x), mean(y)
    std_x, std_y = std_dev(x, mu_x), std_dev(y, mu_y)
    if std_x == 0 or std_y == 0:
        return 0.0
    covariance = sum((x[i] - mu_x) * (y[i] - mu_y) for i in range(len(x))) / len(x)
    return covariance / (std_x * std_y)

def lag_correlation(x, y, lag):
    if lag > 0:
        x_sig = x[:-lag]
        y_sig = y[lag:]
    elif lag < 0:
        x_sig = x[-lag:]
        y_sig = y[:lag]
    else:
        x_sig = x
        y_sig = y
    return correlation(x_sig, y_sig)

def main():
    init_gee()
    
    # Points representing different hydrological dynamics
    points = {
        "A_Floodplain": ee.Geometry.Point([68.01, 24.25]),
        "B_TidalWetland": ee.Geometry.Point([67.45, 23.95]),
        "C_Agriculture": ee.Geometry.Point([68.45, 24.45])
    }
    
    start_date = '2017-01-01'
    end_date = '2025-12-31'
    
    # Load Collections with robust filters
    chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filterDate(start_date, end_date)
    era5 = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR').filterDate(start_date, end_date)
    
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
           .filterDate(start_date, end_date) \
           .filter(ee.Filter.eq('instrumentMode', 'IW')) \
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
           
    print("\nRunning Exploratory Hydrological Analysis across Indus Delta Sample Points (2017-2025)...")
    
    # We will sample 36 months to verify lag patterns (2022-2024)
    months = []
    for y in range(2022, 2025):
        for m in range(1, 13):
            months.append((f"{y}-{m:02d}-01", f"{y}-{m:02d}-28"))
            
    print(f"Sampling {len(months)} monthly intervals...")
    
    for name, geom in points.items():
        print(f"\nAnalyzing Point: {name}...")
        
        rain_series = []
        sm_series = []
        water_series = []
        
        elevation = ee.Image("USGS/SRTMGL1_003").reduceRegion(ee.Reducer.mean(), geom, 30).getInfo().get('elevation', 0)
        print(f"  Elevation: {elevation} m")
        
        for start, end in months:
            # 1. Rainfall
            p_val = chirps.filterDate(start, end).select('precipitation').sum().reduceRegion(ee.Reducer.mean(), geom, 5000).getInfo().get('precipitation', 0)
            rain_series.append(p_val if p_val is not None else 0.0)
            
            # 2. Soil moisture (volumetric soil water layer 1)
            sm_val = era5.filterDate(start, end).select('volumetric_soil_water_layer_1').mean().reduceRegion(ee.Reducer.mean(), geom, 9000).getInfo().get('volumetric_soil_water_layer_1', 0)
            sm_series.append(sm_val if sm_val is not None else 0.0)
            
            # 3. Water occurrence proxy (SAR VV)
            # Filter S1 collection to active month first
            s1_subset = s1.filterDate(start, end)
            size = s1_subset.size().getInfo()
            if size > 0:
                vv_val = s1_subset.select('VV').mean().reduceRegion(ee.Reducer.mean(), geom, 30).getInfo().get('VV', -15.0)
                water_series.append(-vv_val if vv_val is not None else 15.0)
            else:
                water_series.append(15.0)
            
        # Clean series of None values
        rain_series = [r if r is not None else 0.0 for r in rain_series]
        sm_series = [s if s is not None else 0.0 for s in sm_series]
        water_series = [w if w is not None else 15.0 for w in water_series]
        
        # Computations
        r_rain_water = correlation(rain_series, water_series)
        r_sm_water = correlation(sm_series, water_series)
        
        print(f"  Correlation (Rainfall vs. Water Proxy): {r_rain_water:.3f}")
        print(f"  Correlation (Soil Moisture vs. Water Proxy): {r_sm_water:.3f}")
        
        # Lag correlation analysis (Precipitation leading Water Proxy)
        print("  Rainfall Lag Correlations (Rain leading Water):")
        best_lag = 0
        max_corr = -2.0
        for lag in [0, 1, 2, 3]:
            r_lag = lag_correlation(rain_series, water_series, lag)
            print(f"    Lag {lag} month(s): {r_lag:.3f}")
            if r_lag > max_corr:
                max_corr = r_lag
                best_lag = lag
                
        print(f"  Optimal Lag: {best_lag} month(s) (Correlation = {max_corr:.3f})")
        
        # Soil Moisture Autocorrelation (Moisture Memory)
        print("  Moisture Memory Autocorrelation:")
        for lag in [1, 2, 3]:
            r_mem = lag_correlation(sm_series, sm_series, lag)
            print(f"    Lag {lag} month(s): {r_mem:.3f}")

if __name__ == "__main__":
    main()
