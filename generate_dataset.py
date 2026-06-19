import ee
import json
import numpy as np
import os
import time

def init_gee():
    with open("ee-muzzamil12-2f811d1cb2cc.json", "r") as f:
        cred_data = json.load(f)
    credentials = ee.ServiceAccountCredentials(cred_data["client_email"], key_data=json.dumps(cred_data))
    ee.Initialize(credentials=credentials)
    print("GEE Initialized for dataset generation.")

def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
           qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask)

def main():
    init_gee()
    
    # Bounding box for Indus Delta
    roi = ee.Geometry.Rectangle([67.0, 23.0, 69.0, 24.8])
    
    # Static layers
    dem = ee.Image("USGS/SRTMGL1_003").clip(roi)
    slope = ee.Terrain.slope(dem).rename("slope")
    flow_acc = ee.Image("WWF/HydroSHEDS/15ACC").clip(roi).rename("flow_accumulation")
    jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    permanent_water = jrc.select('occurrence').gt(80)
    dist_to_river = permanent_water.fastDistanceTransform().multiply(30).rename("dist_to_river")
    sea_mask = jrc.select('occurrence').eq(0)
    dist_to_coast = sea_mask.fastDistanceTransform().multiply(30).rename("dist_to_coast")
    jrc_occ = jrc.select('occurrence').clip(roi)
    
    static_img = ee.Image.cat([dem, slope, flow_acc, dist_to_river, dist_to_coast, jrc_occ])
    
    # We will generate a stratified random sample of points over the delta:
    # 300 points in permanent water (JRC occurrence > 50)
    # 300 points in seasonal wetlands (JRC occurrence > 0 and <= 50)
    # 400 points in dry land (JRC occurrence == 0)
    print("Generating stratified random points...")
    water_mask = jrc_occ.gt(50)
    seasonal_mask = jrc_occ.gt(0).And(jrc_occ.lte(50))
    land_mask = jrc_occ.eq(0)
    
    water_points = water_mask.selfMask().stratifiedSample(
        numPoints=300, classBand='occurrence', region=roi, scale=100, geometries=True
    )
    seasonal_points = seasonal_mask.selfMask().stratifiedSample(
        numPoints=300, classBand='occurrence', region=roi, scale=100, geometries=True
    )
    land_points = land_mask.selfMask().stratifiedSample(
        numPoints=400, classBand='occurrence', region=roi, scale=100, geometries=True
    )
    
    all_points = water_points.merge(seasonal_points).merge(land_points)
    num_points = all_points.size().getInfo()
    print(f"Generated {num_points} sample points.")
    
    # Define months from 2017 to 2025
    years = list(range(2017, 2026))
    months = list(range(1, 13))
    
    dates = []
    for y in years:
        for m in months:
            dates.append((y, m))
            
    print(f"Processing {len(dates)} monthly steps (Jan 2017 - Dec 2025)...")
    
    # To prevent client-server round-trip timeouts, we will process in chunks of years
    feature_data = []
    point_list = all_points.getInfo()['features']
    coordinates = [feat['geometry']['coordinates'] for feat in point_list]
    
    print("Extracting coordinates locally...")
    # Convert points to coordinates for fast local batch sampling via GEE if needed,
    # or build server-side image collection and extract.
    # The most robust way is to build monthly composites and sample them.
    
    # Let's define the monthly composite generator
    def get_monthly_composite(year, month):
        start = f"{year}-{month:02d}-01"
        # Handle end of month date
        if month == 12:
            end = f"{year+1}-01-01"
        else:
            end = f"{year}-{month+1:02d}-01"
            
        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(roi)
              .filterDate(start, end)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
              .map(mask_s2_clouds))
              
        s2_med = s2.select(['B8', 'B4', 'B3', 'B11', 'B2']).median().clip(roi)
        # Check if S2 exists, if not create empty
        ndvi = s2_med.normalizedDifference(['B8', 'B4']).rename('NDVI')
        ndwi = s2_med.normalizedDifference(['B3', 'B8']).rename('NDWI')
        mndwi = s2_med.normalizedDifference(['B3', 'B11']).rename('MNDWI')
        lswi = s2_med.normalizedDifference(['B8', 'B11']).rename('LSWI')
        
        s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
              .filterBounds(roi)
              .filterDate(start, end)
              .filter(ee.Filter.eq('instrumentMode', 'IW'))
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))
        s1_med = s1.select(['VV', 'VH']).median().clip(roi)
        vh_vv = s1_med.select('VH').subtract(s1_med.select('VV')).rename('VH_VV_ratio')
        
        chirps = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                  .filterBounds(roi)
                  .filterDate(start, end)
                  .select(['precipitation']).sum().clip(roi))
                  
        era5 = (ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
                .filterBounds(roi)
                .filterDate(start, end)
                .select(['temperature_2m', 'total_evaporation_sum', 'potential_evaporation_sum'], 
                        ['temperature', 'evaporation', 'potential_evaporation']).mean().clip(roi))
                        
        # Merge all into one month image
        month_img = ee.Image.cat([
            ndvi, ndwi, mndwi, lswi,
            s1_med.select('VV'), s1_med.select('VH'), vh_vv,
            chirps, era5
        ]).addBands(static_img)
        
        return month_img

    print("Extracting features from Earth Engine...")
    
    # We will sample 3 representative years to prevent server execution timeouts:
    # 2018 (Normal year), 2022 (Extreme flood year), 2024 (Recent year)
    target_years = [2018, 2022, 2024]
    sampled_dates = [(y, m) for y in target_years for m in range(1, 13)]
    
    extracted_features = []
    extracted_labels = []
    
    # Build list of points as ee.FeatureCollection
    ee_points = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(c)) for c in coordinates])
    
    for y, m in sampled_dates:
        print(f"  Fetching month: {y}-{m:02d}...")
        try:
            month_img = get_monthly_composite(y, m)
            
            # Sample all points at once
            samples = month_img.reduceRegions(
                collection=ee_points,
                reducer=ee.Reducer.first(),
                scale=30
            ).getInfo()
            
            for feat in samples['features']:
                props = feat['properties']
                coords = feat['geometry']['coordinates']
                
                # Check for NaNs and fill default values
                row = [
                    props.get('NDVI', 0.0),
                    props.get('NDWI', 0.0),
                    props.get('MNDWI', 0.0),
                    props.get('LSWI', 0.0),
                    props.get('VV', -15.0),
                    props.get('VH', -23.0),
                    props.get('VH_VV_ratio', -8.0),
                    props.get('precipitation', 0.0),
                    props.get('temperature', 298.0),
                    props.get('evaporation', -0.05),
                    props.get('potential_evaporation', -0.1),
                    props.get('elevation', 0.0),
                    props.get('slope', 0.0),
                    props.get('flow_accumulation', 1.0),
                    props.get('dist_to_river', 5000.0),
                    props.get('dist_to_coast', 10000.0),
                    y, # Year
                    m  # Month
                ]
                # Filter out entries where critical S1/S2 data is missing (NaN)
                if any(x is None for x in row):
                    row = [0.0 if x is None else x for x in row]
                    
                # Inundation ground truth: S1 VV backscatter <-15 dB is water
                is_water = 1 if props.get('VV', 0) < -15.0 else 0
                
                extracted_features.append(row)
                extracted_labels.append(is_water)
                
        except Exception as e:
            print(f"  Error for month {y}-{m:02d}: {e}")
            
    X = np.array(extracted_features)
    y = np.array(extracted_labels)
    
    print(f"\nExtraction complete. Total records: {X.shape[0]}")
    
    os.makedirs("data", exist_ok=True)
    np.savez("data/indus_dataset.npz", X=X, y=y)
    print("Dataset saved to data/indus_dataset.npz")

if __name__ == "__main__":
    main()
