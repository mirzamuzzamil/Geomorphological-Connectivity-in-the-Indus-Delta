import ee
import json
import numpy as np
import os
import concurrent.futures

def init_gee():
    with open("ee-muzzamil12-2f811d1cb2cc.json", "r") as f:
        cred_data = json.load(f)
    credentials = ee.ServiceAccountCredentials(cred_data["client_email"], key_data=json.dumps(cred_data))
    ee.Initialize(credentials=credentials)
    print("GEE Initialized globally for multi-threaded generation.")

# Initialize immediately at global level
init_gee()

roi = ee.Geometry.Rectangle([67.0, 23.0, 69.0, 24.8])
static_list = None
ee_points = None

def get_monthly_composite(year, month):
    start = f"{year}-{month:02d}-01"
    if month == 12:
        end = f"{year+1}-01-01"
    else:
        end = f"{year}-{month+1:02d}-01"
        
    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filterBounds(roi)
          .filterDate(start, end)
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')))
    s1_med = s1.select(['VV']).median().clip(roi)
    
    chirps = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
              .filterBounds(roi)
              .filterDate(start, end)
              .select(['precipitation']).sum().clip(roi))
              
    era5 = (ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
            .filterBounds(roi)
            .filterDate(start, end)
            .select(['temperature_2m', 'total_evaporation_sum', 'potential_evaporation_sum'], 
                    ['temperature', 'evaporation', 'potential_evaporation']).mean().clip(roi))
                    
    month_img = ee.Image.cat([s1_med, chirps, era5])
    return month_img

def fetch_month(date_tuple):
    y, m = date_tuple
    if y == 2018 and m == 8:
        return (y, m, [], [], [])
        
    try:
        month_img = get_monthly_composite(y, m)
        samples = month_img.reduceRegions(
            collection=ee_points,
            reducer=ee.Reducer.first(),
            scale=30
        ).getInfo()
        
        features = []
        labels = []
        vv_vals = []
        
        for idx, feat in enumerate(samples['features']):
            props = feat['properties']
            static_props = static_list[idx]
            
            vv_val = props.get('VV', -15.0)
            if vv_val is None:
                vv_val = -15.0
                
            row = [
                props.get('precipitation', 0.0),
                props.get('temperature', 298.0),
                props.get('evaporation', -0.05),
                props.get('potential_evaporation', -0.1),
                static_props.get('elevation', 0.0),
                static_props.get('slope', 0.0),
                static_props.get('flow_accumulation', 1.0),
                static_props.get('dist_to_river', 5000.0),
                static_props.get('dist_to_coast', 10000.0),
                y,
                m
            ]
            row = [0.0 if x is None else x for x in row]
            is_water = 1 if vv_val < -15.0 else 0
            
            features.append(row)
            labels.append(is_water)
            vv_vals.append(vv_val)
            
        print(f"  Successfully fetched: {y}-{m:02d}")
        return (y, m, features, labels, vv_vals)
    except Exception as e:
        print(f"  Error fetching {y}-{m:02d}: {e}")
        return (y, m, [], [], [])

def main():
    global static_list, ee_points
    
    # Static layers setup
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
    
    # Stratified Sampling
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
    
    point_list = all_points.getInfo()['features']
    coordinates = [feat['geometry']['coordinates'] for feat in point_list]
    ee_points = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(c)) for c in coordinates])
    
    print("Extracting static features once...")
    static_samples = static_img.reduceRegions(
        collection=ee_points,
        reducer=ee.Reducer.first(),
        scale=30
    ).getInfo()
    static_list = [feat['properties'] for feat in static_samples['features']]
    print("Static features extracted.")
    
    target_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    sampled_dates = [(y, m) for y in target_years for m in range(1, 13)]
    sampled_dates = [d for d in sampled_dates if not (d[0] == 2018 and d[1] == 8)]
    
    print(f"Beginning multi-threaded download for {len(sampled_dates)} months with 12 workers...")
    
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(fetch_month, date): date for date in sampled_dates}
        for future in concurrent.futures.as_completed(futures):
            date = futures[future]
            try:
                res = future.result()
                all_results.append(res)
            except Exception as exc:
                print(f"Date {date} generated an exception: {exc}")
                
    all_results.sort(key=lambda x: (x[0], x[1]))
    
    final_features = []
    final_labels = []
    final_vv = []
    
    for y, m, feats, lbls, vvs in all_results:
        if len(feats) == 0:
            continue
        final_features.extend(feats)
        final_labels.extend(lbls)
        final_vv.extend(vvs)
        
    X = np.array(final_features)
    y = np.array(final_labels)
    vv = np.array(final_vv)
    
    print(f"\nAll downloads completed. Total records compiled: {X.shape[0]}")
    
    os.makedirs("data", exist_ok=True)
    np.savez("data/indus_dataset_v2.npz", X=X, y=y, vv=vv)
    print("Complete dataset saved to data/indus_dataset_v2.npz")

if __name__ == "__main__":
    main()
