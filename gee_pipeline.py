import ee
import json
import time

def init_gee():
    # Load credentials
    with open("ee-muzzamil12-2f811d1cb2cc.json", "r") as f:
        cred_data = json.load(f)
        
    credentials = ee.ServiceAccountCredentials(
        cred_data["client_email"],
        key_data=json.dumps(cred_data)
    )
    ee.Initialize(credentials=credentials)
    print("Google Earth Engine successfully initialized.")

def get_indus_roi():
    # Bounding box for the Indus River Delta in Pakistan
    # Coordinates: [Min Longitude, Min Latitude, Max Longitude, Max Latitude]
    return ee.Geometry.Rectangle([67.0, 23.0, 69.0, 24.8])

def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
           qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask)

def process_optical(roi, start_date, end_date):
    print("Processing Sentinel-2 Optical Collection...")
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(roi)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
          .map(mask_s2_clouds))
          
    # Function to calculate indices
    def add_indices(img):
        # Bands: B3 (Green), B4 (Red), B8 (NIR), B11 (SWIR1), B12 (SWIR2)
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        ndwi = img.normalizedDifference(['B3', 'B8']).rename('NDWI')
        mndwi = img.normalizedDifference(['B3', 'B11']).rename('MNDWI')
        lswi = img.normalizedDifference(['B8', 'B11']).rename('LSWI')
        
        # Enhanced Vegetation Index (EVI)
        evi = img.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
                'NIR': img.select('B8'),
                'RED': img.select('B4'),
                'BLUE': img.select('B2')
            }).rename('EVI')
            
        return img.addBands([ndvi, ndwi, mndwi, lswi, evi])
        
    s2_with_indices = s2.map(add_indices)
    return s2_with_indices

def process_sar(roi, start_date, end_date):
    print("Processing Sentinel-1 SAR Collection...")
    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filterBounds(roi)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))
          
    def add_ratio(img):
        ratio = img.select('VH').subtract(img.select('VV')).rename('VH_VV_ratio')
        return img.addBands(ratio)
        
    s1_processed = s1.map(add_ratio)
    return s1_processed

def get_terrain_derivatives(roi):
    print("Extracting SRTM Terrain Derivatives...")
    dem = ee.Image("USGS/SRTMGL1_003").clip(roi)
    slope = ee.Terrain.slope(dem).rename("slope")
    aspect = ee.Terrain.aspect(dem).rename("aspect")
    
    # Load HydroSHEDS flow accumulation to represent flow paths
    flow_acc = ee.Image("WWF/HydroSHEDS/15ACC").clip(roi).rename("flow_accumulation")
    
    # Compute distance to permanent water bodies as a river distance proxy
    jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    permanent_water = jrc.select('occurrence').gt(80)
    dist_to_river = permanent_water.fastDistanceTransform().multiply(30).rename("dist_to_river")
    
    # Distance to Coast (Indus Delta shoreline boundary)
    sea_mask = jrc.select('occurrence').eq(0)
    dist_to_coast = sea_mask.fastDistanceTransform().multiply(30).rename("dist_to_coast")
    
    return dem.addBands([slope, aspect, flow_acc, dist_to_river, dist_to_coast])

def process_climate(roi, start_date, end_date):
    print("Processing Rainfall (CHIRPS) and Climate (ERA5-Land)...")
    chirps = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
              .filterBounds(roi)
              .filterDate(start_date, end_date)
              .select(['precipitation'], ['precipitation']))
              
    era5 = (ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
            .filterBounds(roi)
            .filterDate(start_date, end_date)
            .select(['temperature_2m', 'total_evaporation_sum', 'potential_evaporation_sum'], 
                    ['temperature', 'evaporation', 'potential_evaporation']))
            
    return chirps, era5

def main():
    init_gee()
    roi = get_indus_roi()
    
    start_date = '2022-08-01'
    end_date = '2022-10-31' # Capturing the extreme 2022 Pakistan Floods period
    
    # 1. Optical (Sentinel-2)
    s2_col = process_optical(roi, start_date, end_date)
    s2_median = s2_col.select(['NDVI', 'NDWI', 'MNDWI', 'LSWI', 'EVI']).median().clip(roi)
    
    # 2. SAR (Sentinel-1)
    s1_col = process_sar(roi, start_date, end_date)
    s1_median = s1_col.select(['VV', 'VH', 'VH_VV_ratio']).median().clip(roi)
    
    # 3. Terrain
    terrain = get_terrain_derivatives(roi)
    
    # 4. Climate
    chirps_col, era5_col = process_climate(roi, start_date, end_date)
    chirps_sum = chirps_col.sum().clip(roi).rename('total_precipitation')
    era5_mean = era5_col.mean().clip(roi)
    
    # 5. JRC Water Occurrence Label
    jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').clip(roi)
    
    # Merge all static and aggregated dynamic features into a single diagnostic image
    diagnostic_img = ee.Image.cat([
        s2_median,
        s1_median,
        terrain,
        chirps_sum,
        era5_mean,
        jrc
    ])
    
    print("\nHydrological Feature Inventory Image compiled successfully.")
    print("Band names in inventory image:")
    band_names = diagnostic_img.bandNames().getInfo()
    for name in band_names:
        print(f"  - {name}")
        
    # Print sample coordinate metadata to verify
    sample_point = ee.Geometry.Point([68.0, 24.0])
    sample_values = diagnostic_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=sample_point,
        scale=30
    ).getInfo()
    
    print("\nSample values at center coordinate [68.0E, 24.0N]:")
    for k, v in sample_values.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
