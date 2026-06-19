import json
import numpy as np
import ee
from scipy.spatial.distance import cdist

def haversine_distance(coord1, coord2):
    # coord = [lon, lat]
    R = 6371.0088 * 1000.0 # radius of Earth in meters
    lon1, lat1 = np.radians(coord1[0]), np.radians(coord1[1])
    lon2, lat2 = np.radians(coord2[0]), np.radians(coord2[1])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def calculate_morans_i(coords, values):
    n = len(values)
    values = np.array(values, dtype=float)
    mean_val = np.mean(values)
    denom = np.sum((values - mean_val)**2)
    
    if denom == 0:
        return 0.0
        
    # Calculate spatial weight matrix (inverse distance)
    # Exclude self-weights (diagonal = 0)
    coords = np.array(coords)
    dists = cdist(coords, coords, metric='euclidean') # in degrees for weights
    
    weights = np.zeros_like(dists)
    non_zero = dists > 0
    weights[non_zero] = 1.0 / dists[non_zero]
    
    # Row normalize weights
    row_sums = np.sum(weights, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    weights = weights / row_sums
    
    num = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                num += weights[i, j] * (values[i] - mean_val) * (values[j] - mean_val)
                
    morans_i = num / denom
    return morans_i

def main():
    print("Initializing Google Earth Engine...")
    with open("ee-muzzamil12-2f811d1cb2cc.json", "r") as f:
        cred_data = json.load(f)
    credentials = ee.ServiceAccountCredentials(cred_data["client_email"], key_data=json.dumps(cred_data))
    ee.Initialize(credentials=credentials)
    
    roi = ee.Geometry.Rectangle([67.0, 23.0, 69.0, 24.8])
    jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    jrc_occ = jrc.select('occurrence').clip(roi)
    
    # Re-run the sampling logic to obtain representative points and their attributes
    water_mask = jrc_occ.gt(50)
    seasonal_mask = jrc_occ.gt(0).And(jrc_occ.lte(50))
    land_mask = jrc_occ.eq(0)
    
    print("Sampling stratified points from GEE...")
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
    features = all_points.getInfo()['features']
    print(f"Successfully retrieved {len(features)} points.")
    
    coords = []
    occ_vals = []
    
    # Also get elevation to compute spatial statistics on elevation
    dem = ee.Image("USGS/SRTMGL1_003").clip(roi)
    elev_samples = dem.reduceRegions(
        collection=all_points,
        reducer=ee.Reducer.first(),
        scale=30
    ).getInfo()['features']
    
    elevations = []
    for idx, f in enumerate(features):
        c = f['geometry']['coordinates']
        coords.append(c)
        
        # Determine target water class (occurrence value)
        occ = f['properties'].get('occurrence', 0)
        occ_vals.append(occ)
        
        # Elevation
        elev = elev_samples[idx]['properties'].get('first', 0.0)
        if elev is None:
            elev = 0.0
        elevations.append(elev)
        
    coords = np.array(coords)
    n = len(coords)
    
    # Calculate nearest neighbor distance for each point
    nn_dists = []
    for i in range(n):
        min_dist = float('inf')
        for j in range(n):
            if i != j:
                d = haversine_distance(coords[i], coords[j])
                if d < min_dist:
                    min_dist = d
        nn_dists.append(min_dist)
        
    nn_dists = np.array(nn_dists)
    mean_nn_dist = np.mean(nn_dists)
    min_nn_dist = np.min(nn_dists)
    max_nn_dist = np.max(nn_dists)
    
    # Calculate Moran's I
    moran_occ = calculate_morans_i(coords, occ_vals)
    moran_elev = calculate_morans_i(coords, elevations)
    
    print("\n" + "="*50)
    print("SPATIAL AUTOCORRELATION & DISTANCE ANALYSIS")
    print("="*50)
    print(f"Total points analyzed: {n}")
    print(f"Mean Nearest-Neighbor Distance: {mean_nn_dist/1000.0:.3f} km")
    print(f"Min Nearest-Neighbor Distance: {min_nn_dist/1000.0:.3f} km")
    print(f"Max Nearest-Neighbor Distance: {max_nn_dist/1000.0:.3f} km")
    print(f"Moran's I of Water Occurrence: {moran_occ:.4f}")
    print(f"Moran's I of Terrain Elevation: {moran_elev:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
