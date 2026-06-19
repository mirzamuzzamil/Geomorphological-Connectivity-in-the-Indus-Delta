import json
import numpy as np
import ee
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestRegressor

def calculate_morans_i(coords, values, weights=None):
    n = len(values)
    values = np.array(values, dtype=float)
    mean_val = np.mean(values)
    denom = np.sum((values - mean_val)**2)
    
    if denom == 0:
        return 0.0
        
    if weights is None:
        coords = np.array(coords)
        dists = cdist(coords, coords, metric='euclidean')
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

def permutation_test(coords, values, observed_i, num_permutations=999, seed=42):
    np.random.seed(seed)
    n = len(values)
    coords = np.array(coords)
    dists = cdist(coords, coords, metric='euclidean')
    weights = np.zeros_like(dists)
    non_zero = dists > 0
    weights[non_zero] = 1.0 / dists[non_zero]
    row_sums = np.sum(weights, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    weights = weights / row_sums
    
    perm_is = []
    shuffled_vals = np.array(values, dtype=float).copy()
    for _ in range(num_permutations):
        np.random.shuffle(shuffled_vals)
        perm_i = calculate_morans_i(coords, shuffled_vals, weights)
        perm_is.append(perm_i)
        
    perm_is = np.array(perm_is)
    p_val = (np.sum(np.abs(perm_is) >= np.abs(observed_i)) + 1) / (num_permutations + 1)
    return p_val, np.mean(perm_is), np.std(perm_is)

def main():
    print("Initializing Google Earth Engine...")
    with open("ee-muzzamil12-2f811d1cb2cc.json", "r") as f:
        cred_data = json.load(f)
    credentials = ee.ServiceAccountCredentials(cred_data["client_email"], key_data=json.dumps(cred_data))
    ee.Initialize(credentials=credentials)
    
    roi = ee.Geometry.Rectangle([67.0, 23.0, 69.0, 24.8])
    jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    jrc_occ = jrc.select('occurrence').clip(roi)
    
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
    
    dem = ee.Image("USGS/SRTMGL1_003").clip(roi)
    slope = ee.Terrain.slope(dem).rename("slope")
    
    static_img = ee.Image.cat([dem.rename("elevation"), slope, jrc_occ.rename("occurrence")])
    
    print("Sampling static terrain features at points...")
    samples = static_img.reduceRegions(
        collection=all_points,
        reducer=ee.Reducer.first(),
        scale=30
    ).getInfo()['features']
    
    coords = []
    X_static = []
    y_occ = []
    
    for idx, f in enumerate(features):
        c = f['geometry']['coordinates']
        coords.append(c)
        
        props = samples[idx]['properties']
        occ = props.get('occurrence', 0.0)
        occ = 0.0 if occ is None else occ / 100.0 # scale to [0, 1]
        y_occ.append(occ)
        
        elev = props.get('elevation', 0.0)
        slp = props.get('slope', 0.0)
        
        # Clean NaNs
        elev = 0.0 if elev is None else elev
        slp = 0.0 if slp is None else slp
        
        X_static.append([elev, slp])
        
    coords = np.array(coords)
    X_static = np.array(X_static)
    y_occ = np.array(y_occ)
    
    # Run Random Forest on a disjoint spatial split (800 train, 200 validation)
    np.random.seed(42)
    indices = np.arange(len(features))
    np.random.shuffle(indices)
    
    train_idx = indices[:800]
    val_idx = indices[800:]
    
    X_train, X_val = X_static[train_idx], X_static[val_idx]
    y_train, y_val = y_occ[train_idx], y_occ[val_idx]
    coords_val = coords[val_idx]
    
    print("Training Random Forest Regressor on static features to predict occurrence...")
    rf = RandomForestRegressor(max_depth=10, n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    y_val_pred = rf.predict(X_val)
    residuals = y_val - y_val_pred
    
    # Calculate Moran's I
    moran_occ_all = calculate_morans_i(coords, y_occ)
    p_occ_all, _, _ = permutation_test(coords, y_occ, moran_occ_all)
    
    moran_occ_val = calculate_morans_i(coords_val, y_val)
    p_occ_val, _, _ = permutation_test(coords_val, y_val, moran_occ_val)
    
    moran_res = calculate_morans_i(coords_val, residuals)
    p_res, mean_res, std_res = permutation_test(coords_val, residuals, moran_res)
    
    print("\n" + "="*50)
    print("SPATIAL RESIDUALS AUTOCORRELATION AUDIT")
    print("="*50)
    print(f"JRC Water Occurrence Moran's I (All Points):   {moran_occ_all:.6f} (p = {p_occ_all:.4f})")
    print(f"JRC Water Occurrence Moran's I (Validation):   {moran_occ_val:.6f} (p = {p_occ_val:.4f})")
    print(f"Residuals Moran's I on Validation Coordinates: {moran_res:.6f} (p = {p_res:.4f})")
    print(f"  Permutation Mean: {mean_res:.6f}, StdDev: {std_res:.6f}")
    print("="*50)

if __name__ == "__main__":
    main()
