# /// script
# dependencies = [
#   "scikit-learn",
#   "numpy",
#   "scipy"
# ]
# ///

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

def compute_metrics(y_true, y_pred, y_prob):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "auc": auc
    }

def main():
    data = np.load("data/indus_dataset.npz")
    X_raw = data['X']
    y_raw = data['y']
    
    # Raw features in X_raw:
    # 7: precipitation, 8: temperature, 9: evaporation, 10: potential_evaporation
    # 11: elevation, 12: slope, 13: flow_accumulation, 14: dist_to_river, 15: dist_to_coast
    
    years = X_raw[:, 16]
    months = X_raw[:, 17]
    
    target_months = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]
    num_points = 1000
    num_months = len(target_months)
    
    year_map = {2018: 0, 2022: 1, 2024: 2}
    month_map = {m: idx for idx, m in enumerate(target_months)}
    
    # 1. Terrain Only (indices 11-15)
    # 2. Climate Only (indices 7-10)
    # 3. Combined (indices 7-15)
    
    X_grid_all = np.zeros((num_points, 3, num_months, 16))
    y_grid = np.zeros((num_points, 3, num_months))
    
    counter = {}
    for i in range(X_raw.shape[0]):
        yr = int(years[i])
        mo = int(months[i])
        if yr not in year_map or mo not in month_map:
            continue
        y_idx = year_map[yr]
        m_idx = month_map[mo]
        
        key = (yr, mo)
        p_idx = counter.get(key, 0)
        if p_idx < num_points:
            X_grid_all[p_idx, y_idx, m_idx, :] = X_raw[i, :16]
            y_grid[p_idx, y_idx, m_idx] = y_raw[i]
            counter[key] = p_idx + 1

    # Feature splits
    # indices mapping:
    # Climate: 7, 8, 9, 10
    # Terrain: 11, 12, 13, 14, 15
    climate_indices = [7, 8, 9, 10]
    terrain_indices = [11, 12, 13, 14, 15]
    combined_indices = climate_indices + terrain_indices
    
    y_train = np.concatenate([y_grid[:, 0, :], y_grid[:, 2, :]], axis=0).flatten()
    y_test = y_grid[:, 1, :].flatten() # 2022 Flood Period
    
    # Split training set for validation
    split_idx = int(y_train.shape[0] * 0.8)
    
    def run_ablation(feat_indices, name):
        X_train_full = np.concatenate([X_grid_all[:, 0, :, :], X_grid_all[:, 2, :, :]], axis=0)[:, :, feat_indices].reshape(-1, len(feat_indices))
        X_test_full = X_grid_all[:, 1, :, :][:, :, feat_indices].reshape(-1, len(feat_indices))
        
        X_tr = X_train_full[:split_idx]
        y_tr = y_train[:split_idx]
        
        rf = RandomForestClassifier(max_depth=10, n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        
        preds = rf.predict(X_test_full)
        probs = rf.predict_proba(X_test_full)[:, 1]
        
        return compute_metrics(y_test, preds, probs)
        
    metrics_terrain = run_ablation(terrain_indices, "Terrain Only")
    metrics_climate = run_ablation(climate_indices, "Climate Only")
    metrics_combined = run_ablation(combined_indices, "Combined")
    
    print("\n" + "="*60)
    print("EXPERIMENT 4: FEATURE SPLIT ABLATION REPORT")
    print("  Model: Random Forest | Evaluated on 2022 Flood Period")
    print("="*60)
    print(f"{'Feature Split':<15} | {'F1':<6} | {'IoU':<6} | {'Recall':<6} | {'Prec':<6} | {'AUC':<6}")
    print("-"*60)
    print(f"{'Terrain Only':<15} | {metrics_terrain['f1']:.3f} | {metrics_terrain['iou']:.3f} | {metrics_terrain['recall']:.3f} | {metrics_terrain['precision']:.3f} | {metrics_terrain['auc']:.3f}")
    print(f"{'Climate Only':<15} | {metrics_climate['f1']:.3f} | {metrics_climate['iou']:.3f} | {metrics_climate['recall']:.3f} | {metrics_climate['precision']:.3f} | {metrics_climate['auc']:.3f}")
    print(f"{'Combined':<15} | {metrics_combined['f1']:.3f} | {metrics_combined['iou']:.3f} | {metrics_combined['recall']:.3f} | {metrics_combined['precision']:.3f} | {metrics_combined['auc']:.3f}")
    
if __name__ == "__main__":
    main()
