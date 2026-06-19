import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Ensure parent directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.utils import load_indus_dataset, reconstruct_grid, compute_metrics

def main():
    dataset_path = "data/indus_dataset.npz"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}. Please run with full dataset.")
        sys.exit(1)
        
    X_raw, y_raw = load_indus_dataset(dataset_path)
    
    # Raw features in X_raw:
    # 7: precipitation, 8: temperature, 9: evaporation, 10: potential_evaporation
    # 11: elevation, 12: slope, 13: flow_accumulation, 14: dist_to_river, 15: dist_to_coast
    
    X_grid_all, y_grid = reconstruct_grid(X_raw, y_raw, feature_dim=16, is_v2=False)
    
    climate_indices = [7, 8, 9, 10]
    terrain_indices = [11, 12, 13, 14, 15]
    combined_indices = climate_indices + terrain_indices
    
    y_train = np.concatenate([y_grid[:, 0, :], y_grid[:, 2, :]], axis=0).flatten()
    y_test = y_grid[:, 1, :].flatten() # 2022 Flood Period
    
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
    
    print("\n" + "="*65)
    print("EXPERIMENT 4: FEATURE SPLIT ABLATION REPORT")
    print("  Model: Random Forest | Evaluated on 2022 Flood Period")
    print("="*65)
    print(f"{'Feature Split':<15} | {'F1':<6} | {'IoU':<6} | {'Recall':<6} | {'Prec':<6} | {'AUC':<6}")
    print("-"*65)
    print(f"{'Terrain Only':<15} | {metrics_terrain['f1']:.3f} | {metrics_terrain['iou']:.3f} | {metrics_terrain['recall']:.3f} | {metrics_terrain['precision']:.3f} | {metrics_terrain['auc']:.3f}")
    print(f"{'Climate Only':<15} | {metrics_climate['f1']:.3f} | {metrics_climate['iou']:.3f} | {metrics_climate['recall']:.3f} | {metrics_climate['precision']:.3f} | {metrics_climate['auc']:.3f}")
    print(f"{'Combined':<15} | {metrics_combined['f1']:.3f} | {metrics_combined['iou']:.3f} | {metrics_combined['recall']:.3f} | {metrics_combined['precision']:.3f} | {metrics_combined['auc']:.3f}")
    
if __name__ == "__main__":
    main()
