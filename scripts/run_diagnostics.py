import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier

# Ensure parent directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.lstm import LSTMBaseline
from scripts.utils import load_indus_dataset, reconstruct_grid, compute_metrics

def main():
    # Detect dataset
    dataset_path = "data/indus_dataset.npz"
    if not os.path.exists(dataset_path):
        print(f"Error: Full dataset required at data/indus_dataset.npz for data leakage and feature importance audits.")
        sys.exit(1)
        
    print(f"Running diagnostics using: {dataset_path}")
    X_raw, y_raw = load_indus_dataset(dataset_path)
    
    feature_names = [
        "NDVI", "NDWI", "MNDWI", "LSWI", "VV", "VH", "VH_VV_ratio",
        "precipitation", "temperature", "evaporation", "potential_evaporation",
        "elevation", "slope", "flow_accumulation", "dist_to_river", "dist_to_coast"
    ]
    
    # ----------------------------------------------------
    # DIAGNOSTIC 1: Data Leakage Test (Experiment 1)
    # ----------------------------------------------------
    print("\n" + "="*70)
    # Reconstruct grids with all 16 features
    X_grid_all, y_grid = reconstruct_grid(X_raw, y_raw, feature_dim=16, is_v2=False)
    
    print("DIAGNOSTIC 1: DATA LEAKAGE TEST")
    print("Evaluating Random Forest model trained WITH satellite observation bands (VV, VH, NDVI, NDWI, etc.)")
    print("="*70)
    
    # Prepare training (2018 and 2024) and testing (2022 flood) data
    X_train_rf_all = np.concatenate([X_grid_all[:, 0, :, :], X_grid_all[:, 2, :, :]], axis=0).reshape(-1, 16)
    y_train_rf = np.concatenate([y_grid[:, 0, :], y_grid[:, 2, :]], axis=0).flatten()
    X_test_rf_all = X_grid_all[:, 1, :, :].reshape(-1, 16)
    y_test_rf = y_grid[:, 1, :].flatten()
    
    rf_split = int(X_train_rf_all.shape[0] * 0.8)
    X_tr_all, X_te_all = X_train_rf_all[:rf_split], X_train_rf_all[rf_split:]
    y_tr, y_te = y_train_rf[:rf_split], y_train_rf[rf_split:]
    
    rf_all = RandomForestClassifier(max_depth=10, n_estimators=100, random_state=42, n_jobs=-1)
    rf_all.fit(X_tr_all, y_tr)
    
    rf_all_norm_preds = rf_all.predict(X_te_all)
    metrics_all_norm = compute_metrics(y_te, rf_all_norm_preds)
    
    rf_all_test_preds = rf_all.predict(X_test_rf_all)
    metrics_all_test = compute_metrics(y_test_rf, rf_all_test_preds)
    
    print(f"Normal Period Validation F1 (With Leakage):      {metrics_all_norm['f1']:.3f}")
    print(f"Extreme Flood Period Test F1 (With Leakage):    {metrics_all_test['f1']:.3f}")
    print("\n*Note: High scores indicate that model is bypassing physical process learning by predicting water presence directly from concurrent satellite indices.*")
    
    # ----------------------------------------------------
    # DIAGNOSTIC 2: Feature Importance Analysis
    # ----------------------------------------------------
    print("\n" + "="*70)
    print("DIAGNOSTIC 2: FEATURE IMPORTANCE RANKINGS")
    print("="*70)
    
    # Leakage-free features (7 to 15)
    X_train_rf_lf = np.concatenate([X_grid_all[:, 0, :, 7:16], X_grid_all[:, 2, :, 7:16]], axis=0).reshape(-1, 9)
    rf_lf = RandomForestClassifier(max_depth=10, n_estimators=100, random_state=42, n_jobs=-1)
    rf_lf.fit(X_train_rf_lf, y_train_rf)
    
    lf_feature_names = feature_names[7:16]
    importances_lf = rf_lf.feature_importances_
    indices_lf = np.argsort(importances_lf)[::-1]
    
    print("Feature Importances in the Leakage-Free Model:")
    for idx in indices_lf:
        print(f"  - {lf_feature_names[idx]:<25}: {importances_lf[idx]:.4f}")
        
    # ----------------------------------------------------
    # DIAGNOSTIC 3: Sequence Model Threshold Calibration
    # ----------------------------------------------------
    print("\n" + "="*70)
    print("DIAGNOSTIC 3: SEQUENCE MODEL THRESHOLD CALIBRATION")
    print("="*70)
    
    # Extract leakage-free sequence inputs
    X_train_seq = np.concatenate([X_grid_all[:, 0, :, 7:16], X_grid_all[:, 2, :, 7:16]], axis=0) # (2000, 11, 9)
    y_train_seq = np.concatenate([y_grid[:, 0, :], y_grid[:, 2, :]], axis=0)
    
    # Move to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tr_t = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(-1).to(device)
    
    lstm = LSTMBaseline(input_dim=9, hidden_dim=32).to(device)
    optimizer = optim.Adam(lstm.parameters(), lr=0.005)
    criterion = nn.BCELoss()
    
    # Fit for few epochs to extract logits
    print("Fitting baseline LSTM model for calibration...")
    for epoch in range(10):
        lstm.train()
        optimizer.zero_grad()
        loss = criterion(lstm(X_tr_t), y_tr_t)
        loss.backward()
        optimizer.step()
        
    lstm.eval()
    with torch.no_grad():
        lstm_probs = lstm(X_tr_t).cpu().numpy().flatten()
        
    # Find optimal F1 threshold
    best_f1 = -1.0
    best_thresh = 0.5
    for thresh in np.linspace(0.01, 0.99, 99):
        y_pred = (lstm_probs > thresh).astype(int)
        metrics = compute_metrics(y_train_seq.flatten(), y_pred)
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_thresh = thresh
            
    print(f"Optimal F1 Decision Threshold: {best_thresh:.3f}")
    
    # Evaluate at 0.5 vs optimal threshold
    metrics_05 = compute_metrics(y_train_seq.flatten(), (lstm_probs > 0.5).astype(int))
    metrics_opt = compute_metrics(y_train_seq.flatten(), (lstm_probs > best_thresh).astype(int))
    
    print(f"  - Performance at 0.5 Threshold: F1 = {metrics_05['f1']:.3f} | Recall = {metrics_05['recall']:.3f} | Precision = {metrics_05['precision']:.3f}")
    print(f"  - Performance at {best_thresh:.3f} Threshold: F1 = {metrics_opt['f1']:.3f} | Recall = {metrics_opt['recall']:.3f} | Precision = {metrics_opt['precision']:.3f}")

if __name__ == "__main__":
    main()
