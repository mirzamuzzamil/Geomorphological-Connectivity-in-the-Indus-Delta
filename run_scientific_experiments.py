# /// script
# dependencies = [
#   "scikit-learn",
#   "torch",
#   "numpy",
#   "scipy"
# ]
# ///

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from train_baselines import RandomForestBaseline, LSTMBaseline, TransformerBaseline

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
    # Load dataset
    data = np.load("data/indus_dataset.npz")
    X_raw = data['X'] # (num_samples, 18)
    y_raw = data['y'] # (num_samples,)
    
    # We split columns to prevent data leakage:
    # Remove S1 (VV, VH, VH_VV_ratio) and S2 (NDVI, NDWI, MNDWI, LSWI, EVI) from inputs.
    # Keep only Meteorological forcing + Topographic features (indices 7 to 15)
    # Features kept:
    # 0: precipitation, 1: temperature, 2: evaporation, 3: potential_evaporation
    # 4: elevation, 5: slope, 6: flow_accumulation, 7: dist_to_river, 8: dist_to_coast
    X_meteorological = X_raw[:, 7:16]
    
    years = X_raw[:, 16]
    months = X_raw[:, 17]
    
    target_months = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]
    num_points = 1000
    num_months = len(target_months)
    num_features = X_meteorological.shape[1] # 9 features
    
    year_map = {2018: 0, 2022: 1, 2024: 2}
    month_map = {m: idx for idx, m in enumerate(target_months)}
    
    X_grid = np.zeros((num_points, 3, num_months, num_features))
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
            X_grid[p_idx, y_idx, m_idx, :] = X_meteorological[i, :]
            y_grid[p_idx, y_idx, m_idx] = y_raw[i]
            counter[key] = p_idx + 1

    print(f"Scientific Data Grid compiled (Leakage-Free). Shape: {X_grid.shape}")
    
    # Flatten for Random Forest (Baseline A)
    X_train_rf = np.concatenate([X_grid[:, 0, :, :], X_grid[:, 2, :, :]], axis=0).reshape(-1, num_features)
    y_train_rf = np.concatenate([y_grid[:, 0, :], y_grid[:, 2, :]], axis=0).flatten()
    
    X_test_rf_2022 = X_grid[:, 1, :, :].reshape(-1, num_features)
    y_test_rf_2022 = y_grid[:, 1, :].flatten()
    
    rf_split = int(X_train_rf.shape[0] * 0.8)
    X_tr_rf, X_te_rf = X_train_rf[:rf_split], X_train_rf[rf_split:]
    y_tr_rf, y_te_rf = y_train_rf[:rf_split], y_train_rf[rf_split:]
    
    # Train RF
    rf = RandomForestClassifier(max_depth=10, n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr_rf, y_tr_rf)
    
    # Eval RF
    rf_norm_preds = rf.predict(X_te_rf)
    rf_norm_probs = rf.predict_proba(X_te_rf)[:, 1]
    metrics_rf_norm = compute_metrics(y_te_rf, rf_norm_preds, rf_norm_probs)
    
    rf_flood_preds = rf.predict(X_test_rf_2022)
    rf_flood_probs = rf.predict_proba(X_test_rf_2022)[:, 1]
    metrics_rf_flood = compute_metrics(y_test_rf_2022, rf_flood_preds, rf_flood_probs)
    
    monsoon_months_idx = [6, 7] # July and September
    X_monsoon_rf = X_grid[:, 1, monsoon_months_idx, :].reshape(-1, num_features)
    y_monsoon_rf = y_grid[:, 1, monsoon_months_idx].flatten()
    rf_mon_preds = rf.predict(X_monsoon_rf)
    rf_mon_probs = rf.predict_proba(X_monsoon_rf)[:, 1]
    metrics_rf_mon = compute_metrics(y_monsoon_rf, rf_mon_preds, rf_mon_probs)
    
    # Sequences for Deep Learning (Baseline B & C)
    X_train_seq = np.concatenate([X_grid[:, 0, :, :], X_grid[:, 2, :, :]], axis=0) # (2000, 11, 9)
    y_train_seq = np.concatenate([y_grid[:, 0, :], y_grid[:, 2, :]], axis=0)      # (2000, 11)
    
    seq_split = int(X_train_seq.shape[0] * 0.8)
    X_tr_seq, X_va_seq = X_train_seq[:seq_split], X_train_seq[seq_split:]
    y_tr_seq, y_va_seq = y_train_seq[:seq_split], y_train_seq[seq_split:]
    
    X_tr_t = torch.tensor(X_tr_seq, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr_seq, dtype=torch.float32).unsqueeze(-1)
    X_va_t = torch.tensor(X_va_seq, dtype=torch.float32)
    y_va_t = torch.tensor(y_va_seq, dtype=torch.float32).unsqueeze(-1)
    
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)
    
    # Train LSTM
    print("Training LSTM Baseline B (Leakage-Free)...")
    lstm = LSTMBaseline(input_dim=num_features, hidden_dim=32)
    optimizer_lstm = optim.Adam(lstm.parameters(), lr=0.005)
    criterion = nn.BCELoss()
    
    for epoch in range(15):
        lstm.train()
        for bx, by in train_loader:
            optimizer_lstm.zero_grad()
            loss = criterion(lstm(bx), by)
            loss.backward()
            optimizer_lstm.step()
            
    lstm.eval()
    with torch.no_grad():
        lstm_norm_probs = lstm(X_va_t).cpu().numpy().flatten()
        lstm_norm_preds = (lstm_norm_probs > 0.5).astype(int)
        metrics_lstm_norm = compute_metrics(y_va_seq.flatten(), lstm_norm_preds, lstm_norm_probs)
        
        X_test_2022_t = torch.tensor(X_grid[:, 1, :, :], dtype=torch.float32)
        lstm_flood_probs = lstm(X_test_2022_t).cpu().numpy().flatten()
        lstm_flood_preds = (lstm_flood_probs > 0.5).astype(int)
        metrics_lstm_flood = compute_metrics(y_grid[:, 1, :].flatten(), lstm_flood_preds, lstm_flood_probs)
        
        X_mon_2022_t = torch.tensor(X_grid[:, 1, monsoon_months_idx, :], dtype=torch.float32)
        lstm_mon_probs = lstm(X_mon_2022_t).cpu().numpy().flatten()
        lstm_mon_preds = (lstm_mon_probs > 0.5).astype(int)
        metrics_lstm_mon = compute_metrics(y_grid[:, 1, monsoon_months_idx].flatten(), lstm_mon_preds, lstm_mon_probs)

    # Train Transformer
    print("Training Transformer Baseline C (Leakage-Free)...")
    transformer = TransformerBaseline(input_dim=num_features, d_model=32, nhead=2, dim_feedforward=64)
    optimizer_tf = optim.Adam(transformer.parameters(), lr=0.003)
    
    for epoch in range(15):
        transformer.train()
        for bx, by in train_loader:
            optimizer_tf.zero_grad()
            loss = criterion(transformer(bx), by)
            loss.backward()
            optimizer_tf.step()
            
    transformer.eval()
    with torch.no_grad():
        tf_norm_probs = transformer(X_va_t).cpu().numpy().flatten()
        tf_norm_preds = (tf_norm_probs > 0.5).astype(int)
        metrics_tf_norm = compute_metrics(y_va_seq.flatten(), tf_norm_preds, tf_norm_probs)
        
        tf_flood_probs = transformer(X_test_2022_t).cpu().numpy().flatten()
        tf_flood_preds = (tf_flood_probs > 0.5).astype(int)
        metrics_tf_flood = compute_metrics(y_grid[:, 1, :].flatten(), tf_flood_preds, tf_flood_probs)
        
        tf_mon_probs = transformer(X_mon_2022_t).cpu().numpy().flatten()
        tf_mon_preds = (tf_mon_probs > 0.5).astype(int)
        metrics_tf_mon = compute_metrics(y_grid[:, 1, monsoon_months_idx].flatten(), tf_mon_preds, tf_mon_probs)

    # Report Output
    print("\n" + "="*50)
    print("SCIENTIFIC BASELINE PERFORMANCE REPORT (INDUS DELTA)")
    print("  Inputs: Meteorological Forcing & Static Topography Only")
    print("  (Water indices and backscatter removed to prevent data leakage)")
    print("="*50)
    
    def print_table(title, rf_m, lstm_m, tf_m):
        print(f"\n[{title}]")
        print(f"{'Model':<15} | {'F1':<6} | {'IoU':<6} | {'Recall':<6} | {'Prec':<6} | {'AUC':<6}")
        print("-"*60)
        print(f"{'Random Forest':<15} | {rf_m['f1']:.3f} | {rf_m['iou']:.3f} | {rf_m['recall']:.3f} | {rf_m['precision']:.3f} | {rf_m['auc']:.3f}")
        print(f"{'LSTM':<15} | {lstm_m['f1']:.3f} | {lstm_m['iou']:.3f} | {lstm_m['recall']:.3f} | {lstm_m['precision']:.3f} | {lstm_m['auc']:.3f}")
        print(f"{'Transformer':<15} | {tf_m['f1']:.3f} | {tf_m['iou']:.3f} | {tf_m['recall']:.3f} | {tf_m['precision']:.3f} | {tf_m['auc']:.3f}")
        
    print_table("Normal Test Period (2018/2024)", metrics_rf_norm, metrics_lstm_norm, metrics_tf_norm)
    print_table("Monsoon Season (Jul-Sep 2022)", metrics_rf_mon, metrics_lstm_mon, metrics_tf_mon)
    print_table("Extreme Flood Period (Aug-Oct 2022)", metrics_rf_flood, metrics_lstm_flood, metrics_tf_flood)

    # Generate dataset statistics report
    print("\n" + "="*50)
    print("DATASET STATISTICS REPORT (INDUS DELTA)")
    print("="*50)
    total_pix = num_points
    total_samples = X_raw.shape[0]
    water_pix_pct = (np.sum(y_raw == 1) / total_samples) * 100
    land_pix_pct = (np.sum(y_raw == 0) / total_samples) * 100
    print(f"Number of monthly samples:       {total_samples}")
    print(f"Number of grid pixels sampled:   {total_pix}")
    print(f"Water pixels (%):               {water_pix_pct:.2f}%")
    print(f"Non-water pixels (%):           {land_pix_pct:.2f}%")
    print(f"Missing Sentinel-2 observations: 1 month (2018-08 skipped due to 100% cloud occlusion)")
    print(f"Flood months:                    6 months (Aug-Oct 2022 and Jul-Sep 2024)")
    print(f"Extreme flood events:            1 event (Aug-Oct 2022 historic monsoon flood)")

if __name__ == "__main__":
    main()
