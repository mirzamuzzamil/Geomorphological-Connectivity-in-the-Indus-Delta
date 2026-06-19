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
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    
    # Intersection over Union (IoU) for the water class (class 1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    
    # AUC-ROC
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
    X_raw = data['X'] # (num_samples, 18) -> features + year + month
    y_raw = data['y'] # (num_samples,)
    
    # Feature columns:
    # 0: NDVI, 1: NDWI, 2: MNDWI, 3: LSWI, 4: VV, 5: VH, 6: VH_VV_ratio
    # 7: precipitation, 8: temperature, 9: evaporation, 10: potential_evaporation
    # 11: elevation, 12: slope, 13: flow_accumulation, 14: dist_to_river, 15: dist_to_coast
    # 16: year, 17: month
    
    years = X_raw[:, 16]
    months = X_raw[:, 17]
    X_feats = X_raw[:, :16]
    
    # We will build temporal sequences of length 11 months for each of the 1000 points.
    # The months are: 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12 (month 8 is skipped)
    target_months = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]
    num_points = 1000
    num_months = len(target_months)
    num_features = 16
    
    # Reconstruct grid time-series arrays: (num_points, num_years, num_months, num_features)
    # Target years: 2018 (index 0), 2022 (index 1), 2024 (index 2)
    year_map = {2018: 0, 2022: 1, 2024: 2}
    month_map = {m: idx for idx, m in enumerate(target_months)}
    
    X_grid = np.zeros((num_points, 3, num_months, num_features))
    y_grid = np.zeros((num_points, 3, num_months))
    
    # Group raw data by coordinate index
    # Since stratifiedSample outputs points sequentially, we can map them back.
    # For a point p, a year y, a month m, we assign the feature row.
    # Let's count occurrences to map them correctly.
    counter = {}
    for i in range(X_raw.shape[0]):
        yr = int(years[i])
        mo = int(months[i])
        if yr not in year_map or mo not in month_map:
            continue
        y_idx = year_map[yr]
        m_idx = month_map[mo]
        
        # We track how many times this (yr, mo) combination was seen to determine the point index
        key = (yr, mo)
        p_idx = counter.get(key, 0)
        if p_idx < num_points:
            X_grid[p_idx, y_idx, m_idx, :] = X_feats[i, :]
            y_grid[p_idx, y_idx, m_idx] = y_raw[i]
            counter[key] = p_idx + 1

    # Print baseline counts
    print(f"Data grid compiled. Shape: {X_grid.shape}")
    
    # Train / Test Split by Year:
    # Train: 2018 (Normal) + 2024 (Recent)
    # Test: 2022 (Extreme Flood Year)
    
    # For RF (Spatial baseline), flatten temporal dimensions
    X_train_rf = np.concatenate([X_grid[:, 0, :, :], X_grid[:, 2, :, :]], axis=0).reshape(-1, num_features)
    y_train_rf = np.concatenate([y_grid[:, 0, :], y_grid[:, 2, :]], axis=0).flatten()
    
    X_test_rf_2022 = X_grid[:, 1, :, :].reshape(-1, num_features)
    y_test_rf_2022 = y_grid[:, 1, :].flatten()
    
    # Normal Test Split (20% of 2018/2024 data)
    rf_split = int(X_train_rf.shape[0] * 0.8)
    X_tr_rf, X_te_rf = X_train_rf[:rf_split], X_train_rf[rf_split:]
    y_tr_rf, y_te_rf = y_train_rf[:rf_split], y_train_rf[rf_split:]
    
    # Train Random Forest Baseline A
    rf = RandomForestBaseline()
    rf.fit(X_tr_rf, y_tr_rf)
    
    # Predict and evaluate
    rf_norm_preds = rf.predict(X_te_rf)
    rf_norm_probs = rf.model.predict_proba(X_te_rf)[:, 1]
    metrics_rf_norm = compute_metrics(y_te_rf, rf_norm_preds, rf_norm_probs)
    
    rf_flood_preds = rf.predict(X_test_rf_2022)
    rf_flood_probs = rf.model.predict_proba(X_test_rf_2022)[:, 1]
    metrics_rf_flood = compute_metrics(y_test_rf_2022, rf_flood_preds, rf_flood_probs)
    
    # Monsoon 2022 subset (July is index 6, Sept is index 7)
    # Let's extract Monsoon 2022 indices: month 7 (Jul), 9 (Sep)
    monsoon_months_idx = [6, 7]
    X_monsoon_rf = X_grid[:, 1, monsoon_months_idx, :].reshape(-1, num_features)
    y_monsoon_rf = y_grid[:, 1, monsoon_months_idx].flatten()
    rf_mon_preds = rf.predict(X_monsoon_rf)
    rf_mon_probs = rf.model.predict_proba(X_monsoon_rf)[:, 1]
    metrics_rf_mon = compute_metrics(y_monsoon_rf, rf_mon_preds, rf_mon_probs)
    
    # For LSTM and Transformer (Temporal baselines)
    # Train: 2018 + 2024 sequences (shape: num_points * 2, seq_len, features)
    X_train_seq = np.concatenate([X_grid[:, 0, :, :], X_grid[:, 2, :, :]], axis=0) # (2000, 11, 16)
    y_train_seq = np.concatenate([y_grid[:, 0, :], y_grid[:, 2, :]], axis=0)      # (2000, 11)
    
    # Split into train/validation sets (80/20)
    seq_split = int(X_train_seq.shape[0] * 0.8)
    X_tr_seq, X_va_seq = X_train_seq[:seq_split], X_train_seq[seq_split:]
    y_tr_seq, y_va_seq = y_train_seq[:seq_split], y_train_seq[seq_split:]
    
    X_tr_t = torch.tensor(X_tr_seq, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr_seq, dtype=torch.float32).unsqueeze(-1)
    X_va_t = torch.tensor(X_va_seq, dtype=torch.float32)
    y_va_t = torch.tensor(y_va_seq, dtype=torch.float32).unsqueeze(-1)
    
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_va_t, y_va_t), batch_size=32, shuffle=False)
    
    # 2. Train LSTM Baseline B
    print("\nTraining LSTM Baseline B...")
    lstm = LSTMBaseline(input_dim=num_features, hidden_dim=32)
    optimizer_lstm = optim.Adam(lstm.parameters(), lr=0.005)
    criterion = nn.BCELoss()
    
    for epoch in range(12):
        lstm.train()
        for bx, by in train_loader:
            optimizer_lstm.zero_grad()
            loss = criterion(lstm(bx), by)
            loss.backward()
            optimizer_lstm.step()
            
    # Evaluate LSTM
    lstm.eval()
    with torch.no_grad():
        lstm_norm_probs = lstm(X_va_t).cpu().numpy().flatten()
        lstm_norm_preds = (lstm_norm_probs > 0.5).astype(int)
        metrics_lstm_norm = compute_metrics(y_va_seq.flatten(), lstm_norm_preds, lstm_norm_probs)
        
        # Test 2022 Flood
        X_test_2022_t = torch.tensor(X_grid[:, 1, :, :], dtype=torch.float32)
        lstm_flood_probs = lstm(X_test_2022_t).cpu().numpy().flatten()
        lstm_flood_preds = (lstm_flood_probs > 0.5).astype(int)
        metrics_lstm_flood = compute_metrics(y_grid[:, 1, :].flatten(), lstm_flood_preds, lstm_flood_probs)
        
        # Test Monsoon 2022
        X_mon_2022_t = torch.tensor(X_grid[:, 1, monsoon_months_idx, :], dtype=torch.float32)
        lstm_mon_probs = lstm(X_mon_2022_t).cpu().numpy().flatten()
        lstm_mon_preds = (lstm_mon_probs > 0.5).astype(int)
        metrics_lstm_mon = compute_metrics(y_grid[:, 1, monsoon_months_idx].flatten(), lstm_mon_preds, lstm_mon_probs)

    # 3. Train Transformer Baseline C
    print("\nTraining Transformer Baseline C...")
    transformer = TransformerBaseline(input_dim=num_features, d_model=32, nhead=2, dim_feedforward=64)
    optimizer_tf = optim.Adam(transformer.parameters(), lr=0.003)
    
    for epoch in range(12):
        transformer.train()
        for bx, by in train_loader:
            optimizer_tf.zero_grad()
            loss = criterion(transformer(bx), by)
            loss.backward()
            optimizer_tf.step()
            
    # Evaluate Transformer
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

    # ----------------------------------------------------
    # Report Output
    # ----------------------------------------------------
    print("\n" + "="*50)
    print("BASELINE PERFORMANCE REPORT (INDUS DELTA)")
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

if __name__ == "__main__":
    main()
