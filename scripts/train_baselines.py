import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Ensure the parent directory is in the path for importing models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.random_forest import RandomForestBaseline
from models.lstm import LSTMBaseline
from models.transformer import TransformerBaseline
from scripts.utils import load_indus_dataset, reconstruct_grid, compute_metrics

def main():
    # Detect dataset
    dataset_path = "data/indus_dataset_sample.npz"
    if not os.path.exists(dataset_path):
        dataset_path = "data/indus_dataset.npz"
        
    print(f"Loading dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}. Please run with a sample dataset or follow data extraction instructions.")
        sys.exit(1)
        
    X_raw, y_raw = load_indus_dataset(dataset_path)
    
    # Check if this is the sample or the full v1 dataset
    is_v2 = False
    if X_raw.shape[1] == 11:
        is_v2 = True
        num_features = 9
        X_meteorological = X_raw[:, :9]
    else:
        # v1 dataset has 18 columns, climate starts at index 7 to 15
        num_features = 9
        X_meteorological = X_raw[:, 7:16]
        
    # Reconstruct grids
    # Target shape: (num_points, num_years, num_months, num_features) or (num_points, num_steps, num_features)
    X_grid, y_grid = reconstruct_grid(X_raw, y_raw, feature_dim=num_features, is_v2=is_v2)
    
    print(f"Data Grid Compiled. Shape: {X_grid.shape}")
    
    # ----------------------------------------------------
    # Model Splits setup
    # ----------------------------------------------------
    if is_v2:
        # v2 dataset temporal setup: 
        # Train on 2018-2021 (indices 0 to 46), Test on 2022-2024 (indices 47 to 82)
        X_train_seq = X_grid[:, :47, :]
        y_train_seq = y_grid[:, :47]
        X_test_seq = X_grid[:, 47:, :]
        y_test_seq = y_grid[:, 47:]
        
        # Flatten for Random Forest
        X_train_rf = X_train_seq.reshape(-1, num_features)
        y_train_rf = y_train_seq.flatten()
        X_test_rf = X_test_seq.reshape(-1, num_features)
        y_test_rf = y_test_seq.flatten()
    else:
        # original v1 layout: 3 years (2018, 2022, 2024), 11 months each
        # Train on 2018 (year index 0) and 2024 (year index 2)
        # Test on 2022 (year index 1, extreme flood period)
        X_train_rf = np.concatenate([X_grid[:, 0, :, :], X_grid[:, 2, :, :]], axis=0).reshape(-1, num_features)
        y_train_rf = np.concatenate([y_grid[:, 0, :], y_grid[:, 2, :]], axis=0).flatten()
        X_test_rf = X_grid[:, 1, :, :].reshape(-1, num_features)
        y_test_rf = y_grid[:, 1, :].flatten()
        
        # Sequences for Deep Learning
        X_train_seq = np.concatenate([X_grid[:, 0, :, :], X_grid[:, 2, :, :]], axis=0) # (2000, 11, 9)
        y_train_seq = np.concatenate([y_grid[:, 0, :], y_grid[:, 2, :]], axis=0)      # (2000, 11)
        X_test_seq = X_grid[:, 1, :, :] # (1000, 11, 9)
        y_test_seq = y_grid[:, 1, :]    # (1000, 11)
        
    # Split training set for validation (80-20)
    rf_split = int(X_train_rf.shape[0] * 0.8)
    X_tr_rf, X_te_rf = X_train_rf[:rf_split], X_train_rf[rf_split:]
    y_tr_rf, y_te_rf = y_train_rf[:rf_split], y_train_rf[rf_split:]
    
    seq_split = int(X_train_seq.shape[0] * 0.8)
    X_tr_seq, X_va_seq = X_train_seq[:seq_split], X_train_seq[seq_split:]
    y_tr_seq, y_va_seq = y_train_seq[:seq_split], y_train_seq[seq_split:]
    
    # ----------------------------------------------------
    # 1. Train Random Forest (Baseline A)
    # ----------------------------------------------------
    print("\nTraining Random Forest model...")
    rf = RandomForestBaseline(max_depth=10, n_estimators=100)
    rf.fit(X_tr_rf, y_tr_rf)
    
    rf_norm_preds = rf.predict(X_te_rf)
    metrics_rf_norm = compute_metrics(y_te_rf, rf_norm_preds)
    
    rf_test_preds = rf.predict(X_test_rf)
    metrics_rf_test = compute_metrics(y_test_rf, rf_test_preds)
    
    # ----------------------------------------------------
    # 2. PyTorch Dataloader setup
    # ----------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tr_t = torch.tensor(X_tr_seq, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr_seq, dtype=torch.float32).unsqueeze(-1).to(device)
    X_va_t = torch.tensor(X_va_seq, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
    
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)
    
    # ----------------------------------------------------
    # 3. Train LSTM (Baseline B)
    # ----------------------------------------------------
    print("\nTraining LSTM baseline...")
    lstm = LSTMBaseline(input_dim=num_features, hidden_dim=32).to(device)
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
        metrics_lstm_norm = compute_metrics(y_va_seq.flatten(), lstm_norm_preds)
        
        lstm_test_probs = lstm(X_test_t).cpu().numpy().flatten()
        lstm_test_preds = (lstm_test_probs > 0.5).astype(int)
        metrics_lstm_test = compute_metrics(y_test_seq.flatten(), lstm_test_preds)
        
    # ----------------------------------------------------
    # 4. Train Transformer (Baseline C)
    # ----------------------------------------------------
    print("\nTraining Transformer baseline...")
    transformer = TransformerBaseline(input_dim=num_features, d_model=32, nhead=2, dim_feedforward=64).to(device)
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
        metrics_tf_norm = compute_metrics(y_va_seq.flatten(), tf_norm_preds)
        
        tf_test_probs = transformer(X_test_t).cpu().numpy().flatten()
        tf_test_preds = (tf_test_probs > 0.5).astype(int)
        metrics_tf_test = compute_metrics(y_test_seq.flatten(), tf_test_preds)
        
    # ----------------------------------------------------
    # Report Results
    # ----------------------------------------------------
    print("\n" + "="*70)
    print("BASELINE PERFORMANCE REPORT (INDUS DELTA SURFACE WATER)")
    print("="*70)
    
    def print_table(title, rf_m, lstm_m, tf_m):
        print(f"\n[{title}]")
        print(f"{'Model':<15} | {'F1':<6} | {'IoU':<6} | {'Recall':<6} | {'Prec':<6}")
        print("-"*50)
        print(f"{'Random Forest':<15} | {rf_m['f1']:.3f} | {rf_m['iou']:.3f} | {rf_m['recall']:.3f} | {rf_m['precision']:.3f}")
        print(f"{'LSTM':<15} | {lstm_m['f1']:.3f} | {lstm_m['iou']:.3f} | {lstm_m['recall']:.3f} | {lstm_m['precision']:.3f}")
        print(f"{'Transformer':<15} | {tf_m['f1']:.3f} | {tf_m['iou']:.3f} | {tf_m['recall']:.3f} | {tf_m['precision']:.3f}")
        
    print_table("Normal Validation Set", metrics_rf_norm, metrics_lstm_norm, metrics_tf_norm)
    print_table("Testing / Extreme Flood Period", metrics_rf_test, metrics_lstm_test, metrics_tf_test)

if __name__ == "__main__":
    main()
