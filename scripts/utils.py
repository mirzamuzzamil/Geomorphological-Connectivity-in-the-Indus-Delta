import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score

def load_indus_dataset(filepath="data/indus_dataset.npz"):
    """
    Load the Indus Delta dataset.
    Returns:
        X_raw: features matrix
        y_raw: labels array
    """
    data = np.load(filepath)
    return data['X'], data['y']

def reconstruct_grid(X_raw, y_raw, feature_dim=16, is_v2=False):
    """
    Reconstruct tabular data back into a 3D spatial-temporal grid (num_points, num_years, num_months, num_features).
    """
    years = X_raw[:, 16 if not is_v2 else 9]
    months = X_raw[:, 17 if not is_v2 else 10]
    
    if is_v2:
        target_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
        sampled_dates = [(y, m) for y in target_years for m in range(1, 13)]
        sampled_dates = [d for d in sampled_dates if not (d[0] == 2018 and d[1] == 8)]
        date_map = {date: idx for idx, date in enumerate(sampled_dates)}
        
        # Count samples per date key to find number of points dynamically
        counts = {}
        for i in range(X_raw.shape[0]):
            yr = int(years[i])
            mo = int(months[i])
            date_key = (yr, mo)
            if date_key in date_map:
                counts[date_key] = counts.get(date_key, 0) + 1
        num_points = max(counts.values()) if counts else 1000
        
        num_steps = len(sampled_dates)
        
        X_grid = np.zeros((num_points, num_steps, feature_dim))
        y_grid = np.zeros((num_points, num_steps))
        
        counter = {}
        for i in range(X_raw.shape[0]):
            yr = int(years[i])
            mo = int(months[i])
            date_key = (yr, mo)
            if date_key not in date_map:
                continue
            step_idx = date_map[date_key]
            
            p_idx = counter.get(date_key, 0)
            if p_idx < num_points:
                X_grid[p_idx, step_idx, :] = X_raw[i, :feature_dim]
                y_grid[p_idx, step_idx] = y_raw[i]
                counter[date_key] = p_idx + 1
        return X_grid, y_grid
    else:
        # For original indus_dataset.npz (2018, 2022, 2024)
        target_months = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]
        
        year_map = {2018: 0, 2022: 1, 2024: 2}
        month_map = {m: idx for idx, m in enumerate(target_months)}
        
        # Count samples per key
        counts = {}
        for i in range(X_raw.shape[0]):
            yr = int(years[i])
            mo = int(months[i])
            if yr in year_map and mo in month_map:
                key = (yr, mo)
                counts[key] = counts.get(key, 0) + 1
        num_points = max(counts.values()) if counts else 1000
        
        num_months = len(target_months)
        
        X_grid = np.zeros((num_points, 3, num_months, feature_dim))
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
                X_grid[p_idx, y_idx, m_idx, :] = X_raw[i, :feature_dim]
                y_grid[p_idx, y_idx, m_idx] = y_raw[i]
                counter[key] = p_idx + 1
        return X_grid, y_grid

def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Compute binary classification performance metrics: F1, IoU (intersection over union), Recall, Precision, and AUC.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    
    auc = 0.5
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            pass
            
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "auc": auc
    }
