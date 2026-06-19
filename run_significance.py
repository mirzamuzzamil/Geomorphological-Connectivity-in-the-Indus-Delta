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
from scipy.stats import chi2, binom

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

def mcnemar_test(y_true, y_pred1, y_pred2):
    # y_pred1: model 1 prediction
    # y_pred2: model 2 prediction
    # model 1 correct, model 2 wrong
    b = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
    # model 1 wrong, model 2 correct
    c = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
    # both correct
    a = np.sum((y_pred1 == y_true) & (y_pred2 == y_true))
    # both wrong
    d = np.sum((y_pred1 != y_true) & (y_pred2 != y_true))
    
    table = np.array([[a, b], [c, d]])
    
    if b + c < 25:
        p_val = binom.cdf(min(b, c), b + c, 0.5) * 2
        p_val = min(p_val, 1.0)
        stat = float(min(b, c))
        test_type = "Exact Binomial"
    else:
        stat = ((abs(b - c) - 0.5) ** 2) / (b + c)
        p_val = chi2.sf(stat, 1)
        test_type = "Chi-Square (with continuity correction)"
        
    return stat, p_val, test_type, table, b, c

def bootstrap_metrics(y_true, y_pred, y_prob, num_bootstraps=1000, random_seed=42):
    np.random.seed(random_seed)
    n = len(y_true)
    boot_metrics = {
        "f1": [], "iou": [], "precision": [], "recall": [], "auc": []
    }
    
    for i in range(num_bootstraps):
        boot_idx = np.random.choice(n, size=n, replace=True)
        y_t_boot = y_true[boot_idx]
        y_p_boot = y_pred[boot_idx]
        y_pr_boot = y_prob[boot_idx]
        
        m = compute_metrics(y_t_boot, y_p_boot, y_pr_boot)
        for k in boot_metrics:
            boot_metrics[k].append(m[k])
            
    intervals = {}
    for k in boot_metrics:
        sorted_vals = np.sort(boot_metrics[k])
        low = np.percentile(sorted_vals, 2.5)
        high = np.percentile(sorted_vals, 97.5)
        intervals[k] = (low, high)
        
    return intervals

def main():
    data = np.load("data/indus_dataset.npz")
    X_raw = data['X']
    y_raw = data['y']
    
    years = X_raw[:, 16]
    months = X_raw[:, 17]
    
    target_months = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]
    num_points = 1000
    num_months = len(target_months)
    
    year_map = {2018: 0, 2022: 1, 2024: 2}
    month_map = {m: idx for idx, m in enumerate(target_months)}
    
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

    climate_indices = [7, 8, 9, 10]
    terrain_indices = [11, 12, 13, 14, 15]
    combined_indices = climate_indices + terrain_indices
    
    y_train = np.concatenate([y_grid[:, 0, :], y_grid[:, 2, :]], axis=0).flatten()
    y_test = y_grid[:, 1, :].flatten() # 2022 Flood Period
    
    split_idx = int(y_train.shape[0] * 0.8)
    
    def train_and_predict(feat_indices):
        X_train_full = np.concatenate([X_grid_all[:, 0, :, :], X_grid_all[:, 2, :, :]], axis=0)[:, :, feat_indices].reshape(-1, len(feat_indices))
        X_test_full = X_grid_all[:, 1, :, :][:, :, feat_indices].reshape(-1, len(feat_indices))
        
        X_tr = X_train_full[:split_idx]
        y_tr = y_train[:split_idx]
        
        rf = RandomForestClassifier(max_depth=10, n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        
        preds = rf.predict(X_test_full)
        probs = rf.predict_proba(X_test_full)[:, 1]
        return preds, probs

    print("Training models...")
    terrain_preds, terrain_probs = train_and_predict(terrain_indices)
    climate_preds, climate_probs = train_and_predict(climate_indices)
    combined_preds, combined_probs = train_and_predict(combined_indices)
    
    print("Computing bootstrap confidence intervals (1000 iterations)...")
    terrain_boot = bootstrap_metrics(y_test, terrain_preds, terrain_probs)
    climate_boot = bootstrap_metrics(y_test, climate_preds, climate_probs)
    combined_boot = bootstrap_metrics(y_test, combined_preds, combined_probs)
    
    # Compute base metrics
    terrain_base = compute_metrics(y_test, terrain_preds, terrain_probs)
    climate_base = compute_metrics(y_test, climate_preds, climate_probs)
    combined_base = compute_metrics(y_test, combined_preds, combined_probs)

    print("\n" + "="*80)
    print("BOOTSTRAP CONFIDENCE INTERVALS (95% CI)")
    print("="*80)
    for name, base, boot in [("Terrain Only", terrain_base, terrain_boot), 
                             ("Climate Only", climate_base, climate_boot), 
                             ("Combined", combined_base, combined_boot)]:
        print(f"\n[{name}]")
        for k in ["f1", "iou", "precision", "recall", "auc"]:
            print(f"  {k:<10}: {base[k]:.4f} (95% CI: [{boot[k][0]:.4f}, {boot[k][1]:.4f}])")
            
    print("\n" + "="*80)
    print("MCNEMAR SIGNIFICANCE TEST RESULTS")
    print("="*80)
    
    def report_mcnemar(name1, name2, pred1, pred2):
        stat, p, test_type, table, b, c = mcnemar_test(y_test, pred1, pred2)
        print(f"\nComparing {name1} vs {name2}:")
        print(f"  Test type:  {test_type}")
        print(f"  Statistic:  {stat:.4f}")
        print(f"  P-value:    {p:.4e}")
        print(f"  Contingency Table:")
        print(f"    Both Correct: {table[0, 0]:<5} | {name1} Correct, {name2} Incorrect: {table[0, 1]}")
        print(f"    {name1} Incorrect, {name2} Correct: {table[1, 0]:<5} | Both Incorrect: {table[1, 1]}")
        print(f"  Discordant pair counts: B ({name1} correct only) = {b}, C ({name2} correct only) = {c}")
        if p > 0.05:
            print("  Interpretation: The difference is NOT statistically significant (p > 0.05). Models are statistically indistinguishable.")
        else:
            print("  Interpretation: The difference IS statistically significant (p <= 0.05).")
            
    report_mcnemar("Terrain Only", "Combined", terrain_preds, combined_preds)
    report_mcnemar("Climate Only", "Combined", climate_preds, combined_preds)
    report_mcnemar("Climate Only", "Terrain Only", climate_preds, terrain_preds)

if __name__ == "__main__":
    main()
