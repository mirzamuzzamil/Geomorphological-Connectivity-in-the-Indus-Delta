import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import chi2, binom

# Ensure parent directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.utils import load_indus_dataset, reconstruct_grid, compute_metrics

def mcnemar_test(y_true, y_pred1, y_pred2):
    """
    Perform McNemar's test on two model predictions.
    """
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
    """
    Compute 95% bootstrap confidence intervals for classification metrics.
    """
    np.random.seed(random_seed)
    n = len(y_true)
    boot_metrics = {
        "f1": [], "iou": [], "precision": [], "recall": [], "auc": []
    }
    
    for i in range(num_bootstraps):
        boot_idx = np.random.choice(n, size=n, replace=True)
        y_t_boot = y_true[boot_idx]
        y_p_boot = y_pred[boot_idx]
        y_pr_boot = y_prob[boot_idx] if y_prob is not None else None
        
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
    dataset_path = "data/indus_dataset.npz"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}. Run diagnostics or ensure full dataset is present.")
        sys.exit(1)
        
    X_raw, y_raw = load_indus_dataset(dataset_path)
    
    X_grid_all, y_grid = reconstruct_grid(X_raw, y_raw, feature_dim=16, is_v2=False)
    
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

    print("Training models for significance testing...")
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
