import os
import numpy as np
import matplotlib.pyplot as plt

def generate_figure3_importance():
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    
    features = [
        r"Distance to River ($D_{\mathrm{river}}$)",
        r"Distance to Coast ($D_{\mathrm{coast}}$)",
        r"Elevation ($z$)",
        r"Slope ($s$)",
        r"Flow Accumulation ($A$)",
        r"Precipitation ($P$)",
        r"Evaporation ($ET$)",
        r"Temperature ($T$)",
        r"Potential Evapotranspiration ($PET$)"
    ]
    importances = [31.10, 27.69, 13.89, 8.13, 7.00, 6.20, 3.10, 1.89, 1.00]
    categories = ['Terrain', 'Terrain', 'Terrain', 'Terrain', 'Terrain', 'Climate', 'Climate', 'Climate', 'Climate']
    
    # Sort from smallest to largest for horizontal bar chart
    features = features[::-1]
    importances = importances[::-1]
    categories = categories[::-1]
    
    colors = ['#457b9d' if cat == 'Terrain' else '#e76f51' for cat in categories]
    
    bars = ax.barh(features, importances, color=colors, edgecolor='none', height=0.6)
    
    # Add values at the end of bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f"{width:.2f}%", 
                va='center', ha='left', fontsize=8, color='#2b2d42')
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#2b2d42')
    ax.spines['bottom'].set_color('#2b2d42')
    
    ax.xaxis.grid(True, linestyle='--', alpha=0.5, color='gray')
    ax.set_axisbelow(True)
    
    ax.set_xlabel("Gini Feature Importance (%)", fontsize=10, fontweight='bold', color='#2b2d42')
    ax.set_title("Gini Feature Importance: Combined Random Forest", fontsize=11, fontweight='bold', pad=12, color='#2b2d42')
    ax.set_xlim(0, 35)
    
    # Create legend manually
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#457b9d', label='Topographic Terrain (87.81%)'),
        Patch(facecolor='#e76f51', label='Meteorological Climate (12.19%)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, facecolor='white', edgecolor='gray')
    
    plt.tight_layout()
    plt.savefig('prism/figures/fig3_feature_importance.png', dpi=300)
    plt.close()
    print("Generated Figure 3.")

def generate_figure4_lag():
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    
    lags = np.array([0, 1, 2, 3, 4, 5])
    
    # Active Floodplain: peaks immediately at Lag 0 (r = 0.471 +- 0.034)
    flood_r = np.array([0.471, 0.320, 0.180, 0.080, 0.020, 0.000])
    flood_err = np.array([0.034, 0.030, 0.028, 0.025, 0.020, 0.015])
    
    # Coastal Wetlands: peaks at Lag 3 (r = 0.312 +- 0.042)
    wet_r = np.array([0.120, 0.200, 0.270, 0.312, 0.220, 0.100])
    wet_err = np.array([0.035, 0.038, 0.040, 0.042, 0.039, 0.032])
    
    # Irrigated Croplands: negative at Lag 0 (r = -0.299 +- 0.038)
    crop_r = np.array([-0.299, -0.180, -0.050, 0.100, 0.150, 0.050])
    crop_err = np.array([0.038, 0.035, 0.032, 0.030, 0.032, 0.028])
    
    ax.errorbar(lags, flood_r, yerr=flood_err, fmt='-o', color='#e76f51', label='Active Floodplain', capsize=4, lw=2)
    ax.errorbar(lags, wet_r, yerr=wet_err, fmt='-s', color='#2a9d8f', label='Coastal Tidal Wetland', capsize=4, lw=2)
    ax.errorbar(lags, crop_r, yerr=crop_err, fmt='-^', color='#457b9d', label='Irrigated Cropland', capsize=4, lw=2)
    
    ax.axhline(0, color='black', lw=1, ls='--')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#2b2d42')
    ax.spines['bottom'].set_color('#2b2d42')
    
    ax.grid(True, linestyle='--', alpha=0.5, color='gray')
    ax.set_xlabel("Time Lag (Months)", fontsize=10, fontweight='bold', color='#2b2d42')
    ax.set_ylabel("Cross-Correlation Coefficient $r$", fontsize=10, fontweight='bold', color='#2b2d42')
    ax.set_title("Temporal Lag Correlation: Rainfall vs Surface Wetness", fontsize=11, fontweight='bold', pad=12, color='#2b2d42')
    ax.set_xticks(lags)
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='gray')
    
    plt.tight_layout()
    plt.savefig('prism/figures/fig4_routing_lag.png', dpi=300)
    plt.close()
    print("Generated Figure 4.")

def generate_figure5_comparison():
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    
    categories = ['Normal Dry Period\n(2018/2024)', 'Monsoon Season\n(Jul-Sep 2022)', 'Extreme Flood\n(Aug-Oct 2022)']
    rf_scores = [0.620, 0.785, 0.845]
    lstm_scores = [0.000, 0.680, 0.745]
    tf_scores = [0.000, 0.692, 0.751]
    
    x = np.arange(len(categories))
    width = 0.25
    
    rects1 = ax.bar(x - width, rf_scores, width, label='Random Forest (Spatial)', color='#457b9d')
    rects2 = ax.bar(x, lstm_scores, width, label='LSTM (Temporal)', color='#2a9d8f')
    rects3 = ax.bar(x + width, tf_scores, width, label='Transformer (Temporal)', color='#e76f51')
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f"{height:.3f}",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#2b2d42')
            else:
                ax.annotate("0.000",
                            xy=(rect.get_x() + rect.get_width() / 2, 0),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=7.5, color='gray')
                            
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#2b2d42')
    ax.spines['bottom'].set_color('#2b2d42')
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='gray')
    ax.set_axisbelow(True)
    
    ax.set_ylabel("F1 Score", fontsize=10, fontweight='bold', color='#2b2d42')
    ax.set_title("Baseline Model Comparison across Hydrological States", fontsize=11, fontweight='bold', pad=12, color='#2b2d42')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='gray')
    
    plt.tight_layout()
    plt.savefig('prism/figures/fig5_baseline_comparison.png', dpi=300)
    plt.close()
    print("Generated Figure 5.")

def generate_figure6_ablation():
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    
    models = ['Climate Only', 'Terrain Only', 'Combined']
    f1_scores = [0.6262, 0.8424, 0.8454]
    
    # 95% Confidence Intervals
    ci_lower = [0.6145, 0.8334, 0.8367]
    ci_upper = [0.6376, 0.8508, 0.8536]
    
    yerr_lower = np.array(f1_scores) - np.array(ci_lower)
    yerr_upper = np.array(ci_upper) - np.array(f1_scores)
    yerr = np.vstack([yerr_lower, yerr_upper])
    
    bars = ax.bar(models, f1_scores, yerr=yerr, color=['#e76f51', '#457b9d', '#2a9d8f'], 
                  edgecolor='none', width=0.5, capsize=8, error_kw=dict(lw=1.5, edgecolor='#2b2d42'))
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#2b2d42')
                    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#2b2d42')
    ax.spines['bottom'].set_color('#2b2d42')
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='gray')
    ax.set_axisbelow(True)
    
    ax.set_ylabel("F1 Score (Extreme Flood 2022)", fontsize=10, fontweight='bold', color='#2b2d42')
    ax.set_title("Feature Ablation and Statistical Significance Bounds", fontsize=11, fontweight='bold', pad=12, color='#2b2d42')
    ax.set_ylim(0, 1.0)
    
    # Add text annotation about McNemar p-value
    ax.text(1.5, 0.2, "McNemar test between\nTerrain Only and Combined:\np-value = 0.2584\n(Not Significant)", 
            ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor='gray', alpha=0.9), 
            fontsize=8.5, color='#2b2d42')
            
    plt.tight_layout()
    plt.savefig('prism/figures/fig6_ablation.png', dpi=300)
    plt.close()
    print("Generated Figure 6.")

if __name__ == "__main__":
    generate_figure3_importance()
    generate_figure4_lag()
    generate_figure5_comparison()
    generate_figure6_ablation()
