import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont

# Set up matplotlib style for scientific journals
plt.rcParams.update({
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
    'axes.edgecolor':     '#222222',
    'axes.linewidth':     0.8,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'grid.color':         '#dddddd',
    'grid.linewidth':     0.5,
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':          11,
    'axes.labelsize':     12,
    'axes.titlesize':     13,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    10,
    'legend.frameon':     True,
    'legend.framealpha':  0.9,
    'legend.edgecolor':   '#cccccc',
    'lines.linewidth':    1.5,
    'savefig.dpi':        600,
    'savefig.bbox':       'tight',
    'savefig.facecolor':  'white',
})

GREY   = '#444444'
BLUE   = '#2166ac'
RED    = '#d6604d'
GREEN  = '#4dac26'
ORANGE = '#f4a582'
OUT_DIR = 'results/figures'

def create_schematic(filename, text, size=(800, 600), bg_color=(240, 244, 248), text_color=(33, 37, 41)):
    img = Image.new('RGB', size, color=bg_color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, size[0]-10, size[1]-10], outline=(100, 110, 120), width=3)
    
    font = None
    try:
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf"
        ]
        for path in font_paths:
            if os.path.exists(path):
                font = ImageFont.truetype(path, size=24)
                break
    except:
        pass
    
    lines = text.split('\n')
    y = size[1] // 2 - (len(lines) * 15)
    for line in lines:
        if font:
            w = draw.textlength(line, font=font)
            draw.text(((size[0] - w) // 2, y), line, fill=text_color, font=font)
            y += font.size + 10
        else:
            draw.text((size[0]//2 - 100, y), line, fill=text_color)
            y += 20
            
    img.save(filename)
    print(f"Generated schematic: {filename}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # ── Figure 1: Study Area Schematic ──────────────────────────────────────────
    fig1_text = (
        "Figure 1: Indus River Delta Study Area Schematic\n\n"
        "Geographic Boundaries:\n"
        "Latitude: 23.0 N to 24.8 N\n"
        "Longitude: 67.0 E to 69.0 E\n\n"
        "Sampling Stratification:\n"
        "- Permanent Water (JRC > 80%): 300 points\n"
        "- Seasonal Wetlands (JRC 1-80%): 300 points\n"
        "- Dry Land (JRC = 0%): 400 points\n\n"
        "[Note: Map coordinates and digital elevation boundaries\n"
        "used for training and validation]"
    )
    create_schematic(f"{OUT_DIR}/fig1_study_area.png", fig1_text)
    
    # ── Figure 2: Workflow Schematic ───────────────────────────────────────────
    fig2_text = (
        "Figure 2: Methodological Workflow Schematic\n\n"
        "1. Google Earth Engine Data Extraction (83,000 observations)\n"
        "   - Sentinel-1 (SAR VV/VH), Sentinel-2 (Indices), SRTM DEM, CHIRPS, ERA5-Land\n"
        "2. Feature Splitting (Leakage-Free Configuration)\n"
        "   - Climate Only vs. Terrain Only vs. Combined\n"
        "3. Machine Learning Classifiers (Random Forest, LSTM, Transformer)\n"
        "4. Diagnostic Evaluations (Gini Feature Importance, Routing Lag, Ablations)\n"
        "5. Statistical Significance Audits (Bootstrap CI, Edwards' McNemar Test)"
    )
    create_schematic(f"{OUT_DIR}/fig2_workflow.png", fig2_text)
    
    # ── Figure 3: Feature Importance ──────────────────────────────────────────────
    features = [
        'River Distance', 'Coastal Distance', 'Elevation', 'Slope', 'Flow Accumulation',
        'Precipitation', 'Evaporation', 'Temperature', 'PET'
    ]
    importance = [0.3110, 0.2769, 0.1389, 0.0813, 0.0700, 0.0620, 0.0310, 0.0189, 0.0100]
    colors = [BLUE if i < 5 else RED for i in range(len(features))]
    
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.barh(features[::-1], importance[::-1], color=colors[::-1], height=0.6, edgecolor='white', linewidth=0.3)
    ax.set_xlabel('Mean Decrease in Gini Impurity (Normalised)')
    ax.set_title('Figure 3. Random Forest Feature Importance — Indus Delta', pad=8)
    ax.axvline(0, color='#222222', linewidth=0.8)
    ax.set_xlim(0, 0.42)
    ax.xaxis.grid(True, linestyle='--', linewidth=0.5, color='#dddddd')
    ax.set_axisbelow(True)
    
    for bar, val in zip(bars, importance[::-1]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', ha='left', fontsize=9.5, color=GREY)
        
    terrain_sum = sum(importance[:5])
    ax.text(0.38, 0.08, f'Terrain total: {terrain_sum:.1%}', transform=ax.transAxes, fontsize=10, color=BLUE, ha='right', style='italic')
    
    patch_t = mpatches.Patch(color=BLUE, label='Terrain features (static)')
    patch_c = mpatches.Patch(color=RED,  label='Climate features (dynamic)')
    ax.legend(handles=[patch_t, patch_c], loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig3_feature_importance.png', dpi=600)
    plt.close()
    
    # ── Figure 4: Routing Lag ──────────────────────────────────────────────────────
    lags = [0, 1, 2, 3, 4, 5]
    lag_floodplain = [0.221, 0.312, 0.401, 0.471, 0.431, 0.388]
    lag_wetland    = [0.088, 0.154, 0.241, 0.312, 0.298, 0.271]
    lag_cropland   = [0.041, -0.082, -0.178, -0.243, -0.299, -0.281]
    
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(lags, lag_floodplain, 'o-', color=BLUE,   label='Active Floodplain', markersize=5)
    ax.plot(lags, lag_wetland,    's-', color=GREEN,   label='Coastal Tidal Wetland', markersize=5)
    ax.plot(lags, lag_cropland,   '^-', color=RED,     label='Irrigated Cropland (divergence)', markersize=5)
    ax.axhline(0, color='#888888', linewidth=0.8, linestyle='--')
    ax.axvline(3, color='#888888', linewidth=0.8, linestyle=':')
    ax.text(3.07, 0.02, '3-month\nlag peak', fontsize=9.5, color=GREY, va='bottom')
    ax.set_xlabel('Lag (months)')
    ax.set_ylabel('Cross-correlation coefficient $r$')
    ax.set_title('Figure 4. Temporal Routing Lag — Precipitation vs. Sentinel-1 Backscatter', pad=8)
    ax.set_xticks(lags)
    ax.set_ylim(-0.45, 0.60)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='#dddddd')
    ax.set_axisbelow(True)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig4_routing_lag.png', dpi=600)
    plt.close()
    
    # ── Figure 5: Baseline Comparison ─────────────────────────────────────────────
    models     = ['RF\n(Terrain)', 'RF\n(Climate)', 'RF\n(Combined)', 'LSTM', 'Transformer']
    f1_normal  = [0.148, 0.062, 0.151, 0.000, 0.000]
    f1_monsoon = [0.835, 0.601, 0.838, 0.752, 0.767]
    f1_flood   = [0.842, 0.626, 0.845, 0.745, 0.751]
    
    x   = np.arange(len(models))
    w   = 0.26
    
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    b1 = ax.bar(x - w,   f1_normal,  w, label='Normal/Dry seasons', color='#aec7e8', edgecolor='white')
    b2 = ax.bar(x,       f1_monsoon, w, label='Monsoon (Jul–Sep 2022)', color=BLUE, edgecolor='white')
    b3 = ax.bar(x + w,   f1_flood,   w, label='Extreme Flood (Aug–Oct 2022)', color='#08306b', edgecolor='white')
    
    ax.set_ylabel('F1-score (Water Class)')
    ax.set_title('Figure 5. Baseline Model Comparison — F1-scores across Hydrological Periods', pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='#dddddd')
    ax.set_axisbelow(True)
    ax.legend(loc='upper right')
    
    ax.annotate('Collapse\n(F1=0.000)', xy=(3 - w, 0.02), xytext=(3 - w, 0.22),
                fontsize=9, ha='center', color=RED,
                arrowprops=dict(arrowstyle='->', color=RED, lw=1))
    
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig5_baseline_comparison.png', dpi=600)
    plt.close()
    
    # ── Figure 6: Ablation Study ──────────────────────────────────────────────────
    metrics   = ['F1', 'IoU', 'Precision', 'Recall', 'AUC']
    terrain   = [0.842, 0.728, 0.894, 0.797, 0.938]
    climate   = [0.626, 0.456, 0.640, 0.613, 0.743]
    combined  = [0.845, 0.732, 0.893, 0.803, 0.931]
    
    ci_t = [(0.833, 0.851), (0.714, 0.740), (0.884, 0.903), (0.784, 0.809), (0.934, 0.943)]
    ci_c = [(0.615, 0.638), (0.444, 0.468), (0.626, 0.654), (0.598, 0.627), (0.733, 0.752)]
    ci_cb= [(0.837, 0.854), (0.719, 0.745), (0.884, 0.902), (0.790, 0.814), (0.926, 0.936)]
    
    err_t  = [(v - lo, hi - v) for v, (lo, hi) in zip(terrain,  ci_t)]
    err_c  = [(v - lo, hi - v) for v, (lo, hi) in zip(climate,  ci_c)]
    err_cb = [(v - lo, hi - v) for v, (lo, hi) in zip(combined, ci_cb)]
    
    x = np.arange(len(metrics))
    w = 0.26
    
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for i, (vals, errs, label, color) in enumerate([
            (terrain,  err_t,  'Terrain Only', BLUE),
            (climate,  err_c,  'Climate Only', RED),
            (combined, err_cb, 'Combined',     GREEN),
    ]):
        lo = [e[0] for e in errs]
        hi = [e[1] for e in errs]
        ax.bar(x + (i-1)*w, vals, w, label=label, color=color, alpha=0.85, edgecolor='white')
        ax.errorbar(x + (i-1)*w, vals, yerr=[lo, hi], fmt='none', color='#333333', capsize=3, linewidth=0.9)
        
    ax.set_ylabel('Score')
    ax.set_title('Figure 6. Feature Split Ablation — 95% Bootstrap CI, Extreme Flood Test Period', pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10.5)
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='#dddddd')
    ax.set_axisbelow(True)
    ax.legend(loc='lower right')
    
    ax.text(0.01, 0.97, 'McNemar p = 0.2584\n(Terrain ≈ Combined, not significant)',
            transform=ax.transAxes, fontsize=9.5, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fffbe6', edgecolor='#cccc88', alpha=0.9))
            
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig6_ablation.png', dpi=600)
    plt.close()
    
    # ── Figure 7: Conceptual Motivation ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12.5, 4.5), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 4.5)
    
    def draw_box(x, y, w, h, title, points, facecolor, edgecolor):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.2, title, ha='center', va='top', fontsize=11, fontweight='bold', color='#0f172a')
        y_offset = y + h - 0.7
        for pt in points:
            ax.text(x + 0.15, y_offset, "• " + pt, ha='left', va='top', fontsize=8.5, color='#334155')
            y_offset -= 0.3
            
    draw_box(0.4, 0.4, 2.7, 3.4, "1. Connectivity\n(Spatial Boundary)", 
             ["87.81% Gini feature weight", "River/coast distance & elevation", "Constrains inundation boundaries", "Determines WHERE flooding occurs"], 
             "#f0f7ff", "#1d4ed8")
             
    draw_box(3.4, 0.4, 2.7, 3.4, "2. Temporal Memory\n(Dynamic Trigger)", 
             ["12.19% climate feature weight", "3-month coastal routing lag", "Cropland irrigation divergence", "Regulates WHEN flooding occurs"], 
             "#f0fdf4", "#15803d")
             
    draw_box(6.4, 0.4, 2.7, 3.4, "3. Empirical Failures\n(Model Collapse)", 
             ["Spatial models (RF) overfit locally", "Unable to generalize spatially", "LSTMs & Transformers collapse", "F1 = 0.000 in normal dry seasons"], 
             "#fff5f5", "#b91c1c")
             
    draw_box(9.4, 0.4, 2.7, 3.4, "4. Future Work\n(Physically Constrained)", 
             ["Motivates state-space models", "Integrate mass balance limits", "Couple storage and routing", "Enhance spatial generalization"], 
             "#fffbeb", "#d97706")
             
    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(label, xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=2, color='#2b2d42', shrinkA=0, shrinkB=0),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
                    
    draw_arrow(3.15, 2.1, 3.35, 2.1)
    draw_arrow(6.15, 2.1, 6.35, 2.1)
    draw_arrow(9.15, 2.1, 9.35, 2.1)
    
    ax.text(6.25, 4.2, "CONCEPTUAL PATHWAY: FROM PROCESS DIAGNOSTICS TO PHYSICAL ARCHITECTURES", ha='center', va='center', fontsize=12.5, fontweight='bold', color='#0f172a')
    
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig7_motivation.png', dpi=600)
    plt.close()
    
    print(f"\nAll 7 journal figures generated successfully and saved to {OUT_DIR}/")

if __name__ == "__main__":
    main()
