"""
Journal-quality figure regeneration for HESS submission.
White background, no decorative colors, Inter/Arial font, 300 dpi.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

# ── Style ─────────────────────────────────────────────────────────────────────
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
OUT    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'prism', 'figures')

# ── Figure 3: Feature Importance ──────────────────────────────────────────────
features = [
    'River Distance',
    'Coastal Distance',
    'Elevation',
    'Slope',
    'Flow Accumulation',
    'Precipitation',
    'Evaporation',
    'Temperature',
    'PET',
]
# Terrain: 0.3110+0.2769+0.1389+0.0813+0.0700 = 0.8781 (87.81%)
# Climate: 0.0620+0.0310+0.0189+0.0100         = 0.1219 (12.19%)
# Total  : 1.0000
importance = [0.3110, 0.2769, 0.1389, 0.0813, 0.0700,
              0.0620, 0.0310, 0.0189, 0.0100]
colors = [BLUE if i < 5 else RED for i in range(len(features))]

fig, ax = plt.subplots(figsize=(6.5, 3.8))
bars = ax.barh(features[::-1], importance[::-1], color=colors[::-1],
               height=0.6, edgecolor='white', linewidth=0.3)
ax.set_xlabel('Mean Decrease in Gini Impurity (Normalised)')
ax.set_title('Figure 3. Random Forest Feature Importance — Indus Delta', pad=8)
ax.axvline(0, color='#222222', linewidth=0.8)
ax.set_xlim(0, 0.42)
ax.xaxis.grid(True, linestyle='--', linewidth=0.5, color='#dddddd')
ax.set_axisbelow(True)

# Annotate bars
for bar, val in zip(bars, importance[::-1]):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', ha='left', fontsize=9.5, color=GREY)

# Cumulative sum annotation
cum = np.cumsum(sorted(importance, reverse=True))
terrain_sum = sum(importance[:5])
ax.text(0.38, 0.08, f'Terrain total: {terrain_sum:.1%}',
        transform=ax.transAxes, fontsize=10, color=BLUE,
        ha='right', style='italic')

patch_t = mpatches.Patch(color=BLUE, label='Terrain features (static)')
patch_c = mpatches.Patch(color=RED,  label='Climate features (dynamic)')
ax.legend(handles=[patch_t, patch_c], loc='lower right')

plt.tight_layout()
plt.savefig(f'{OUT}/fig3_feature_importance.png', dpi=600)
plt.savefig(f'{OUT}/fig3_feature_importance.pdf', format='pdf', dpi=600)
plt.close()
print('fig3 done')

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
ax.set_title('Figure 4. Temporal Routing Lag — Precipitation vs.\ Sentinel-1 Backscatter', pad=8)
ax.set_xticks(lags)
ax.set_ylim(-0.45, 0.60)
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='#dddddd')
ax.set_axisbelow(True)
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(f'{OUT}/fig4_routing_lag.png', dpi=600)
plt.savefig(f'{OUT}/fig4_routing_lag.pdf', format='pdf', dpi=600)
plt.close()
print('fig4 done')

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

# Annotate collapse
ax.annotate('Collapse\n(F1=0.000)', xy=(3 - w, 0.02), xytext=(3 - w, 0.22),
            fontsize=9, ha='center', color=RED,
            arrowprops=dict(arrowstyle='->', color=RED, lw=1))

plt.tight_layout()
plt.savefig(f'{OUT}/fig5_baseline_comparison.png', dpi=600)
plt.savefig(f'{OUT}/fig5_baseline_comparison.pdf', format='pdf', dpi=600)
plt.close()
print('fig5 done')

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
    ax.errorbar(x + (i-1)*w, vals,
                yerr=[lo, hi],
                fmt='none', color='#333333', capsize=3, linewidth=0.9)

ax.set_ylabel('Score')
ax.set_title('Figure 6. Feature Split Ablation — 95% Bootstrap CI, Extreme Flood Test Period', pad=8)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=10.5)
ax.set_ylim(0, 1.05)
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='#dddddd')
ax.set_axisbelow(True)
ax.legend(loc='lower right')

# McNemar annotation
ax.text(0.01, 0.97, 'McNemar p = 0.2584\n(Terrain ≈ Combined, not significant)',
        transform=ax.transAxes, fontsize=9.5, va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fffbe6', edgecolor='#cccc88', alpha=0.9))

plt.tight_layout()
plt.savefig(f'{OUT}/fig6_ablation.png', dpi=600)
plt.savefig(f'{OUT}/fig6_ablation.pdf', format='pdf', dpi=600)
plt.close()
print('fig6 done')

print('\nAll journal figures saved to', OUT)
