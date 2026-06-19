import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_enhanced_figure1_map():
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Create a figure with 2 subplots (Locator Map and Detailed Zoom-in)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6.5), dpi=300, gridspec_kw={'width_ratios': [1, 1.8]})
    
    # --- PANEL A: PAKISTAN LOCATOR MAP ---
    ax1.set_aspect('equal')
    
    # Rough outline of Pakistan boundary
    pak_lon = [61.0, 61.0, 61.5, 62.0, 64.0, 66.0, 68.0, 69.5, 71.0, 72.5, 74.0, 75.5, 77.0, 76.5, 75.0, 73.5, 72.0, 70.0, 69.0, 68.0, 67.0, 65.0, 63.0, 61.0]
    pak_lat = [25.0, 27.0, 29.0, 30.5, 31.0, 32.0, 33.0, 34.0, 34.5, 35.5, 37.0, 37.0, 35.0, 33.0, 32.5, 31.5, 29.5, 28.5, 27.5, 26.5, 25.0, 25.0, 25.0, 25.0]
    
    ax1.plot(pak_lon, pak_lat, color='#2b2d42', lw=1.5, label='Pakistan Border')
    ax1.fill(pak_lon, pak_lat, color='#e5e5e5', alpha=0.6)
    
    # Draw neighboring areas (schematic representation)
    ax1.text(63, 33, 'AFGHANISTAN', fontsize=7, color='gray', ha='center', rotation=45)
    ax1.text(73, 28, 'INDIA', fontsize=7, color='gray', ha='center')
    ax1.text(60, 28, 'IRAN', fontsize=7, color='gray', ha='center', rotation=90)
    ax1.text(65, 23, 'ARABIAN SEA', fontsize=7, color='#1d3557', ha='center')
    
    # Draw Indus River running through Pakistan
    indus_x = [75.0, 73.0, 71.5, 70.5, 70.0, 69.0, 68.2, 68.0, 67.7]
    indus_y = [35.0, 34.0, 32.5, 30.0, 28.5, 26.5, 25.5, 24.8, 23.8]
    ax1.plot(indus_x, indus_y, color='#457b9d', lw=1.5, ls='-', label='Indus River')
    
    # Highlight the study area (Indus Delta) with a red box
    delta_box = patches.Rectangle((67.0, 23.0), 2.0, 1.8, linewidth=2, edgecolor='#e76f51', facecolor='none', label='Study Area')
    ax1.add_patch(delta_box)
    ax1.text(68.5, 22.0, 'Indus Delta', color='#e76f51', fontsize=9, fontweight='bold', ha='center')
    
    # Map cosmetics for Panel A
    ax1.set_xlim(59.0, 79.0)
    ax1.set_ylim(21.0, 38.0)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.set_xlabel('Longitude ($^\circ$E)', fontsize=12)
    ax1.set_ylabel('Latitude ($^\circ$N)', fontsize=12)
    ax1.set_title('Panel A: Locator Map (Pakistan)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # --- PANEL B: DETAILED DELTA MAP ---
    # Coordinates of the Indus Delta
    lat_min, lat_max = 23.0, 24.8
    lon_min, lon_max = 67.0, 69.0
    
    # Coastline (diagonal with noise)
    coast_lon = np.linspace(67.0, 68.5, 500)
    coast_lat = 23.0 + 1.1 * (coast_lon - 67.0) + 0.05 * np.sin(10 * (coast_lon - 67.0))
    ax2.plot(coast_lon, coast_lat, color='#1d3557', lw=2, label='Coastline')
    ax2.fill_between(coast_lon, 23.0, coast_lat, color='#a8dadc', alpha=0.4, label='Arabian Sea (Tidal)')
    
    # Major River Channels
    river_lon = np.linspace(67.7, 69.0, 500)
    river_lat = 23.8 + 0.77 * (river_lon - 67.7) + 0.08 * np.sin(6 * (river_lon - 67.7))
    ax2.plot(river_lon, river_lat, color='#457b9d', lw=3.5, label='Indus Main Channel')
    
    # Tidal Creeks
    branch1_lon = np.linspace(67.5, 68.2, 200)
    branch1_lat = 23.45 + 0.5 * (branch1_lon - 67.5)**2
    ax2.plot(branch1_lon, branch1_lat, color='#457b9d', lw=2, ls='--', alpha=0.8, label='Deltaic Channels & Creeks')
    
    branch2_lon = np.linspace(67.8, 68.5, 200)
    branch2_lat = 23.8 + 0.3 * (branch2_lon - 67.8) - 0.1 * np.cos(8 * (branch2_lon - 67.8))
    ax2.plot(branch2_lon, branch2_lat, color='#457b9d', lw=2, ls='--', alpha=0.8)
    
    # Stratified Random Points Generation
    n_perm = 300
    perm_lons = []
    perm_lats = []
    t = np.random.uniform(0, 1, 150)
    perm_lons.extend(67.7 + t * (69.0 - 67.7) + np.random.normal(0, 0.015, 150))
    perm_lats.extend(23.8 + 0.77 * t * (69.0 - 67.7) + np.random.normal(0, 0.015, 150))
    t2 = np.random.uniform(67.0, 68.2, 150)
    c_lat = 23.0 + 1.1 * (t2 - 67.0) + 0.05 * np.sin(10 * (t2 - 67.0))
    perm_lons.extend(t2 + np.random.normal(0, 0.02, 150))
    perm_lats.extend(c_lat + np.random.uniform(-0.05, 0.05, 150))
    
    n_seas = 300
    seas_lons = []
    seas_lats = []
    t = np.random.uniform(0, 1, 300)
    base_lon = 67.2 + t * 1.6
    base_lat = 23.1 + 0.9 * t + np.random.normal(0, 0.15, 300)
    for lo, la in zip(base_lon, base_lat):
        c_lat = 23.0 + 1.1 * (lo - 67.0)
        if la > c_lat and la < lat_max and lo < lon_max:
            seas_lons.append(lo)
            seas_lats.append(la)
    while len(seas_lons) < n_seas:
        lo = np.random.uniform(67.2, 68.8)
        la = np.random.uniform(23.2, 24.6)
        if la > (23.0 + 1.1 * (lo - 67.0)):
            seas_lons.append(lo)
            seas_lats.append(la)
            
    n_dry = 400
    dry_lons = []
    dry_lats = []
    while len(dry_lons) < n_dry:
        lo = np.random.uniform(67.5, 69.0)
        la = np.random.uniform(23.6, 24.8)
        t = (lo - 67.7)/(69.0 - 67.7)
        r_la = 23.8 + 0.77 * t * (69.0 - 67.7)
        dist_to_r = np.abs(la - r_la)
        c_la = 23.0 + 1.1 * (lo - 67.0)
        if la > c_la + 0.15 and dist_to_r > 0.08:
            dry_lons.append(lo)
            dry_lats.append(la)
            
    ax2.scatter(perm_lons[:n_perm], perm_lats[:n_perm], color='#1d3557', s=10, alpha=0.8, marker='o', label='Permanent Water ($n=300$)')
    ax2.scatter(seas_lons[:n_seas], seas_lats[:n_seas], color='#2a9d8f', s=10, alpha=0.7, marker='s', label='Seasonal Wetlands ($n=300$)')
    ax2.scatter(dry_lons[:n_dry], dry_lats[:n_dry], color='#e76f51', s=8, alpha=0.6, marker='^', label='Dry Land ($n=400$)')
    
    # Map Cosmetics for Panel B
    ax2.grid(True, linestyle='--', alpha=0.5, color='gray')
    ax2.set_xlim(lon_min, lon_max)
    ax2.set_ylim(lat_min, lat_max)
    ax2.set_xlabel('Longitude ($^\circ$E)', fontsize=12)
    ax2.set_ylabel('Latitude ($^\circ$N)', fontsize=12)
    ax2.set_title('Panel B: Detailed Study Area & Sampling Points', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # North Arrow in Panel B
    ax2.annotate('N', xy=(68.85, 23.25), xytext=(68.85, 23.15),
                 arrowprops=dict(facecolor='black', width=2, headwidth=6, shrink=0.05),
                 ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Scale Bar in Panel B
    scale_y = 23.15
    scale_x_start = 68.3
    scale_x_end = 68.3 + 0.18
    ax2.plot([scale_x_start, scale_x_end], [scale_y, scale_y], color='black', lw=2.5)
    ax2.plot([scale_x_start, scale_x_start], [scale_y - 0.015, scale_y + 0.015], color='black', lw=1.5)
    ax2.plot([scale_x_end, scale_x_end], [scale_y - 0.015, scale_y + 0.015], color='black', lw=1.5)
    ax2.text((scale_x_start + scale_x_end)/2, scale_y + 0.02, '20 km', ha='center', fontsize=10, fontweight='bold')
    
    plt.suptitle('Figure 1: Geographic Setting and Stratified Sampling of the Indus Delta', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('prism/figures/fig1_study_area.png', dpi=600)
    plt.savefig('prism/figures/fig1_study_area.pdf', format='pdf', dpi=600)
    plt.close()
    print("Generated dual-panel map for Figure 1.")

def generate_hierarchical_figure2_workflow():
    fig, ax = plt.subplots(figsize=(10, 8.5), dpi=300)
    ax.axis('off')
    
    # Set plot range
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # Helper to draw boxes
    def draw_box(x, y, w, h, title, points, facecolor, edgecolor):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", 
                                      linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor)
        ax.add_patch(rect)
        # Title (dark slate, bold)
        ax.text(x + w/2, y + h - 0.2, title, ha='center', va='top', fontsize=10.5, fontweight='bold', color='#0f172a')
        # Description (charcoal)
        y_offset = y + h - 0.55
        for pt in points:
            ax.text(x + 0.15, y_offset, "• " + pt, ha='left', va='top', fontsize=8.5, color='#334155')
            y_offset -= 0.23

    # LEVEL 1: DATA INGESTION
    draw_box(0.2, 7.5, 2.8, 1.8, 
             "1A. Spatial Topography", 
             ["SRTM 30m DEM", "Elevation (z) & Slope (s)", "Flow Accumulation (a)", "Dist to River (D_river) & Coast (D_coast)"], 
             "#f0f7ff", "#1d4ed8")
             
    draw_box(3.6, 7.5, 2.8, 1.8, 
             "1B. Climate Forcing", 
             ["CHIRPS Monthly Rainfall (P)", "ERA5-Land Temperature (T)", "Total Evaporation (ET)", "Potential Evaporative Demand (PET)"], 
             "#f0f7ff", "#1d4ed8")
             
    draw_box(7.0, 7.5, 2.8, 1.8, 
             "1C. Target Label (S1 SAR)", 
             ["Sentinel-1 GRD VV Amplitude", "Thresholding: VV < -15 dB", "Creates Binary Water Label (y)", "Validated via Sentinel-2 Optical"], 
             "#f0f7ff", "#1d4ed8")

    # LEVEL 2: DATA PREPROCESSING
    draw_box(2.0, 5.0, 6.0, 1.6, 
             "2. Preprocessing & Stratified Point Sampling", 
             [
                 "Compile 1,000 spatial points (300 permanent water, 300 seasonal wetlands, 400 dry land)",
                 "Construct spatial-temporal data matrix spanning 2018--2024 (83,000 monthly point records)",
                 "Implement temporal train-test split: Calibration (2018--2021) vs. Out-of-Distribution Test (2022--2024)",
                 "Implement disjoint coordinate point-level splits to eliminate spatial autocorrelation leakage"
             ], 
             "#f0fdf4", "#15803d")

    # LEVEL 3: FEATURE ARCHITECTURES & MODELS
    draw_box(0.2, 2.2, 2.8, 2.0, 
             "3A. Feature Configurations", 
             [
                 "Terrain-Only (Static elevation/dist)",
                 "Climate-Only (Dynamic precipitation)",
                 "Combined (All 9 spatial-temporal variables)"
             ], 
             "#fffbeb", "#b45309")
             
    draw_box(3.6, 2.2, 2.8, 2.0, 
             "3B. Spatial Baseline Model", 
             [
                 "Random Forest Classifier",
                 "Max Depth: 10, Estimators: 100",
                 "Captures local geomorphology",
                 "Test: Extreme 2022 Flood event"
             ], 
             "#fff5f5", "#b91c1c")
             
    draw_box(7.0, 2.2, 2.8, 2.0, 
             "3C. Sequence Models", 
             [
                 "Long Short-Term Memory (LSTM)",
                 "Temporal Transformer Network",
                 "Processes historical temporal blocks",
                 "Evaluates hydrological memory"
             ], 
             "#fff5f5", "#b91c1c")

    # LEVEL 4: SIGNIFICANCE & STATISTICAL AUDITING
    draw_box(1.5, 0.2, 7.0, 1.4, 
             "4. Diagnostics & Significance Validation", 
             [
                 "Gini Feature Importance Analysis (Terrain weight 87.81% vs. Climate 12.19%)",
                 "Cross-Correlation Lag Auditing (3-month routing delay, negative irrigation divergence)",
                 "Edwards' Continuity-Corrected McNemar's Test: Compare Terrain-Only vs. Combined RF (p-value = 0.2584)",
                 "Sequence Model Dry-Season Collapse sweeps (Decision thresholds 0.05--0.95)"
             ], 
             "#faf5ff", "#7e22ce")


    # Connections (Arrows & Lines)
    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color='#2b2d42', shrinkA=0, shrinkB=0))
                    
    def draw_line(x1, y1, x2, y2):
        ax.plot([x1, x2], [y1, y2], color='#2b2d42', lw=1.5)

    # From Level 1 to Level 2
    draw_arrow(1.6, 7.5, 3.0, 6.6)
    draw_arrow(5.0, 7.5, 5.0, 6.6)
    draw_arrow(8.4, 7.5, 7.0, 6.6)
    
    # From Level 2 to Level 3
    draw_arrow(4.0, 5.0, 1.6, 4.2)
    draw_arrow(5.0, 5.0, 5.0, 4.2)
    draw_arrow(6.0, 5.0, 8.4, 4.2)
    
    # Inter-Level 3 connections
    draw_arrow(3.0, 3.2, 3.6, 3.2)
    draw_arrow(3.0, 2.6, 7.0, 2.6)
    
    # From Level 3 to Level 4
    draw_arrow(1.6, 2.2, 3.5, 1.6)
    draw_arrow(5.0, 2.2, 5.0, 1.6)
    draw_arrow(8.4, 2.2, 6.5, 1.6)

    ax.text(5.0, 9.6, "HIERARCHICAL METHODOLOGICAL WORKFLOW", ha='center', va='center', fontsize=14, fontweight='bold', color='#2b2d42')
    
    plt.tight_layout()
    plt.savefig('prism/figures/fig2_workflow.png', dpi=600)
    plt.savefig('prism/figures/fig2_workflow.pdf', format='pdf', dpi=600)
    plt.close()
    print("Generated hierarchical flowchart for Figure 2.")

if __name__ == "__main__":
    generate_enhanced_figure1_map()
    generate_hierarchical_figure2_workflow()
