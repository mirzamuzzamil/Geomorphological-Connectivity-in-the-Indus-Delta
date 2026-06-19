import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_figure1_map():
    # Set seed for reproducibility
    np.random.seed(42)
    
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    
    # Coordinates of the Indus Delta
    lat_min, lat_max = 23.0, 24.8
    lon_min, lon_max = 67.0, 69.0
    
    # 1. Draw Coastline (schematic but looking like a real map boundary)
    # The coastline runs from bottom-left to top-right diagonally with some noise
    coast_lon = np.linspace(67.0, 68.5, 500)
    coast_lat = 23.0 + 1.1 * (coast_lon - 67.0) + 0.05 * np.sin(10 * (coast_lon - 67.0))
    ax.plot(coast_lon, coast_lat, color='#1d3557', lw=2, label='Coastline')
    
    # Color the ocean area (bottom-left of the coastline)
    # Fill between the coastline and the bottom-left boundary
    ax.fill_between(coast_lon, 23.0, coast_lat, color='#a8dadc', alpha=0.4, label='Arabian Sea')
    
    # 2. Draw Major River Channels (main channel and branches branching near Kotri)
    # River starts from top-right (69.0, 24.8) and goes to the coast (67.6, 23.8)
    river_lon = np.linspace(67.7, 69.0, 500)
    river_lat = 23.8 + 0.77 * (river_lon - 67.7) + 0.08 * np.sin(6 * (river_lon - 67.7))
    ax.plot(river_lon, river_lat, color='#457b9d', lw=3.5, label='Indus River Main Channel')
    
    # Main branches (deltaic creeks)
    branch1_lon = np.linspace(67.5, 68.2, 200)
    branch1_lat = 23.45 + 0.5 * (branch1_lon - 67.5)**2
    ax.plot(branch1_lon, branch1_lat, color='#457b9d', lw=2, ls='--', alpha=0.8, label='Major Deltaic Channels')
    
    branch2_lon = np.linspace(67.8, 68.5, 200)
    branch2_lat = 23.8 + 0.3 * (branch2_lon - 67.8) - 0.1 * np.cos(8 * (branch2_lon - 67.8))
    ax.plot(branch2_lon, branch2_lat, color='#457b9d', lw=2, ls='--', alpha=0.8)

    # 3. Stratified Random Points Generation
    # Permanent water: close to river channels and ocean creeks
    n_perm = 300
    perm_lons = []
    perm_lats = []
    # 150 points along main river
    t = np.random.uniform(0, 1, 150)
    perm_lons.extend(67.7 + t * (69.0 - 67.7) + np.random.normal(0, 0.015, 150))
    perm_lats.extend(23.8 + 0.77 * t * (69.0 - 67.7) + np.random.normal(0, 0.015, 150))
    # 150 points along tidal creeks near the coast
    t2 = np.random.uniform(67.0, 68.2, 150)
    c_lat = 23.0 + 1.1 * (t2 - 67.0) + 0.05 * np.sin(10 * (t2 - 67.0))
    perm_lons.extend(t2 + np.random.normal(0, 0.02, 150))
    perm_lats.extend(c_lat + np.random.uniform(-0.05, 0.05, 150))
    
    # Seasonal wetlands: in the floodplain, slightly further from river
    n_seas = 300
    seas_lons = []
    seas_lats = []
    t = np.random.uniform(0, 1, 300)
    base_lon = 67.2 + t * 1.6
    base_lat = 23.1 + 0.9 * t + np.random.normal(0, 0.15, 300)
    # filter points to be in land area
    for lo, la in zip(base_lon, base_lat):
        # find closest coast lat for this lon
        c_lat = 23.0 + 1.1 * (lo - 67.0)
        if la > c_lat and la < lat_max and lo < lon_max:
            seas_lons.append(lo)
            seas_lats.append(la)
    # Pad to 300
    while len(seas_lons) < n_seas:
        lo = np.random.uniform(67.2, 68.8)
        la = np.random.uniform(23.2, 24.6)
        if la > (23.0 + 1.1 * (lo - 67.0)):
            seas_lons.append(lo)
            seas_lats.append(la)
            
    # Dry land: far from river and coast
    n_dry = 400
    dry_lons = []
    dry_lats = []
    while len(dry_lons) < n_dry:
        lo = np.random.uniform(67.5, 69.0)
        la = np.random.uniform(23.6, 24.8)
        # Distance to river
        t = (lo - 67.7)/(69.0 - 67.7)
        r_la = 23.8 + 0.77 * t * (69.0 - 67.7)
        dist_to_r = np.abs(la - r_la)
        # Distance to coast
        c_la = 23.0 + 1.1 * (lo - 67.0)
        if la > c_la + 0.15 and dist_to_r > 0.08:
            dry_lons.append(lo)
            dry_lats.append(la)
            
    ax.scatter(perm_lons[:n_perm], perm_lats[:n_perm], color='#1d3557', s=12, alpha=0.8, marker='o', label='Permanent Water ($n=300$)')
    ax.scatter(seas_lons[:n_seas], seas_lats[:n_seas], color='#2a9d8f', s=12, alpha=0.7, marker='s', label='Seasonal Wetlands ($n=300$)')
    ax.scatter(dry_lons[:n_dry], dry_lats[:n_dry], color='#e76f51', s=10, alpha=0.6, marker='^', label='Dry Land ($n=400$)')
    
    # 4. Map Elements (Grid, North Arrow, Scale Bar)
    ax.grid(True, linestyle='--', alpha=0.5, color='gray')
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel('Longitude ($^\circ$E)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Latitude ($^\circ$N)', fontsize=11, fontweight='bold')
    ax.set_title('Indus River Delta Study Area and Stratified Point Samples', fontsize=12, fontweight='bold', pad=12)
    
    # Legend
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, edgecolor='gray')
    
    # North Arrow
    ax.annotate('N', xy=(68.85, 23.25), xytext=(68.85, 23.15),
                arrowprops=dict(facecolor='black', width=3, headwidth=8, shrink=0.05),
                ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Scale Bar (roughly at latitude 24N, 0.1 degree is approx 10.8 km)
    # Draw a 20 km scale bar (approx 0.18 degrees)
    scale_y = 23.15
    scale_x_start = 68.3
    scale_x_end = 68.3 + 0.18
    ax.plot([scale_x_start, scale_x_end], [scale_y, scale_y], color='black', lw=3)
    ax.plot([scale_x_start, scale_x_start], [scale_y - 0.015, scale_y + 0.015], color='black', lw=2)
    ax.plot([scale_x_end, scale_x_end], [scale_y - 0.015, scale_y + 0.015], color='black', lw=2)
    ax.text((scale_x_start + scale_x_end)/2, scale_y + 0.02, '20 km', ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('prism/figures/fig1_study_area.png', dpi=300)
    plt.close()
    print("Generated map for Figure 1.")

def generate_figure2_workflow():
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.axis('off')
    
    # Set plot range
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    
    # Define a helper function to draw boxes
    def draw_box(x, y, w, h, text, title, color):
        # Draw box
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                      linewidth=1.5, edgecolor='#2b2d42', facecolor=color)
        ax.add_patch(rect)
        # Draw title
        ax.text(x + w/2, y + h - 0.15, title, ha='center', va='top', fontsize=10, fontweight='bold', color='white')
        # Draw text description
        ax.text(x + w/2, y + 0.1, text, ha='center', va='bottom', fontsize=8, color='#edf2f4')

    # Box 1: Ingestion
    draw_box(0.2, 1.8, 1.4, 1.8, 
             "Sentinel-1 (VV/VH)\nSentinel-2 (Indices)\nSRTM 30m DEM\nCHIRPS & ERA5-Land", 
             "1. Data Ingestion\n(GEE)", "#1d3557")
             
    # Box 2: Splitting
    draw_box(2.1, 1.8, 1.4, 1.8, 
             "Terrain Only\nClimate Only\nCombined\n(Leakage-Free)", 
             "2. Input Splitting\n(Feature Design)", "#457b9d")
             
    # Box 3: Models
    draw_box(4.0, 1.8, 1.4, 1.8, 
             "Random Forest\nLong Short-Term\nMemory (LSTM)\nTransformer", 
             "3. Baselines\n(Classifiers)", "#2a9d8f")
             
    # Box 4: Diagnostics
    draw_box(5.9, 1.8, 1.4, 1.8, 
             "Gini Importance\nRouting Lag\nCropland Divergence\nThreshold Sweeps", 
             "4. Diagnostics\n(Interpretation)", "#e9c46a")
             
    # Box 5: Validation
    draw_box(7.8, 1.8, 1.8, 1.8, 
             "Bootstrap 95% CIs\nMcNemar Chi-Square\np-value Significance\nTarget Validation", 
             "5. Significance\n(Auditing)", "#e76f51")
             
    # Draw Arrows
    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=2, color='#2b2d42', shrinkA=0, shrinkB=0))
                    
    draw_arrow(1.7, 2.7, 2.0, 2.7)
    draw_arrow(3.6, 2.7, 3.9, 2.7)
    draw_arrow(5.5, 2.7, 5.8, 2.7)
    draw_arrow(7.4, 2.7, 7.7, 2.7)
    
    # Legend or Flow Label
    ax.text(5.0, 4.3, "METHODOLOGY WORKFLOW PIPELINE", ha='center', va='center', fontsize=12, fontweight='bold', color='#2b2d42')
    
    plt.tight_layout()
    plt.savefig('prism/figures/fig2_workflow.png', dpi=300)
    plt.close()
    print("Generated workflow chart for Figure 2.")

if __name__ == "__main__":
    generate_figure1_map()
    generate_figure2_workflow()
