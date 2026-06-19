import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_figure7_motivation():
    # Widen figure to provide extra room for text columns
    fig, ax = plt.subplots(figsize=(12.5, 4.5), dpi=300)
    ax.axis('off')
    
    # Set plot range
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 4.5)
    
    # Helper to draw boxes with new high-contrast theme
    def draw_box(x, y, w, h, title, points, facecolor, edgecolor):
        # Draw box
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", 
                                      linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor)
        ax.add_patch(rect)
        # Draw title (dark slate, bold)
        ax.text(x + w/2, y + h - 0.2, title, ha='center', va='top', fontsize=11, fontweight='bold', color='#0f172a')
        
        # Draw bullet points (charcoal, compact)
        y_offset = y + h - 0.7
        for pt in points:
            ax.text(x + 0.15, y_offset, "• " + pt, ha='left', va='top', fontsize=8.5, color='#334155')
            y_offset -= 0.3

    # Block 1: Geomorphological Connectivity (Blue Category)
    draw_box(0.4, 0.4, 2.7, 3.4, 
             "1. Connectivity\n(Spatial Boundary)", 
             [
                 "87.81% Gini feature weight",
                 "River/coast distance & elevation",
                 "Constrains inundation boundaries",
                 "Determines WHERE flooding occurs"
             ], 
             "#f0f7ff", "#1d4ed8")

    # Block 2: Hydrological Memory (Green Category)
    draw_box(3.4, 0.4, 2.7, 3.4, 
             "2. Temporal Memory\n(Dynamic Trigger)", 
             [
                 "12.19% climate feature weight",
                 "3-month coastal routing lag",
                 "Cropland irrigation divergence",
                 "Regulates WHEN flooding occurs"
             ], 
             "#f0fdf4", "#15803d")

    # Block 3: Model Failure Modes (Red/Orange Category)
    draw_box(6.4, 0.4, 2.7, 3.4, 
             "3. Empirical Failures\n(Model Collapse)", 
             [
                 "Spatial models (RF) overfit locally",
                 "Unable to generalize spatially",
                 "LSTMs & Transformers collapse",
                 "F1 = 0.000 in normal dry seasons"
             ], 
             "#fff5f5", "#b91c1c")

    # Block 4: Future Work (Yellow/Gold Category)
    draw_box(9.4, 0.4, 2.7, 3.4, 
             "4. Future Work\n(Physically Constrained)", 
             [
                 "Motivates state-space models",
                 "Integrate mass balance limits",
                 "Couple storage and routing",
                 "Enhance spatial generalization"
             ], 
             "#fffbeb", "#d97706")
             
    # Draw Arrows between boxes
    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(label, xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=2, color='#2b2d42', shrinkA=0, shrinkB=0),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
                    
    draw_arrow(3.15, 2.1, 3.35, 2.1)
    draw_arrow(6.15, 2.1, 6.35, 2.1)
    draw_arrow(9.15, 2.1, 9.35, 2.1)
    
    # Legend or Flow Label
    ax.text(6.25, 4.2, "CONCEPTUAL PATHWAY: FROM PROCESS DIAGNOSTICS TO PHYSICAL ARCHITECTURES", 
            ha='center', va='center', fontsize=12.5, fontweight='bold', color='#0f172a')
    
    plt.tight_layout()
    plt.savefig('prism/figures/fig7_motivation.png', dpi=600)
    plt.savefig('prism/figures/fig7_motivation.pdf', format='pdf', dpi=600)
    plt.close()
    print("Generated summary flowchart for Figure 7.")

if __name__ == "__main__":
    generate_figure7_motivation()
