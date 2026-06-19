import os
from PIL import Image, ImageDraw, ImageFont

def create_schematic(filename, text, size=(800, 600), bg_color=(240, 244, 248), text_color=(33, 37, 41)):
    # Create an image with a light gray-blue background
    img = Image.new('RGB', size, color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Draw a stylish border
    draw.rectangle([10, 10, size[0]-10, size[1]-10], outline=(100, 110, 120), width=3)
    
    # Use default font or try to load a standard system font
    font = None
    try:
        # Try a few common font paths on macOS
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf"
        ]
        for path in font_paths:
            if os.path.exists(path):
                font = ImageFont.truetype(path, size=24)
                break
    except Exception:
        pass
    
    # Draw text
    lines = text.split('\n')
    y = size[1] // 2 - (len(lines) * 15)
    for line in lines:
        # Simple text drawing
        if font:
            w = draw.textlength(line, font=font)
            draw.text(((size[0] - w) // 2, y), line, fill=text_color, font=font)
            y += font.size + 10
        else:
            draw.text((size[0]//2 - 100, y), line, fill=text_color)
            y += 20
            
    img.save(filename)
    print(f"Generated schematic: {filename}")

if __name__ == "__main__":
    os.makedirs("prism/figures", exist_ok=True)
    
    # Figure 1: Study Area Schematic
    fig1_text = (
        "Figure 1: Indus River Delta Study Area Schematic\n\n"
        "Geographic Boundaries:\n"
        "Latitude: 23.0 N to 24.8 N\n"
        "Longitude: 67.0 E to 69.0 E\n\n"
        "Sampling Stratification:\n"
        "- Permanent Water (JRC > 50%): 300 points\n"
        "- Seasonal Wetlands (JRC 1-50%): 300 points\n"
        "- Dry Land (JRC = 0%): 400 points\n\n"
        "[Note: Map coordinates and digital elevation boundaries\n"
        "used for training and validation]"
    )
    create_schematic("prism/figures/fig1_study_area.png", fig1_text)
    
    # Figure 2: Workflow Schematic
    fig2_text = (
        "Figure 2: Methodological Workflow Schematic\n\n"
        "1. Google Earth Engine Data Extraction (35,000 spatial-temporal records)\n"
        "   - Sentinel-1 (SAR VV/VH), Sentinel-2 (Indices), SRTM DEM, CHIRPS, ERA5-Land\n"
        "2. Feature Splitting (Leakage-Free Configuration)\n"
        "   - Climate Only vs. Terrain Only vs. Combined\n"
        "3. Machine Learning Classifiers (Random Forest, LSTM, Transformer)\n"
        "4. Diagnostic Evaluations (Gini Feature Importance, Routing Lag, Ablations)\n"
        "5. Statistical Significance Audits (Bootstrap CI, Edwards' McNemar Test)"
    )
    create_schematic("prism/figures/fig2_workflow.png", fig2_text)
