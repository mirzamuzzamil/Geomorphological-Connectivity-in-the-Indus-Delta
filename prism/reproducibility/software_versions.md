# Software Versions and Dependencies

The baseline modeling, data extraction, and statistical analyses were executed using the following software stack:

## Core Environment
- **Operating System:** macOS (or Linux compatible)
- **Python Version:** `3.10.x` or `3.11.x`

## Core Libraries & Packages
- **PyTorch:** `2.1.2+cpu` (or newer)
- **Scikit-Learn:** `1.3.2`
- **Google Earth Engine Python API (earthengine-api):** `0.1.381`
- **Numpy:** `1.26.2`
- **Scipy:** `1.11.4` (used for cross-correlation and McNemar test calculations)
- **Pandas:** `2.1.4`
- **Pillow (PIL):** `10.1.0` (used for schematic drawing)
- **Matplotlib:** `3.8.2` (used for figure generation)

## Installation Guide
To install all necessary dependencies locally, execute the following command:
```bash
pip install torch==2.1.2 scikit-learn==1.3.2 earthengine-api==0.1.381 numpy==1.26.2 scipy==1.11.4 pandas==2.1.4 Pillow==10.1.0 matplotlib==3.8.2
```
