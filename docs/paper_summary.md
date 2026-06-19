# Paper Summary — Indus Delta Surface Water Dynamics

## Title
**Geomorphological Connectivity and Temporal Hydrological Controls of Surface Water Dynamics in the Indus River Delta**

---

## 1. Abstract Summary
Empirical machine learning models are widely applied for mapping surface water dynamics. However, they often suffer from spatial overfitting and data leakage when trained with concurrent satellite observations. 
This study establishes a leakage-free benchmark to isolate the spatial boundaries (geomorphological connectivity) and temporal triggers (meteorological forcing and memory) controlling surface water dynamics in the highly regulated Indus River Delta, Pakistan. 
We evaluate three baseline architectures—Random Forest, LSTM, and Transformer—across normal and extreme flood regimes.

---

## 2. Core Scientific Questions
1. **To what extent does static geomorphology constrain surface water boundaries in deltaic plains?**
2. **What is the temporal lag between meteorological rainfall triggers and inundation occurrence?**
3. **How do sequence models perform under spatial coordinate-withheld settings in dry seasons?**

---

## 3. Key Findings
* **Topographic Dominance:** Static geomorphological features (elevation, slope, distance to river, distance to coast) account for **87.81%** of the model feature importance weight. The difference between the Terrain-only model ($F_1 = 0.8424$) and the Combined model ($F_1 = 0.8454$) is statistically insignificant (McNemar $p = 0.2584$).
* **Hydrological Routing Lag:** Cross-correlation analysis reveals a **3-month lag** between CHIRPS precipitation triggers and Sentinel-1 backscatter-derived inundation in the tidal wetlands ($r = 0.312$).
* **Irrigation Divergence:** Irrigated agricultural zones exhibit a negative correlation at Lag 0 ($r = -0.299$), reflecting anthropogenic water management diverting from natural precipitation cycles.
* **Sequence Model Collapse:** Under coordinate-withheld settings in normal dry seasons, both PyTorch LSTMs and Transformers default to predicting land, leading to an $F_1$-score collapse to **0.000**, highlighting limitations of standard sequence models in handling sub-pixel deltaic channels under coarse meteorological inputs.

---

## 4. Methodological Highlights
* **Leakage-Free Audit:** Sentinel-1 radar backscatter (VV/VH) and Sentinel-2 optical bands (NDVI, NDWI) are strictly withheld from model inputs and used exclusively to derive validation labels.
* **Spatially Disjoint splits:** Spatially independent validation sets prevent spatial overfitting and coordinate memorization.
* **Edwards' McNemar Tests:** Rigorous chi-square significance testing with continuity corrections evaluates the impact of climate forcing vs geomorphology.
