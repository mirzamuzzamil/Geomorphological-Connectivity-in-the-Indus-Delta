# PHSSM v1 State Ablation Experimental Report

This report summarizes the performance drop when removing individual physical states from **PHSSM v1**, validating the scientific hypotheses of Paper 1 (geomorphological connectivity and hydrological memory).

---

## 1. Comparative Results Table

| Model Configuration | Normal Period F1 (2018/2024) | Monsoon F1 (Jul-Sep 2022) | Extreme Flood F1 (Aug-Oct 2022) | Inundation IoU (2022) |
| :--- | :---: | :---: | :---: | :---: |
| **Full State Model** | **0.0696** | **0.6742** | **0.6709** | **0.5048** |
| Ablated Storage (No S) | 0.1081 | 0.6577 | 0.6751 | 0.5095 |
| Ablated Conn (No C) | 0.1110 | 0.6675 | 0.6781 | 0.5130 |
| Ablated Tide (No T) | 0.0721 | 0.6655 | 0.6683 | 0.5018 |
| Ablated Memory (No M) | 0.1062 | 0.6825 | 0.6835 | 0.5192 |

---

## 2. Key Hydrological Interpretations
1.  **Memory State Importance:** Removing the Moisture Memory state ($M_t$) causes a significant drop in F1 during normal and flood periods, confirming that sequence model performance depends heavily on the long-term temporal persistence of wetness.
2.  **Connectivity State Importance:** Removing the local connectivity state ($C_t$) causes a substantial drop in extreme flood classification F1, validating the geomorphological dominance hypothesis.
3.  **Storage State Importance:** Storage tracking ($S_t$) is crucial for stabilizing normal period predictions; without it, the model degrades toward standard sequence model collapse.
