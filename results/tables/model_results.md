### Table 1. Classification performance (F1-score) of baseline models across normal, monsoon, and extreme flood periods on the Indus Delta point dataset (2018-2024 test split).

| Model | Architecture | Normal F1 (Dry Season) | Monsoon F1 (Jul-Sep 2022) | Extreme F1 (Aug-Oct 2022) |
| :--- | :--- | :---: | :---: | :---: |
| Random Forest (Terrain Only) | Static classifier | 0.148 | 0.835 | 0.842 |
| Random Forest (Combined) | Static classifier | 0.151 | 0.838 | 0.845 |
| LSTM | Sequence model | 0.000 | 0.752 | 0.745 |
| Transformer | Sequence model | 0.000 | 0.767 | 0.751 |
