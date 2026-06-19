import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix

y_true = np.array([1]*336 + [0]*664)
# TP = 298, FN = 38 (true is 1, pred is 0)
# TN = 626, FP = 38 (true is 0, pred is 1)
y_pred = np.array([1]*298 + [0]*38 + [1]*38 + [0]*626)

print("Accuracy: ", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:   ", recall_score(y_true, y_pred))
print("F1:       ", f1_score(y_true, y_pred))
print("Kappa:    ", cohen_kappa_score(y_true, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
