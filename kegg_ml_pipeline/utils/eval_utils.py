from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def compute_auroc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_proba))


def compute_auprc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_proba))


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    print("Confusion Matrix:")
    print(f"  TN={tn}  FP={fp}")
    print(f"  FN={fn}  TP={tp}")
