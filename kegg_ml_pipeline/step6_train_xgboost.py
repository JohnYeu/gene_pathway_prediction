from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
import sys

import numpy as np
from sklearn.metrics import f1_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

import config
from utils.eval_utils import compute_auroc, compute_auprc, print_confusion_matrix


def train_with_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict,
    n_splits: int = config.CV_FOLDS,
    results_dir: str = config.RESULTS_DIR,
) -> list[float]:
    """Run stratified k-fold cross-validation and return per-fold AUROC scores.

    Each fold trains a fresh XGBClassifier so no state leaks between folds.
    After all folds, saves a bar chart to ``results_dir/step6_cv_scores.png`` showing
    per-fold AUROC, the mean, and the acceptance threshold.

    Args:
        X_train: Feature matrix of shape ``(n, d)`` used for training.
        y_train: Binary label vector of length ``n``.
        params: XGBoost hyper-parameter dict (e.g. config.XGB_PARAMS).
        n_splits: Number of cross-validation folds.
        results_dir: Directory where ``step6_cv_scores.png`` will be written.

    Returns:
        List of AUROC floats, one per fold.
    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=params.get("random_state", config.RANDOM_STATE),
    )
    cv_scores: list[float] = []

    for i, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        model = XGBClassifier(**dict(params))
        model.fit(X_tr, y_tr)
        y_proba = model.predict_proba(X_val)[:, 1]
        score = compute_auroc(y_val, y_proba)
        cv_scores.append(score)
        print(f"Fold {i}/{n_splits}  AUROC={score:.4f}")

    mean_auroc = float(np.mean(cv_scores))
    std_auroc = float(np.std(cv_scores))
    print(f"CV AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")

    os.makedirs(results_dir, exist_ok=True)
    fig, ax = plt.subplots()
    ax.bar(range(1, n_splits + 1), cv_scores)
    ax.axhline(mean_auroc, color="steelblue", linestyle="-", label=f"Mean={mean_auroc:.4f}")
    ax.axhline(
        config.AUROC_THRESHOLD,
        color="red",
        linestyle="--",
        label=f"Threshold={config.AUROC_THRESHOLD}",
    )
    ax.set_xlabel("Fold")
    ax.set_ylabel("AUROC")
    ax.set_title("Cross-validation AUROC")
    ax.legend()
    plt.tight_layout()
    cv_plot_path = os.path.join(results_dir, "step6_cv_scores.png")
    plt.savefig(cv_plot_path)
    plt.close()
    print(f"Saved → {cv_plot_path}")

    return cv_scores


def train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict,
    model_path: str = config.MODEL_PATH,
) -> XGBClassifier:
    """Train a final XGBClassifier on the full training set and save it to disk.

    The model is serialised with XGBoost's native JSON format so that step 8
    can reload it with ``XGBClassifier().load_model(model_path)``.

    Args:
        X_train: Full training feature matrix.
        y_train: Full training label vector.
        params: XGBoost hyper-parameter dict.
        model_path: Destination path for the serialised model file.

    Returns:
        Fitted XGBClassifier instance.
    """
    model = XGBClassifier(**dict(params))
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    print(f"Saved → {model_path}")

    return model


def evaluate_model(
    model: XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    results_dir: str = config.RESULTS_DIR,
) -> dict[str, float]:
    """Evaluate a trained model on the held-out test set.

    Computes AUROC, AUPRC, Accuracy, F1, and prints a confusion matrix.
    Also saves a ROC curve plot to ``results_dir/step6_roc_curve.png``.

    Args:
        model: Trained XGBClassifier.
        X_test: Test feature matrix.
        y_test: Test label vector.
        results_dir: Directory where ``step6_roc_curve.png`` will be written.

    Returns:
        Dict with keys ``auroc``, ``auprc``, ``accuracy``, ``f1``.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auroc = compute_auroc(y_test, y_proba)
    auprc = compute_auprc(y_test, y_proba)
    acc = float((y_pred == y_test).mean())
    f1 = float(f1_score(y_test, y_pred))

    print(f"Test AUROC : {auroc:.4f}")
    print(f"Test AUPRC : {auprc:.4f}")
    print(f"Accuracy   : {acc:.4f}")
    print(f"F1 Score   : {f1:.4f}")
    print_confusion_matrix(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    os.makedirs(results_dir, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUROC={auroc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    roc_path = os.path.join(results_dir, "step6_roc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"Saved → {roc_path}")

    return {"auroc": auroc, "auprc": auprc, "accuracy": acc, "f1": f1}


def check_auroc_threshold(
    cv_scores: list[float],
    threshold: float = config.AUROC_THRESHOLD,
) -> bool:
    """Check whether the mean CV AUROC meets the acceptance threshold.

    Prints a warning with concrete tuning suggestions when the threshold is
    not met. Never raises an exception or exits the process — the caller
    decides how to handle a failing check.

    Args:
        cv_scores: Per-fold AUROC scores returned by ``train_with_cv``.
        threshold: Minimum acceptable mean AUROC.

    Returns:
        True if mean AUROC >= threshold, False otherwise.
    """
    mean_auroc = float(np.mean(cv_scores))
    if mean_auroc < threshold:
        print(
            f"[WARNING] CV AUROC {mean_auroc:.4f} < threshold {threshold:.4f}. "
            "Tuning suggestions:\n"
            "  - Increase n_estimators (e.g. 500-1000)\n"
            "  - Adjust max_depth (try 4-8)\n"
            "  - Lower learning_rate (e.g. 0.01-0.03)\n"
            "  - Adjust scale_pos_weight to match class imbalance\n"
            "  - Consider Optuna for automated hyperparameter search"
        )
        return False
    print(f"[OK] CV AUROC {mean_auroc:.4f} >= threshold {threshold:.4f}")
    return True


if __name__ == "__main__":
    mock = "--mock" in sys.argv
    train_path  = config.MOCK_TRAIN_DATA  if mock else config.TRAIN_DATA
    test_path   = config.MOCK_TEST_DATA   if mock else config.TEST_DATA
    model_path  = config.MOCK_MODEL_PATH  if mock else config.MODEL_PATH
    results_dir = config.MOCK_RESULTS_DIR if mock else config.RESULTS_DIR

    train_data = np.load(train_path)
    test_data  = np.load(test_path)

    if list(train_data["all_go_terms"]) != list(test_data["all_go_terms"]):
        raise ValueError("train/test all_go_terms mismatch — re-run step5 to regenerate both files")

    X_train, y_train = train_data["X"], train_data["y"]
    X_test, y_test   = test_data["X"],  test_data["y"]

    source_tag = "mock" if mock else "real"
    print(f"[{source_tag}] Train: {X_train.shape}  Test: {X_test.shape}")

    cv_scores = train_with_cv(X_train, y_train, config.XGB_PARAMS, results_dir=results_dir)
    check_auroc_threshold(cv_scores)
    model = train_final_model(X_train, y_train, config.XGB_PARAMS, model_path=model_path)
    evaluate_model(model, X_test, y_test, results_dir=results_dir)
