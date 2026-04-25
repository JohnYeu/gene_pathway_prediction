from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
import sys

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

import config


def _build_feature_names(all_go_terms: list[str]) -> list[str]:
    """Build the full ordered feature name list matching the feature matrix columns.

    The order must be identical to the concatenation order used in
    step4_feature_extraction.extract_features():
      GO frequency vector  (len(all_go_terms) dimensions)
      Jaccard stats        (4 dimensions)
      Size features        (2 dimensions)

    Args:
        all_go_terms: Stable-sorted list of GO term identifiers.

    Returns:
        List of length ``len(all_go_terms) + 6``.
    """
    return (
        list(all_go_terms)
        + ["jaccard_mean", "jaccard_min", "jaccard_max", "jaccard_std"]
        + ["log_size", "mean_go_per_gene"]
    )


def global_shap_analysis(
    model: XGBClassifier,
    X_train: np.ndarray,
    all_go_terms: list[str],
    top_n: int = 30,
    shap_dir: str = config.SHAP_DIR,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Compute global SHAP importance and save a beeswarm summary plot.

    Uses TreeExplainer for speed. The resulting feature importance DataFrame
    is also saved as ``step7_global_importance.csv`` so that downstream steps
    (e.g. step8) can load it without re-running SHAP.

    Args:
        model: Trained XGBClassifier.
        X_train: Training feature matrix used to compute SHAP values.
        all_go_terms: Stable-sorted list of GO term identifiers.
        top_n: Number of top features shown in the summary plot.
        shap_dir: Directory where output files will be written.

    Returns:
        Tuple ``(shap_values, importance_df)`` where ``shap_values`` has shape
        ``(n_train, n_features)`` and ``importance_df`` has columns
        ``["feature", "mean_abs_shap"]`` sorted descending.
    """
    feature_names = _build_feature_names(all_go_terms)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_train)

    os.makedirs(shap_dir, exist_ok=True)

    # Beeswarm summary plot
    shap.summary_plot(
        shap_vals, X_train, feature_names=feature_names,
        max_display=top_n, show=False,
    )
    plt.tight_layout()
    summary_path = os.path.join(shap_dir, "step7_global_summary.png")
    plt.savefig(summary_path)
    plt.close()
    print(f"Saved → {summary_path}")

    # Feature importance DataFrame
    importance = np.abs(shap_vals).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance})
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print(f"\nTop-{top_n} features by mean |SHAP|:")
    print(df.head(top_n).to_string(index=False))

    csv_path = os.path.join(shap_dir, "step7_global_importance.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")

    return shap_vals, df


def local_shap_analysis(
    model: XGBClassifier,
    X_sample: np.ndarray,
    sample_id: str,
    all_go_terms: list[str],
    pathway_name: str = "",
    shap_dir: str = config.SHAP_DIR,
    file_prefix: str = "step7",
) -> None:
    """Compute and visualise local SHAP explanation for a single sample.

    Saves a waterfall plot and prints the top 5 positive and negative
    contributing features.

    The output filename pattern is ``{file_prefix}_local_{sample_id}.png``,
    so the file itself records which pipeline step produced it.  The default
    ``file_prefix="step7"`` matches step-7 standalone usage (training-sample
    explanations); step 9 passes ``file_prefix="step9"`` when it generates
    waterfalls for top-K scored candidates.

    Args:
        model: Trained XGBClassifier.
        X_sample: Feature vector of shape ``(d,)`` or ``(1, d)``.
        sample_id: Identifier used in the output filename.
        all_go_terms: Stable-sorted list of GO term identifiers.
        pathway_name: Optional display name shown in the plot title.
        shap_dir: Directory where the waterfall PNG will be written.
        file_prefix: Step prefix to embed in the output filename.
    """
    X_sample = np.asarray(X_sample)
    if X_sample.ndim == 1:
        X_sample = X_sample.reshape(1, -1)
    if X_sample.shape[0] != 1:
        raise ValueError(
            f"X_sample must contain exactly 1 row, got {X_sample.shape[0]}. "
            "Use batch_local_shap() for multiple samples."
        )

    feature_names = _build_feature_names(all_go_terms)
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_sample)
    explanation.feature_names = feature_names

    os.makedirs(shap_dir, exist_ok=True)

    title = (
        f"Local SHAP: {pathway_name} ({sample_id})"
        if pathway_name
        else f"Local SHAP: {sample_id}"
    )
    shap.plots.waterfall(explanation[0], max_display=20, show=False)
    plt.title(title)
    plt.tight_layout()
    plot_path = os.path.join(shap_dir, f"{file_prefix}_local_{sample_id}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved → {plot_path}")

    # Print top positive / negative contributors
    sv = explanation[0].values
    top_pos_idx = sorted(range(len(sv)), key=lambda i: sv[i], reverse=True)[:5]
    top_neg_idx = sorted(range(len(sv)), key=lambda i: sv[i])[:5]

    print(f"  Top positive contributors ({sample_id}):")
    for idx in top_pos_idx:
        print(f"    {feature_names[idx]:30s}  {sv[idx]:+.4f}")
    print(f"  Top negative contributors ({sample_id}):")
    for idx in top_neg_idx:
        print(f"    {feature_names[idx]:30s}  {sv[idx]:+.4f}")


def batch_local_shap(
    model: XGBClassifier,
    X_candidates: np.ndarray,
    candidate_ids: list[str],
    all_go_terms: list[str],
    top_k: int = 5,
) -> pd.DataFrame:
    """Compute SHAP summaries for a batch of candidate gene sets.

    Returns a long-format DataFrame listing the top-k features (by |SHAP|)
    for each candidate. This function is intended to be called directly by
    step8/step9 with their own candidate data — it does not write any files.

    Args:
        model: Trained XGBClassifier.
        X_candidates: Feature matrix of shape ``(n_candidates, n_features)``.
        candidate_ids: Identifier for each candidate row.
        all_go_terms: Stable-sorted list of GO term identifiers.
        top_k: Number of top features to report per candidate.

    Returns:
        DataFrame with columns ``[candidate_id, rank, feature_name, shap_value]``.

    Raises:
        ValueError: If ``len(candidate_ids) != len(X_candidates)`` or
                    ``top_k < 1``.
    """
    if len(candidate_ids) != len(X_candidates):
        raise ValueError(
            f"candidate_ids length ({len(candidate_ids)}) != "
            f"X_candidates rows ({len(X_candidates)})"
        )
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    feature_names = _build_feature_names(all_go_terms)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_candidates)

    records = []
    for i, cid in enumerate(candidate_ids):
        sv = shap_vals[i]
        top_idx = np.argsort(np.abs(sv))[::-1][:top_k]
        for rank, idx in enumerate(top_idx, 1):
            records.append({
                "candidate_id": cid,
                "rank": rank,
                "feature_name": feature_names[idx],
                "shap_value": float(sv[idx]),
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    mock = "--mock" in sys.argv
    model_path = config.MOCK_MODEL_PATH if mock else config.MODEL_PATH
    train_path = config.MOCK_TRAIN_DATA  if mock else config.TRAIN_DATA
    shap_dir   = os.path.join(config.MOCK_RESULTS_DIR, "shap") if mock else config.SHAP_DIR

    model = XGBClassifier()
    model.load_model(model_path)

    train_data   = np.load(train_path)
    X_train      = train_data["X"]
    all_go_terms = list(train_data["all_go_terms"])

    source_tag = "mock" if mock else "real"
    print(f"[{source_tag}] X_train: {X_train.shape}  GO terms: {len(all_go_terms)}")

    # Feature dimension consistency checks
    expected_d = len(_build_feature_names(all_go_terms))
    if X_train.shape[1] != expected_d:
        raise ValueError(
            f"Feature mismatch: X_train has {X_train.shape[1]} columns, "
            f"but all_go_terms implies {expected_d} features. "
            "Re-run step4/step5 to regenerate train data."
        )
    model_d = model.get_booster().num_features()
    if model_d != expected_d:
        raise ValueError(
            f"Model was trained on {model_d} features, "
            f"but current train data has {expected_d}. "
            "Re-run step6 to retrain the model."
        )

    # Global SHAP analysis
    shap_vals, importance_df = global_shap_analysis(
        model, X_train, all_go_terms, shap_dir=shap_dir,
    )

    # Local SHAP analysis on first 3 training samples as examples
    for i in range(min(3, len(X_train))):
        local_shap_analysis(
            model, X_train[i], sample_id=str(i),
            all_go_terms=all_go_terms, shap_dir=shap_dir,
        )

    # Batch summary on first 10 training samples
    n_batch = min(10, len(X_train))
    batch_df = batch_local_shap(
        model, X_train[:n_batch],
        candidate_ids=[str(i) for i in range(n_batch)],
        all_go_terms=all_go_terms,
    )
    print("\nBatch SHAP summary (first 20 rows):")
    print(batch_df.head(20).to_string(index=False))
