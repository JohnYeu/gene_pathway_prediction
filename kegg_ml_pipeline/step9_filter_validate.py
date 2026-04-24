from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

import config
from step3_go_annotation import load_go_annotation
from step4_feature_extraction import build_feature_matrix
from utils.io_utils import load_pathways_tsv
from utils.mock_data import generate_mock_universe

# ---------------------------------------------------------------------------
# Module-level column constants — referenced in generate_final_report and
# __main__ so that both empty and non-empty paths stay consistent.
# ---------------------------------------------------------------------------

_BASE_COLS: list[str] = [
    "pathway_id", "pathway_name", "source", "scored_gene_count",
    "score", "is_candidate", "max_jaccard", "closest_pathway",
    "overlap_class", "top_enriched_go", "min_go_pvalue", "top_shap_features",
]

# These columns appear only in step8 DEG-mode output; preserved when present.
_COVERAGE_COLS: list[str] = [
    "overlap_count", "pathway_gene_count", "deg_gene_count",
    "pathway_coverage", "deg_coverage",
]

# Minimum set of columns that must be present in the step8 output CSV.
_REQUIRED_COLS: frozenset[str] = frozenset({
    "pathway_id", "pathway_name", "source",
    "scored_genes", "scored_gene_count", "score", "is_candidate",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_jaccard(set_a: set, set_b: set) -> float:
    """Compute the Jaccard similarity between two sets.

    Args:
        set_a: First set.
        set_b: Second set.

    Returns:
        Jaccard similarity in [0, 1]. Returns 0.0 when either set is empty.
    """
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def filter_by_jaccard(
    candidates: pd.DataFrame,
    all_pathways: dict[str, dict],
    threshold: float = config.JACCARD_THRESHOLD,
) -> pd.DataFrame:
    """Annotate each candidate with its maximum Jaccard similarity to known pathways.

    For each candidate, the ``scored_genes`` column (comma-separated string) is
    parsed into a gene set and compared against every pathway in ``all_pathways``.
    The result is annotated with three new columns:

    - ``max_jaccard``: highest Jaccard score across all known pathways.
    - ``closest_pathway``: pathway_id of the most similar known pathway.
    - ``overlap_class``: ``"high_overlap"`` when max_jaccard >= threshold,
      ``"low_overlap"`` otherwise.

    Note: step8 DEG-mode candidates are always subsets of known pathway gene
    sets, so ``overlap_class`` describes similarity level only — it does not
    imply novelty.

    Args:
        candidates: DataFrame with at least a ``scored_genes`` column (comma-
                    separated gene IDs, matching the step8 output format).
        all_pathways: Known pathway dict ``{pathway_id: {"genes": set, ...}}``.
        threshold: Jaccard threshold for "high_overlap" vs "low_overlap".

    Returns:
        The input DataFrame with three appended columns.

    Raises:
        ValueError: If ``threshold`` is outside [0, 1].
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    if candidates.empty:
        return candidates.assign(
            max_jaccard=pd.Series(dtype=float),
            closest_pathway=pd.Series(dtype=str),
            overlap_class=pd.Series(dtype=str),
        )

    max_jaccards: list[float] = []
    closest_pathways: list[str] = []
    overlap_classes: list[str] = []

    for _, row in candidates.iterrows():
        cand_genes = {g.strip() for g in str(row["scored_genes"]).split(",") if g.strip()}
        best_j = 0.0
        best_pid = ""
        for pid, p in all_pathways.items():
            j = compute_jaccard(cand_genes, set(p.get("genes", set())))
            if j > best_j:
                best_j = j
                best_pid = pid
        max_jaccards.append(best_j)
        closest_pathways.append(best_pid)
        overlap_classes.append("high_overlap" if best_j >= threshold else "low_overlap")

    result = candidates.copy()
    result["max_jaccard"]      = max_jaccards
    result["closest_pathway"]  = closest_pathways
    result["overlap_class"]    = overlap_classes
    return result


def _build_go_index(
    gene_go: dict[str, set[str]],
    all_go_terms: list[str],
) -> dict[str, set[str]]:
    """Build a GO-term → gene-set reverse index restricted to all_go_terms.

    Pre-computing this index once and reusing it across all candidates reduces
    the per-candidate GO enrichment test from O(|gene_go| × |all_go_terms|)
    to O(|candidate_genes|) per term.

    Args:
        gene_go: Mapping from gene ID to its set of GO term IDs.
        all_go_terms: Ordered list of GO terms used as features during training.

    Returns:
        Dict mapping each GO term in all_go_terms to the set of genes annotated
        with that term.
    """
    go_index: dict[str, set[str]] = {t: set() for t in all_go_terms}
    for gene, terms in gene_go.items():
        for t in terms:
            if t in go_index:
                go_index[t].add(gene)
    return go_index


def go_enrichment_validation(
    candidate_genes: list[str],
    gene_go: dict[str, set[str]],
    all_go_terms: list[str],
    go_index: dict[str, set[str]] | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Run a hypergeometric GO enrichment test on a candidate gene set.

    The background universe is all genes present in ``gene_go``. Only GO terms
    in ``all_go_terms`` (the training feature space) are tested, and Bonferroni
    correction uses ``len(all_go_terms)`` as the number of hypotheses — keeping
    the correction denominator consistent with the model's feature space.

    Args:
        candidate_genes: List of gene IDs in the candidate set.
        gene_go: Mapping from gene ID to its set of GO term IDs.
        all_go_terms: Ordered list of GO terms used as features during training.
        go_index: Pre-computed reverse index from ``_build_go_index()``. If
                  None, it is built internally (correct but slow for many
                  candidates).
        top_n: Maximum number of enriched GO terms to return.

    Returns:
        DataFrame with columns ``[go_term, overlap_count, expected, p_value,
        p_adjusted]`` sorted by ``p_value`` ascending, at most ``top_n`` rows.
        Returns an empty DataFrame when no candidate genes have GO annotations.

    Raises:
        ValueError: If ``top_n < 1``.
    """
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")

    if go_index is None:
        go_index = _build_go_index(gene_go, all_go_terms)

    candidate_set = set(candidate_genes) & gene_go.keys()
    n = len(candidate_set)
    if n == 0:
        return pd.DataFrame(
            columns=["go_term", "overlap_count", "expected", "p_value", "p_adjusted"]
        )

    M = len(gene_go)
    records = []
    for t in all_go_terms:
        K = len(go_index[t])
        if K == 0:
            continue
        k = len(candidate_set & go_index[t])
        if k == 0:
            continue
        p_value = float(stats.hypergeom.sf(k - 1, M, K, n))
        expected = n * K / M
        p_adjusted = min(p_value * len(all_go_terms), 1.0)
        records.append({
            "go_term":       t,
            "overlap_count": k,
            "expected":      expected,
            "p_value":       p_value,
            "p_adjusted":    p_adjusted,
        })

    if not records:
        return pd.DataFrame(
            columns=["go_term", "overlap_count", "expected", "p_value", "p_adjusted"]
        )

    df = (
        pd.DataFrame(records)
        .sort_values("p_value")
        .head(top_n)
        .reset_index(drop=True)
    )
    return df


def generate_final_report(
    candidates_filtered: pd.DataFrame,
    shap_summary: pd.DataFrame,
    go_enrichment_results: dict[str, pd.DataFrame],
    output_path: str = config.FINAL_REPORT,
) -> pd.DataFrame:
    """Integrate scores, Jaccard labels, GO enrichment, and SHAP features into
    a single final report CSV.

    ``shap_summary`` must be in the long format returned by
    ``step7_shap_analysis.batch_local_shap()``:
    columns ``[candidate_id, rank, feature_name, shap_value]``.

    Empty candidates are handled gracefully: a header-only CSV is written,
    preserving any coverage columns present in the (empty) input so that
    downstream consumers can parse the schema regardless of whether data exists.

    Args:
        candidates_filtered: DataFrame produced by ``filter_by_jaccard()``.
                             Must contain the step8 base columns plus the three
                             Jaccard annotation columns.
        shap_summary: Long-format per-candidate SHAP DataFrame.
        go_enrichment_results: Dict mapping pathway_id to its GO enrichment
                               result DataFrame.
        output_path: Destination path for the output CSV.

    Returns:
        Final report DataFrame. The ``scored_genes`` column is excluded from the
        CSV (too wide) but retained in the returned DataFrame.
    """
    extra_coverage = [c for c in _COVERAGE_COLS if c in candidates_filtered.columns]

    if candidates_filtered.empty:
        # CSV uses only the non-verbose base + coverage columns (no scored_genes).
        csv_cols = _BASE_COLS + extra_coverage
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        pd.DataFrame(columns=csv_cols).to_csv(output_path, index=False)
        print(f"[WARNING] No candidates to report. Saved → {output_path}")
        # Return value preserves the full input schema (including scored_genes)
        # plus any annotation columns not yet present, so callers can rely on
        # a consistent column set regardless of whether data exists.
        _annotation_cols = [
            "max_jaccard", "closest_pathway", "overlap_class",
            "top_enriched_go", "min_go_pvalue", "top_shap_features",
        ]
        return_cols = list(candidates_filtered.columns) + [
            c for c in _annotation_cols if c not in candidates_filtered.columns
        ]
        return pd.DataFrame(columns=return_cols)

    df = candidates_filtered.copy()

    # ── SHAP: extract top-3 features per candidate ──────────────────────────
    if not shap_summary.empty and {"candidate_id", "rank", "feature_name"}.issubset(
        shap_summary.columns
    ):
        top3_shap = (
            shap_summary[shap_summary["rank"] <= 3]
            .groupby("candidate_id")["feature_name"]
            .apply(lambda s: "|".join(s.tolist()))
            .rename("top_shap_features")
        )
        df = df.merge(
            top3_shap.reset_index().rename(columns={"candidate_id": "pathway_id"}),
            on="pathway_id",
            how="left",
        )
    else:
        df["top_shap_features"] = ""

    df["top_shap_features"] = df["top_shap_features"].fillna("")

    # ── GO enrichment: top-3 terms and minimum adjusted p-value ─────────────
    top_go_list:  list[str]   = []
    min_pval_list: list[float] = []

    for pid in df["pathway_id"]:
        enrich_df = go_enrichment_results.get(pid, pd.DataFrame())
        if enrich_df.empty or "go_term" not in enrich_df.columns:
            top_go_list.append("")
            min_pval_list.append(1.0)
        else:
            top3 = enrich_df.head(3)
            top_go_list.append("|".join(top3["go_term"].tolist()))
            min_pval_list.append(float(enrich_df["p_adjusted"].min()))

    df["top_enriched_go"] = top_go_list
    df["min_go_pvalue"]   = min_pval_list

    # ── Assemble final column order ──────────────────────────────────────────
    csv_cols = _BASE_COLS + extra_coverage
    # Keep scored_genes in memory but exclude from CSV
    csv_df = df[[c for c in csv_cols if c in df.columns]]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    csv_df.to_csv(output_path, index=False)
    n_high = int((df["overlap_class"] == "high_overlap").sum())
    n_low  = int((df["overlap_class"] == "low_overlap").sum())
    print(f"Final report: {len(df)} candidates  "
          f"high_overlap={n_high}  low_overlap={n_low}")
    print(f"Saved → {output_path}")

    # ── Markdown summary (front 10 rows, key columns only) ──────────────────
    summary_cols = [c for c in
                    ["pathway_id", "pathway_name", "score", "overlap_class",
                     "max_jaccard", "top_enriched_go"]
                    if c in df.columns]
    try:
        print(df[summary_cols].head(10).to_markdown(index=False))
    except ImportError:
        print(df[summary_cols].head(10).to_string(index=False))

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mock = "--mock" in sys.argv
    scores_path = config.MOCK_CANDIDATE_SCORES if mock else config.CANDIDATE_SCORES
    output_path = config.MOCK_FINAL_REPORT     if mock else config.FINAL_REPORT
    model_path  = config.MOCK_MODEL_PATH       if mock else config.MODEL_PATH
    train_path  = config.MOCK_TRAIN_DATA       if mock else config.TRAIN_DATA

    # Load step8 output with schema validation
    candidates_df = pd.read_csv(scores_path)
    missing = _REQUIRED_COLS - set(candidates_df.columns)
    if missing:
        raise ValueError(
            f"step8 output {scores_path!r} is missing columns: {sorted(missing)}. "
            "Re-run step8 to regenerate."
        )

    # Empty CSV (step8 DEG empty result) has is_candidate dtype=object.
    # Filtering with df[df["is_candidate"]] on an object column drops all columns.
    # Must check emptiness first, then normalise to bool before filtering.
    if candidates_df.empty:
        candidates_filtered = candidates_df.copy()
    else:
        candidates_df["is_candidate"] = (
            candidates_df["is_candidate"].astype(str).str.lower() == "true"
        )
        candidates_filtered = (
            candidates_df[candidates_df["is_candidate"]].copy().reset_index(drop=True)
        )

    source_tag = "mock" if mock else "real"
    print(f"[{source_tag}] {len(candidates_filtered)} candidates (is_candidate=True)")

    # Load pathways and GO annotations
    if mock:
        u            = generate_mock_universe()
        all_pathways = u["pathways"]
        gene_go      = {g: set(ts) for g, ts in u["gene_go"].items()}
    else:
        all_pathways = load_pathways_tsv(config.ALL_PATHWAYS)
        gene_go      = load_go_annotation(config.GO_ANNOTATION, mock=False)

    train_data   = np.load(train_path)
    all_go_terms = list(train_data["all_go_terms"])

    if candidates_filtered.empty:
        print("[WARNING] No candidates found. Writing empty report.")
        generate_final_report(
            candidates_filtered, pd.DataFrame(), {}, output_path=output_path
        )
        sys.exit(0)

    # Jaccard filtering
    candidates_filtered = filter_by_jaccard(candidates_filtered, all_pathways)
    n_high = int((candidates_filtered["overlap_class"] == "high_overlap").sum())
    n_low  = int((candidates_filtered["overlap_class"] == "low_overlap").sum())
    print(f"  high_overlap={n_high}  low_overlap={n_low}")

    # GO enrichment — build reverse index once, reuse across all candidates
    go_index = _build_go_index(gene_go, all_go_terms)
    go_enrichment_results: dict[str, pd.DataFrame] = {}
    for _, row in candidates_filtered.iterrows():
        pid   = row["pathway_id"]
        genes = [g.strip() for g in str(row["scored_genes"]).split(",") if g.strip()]
        go_enrichment_results[pid] = go_enrichment_validation(
            genes, gene_go, all_go_terms, go_index=go_index
        )

    # Per-candidate SHAP (requires model)
    from xgboost import XGBClassifier
    from step7_shap_analysis import batch_local_shap, _build_feature_names

    model = XGBClassifier()
    model.load_model(model_path)

    # Feature-dimension consistency check (matches step7/step8 pattern)
    expected_d = len(_build_feature_names(all_go_terms))
    if model.get_booster().num_features() != expected_d:
        raise ValueError(
            f"Model has {model.get_booster().num_features()} features but "
            f"all_go_terms implies {expected_d}. Re-run step6."
        )

    gene_lists = [
        [g.strip() for g in str(row["scored_genes"]).split(",") if g.strip()]
        for _, row in candidates_filtered.iterrows()
    ]
    X_cands = build_feature_matrix(gene_lists, gene_go, all_go_terms)
    shap_summary = batch_local_shap(
        model, X_cands,
        candidate_ids=candidates_filtered["pathway_id"].tolist(),
        all_go_terms=all_go_terms,
        top_k=5,
    )

    # Generate and save final report
    report_df = generate_final_report(
        candidates_filtered, shap_summary, go_enrichment_results,
        output_path=output_path,
    )
