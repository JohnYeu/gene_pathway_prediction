from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import config
from step3_go_annotation import load_go_annotation
from step4_feature_extraction import build_feature_matrix, extract_features
from utils.io_utils import load_pathways_tsv
from utils.mock_data import generate_mock_universe


def load_deg_list(deg_file: str) -> list[str]:
    """Load a differentially expressed gene list from a file.

    Supports two formats:
    - Plain text (.txt or no extension): one gene ID per line.
    - Tabular (.csv / .tsv / .txt containing tabs): parsed with pandas.
      Column selection priority (case-insensitive):
        1. Any column whose name contains "gene"
        2. Any column whose name contains "symbol"
        3. Any column whose name contains "id" but not "gene" or "symbol"
           (avoids mis-selecting "sample_id", "comparison_id", etc.)
        4. First column as fallback

    Empty lines and lines starting with "#" are ignored.
    Gene IDs are returned as-is (no upper-casing).

    Args:
        deg_file: Path to the DEG list file.

    Returns:
        List of gene ID strings.

    Raises:
        FileNotFoundError: If ``deg_file`` does not exist.
    """
    if not os.path.exists(deg_file):
        raise FileNotFoundError(f"DEG file not found: {deg_file}")

    # Detect tabular format by extension or by sniffing the first *non-comment* line.
    # Skipping blank lines and lines starting with "#" avoids misclassifying a
    # comment-headed .txt as plain-text when the real header is on line 2, and
    # avoids using a comment line as the separator probe for .tsv/.csv files.
    def _first_data_line(path: str) -> str:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for ln in fh:
                stripped = ln.strip()
                if stripped and not stripped.startswith("#"):
                    return stripped
        return ""

    is_tabular = deg_file.endswith(".csv") or deg_file.endswith(".tsv")
    if not is_tabular and deg_file.endswith(".txt"):
        first = _first_data_line(deg_file)
        is_tabular = "\t" in first or "," in first

    if is_tabular:
        first_data = _first_data_line(deg_file)
        sep = "\t" if (deg_file.endswith(".tsv") or "\t" in first_data) else ","
        df = pd.read_csv(deg_file, sep=sep, dtype=str, comment="#")
        cols_lower = {c.lower(): c for c in df.columns}

        selected = None
        for col_lower, col_orig in cols_lower.items():
            if "gene" in col_lower:
                selected = col_orig
                break
        if selected is None:
            for col_lower, col_orig in cols_lower.items():
                if "symbol" in col_lower:
                    selected = col_orig
                    break
        if selected is None:
            for col_lower, col_orig in cols_lower.items():
                if "id" in col_lower and "gene" not in col_lower and "symbol" not in col_lower:
                    selected = col_orig
                    break
        if selected is None:
            selected = df.columns[0]

        genes = [
            str(v).strip()
            for v in df[selected].dropna()
            if str(v).strip() and not str(v).strip().startswith("#")
        ]
    else:
        genes = []
        with open(deg_file, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                gene = line.strip()
                if gene and not gene.startswith("#"):
                    genes.append(gene)

    return genes


def score_gene_list(
    gene_list: list[str],
    model: XGBClassifier,
    gene_go: dict[str, set[str]],
    all_go_terms: list[str],
) -> float:
    """Score a single gene set as a pathway-like probability.

    Args:
        gene_list: Gene identifiers to score.
        model: Trained XGBClassifier.
        gene_go: Mapping from gene ID to the set of its GO terms.
        all_go_terms: Stable-sorted list of all GO terms used during training.

    Returns:
        Float in [0, 1] representing the probability that the gene set
        belongs to a real biological pathway.

    Raises:
        ValueError: If the gene list contains fewer than 3 unique genes.
    """
    if len(set(gene_list)) < 3:
        raise ValueError(
            f"gene_list must contain at least 3 unique genes, "
            f"got {len(set(gene_list))}. "
            "Model was trained on pathways with ≥3 genes."
        )
    X = extract_features(gene_list, gene_go, all_go_terms).reshape(1, -1)
    return float(model.predict_proba(X)[:, 1][0])


def score_candidate_pathways(
    candidates: dict[str, dict],
    model: XGBClassifier,
    gene_go: dict[str, set[str]],
    all_go_terms: list[str],
    threshold: float = config.SCORE_THRESHOLD,
    output_path: str = config.CANDIDATE_SCORES,
) -> pd.DataFrame:
    """Score a collection of candidate gene sets and save results to CSV.

    Each candidate is scored as a pathway-like probability using the trained
    XGBoost model. Results are sorted by score descending.

    Args:
        candidates: Dict mapping pathway/candidate ID to a metadata dict with
                    keys ``"genes"`` (set or list of gene IDs), ``"name"``
                    (display name), and ``"source"`` (e.g. "KEGG"). This
                    format matches the output of ``load_pathways_tsv()``.
        model: Trained XGBClassifier.
        gene_go: Mapping from gene ID to the set of its GO terms.
        all_go_terms: Stable-sorted list of all GO terms used during training.
        threshold: Candidates with score >= threshold are marked as
                   ``is_candidate=True``.
        output_path: Destination path for the output CSV file.

    Returns:
        DataFrame with columns
        ``[pathway_id, pathway_name, source, scored_genes,
           scored_gene_count, score, is_candidate]``
        sorted by score descending.

    Raises:
        ValueError: If ``candidates`` is empty, ``threshold`` is outside
                    [0, 1], or any candidate has fewer than 3 unique genes.
    """
    if not candidates:
        raise ValueError("candidates is empty")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    invalid = [
        pid for pid, p in candidates.items()
        if len(set(p.get("genes", []))) < 3
    ]
    if invalid:
        raise ValueError(
            f"{len(invalid)} candidate(s) have fewer than 3 unique genes: "
            f"{invalid[:5]}. Model was trained on gene sets with ≥3 genes."
        )

    pathway_ids = list(candidates.keys())
    gene_lists  = [list(p["genes"]) for p in candidates.values()]
    X           = build_feature_matrix(gene_lists, gene_go, all_go_terms)
    scores      = model.predict_proba(X)[:, 1]

    df = pd.DataFrame({
        "pathway_id":        pathway_ids,
        "pathway_name":      [candidates[pid].get("name", "")   for pid in pathway_ids],
        "source":            [candidates[pid].get("source", "")  for pid in pathway_ids],
        "scored_genes":      [
            ",".join(sorted(set(str(g) for g in candidates[pid]["genes"])))
            for pid in pathway_ids
        ],
        "scored_gene_count": [len(set(candidates[pid]["genes"])) for pid in pathway_ids],
        "score":             scores.astype(float),
        "is_candidate":      scores >= threshold,
    })
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    n_above = int(df["is_candidate"].sum())
    print(f"Scored {len(df)} candidates, {n_above} above threshold {threshold}")
    print(f"Saved → {output_path}")
    return df


if __name__ == "__main__":
    mock = "--mock" in sys.argv
    model_path  = config.MOCK_MODEL_PATH       if mock else config.MODEL_PATH
    train_path  = config.MOCK_TRAIN_DATA        if mock else config.TRAIN_DATA
    output_path = config.MOCK_CANDIDATE_SCORES  if mock else config.CANDIDATE_SCORES

    model = XGBClassifier()
    model.load_model(model_path)

    gene_go      = load_go_annotation(config.GO_ANNOTATION, mock=mock)
    train_data   = np.load(train_path)
    all_go_terms = list(train_data["all_go_terms"])

    # Feature dimension consistency check
    from step7_shap_analysis import _build_feature_names
    expected_d = len(_build_feature_names(all_go_terms))
    if model.get_booster().num_features() != expected_d:
        raise ValueError(
            f"Model has {model.get_booster().num_features()} features but "
            f"all_go_terms implies {expected_d}. Re-run step6."
        )

    if mock:
        all_pathways = generate_mock_universe()["pathways"]
    else:
        all_pathways = load_pathways_tsv(config.ALL_PATHWAYS)

    # Parse optional DEG file argument
    deg_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(deg_args) > 1:
        raise ValueError(
            f"Expected at most 1 DEG file argument, got {len(deg_args)}: {deg_args}"
        )

    source_tag = "mock" if mock else "real"

    if deg_args:
        # DEG mode: score intersection of DEG list with each known pathway
        deg_genes = set(load_deg_list(deg_args[0]))
        candidates: dict[str, dict] = {}
        coverage_meta: dict[str, dict] = {}

        for pid, p in all_pathways.items():
            overlap = deg_genes & set(p["genes"])
            if len(overlap) >= 3:
                candidates[pid] = {
                    "name":   p.get("name", ""),
                    "genes":  overlap,
                    "source": p.get("source", ""),
                }
                coverage_meta[pid] = {
                    "overlap_count":      len(overlap),
                    "pathway_gene_count": len(p["genes"]),
                    "deg_gene_count":     len(deg_genes),
                    "pathway_coverage":   len(overlap) / len(p["genes"]) if p["genes"] else 0.0,
                    "deg_coverage":       len(overlap) / len(deg_genes)  if deg_genes  else 0.0,
                }

        print(f"[{source_tag}][DEG mode] {len(candidates)} pathways with ≥3 gene overlap")

        if not candidates:
            print(
                "[WARNING] No pathway had ≥3 gene overlap with the DEG list. "
                "Check gene ID format. Writing empty result."
            )
            empty_cols = [
                "pathway_id", "pathway_name", "source",
                "scored_genes", "scored_gene_count", "score", "is_candidate",
                "overlap_count", "pathway_gene_count", "deg_gene_count",
                "pathway_coverage", "deg_coverage",
            ]
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            pd.DataFrame(columns=empty_cols).to_csv(output_path, index=False)
            print(f"Saved → {output_path}  (0 candidates)")
        else:
            df = score_candidate_pathways(
                candidates, model, gene_go, all_go_terms,
                output_path=output_path,
            )
            # Append coverage columns and overwrite CSV
            meta_df = (
                pd.DataFrame(coverage_meta)
                .T.rename_axis("pathway_id")
                .reset_index()
            )
            df = df.merge(meta_df, on="pathway_id", how="left")
            df.to_csv(output_path, index=False)
            print(df.head(10).to_string(index=False))

    else:
        # Demo mode: score all known pathways to validate the model
        candidates = {
            pid: p for pid, p in all_pathways.items()
            if len(p.get("genes", set())) >= 3
        }
        print(f"[{source_tag}][demo mode] Scoring {len(candidates)} known pathways...")
        df = score_candidate_pathways(
            candidates, model, gene_go, all_go_terms,
            output_path=output_path,
        )
        print(df.head(10).to_string(index=False))
