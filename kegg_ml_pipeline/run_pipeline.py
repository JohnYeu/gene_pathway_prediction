"""One-click entry point for the KEGG ML pipeline.

Usage examples::

    # Full pipeline (steps 1-9)
    python run_pipeline.py

    # Full pipeline with mock data (no internet access or real data files needed)
    python run_pipeline.py --mock

    # Run a single step
    python run_pipeline.py --step 6

    # Resume from a given step (inclusive)
    python run_pipeline.py --from-step 4

    # DEG mode: restrict step-8 scoring to DEG-pathway overlaps
    python run_pipeline.py --deg deg_list.txt

    # Combinations
    python run_pipeline.py --mock --step 8 --deg deg_list.txt

Note on mock partial runs
--------------------------
``--mock --step 2`` and ``--mock --step 4`` are intentional smoke checks:

  * Step 2 in mock mode generates pathways in memory only (``save=False``),
    so no TSV is written to ``data/all_pathways.tsv``.
  * Step 4 in mock mode validates feature dimensions on 5 sample rows only;
    it does not write ``data/feature_matrix.npz``.

``--mock --from-step 3`` (and any later mock partial run) works correctly:
``_load_state()`` always reconstructs ``all_pathways`` and ``gene_go`` from
``generate_mock_universe()`` in mock mode, so no pathway TSV on disk is needed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

# Lock the working directory to this script's location so that all relative
# paths in config.py resolve correctly regardless of where the caller invokes
# the script from (e.g. ``python kegg_ml_pipeline/run_pipeline.py``).
os.chdir(Path(__file__).resolve().parent)

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import config
import step1_setup
import step2a_kegg
import step2b_aracyc
import step3_go_annotation
import step4_feature_extraction
import step5_build_dataset
import step6_train_xgboost
import step7_shap_analysis
import step8_score_candidates
import step9_filter_validate
from utils.io_utils import load_pathways_tsv
from utils.mock_data import generate_mock_universe


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for run_pipeline.py."""
    parser = argparse.ArgumentParser(
        description="KEGG/AraCyc ML pipeline for Arabidopsis pathway discovery.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help=(
            "Use synthetic mock data.  No internet access or real data files "
            "are required.  Output is written under results/mock/."
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--step",
        type=int,
        metavar="N",
        help="Run only step N (1-9).",
    )
    group.add_argument(
        "--from-step",
        type=int,
        metavar="N",
        dest="from_step",
        help="Run steps N through 9 (inclusive).",
    )
    parser.add_argument(
        "--deg",
        type=str,
        metavar="FILE",
        help=(
            "Path to a differentially expressed gene list (plain-text, CSV, or TSV). "
            "When supplied, step 8 scores DEG-pathway overlaps instead of all pathways."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _now() -> str:
    """Return the current wall-clock time as HH:MM:SS."""
    return datetime.now().strftime("%H:%M:%S")


# Human-readable recovery hints printed when a step fails.
_FIX_HINTS: dict[int, str] = {
    1: "Run: pip install -r requirements.txt  (Python >= 3.10 required)",
    2: "No internet?  Use: python run_pipeline.py --mock",
    3: (
        "Missing GO annotation file.  Download ATH_GO_GOSLIM.txt from "
        "https://www.arabidopsis.org/download/go  or use --mock"
    ),
    4: "Re-run from step 2: python run_pipeline.py --from-step 2",
    5: "Re-run from step 3: python run_pipeline.py --from-step 3",
    6: "CV AUROC below threshold — consider tuning XGB_PARAMS in config.py",
    7: "Ensure step 6 completed.  Re-run: python run_pipeline.py --from-step 6",
    8: "Check gene ID format in the DEG file.  Use --mock to verify the pipeline.",
    9: "Ensure step 8 completed.  Re-run: python run_pipeline.py --from-step 8",
}


def _print_fix_hint(step_n: int) -> None:
    """Print a recovery suggestion for a failed step."""
    hint = _FIX_HINTS.get(step_n)
    if hint:
        print(f"  Hint: {hint}")


def _run_step(n: int, name: str, func: Callable, *args: Any, **kwargs: Any) -> Any:
    """Execute a single pipeline step with a header banner and elapsed timing.

    Calls ``func(*args, **kwargs)`` inside a try/except.  On failure, prints
    the exception message, calls ``_print_fix_hint``, and exits with code 1 so
    the caller does not need to handle exceptions individually.

    Args:
        n:    Step number (1-9), used in the banner and fix-hint lookup.
        name: Human-readable step description shown in the banner.
        func: Callable to invoke.
        *args:    Positional arguments forwarded to func.
        **kwargs: Keyword arguments forwarded to func.

    Returns:
        Whatever func returns on success.
    """
    print(f"\n{'=' * 60}")
    print(f"[Step {n}] {name}  ({_now()})")
    print("=" * 60)
    t0 = time.time()
    try:
        result = func(*args, **kwargs)
        print(f"[Step {n}] Finished in {time.time() - t0:.1f}s")
        return result
    except Exception as exc:
        print(f"\n[Step {n}] FAILED after {time.time() - t0:.1f}s")
        print(f"  Error: {exc}")
        _print_fix_hint(n)
        sys.exit(1)


# ---------------------------------------------------------------------------
# State recovery for partial runs
# ---------------------------------------------------------------------------

def _load_state(start_step: int, mock: bool) -> dict[str, Any]:
    """Load the minimum set of disk artifacts needed to start at ``start_step``.

    The pipeline state is split into two regimes:

    Pre-training (steps 3-5)
        ``all_pathways`` is loaded from ``config.ALL_PATHWAYS`` (real) or
        reconstructed from the mock universe (mock).  ``gene_go`` and
        ``all_go_terms`` are needed only when starting at step 4 or 5 — they
        are the output of step 3, which must have run already.  For start_step=3,
        ``gene_go`` is intentionally absent from state so that step 3 can create
        it.  ``all_go_terms`` is derived from ``gene_go`` once available.

        Mock note: ``all_pathways`` and ``gene_go`` are always reconstructed from
        ``generate_mock_universe()`` because the mock pipeline does not write
        ``data/all_pathways.tsv`` to disk (step 2 uses ``save=False``).

    Post-training (steps 6-9)
        ``X_train``, ``y_train``, ``X_test``, ``y_test``, and ``all_go_terms``
        are loaded from the NPZ archives written by step 5.  ``all_pathways``
        and ``gene_go`` are also loaded here because steps 8 and 9 need them.
        The XGBoost model is loaded for start steps 7-9.  ``df8`` (step 8
        scores) is loaded from CSV only when starting at step 9; for all other
        start steps it will be populated in memory by step 8 itself.

    Args:
        start_step: First step that will be executed in this run.
        mock:       Whether to use mock data paths and the mock universe.

    Returns:
        Dict with the keys required for ``start_step`` and any subsequent steps.
    """
    state: dict[str, Any] = {}

    # ── Pre-training: steps 3-5 ───────────────────────────────────────────────
    # (step 2 starts from scratch, so nothing needs to be pre-loaded for it)
    if 3 <= start_step <= 5:
        if mock:
            # Mock pipeline never writes ALL_PATHWAYS — always regenerate.
            u = generate_mock_universe()
            state["all_pathways"] = u["pathways"]
            state["gene_go"]      = {g: set(ts) for g, ts in u["gene_go"].items()}
            state["all_go_terms"] = step3_go_annotation.get_all_go_terms(
                state["gene_go"]
            )
        else:
            # step 2 must have run; load its artifact from disk.
            state["all_pathways"] = load_pathways_tsv(config.ALL_PATHWAYS)
            if start_step >= 4:
                # step 3 must have run; re-derive its output from the source file
                # rather than caching it, so the state always reflects the current
                # annotation file rather than any intermediate artefact.
                state["gene_go"] = step3_go_annotation.load_go_annotation(
                    config.GO_ANNOTATION, mock=False
                )
                state["all_go_terms"] = step3_go_annotation.get_all_go_terms(
                    state["gene_go"]
                )

    # ── Post-training: steps 6-9 ──────────────────────────────────────────────
    if start_step >= 6:
        train_path = config.MOCK_TRAIN_DATA if mock else config.TRAIN_DATA
        test_path  = config.MOCK_TEST_DATA  if mock else config.TEST_DATA

        # Training NPZ is the authoritative source for all_go_terms and the
        # feature matrix used during training.
        train_data = np.load(train_path)
        test_data  = np.load(test_path)

        # Guard against train/test NPZ files from different step-5 runs.
        # Mismatched all_go_terms means the feature spaces differ, which would
        # cause the model to train and evaluate on inconsistent features and
        # produce silently wrong metrics.
        if list(train_data["all_go_terms"]) != list(test_data["all_go_terms"]):
            print(
                f"[ERROR] all_go_terms mismatch between {train_path} and {test_path}. "
                "Re-run step 5 to regenerate both files from the same GO vocabulary."
            )
            sys.exit(1)

        state["X_train"]      = train_data["X"]
        state["y_train"]      = train_data["y"]
        state["X_test"]       = test_data["X"]
        state["y_test"]       = test_data["y"]
        state["all_go_terms"] = list(train_data["all_go_terms"])

        # Steps 8-9 also need pathway dicts and GO annotations.
        if mock:
            u = generate_mock_universe()
            state["all_pathways"] = u["pathways"]
            state["gene_go"]      = {g: set(ts) for g, ts in u["gene_go"].items()}
        else:
            state["all_pathways"] = load_pathways_tsv(config.ALL_PATHWAYS)
            state["gene_go"]      = step3_go_annotation.load_go_annotation(
                config.GO_ANNOTATION, mock=False
            )

        # Steps 7-9 require the trained model.
        if start_step >= 7:
            model_path = config.MOCK_MODEL_PATH if mock else config.MODEL_PATH
            model = XGBClassifier()
            model.load_model(model_path)
            state["model"] = model

        # Step 9 must start with the step-8 CSV.  For all other start steps,
        # ``df8`` is populated in memory by step 8 itself.
        if start_step == 9:
            scores_path = (
                config.MOCK_CANDIDATE_SCORES if mock else config.CANDIDATE_SCORES
            )
            candidates_df = pd.read_csv(scores_path)
            missing = step9_filter_validate._REQUIRED_COLS - set(candidates_df.columns)
            if missing:
                print(
                    f"[ERROR] step-8 output {scores_path!r} is missing columns: "
                    f"{sorted(missing)}. Re-run step 8 first."
                )
                sys.exit(1)
            # _step9 will handle the is_candidate filter; store the raw df here.
            state["df8"] = candidates_df

    return state


# ---------------------------------------------------------------------------
# Individual step wrappers
# ---------------------------------------------------------------------------

def _step1(mock: bool) -> None:
    """Verify that all required Python packages are installed."""
    ok = _run_step(1, "Environment setup", step1_setup.run_setup)
    if not ok:
        print("[Step 1] Setup reported a failure — check output above.")
        sys.exit(1)


def _step2(state: dict, mock: bool) -> None:
    """Fetch KEGG pathways and parse AraCyc; merge into a single pathway dict."""
    def _run() -> dict:
        kegg   = step2a_kegg.get_ath_pathways(mock=mock)
        aracyc = step2b_aracyc.parse_aracyc_pathways(config.ARACYC_RAW, mock=mock)
        # save=False in mock mode: avoids overwriting data/all_pathways.tsv
        # with synthetic data, keeping the real-data artefact intact.
        return step2b_aracyc.merge_and_save_all_pathways(kegg, aracyc, save=not mock)

    state["all_pathways"] = _run_step(2, "Fetch & merge pathways", _run)

    if mock:
        # Warn: the pathway dict lives only in memory for the rest of this run.
        # A subsequent --mock --from-step 3 will fail because no TSV was written.
        print(
            "[Step 2] Mock smoke check: pathways are in memory only "
            "(data/all_pathways.tsv was NOT written). "
            "Use --mock --from-step 2 to re-run the whole mock sequence."
        )


def _step3(state: dict, mock: bool) -> None:
    """Load TAIR GO annotations and derive the GO term vocabulary."""
    def _run() -> tuple[dict, list]:
        gene_go      = step3_go_annotation.load_go_annotation(
            config.GO_ANNOTATION, mock=mock
        )
        all_go_terms = step3_go_annotation.get_all_go_terms(gene_go)
        return gene_go, all_go_terms

    gene_go, all_go_terms = _run_step(3, "Load GO annotations", _run)
    state["gene_go"]      = gene_go
    state["all_go_terms"] = all_go_terms


def _step4(state: dict, mock: bool) -> None:
    """Smoke-check feature extraction on the first 5 pathways.

    In run_pipeline this step only validates the feature dimension; it does NOT
    write data/feature_matrix.npz.  Run ``python step4_feature_extraction.py``
    directly to produce that file.
    """
    all_pathways = state["all_pathways"]
    gene_go      = state["gene_go"]
    all_go_terms = state["all_go_terms"]

    def _run() -> None:
        gene_lists = [
            sorted(p["genes"])
            for p in all_pathways.values()
            if len(p.get("genes", set())) >= 3
        ][:5]
        X_sample = step4_feature_extraction.build_feature_matrix(
            gene_lists, gene_go, all_go_terms
        )
        expected_d = len(all_go_terms) + 6
        if X_sample.shape[1] != expected_d:
            raise ValueError(
                f"Feature dimension mismatch: got {X_sample.shape[1]}, "
                f"expected {expected_d} (len(all_go_terms) + 6)"
            )
        print(f"Feature check OK: sample shape={X_sample.shape}")
        if mock:
            # Warn: no npz was written; same note as step 2.
            print(
                "[Step 4] Mock smoke check: data/feature_matrix.npz was NOT written. "
                "Run python step4_feature_extraction.py to produce that file."
            )

    _run_step(4, "Feature extraction smoke check", _run)


def _step5(state: dict, mock: bool) -> None:
    """Build the labelled dataset and split into train / test archives."""
    all_pathways = state["all_pathways"]
    gene_go      = state["gene_go"]
    all_go_terms = state["all_go_terms"]
    train_path   = config.MOCK_TRAIN_DATA if mock else config.TRAIN_DATA
    test_path    = config.MOCK_TEST_DATA  if mock else config.TEST_DATA

    def _run() -> tuple:
        X, y = step5_build_dataset.build_dataset(all_pathways, gene_go, all_go_terms)
        return step5_build_dataset.split_and_save(
            X, y, all_go_terms, train_path=train_path, test_path=test_path,
        )

    X_train, y_train, X_test, y_test = _run_step(5, "Build dataset", _run)
    state["X_train"] = X_train
    state["y_train"] = y_train
    state["X_test"]  = X_test
    state["y_test"]  = y_test


def _step6(state: dict, mock: bool) -> None:
    """Run stratified cross-validation, then train and evaluate the final model."""
    X_train      = state["X_train"]
    y_train      = state["y_train"]
    X_test       = state["X_test"]
    y_test       = state["y_test"]
    model_path   = config.MOCK_MODEL_PATH  if mock else config.MODEL_PATH
    results_dir  = config.MOCK_RESULTS_DIR if mock else config.RESULTS_DIR

    def _run() -> XGBClassifier:
        cv_scores = step6_train_xgboost.train_with_cv(
            X_train, y_train, config.XGB_PARAMS, results_dir=results_dir,
        )
        step6_train_xgboost.check_auroc_threshold(cv_scores)
        model = step6_train_xgboost.train_final_model(
            X_train, y_train, config.XGB_PARAMS, model_path=model_path,
        )
        step6_train_xgboost.evaluate_model(model, X_test, y_test, results_dir=results_dir)
        return model

    state["model"] = _run_step(6, "Train XGBoost model", _run)


def _step7(state: dict, mock: bool) -> None:
    """Compute global SHAP importance and save the beeswarm summary plot.

    In run_pipeline only ``global_shap_analysis`` is called (fast path).
    Run ``python step7_shap_analysis.py`` directly to also produce per-sample
    local waterfall plots and a batch SHAP summary.
    """
    model        = state["model"]
    X_train      = state["X_train"]
    all_go_terms = state["all_go_terms"]
    shap_dir = (
        os.path.join(config.MOCK_RESULTS_DIR, "shap") if mock else config.SHAP_DIR
    )
    _run_step(
        7, "Global SHAP analysis",
        step7_shap_analysis.global_shap_analysis,
        model, X_train, all_go_terms, shap_dir=shap_dir,
    )


def _step8(state: dict, mock: bool, deg: str | None) -> None:
    """Score candidate pathways; optionally limit to DEG-pathway overlaps."""
    model        = state["model"]
    gene_go      = state["gene_go"]
    all_go_terms = state["all_go_terms"]
    all_pathways = state["all_pathways"]
    output_path  = config.MOCK_CANDIDATE_SCORES if mock else config.CANDIDATE_SCORES

    def _run() -> pd.DataFrame:
        if deg:
            # DEG mode: score the intersection of the supplied gene list with
            # each known pathway (mirrors step8 __main__ DEG branch exactly).
            deg_genes = set(step8_score_candidates.load_deg_list(deg))
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
                        "pathway_coverage": (
                            len(overlap) / len(p["genes"]) if p["genes"] else 0.0
                        ),
                        "deg_coverage": (
                            len(overlap) / len(deg_genes) if deg_genes else 0.0
                        ),
                    }

            source_tag = "mock" if mock else "real"
            print(
                f"[{source_tag}][DEG mode] "
                f"{len(candidates)} pathways with >=3 gene overlap"
            )

            if not candidates:
                # Empty result: write a header-only CSV and return an empty frame.
                # score_candidate_pathways() would raise ValueError on an empty
                # candidates dict, so we must short-circuit here.
                empty_cols = [
                    "pathway_id", "pathway_name", "source", "scored_genes",
                    "scored_gene_count", "score", "is_candidate",
                    "overlap_count", "pathway_gene_count", "deg_gene_count",
                    "pathway_coverage", "deg_coverage",
                ]
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                pd.DataFrame(columns=empty_cols).to_csv(output_path, index=False)
                print(
                    "[WARNING] No pathway had >=3 gene overlap with the DEG list. "
                    f"Saved empty → {output_path}"
                )
                return pd.DataFrame(columns=empty_cols)

            df = step8_score_candidates.score_candidate_pathways(
                candidates, model, gene_go, all_go_terms, output_path=output_path,
            )
            meta_df = (
                pd.DataFrame(coverage_meta)
                .T.rename_axis("pathway_id")
                .reset_index()
            )
            df = df.merge(meta_df, on="pathway_id", how="left")
            df.to_csv(output_path, index=False)
            return df

        # Demo mode: score all known pathways with >= 3 genes.
        candidates = {
            pid: p
            for pid, p in all_pathways.items()
            if len(p.get("genes", set())) >= 3
        }
        source_tag = "mock" if mock else "real"
        print(
            f"[{source_tag}][demo mode] "
            f"Scoring {len(candidates)} known pathways..."
        )
        return step8_score_candidates.score_candidate_pathways(
            candidates, model, gene_go, all_go_terms, output_path=output_path,
        )

    state["df8"] = _run_step(8, "Score candidate pathways", _run)


def _step9(state: dict, mock: bool) -> pd.DataFrame:
    """Jaccard deduplication, GO enrichment, per-candidate SHAP, final report."""
    model        = state["model"]
    gene_go      = state["gene_go"]
    all_go_terms = state["all_go_terms"]
    all_pathways = state["all_pathways"]
    output_path  = config.MOCK_FINAL_REPORT if mock else config.FINAL_REPORT

    # state["df8"] contains the full step-8 output (all scored candidates).
    # Filter to is_candidate=True here — this normalises both the in-memory
    # bool column (from step 8) and the string column (from reading a CSV when
    # starting at step 9 via --step 9 or --from-step 9).
    raw_df8 = state["df8"]
    if raw_df8.empty:
        candidates_filtered = raw_df8.copy()
    else:
        raw_df8 = raw_df8.copy()
        raw_df8["is_candidate"] = (
            raw_df8["is_candidate"].astype(str).str.lower() == "true"
        )
        candidates_filtered = (
            raw_df8[raw_df8["is_candidate"]].copy().reset_index(drop=True)
        )

    source_tag = "mock" if mock else "real"
    print(f"[{source_tag}] {len(candidates_filtered)} candidates (is_candidate=True)")

    def _run() -> pd.DataFrame:
        if candidates_filtered.empty:
            print("[WARNING] No candidates found. Writing empty report.")
            return step9_filter_validate.generate_final_report(
                candidates_filtered, pd.DataFrame(), {}, output_path=output_path,
            )

        # Jaccard annotation (adds max_jaccard, closest_pathway, overlap_class)
        cf = step9_filter_validate.filter_by_jaccard(candidates_filtered, all_pathways)
        n_high = int((cf["overlap_class"] == "high_overlap").sum())
        n_low  = int((cf["overlap_class"] == "low_overlap").sum())
        print(f"  high_overlap={n_high}  low_overlap={n_low}")

        # GO enrichment — build the reverse index once and reuse it across all
        # candidates to keep the per-candidate cost at O(|candidate_genes|).
        go_index = step9_filter_validate._build_go_index(gene_go, all_go_terms)
        go_enrichment_results: dict[str, pd.DataFrame] = {}
        for _, row in cf.iterrows():
            pid   = row["pathway_id"]
            genes = [
                g.strip()
                for g in str(row["scored_genes"]).split(",")
                if g.strip()
            ]
            go_enrichment_results[pid] = step9_filter_validate.go_enrichment_validation(
                genes, gene_go, all_go_terms, go_index=go_index,
            )

        # Per-candidate SHAP (batch, long-format)
        from step7_shap_analysis import batch_local_shap, _build_feature_names

        expected_d = len(_build_feature_names(all_go_terms))
        if model.get_booster().num_features() != expected_d:
            raise ValueError(
                f"Model has {model.get_booster().num_features()} features but "
                f"all_go_terms implies {expected_d}. Re-run step 6."
            )
        gene_lists = [
            [g.strip() for g in str(row["scored_genes"]).split(",") if g.strip()]
            for _, row in cf.iterrows()
        ]
        X_cands = step4_feature_extraction.build_feature_matrix(
            gene_lists, gene_go, all_go_terms,
        )
        shap_summary = batch_local_shap(
            model, X_cands,
            candidate_ids=cf["pathway_id"].tolist(),
            all_go_terms=all_go_terms,
            top_k=5,
        )

        return step9_filter_validate.generate_final_report(
            cf, shap_summary, go_enrichment_results, output_path=output_path,
        )

    return _run_step(9, "Filter, validate & final report", _run)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and run the requested pipeline steps in order."""
    parser = _build_parser()
    args   = parser.parse_args()
    mock   = args.mock

    # Determine which steps to run.
    if args.step is not None:
        if not 1 <= args.step <= 9:
            parser.error(f"--step must be between 1 and 9, got {args.step}")
        steps = [args.step]
    elif args.from_step is not None:
        if not 1 <= args.from_step <= 9:
            parser.error(f"--from-step must be between 1 and 9, got {args.from_step}")
        steps = list(range(args.from_step, 10))
    else:
        steps = list(range(1, 10))

    start_step = steps[0]
    full_run   = (steps == list(range(1, 10)))

    print(f"KEGG ML Pipeline  ({_now()})")
    print(f"  Mode  : {'mock' if mock else 'real'}")
    print(f"  Steps : {steps}")
    if args.deg:
        print(f"  DEG   : {args.deg}")

    # For partial runs, load persisted artifacts from disk before executing
    # the first requested step.
    state: dict[str, Any] = {}
    if not full_run:
        print(f"\nLoading state for start step {start_step}...")
        state = _load_state(start_step, mock)

    # Execute the requested steps in order.
    for n in steps:
        if n == 1:
            _step1(mock)
        elif n == 2:
            _step2(state, mock)
        elif n == 3:
            _step3(state, mock)
        elif n == 4:
            _step4(state, mock)
        elif n == 5:
            _step5(state, mock)
        elif n == 6:
            _step6(state, mock)
        elif n == 7:
            _step7(state, mock)
        elif n == 8:
            _step8(state, mock, args.deg)
        elif n == 9:
            _step9(state, mock)

    # Print a short summary after all steps complete.
    final_path = config.MOCK_FINAL_REPORT if mock else config.FINAL_REPORT
    print(f"\n{'=' * 60}")
    print("Pipeline complete!")
    print("=" * 60)
    # Only show the final report when step 9 was part of this run; otherwise
    # the CSV on disk may be from a previous invocation and would mislead the
    # user into thinking this run produced it.
    if 9 in steps and os.path.exists(final_path):
        print(f"Final report: {final_path}")
        try:
            df = pd.read_csv(final_path)
            if not df.empty:
                print(df.head(5).to_string(index=False))
        except Exception:
            pass


if __name__ == "__main__":
    main()
