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
``--mock --step 2`` is the only mock smoke-check: it generates pathways in
memory (``save=False``) and never writes ``data/step2_all_pathways.tsv``, so
``--mock --from-step 3`` would fail to find that artefact on disk — except
that ``_load_state()`` always reconstructs ``all_pathways`` and ``gene_go``
from ``generate_mock_universe()`` in mock mode, so partial mock runs from
step 3 onwards work correctly without any TSV on disk.

All other mock steps write their normal artefacts under ``data/mock_stepN_*``
or ``results/mock/``.  In particular ``--mock --step 4`` now writes the full
``data/mock_step4_feature_matrix.npz`` (it is no longer a smoke-check).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
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


def _safe_filename(s: str) -> str:
    """Sanitise an arbitrary string for use as a filename component.

    Keeps only ``[A-Za-z0-9_.-]``; every other character (``:``, ``/``, ``+``,
    whitespace, etc.) is replaced with ``_``.  Pathway IDs such as
    ``aracyc:mock_0`` or KEGG pathway URLs would otherwise break filesystem
    paths or be platform-dependent (Windows reserved characters), so we
    normalise here for cross-platform supplementary outputs.
    """
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


# Human-readable recovery hints printed when a step fails.
_FIX_HINTS: dict[int, str] = {
    1: "Run: pip install -r requirements.txt  (Python >= 3.10 required)",
    2: "No internet?  Use: python run_pipeline.py --mock",
    3: (
        "Missing GO annotation source.  Place either data/tair.gaf.gz or "
        "data/ATH_GO_GOSLIM.txt under kegg_ml_pipeline/data/, or use --mock."
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
        ``data/step2_all_pathways.tsv`` to disk (step 2 uses ``save=False``).

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

        # Provenance must also match between train and test. Two NPZs from
        # different GO snapshots can have identical surviving vocabularies
        # (the filter happens to keep the same term set on both sources), but
        # the gene→GO mappings underneath the vector dimensions differ, so
        # mixing them in step 6 evaluation is silently wrong.
        def _read_provenance(npz):
            sha = (str(npz["go_source_sha256"].item())
                   if "go_source_sha256" in npz.files else "")
            mg  = (int(npz["go_min_genes"].item())
                   if "go_min_genes" in npz.files else -1)
            mf  = (float(npz["go_max_fraction"].item())
                   if "go_max_fraction" in npz.files else -1.0)
            return sha, mg, mf

        train_prov = _read_provenance(train_data)
        test_prov  = _read_provenance(test_data)
        if train_prov != test_prov:
            print(
                f"[ERROR] GO provenance mismatch between {train_path} "
                f"(sha={train_prov[0][:12]}..., min_genes={train_prov[1]}, "
                f"max_fraction={train_prov[2]}) and {test_path} "
                f"(sha={test_prov[0][:12]}..., min_genes={test_prov[1]}, "
                f"max_fraction={test_prov[2]}). "
                "Re-run step 5 to regenerate both files from the same GO snapshot."
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

        # Vocab consistency check: gene_go (after the step-3 GO filter) must
        # share the same vocabulary as train.npz's all_go_terms. A mismatch
        # means the GO filter parameters changed since training, and any
        # downstream feature extraction would silently use the wrong width.
        if not mock:
            cache_vocab = sorted(
                {t for ts in state["gene_go"].values() for t in ts}
            )
            if cache_vocab != list(state["all_go_terms"]):
                print(
                    f"[ERROR] gene_go vocab ({len(cache_vocab)}) != "
                    f"train.npz all_go_terms ({len(state['all_go_terms'])}). "
                    "Re-run from step 3 — likely the GO filter parameters "
                    "changed."
                )
                sys.exit(1)

            # Provenance check: the all_go_terms vocab match above only
            # catches changes that alter which terms survive the filter. A
            # source update can rewire gene→GO mappings while leaving the
            # retained term set intact (e.g. a gene gains/loses an annotation
            # to a term that was already kept). Comparing the source SHA and
            # the filter params catches that case so step 8/9 cannot quietly
            # feed a new feature space into an old model.
            saved_sha, saved_min, saved_max = train_prov
            if not saved_sha or saved_min < 0 or saved_max < 0:
                # Real-mode hard fail: a missing-provenance NPZ means we
                # cannot prove train, test, model, and current GO source are
                # all aligned, which is exactly the silent-inconsistency case
                # the provenance fields were added to prevent.
                print(
                    f"[ERROR] {train_path} lacks GO provenance fields "
                    "(produced by an older pipeline version). Cannot verify "
                    "the trained model still matches the current GO source. "
                    "Re-run from step 5: "
                    "python run_pipeline.py --from-step 5"
                )
                sys.exit(1)

            # current_go_source_sha256() prefers the raw ATH file but falls
            # back to the cache meta header — needed for users who keep only
            # data/step3_gene_go.tsv after deleting the ATH/GAF source.
            current_sha = step3_go_annotation.current_go_source_sha256()
            if not current_sha:
                print(
                    "[ERROR] Cannot determine current GO source SHA256: "
                    f"neither {config.GO_ANNOTATION!r} nor "
                    f"{config.GO_CACHE!r} is available. "
                    "Re-run from step 3."
                )
                sys.exit(1)
            if saved_sha != current_sha:
                print(
                    f"[ERROR] GO source SHA256 mismatch: train.npz="
                    f"{saved_sha[:12]}... vs current="
                    f"{current_sha[:12]}.... "
                    "The GO annotation source changed since training; "
                    "re-run from step 3."
                )
                sys.exit(1)
            if saved_min != config.GO_MIN_GENES:
                print(
                    f"[ERROR] GO_MIN_GENES mismatch: train.npz="
                    f"{saved_min} vs config={config.GO_MIN_GENES}. "
                    "Re-run from step 3."
                )
                sys.exit(1)
            if abs(saved_max - config.GO_MAX_GENE_FRACTION) > 1e-9:
                print(
                    f"[ERROR] GO_MAX_GENE_FRACTION mismatch: train.npz="
                    f"{saved_max} vs config={config.GO_MAX_GENE_FRACTION}. "
                    "Re-run from step 3."
                )
                sys.exit(1)

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
        # save=False in mock mode: avoids overwriting data/step2_all_pathways.tsv
        # with synthetic data, keeping the real-data artefact intact.
        return step2b_aracyc.merge_and_save_all_pathways(kegg, aracyc, save=not mock)

    state["all_pathways"] = _run_step(2, "Fetch & merge pathways", _run)

    if mock:
        # Warn: the pathway dict lives only in memory for the rest of this run.
        # _load_state() always rebuilds mock pathways from generate_mock_universe(),
        # so --mock --from-step 3 still works without any TSV on disk; this print
        # just makes the in-memory-only behaviour explicit for the user.
        print(
            "[Step 2] Mock smoke check: pathways are in memory only "
            "(data/step2_all_pathways.tsv was NOT written)."
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
    """Build the full feature matrix and save it as a NumPy archive.

    The on-disk archive is intentionally redundant with the train/test NPZ
    files written by step 5 — it preserves the *unsplit* per-pathway feature
    matrix and aligned ``pathway_ids``, which is the natural input table for
    paper supplementary analyses (e.g. UMAP, clustering, manual inspection).
    """
    all_pathways = state["all_pathways"]
    gene_go      = state["gene_go"]
    all_go_terms = state["all_go_terms"]
    feature_path = config.MOCK_FEATURE_MATRIX if mock else config.FEATURE_MATRIX

    def _run() -> None:
        valid = {
            pid: p
            for pid, p in all_pathways.items()
            if len(p.get("genes", set())) >= 3
        }
        pathway_ids = list(valid.keys())
        gene_lists  = [list(p["genes"]) for p in valid.values()]
        X = step4_feature_extraction.build_feature_matrix(
            gene_lists, gene_go, all_go_terms,
        )

        expected_d = len(all_go_terms) + 6
        if X.shape[1] != expected_d:
            raise ValueError(
                f"Feature dimension mismatch: got {X.shape[1]}, "
                f"expected {expected_d} (len(all_go_terms) + 6)"
            )
        step4_feature_extraction.save_feature_matrix(
            X, all_go_terms, pathway_ids, path=feature_path,
        )

    _run_step(4, "Build feature matrix", _run)


def _step5(state: dict, mock: bool) -> None:
    """Build the labelled dataset and split into train / test archives."""
    all_pathways = state["all_pathways"]
    gene_go      = state["gene_go"]
    all_go_terms = state["all_go_terms"]
    train_path   = config.MOCK_TRAIN_DATA if mock else config.TRAIN_DATA
    test_path    = config.MOCK_TEST_DATA  if mock else config.TEST_DATA

    # Real-mode provenance: SHA of the GO source + the filter params that
    # produced gene_go. Step 6+ resume compares these against current state to
    # detect a model trained against a different GO snapshot. Mock mode skips
    # provenance (sentinel values) since the data is regenerated each run.
    #
    # Resolved inside _run() so any failure (e.g. cache-only mode where neither
    # ATH nor GAF is on disk and the cache meta is also missing) routes through
    # _run_step's fix-hint UX instead of crashing before the step banner prints.
    def _run() -> tuple:
        if mock:
            sha = ""
            min_genes = -1
            max_fraction = -1.0
        else:
            # current_go_source_sha256() prefers GO_ANNOTATION but falls back
            # to GO_CACHE meta — keeps step 5 runnable in cache-only mode,
            # consistent with what load_go_annotation() already supports.
            sha = step3_go_annotation.current_go_source_sha256()
            if not sha:
                raise RuntimeError(
                    "Cannot determine GO source SHA256 for provenance: "
                    f"neither {config.GO_ANNOTATION!r} nor {config.GO_CACHE!r} "
                    "is available. Re-run from step 3."
                )
            min_genes = config.GO_MIN_GENES
            max_fraction = config.GO_MAX_GENE_FRACTION

        X, y = step5_build_dataset.build_dataset(all_pathways, gene_go, all_go_terms)
        return step5_build_dataset.split_and_save(
            X, y, all_go_terms,
            train_path=train_path, test_path=test_path,
            go_source_sha256=sha,
            go_min_genes=min_genes,
            go_max_fraction=max_fraction,
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
        test_metrics = step6_train_xgboost.evaluate_model(
            model, X_test, y_test, results_dir=results_dir,
        )
        # Persist metrics for the end-of-pipeline metrics_summary.csv writer.
        # state captures from the *fresh* run only — partial runs that skip
        # step 6 will not have these keys, which is the intended gating.
        state["cv_scores"]    = cv_scores
        state["test_metrics"] = test_metrics
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
    """Jaccard deduplication, GO enrichment, per-candidate SHAP, final report.

    Beyond ``step9_final_candidate_pathways.csv``, this step also produces three
    paper-supplementary artefacts (all named with a ``step9_`` prefix so the
    file records its producing step):
      * ``results/shap/step9_local_candidate_<pid>.png`` — top-K waterfall plots
      * ``results/shap/step9_batch_summary.csv``         — per-candidate top-5 SHAP
      * ``results/step9_go_enrichment_per_candidate.csv``— full GO enrichment table

    The two CSV outputs are always written with a stable header even when
    there are no candidates, so downstream scripts can rely on the schema
    without ``os.path.exists()`` checks.
    """
    model        = state["model"]
    gene_go      = state["gene_go"]
    all_go_terms = state["all_go_terms"]
    all_pathways = state["all_pathways"]
    output_path  = config.MOCK_FINAL_REPORT if mock else config.FINAL_REPORT
    shap_dir = (
        os.path.join(config.MOCK_RESULTS_DIR, "shap") if mock else config.SHAP_DIR
    )
    shap_csv_path = (
        config.MOCK_SHAP_BATCH_SUMMARY if mock else config.SHAP_BATCH_SUMMARY
    )
    go_full_path = (
        config.MOCK_GO_ENRICHMENT_PER_CANDIDATE
        if mock
        else config.GO_ENRICHMENT_PER_CANDIDATE
    )

    # Stable column schemas for the two supplementary CSVs — used in both the
    # populated and empty branches so the file always has the same header.
    BATCH_SUMMARY_COLS = ["candidate_id", "rank", "feature_name", "shap_value"]
    GO_FULL_COLS = [
        "pathway_id", "rank", "go_term",
        "overlap_count", "expected", "p_value", "p_adjusted",
    ]

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

    def _write_empty_supplementary() -> None:
        """Write header-only batch_summary.csv and go_enrichment_per_candidate.csv,
        and clean up stale local_candidate_*.png from prior runs.
        """
        os.makedirs(os.path.dirname(shap_csv_path) or ".", exist_ok=True)
        pd.DataFrame(columns=BATCH_SUMMARY_COLS).to_csv(shap_csv_path, index=False)
        print(f"Saved → {shap_csv_path}  (0 rows)")

        os.makedirs(os.path.dirname(go_full_path) or ".", exist_ok=True)
        pd.DataFrame(columns=GO_FULL_COLS).to_csv(go_full_path, index=False)
        print(f"Saved → {go_full_path}  (0 rows)")

        # Clean stale waterfall images so an `ls step9_local_candidate_*.png`
        # count reflects the current run, not residue from a prior run.
        if os.path.isdir(shap_dir):
            for stale in glob.glob(
                os.path.join(shap_dir, "step9_local_candidate_*.png")
            ):
                os.remove(stale)

    def _run() -> pd.DataFrame:
        if candidates_filtered.empty:
            print("[WARNING] No candidates found. Writing empty report + empty supplementary CSVs.")
            _write_empty_supplementary()
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

        # First pass: top-10 (default truncation) — feeds the final report's
        # top_enriched_go column.
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

        # Second pass: full enrichment (top_n=len(all_go_terms) disables the
        # internal .head(top_n) truncation) — feeds the supplementary CSV.
        # We keep the two passes decoupled so the final report stays compact
        # while the supplementary CSV is exhaustive.
        go_full_results: dict[str, pd.DataFrame] = {}
        for _, row in cf.iterrows():
            pid   = row["pathway_id"]
            genes = [
                g.strip()
                for g in str(row["scored_genes"]).split(",")
                if g.strip()
            ]
            go_full_results[pid] = step9_filter_validate.go_enrichment_validation(
                genes, gene_go, all_go_terms,
                go_index=go_index,
                top_n=len(all_go_terms),
            )

        # Per-candidate SHAP (batch, long-format) and top-K waterfall plots
        from step7_shap_analysis import (
            _build_feature_names,
            batch_local_shap,
            local_shap_analysis,
        )

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

        # Clean stale step9_local_candidate_*.png from prior runs, then write
        # the current top-K waterfalls.  Only delete this exact prefix so
        # step7_local_*.png and step7_global_summary.png survive.
        os.makedirs(shap_dir, exist_ok=True)
        for stale in glob.glob(
            os.path.join(shap_dir, "step9_local_candidate_*.png")
        ):
            os.remove(stale)

        top_k = min(config.SHAP_LOCAL_TOP_K, len(cf))
        top_cf = cf.nlargest(top_k, "score").reset_index(drop=True)
        # X_cands rows align with cf order (pre-sort), so map pid -> index in cf.
        cf_index_by_pid = {
            row["pathway_id"]: i
            for i, row in cf.reset_index(drop=True).iterrows()
        }
        for _, row in top_cf.iterrows():
            pid = row["pathway_id"]
            idx = cf_index_by_pid[pid]
            local_shap_analysis(
                model, X_cands[idx],
                sample_id=f"candidate_{_safe_filename(pid)}",
                all_go_terms=all_go_terms,
                pathway_name=row.get("pathway_name", ""),
                shap_dir=shap_dir,
                # Embed step-9 in the output filename so the file records its
                # producing step (paper-supplementary friendly).
                file_prefix="step9",
            )

        # Save batch SHAP summary CSV — always with the stable header schema.
        os.makedirs(os.path.dirname(shap_csv_path) or ".", exist_ok=True)
        if shap_summary.empty:
            pd.DataFrame(columns=BATCH_SUMMARY_COLS).to_csv(shap_csv_path, index=False)
        else:
            shap_summary.to_csv(shap_csv_path, index=False)
        print(f"Saved → {shap_csv_path}  ({len(shap_summary)} rows)")

        # Save full per-candidate GO enrichment CSV — long format, header stable.
        go_long_records: list[dict] = []
        for pid, edf in go_full_results.items():
            if edf.empty:
                continue
            for rank, (_, erow) in enumerate(edf.iterrows(), 1):
                go_long_records.append({
                    "pathway_id":    pid,
                    "rank":          rank,
                    "go_term":       erow["go_term"],
                    "overlap_count": int(erow["overlap_count"]),
                    "expected":      float(erow["expected"]),
                    "p_value":       float(erow["p_value"]),
                    "p_adjusted":    float(erow["p_adjusted"]),
                })
        go_long_df = (
            pd.DataFrame(go_long_records, columns=GO_FULL_COLS)
            if go_long_records
            else pd.DataFrame(columns=GO_FULL_COLS)
        )
        os.makedirs(os.path.dirname(go_full_path) or ".", exist_ok=True)
        go_long_df.to_csv(go_full_path, index=False)
        print(f"Saved → {go_full_path}  ({len(go_long_df)} rows)")

        return step9_filter_validate.generate_final_report(
            cf, shap_summary, go_enrichment_results, output_path=output_path,
        )

    report_df = _run_step(9, "Filter, validate & final report", _run)
    # Always persist the report so end-of-pipeline metrics_summary.csv has a
    # consistent input — even when candidates_filtered is empty (in which case
    # report_df is also empty but still has the standard column schema).
    state["report_df"] = report_df
    return report_df


def _write_metrics_summary(state: dict, mock: bool, steps: list[int]) -> None:
    """Write a single-row CSV aggregating CV/test metrics and candidate counts.

    Strict gating: only write when *this* invocation actually executed both
    step 6 (which produces ``cv_scores`` and ``test_metrics``) and step 9
    (which produces ``report_df``).  Any partial run that does not include
    both leaves the previous metrics_summary.csv untouched, so a stale write
    cannot pollute paper-ready supplementary files.
    """
    steps_set = set(steps)
    fresh_run = {6, 9} <= steps_set
    has_state = all(
        k in state for k in ("cv_scores", "test_metrics", "report_df")
    )
    if not (fresh_run and has_state):
        return

    cv  = state["cv_scores"]
    tm  = state["test_metrics"]
    rdf = state["report_df"]
    summary = {
        "mode":                 "mock" if mock else "real",
        "timestamp":            datetime.now().isoformat(timespec="seconds"),
        "cv_auroc_mean":        float(np.mean(cv)),
        "cv_auroc_std":         float(np.std(cv)),
        "test_auroc":           float(tm["auroc"]),
        "test_auprc":           float(tm["auprc"]),
        "test_accuracy":        float(tm["accuracy"]),
        "test_f1":              float(tm["f1"]),
        "n_candidates_total":   int(len(rdf)),
        "n_high_overlap": (
            int((rdf["overlap_class"] == "high_overlap").sum())
            if "overlap_class" in rdf.columns else 0
        ),
        "n_low_overlap": (
            int((rdf["overlap_class"] == "low_overlap").sum())
            if "overlap_class" in rdf.columns else 0
        ),
    }
    path = config.MOCK_METRICS_SUMMARY if mock else config.METRICS_SUMMARY
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame([summary]).to_csv(path, index=False)
    print(f"Saved → {path}")


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

    # Aggregate metrics into a single-row summary CSV — only written when this
    # run actually executed both step 6 and step 9 (see _write_metrics_summary
    # for the gating rules).  Partial runs leave any existing file untouched.
    _write_metrics_summary(state, mock, steps)

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
