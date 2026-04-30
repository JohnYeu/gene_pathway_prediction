# Centralized path configuration used by later pipeline stages.
# Keeping these values in one module makes it easier to switch between
# real-data runs and future mock/test runs without rewriting scripts.

# All generated intermediate products carry a `stepN_` prefix so the file
# itself records which pipeline step produced it (paper-supplementary friendly).
# Inputs (ARACYC_RAW, GO_ANNOTATION) keep their original names.

# ── Step 2: KEGG + AraCyc pathway parsing & merging ─────────────────────────
# Parsed KEGG pathway cache generated from the KEGG raw files / API.
KEGG_CACHE = "data/step2_kegg_pathways.tsv"

# Parsed AraCyc cache written by the AraCyc preprocessing step.
ARACYC_CACHE = "data/step2_aracyc_pathways.tsv"

# Actual raw AraCyc input available in this repository (INPUT — not renamed).
ARACYC_RAW = "data/aracyc_pathways.20251021"

# Final merged pathway table combining KEGG and AraCyc.
ALL_PATHWAYS = "data/step2_all_pathways.tsv"

# ── Step 3: GO annotation parsing ───────────────────────────────────────────
# Optional upstream GAF source. When ATH_GO_GOSLIM.txt is missing but this file
# exists, step 3 derives ATH_GO_GOSLIM.txt from it. mtime is intentionally NOT
# used to decide regeneration (unzip / git checkout / cp all reset mtime).
GO_GAF = "data/tair.gaf.gz"

# TAIR GO annotation source file (INPUT — not renamed).
GO_ANNOTATION = "data/ATH_GO_GOSLIM.txt"

# Serialized gene -> GO mapping cache (TSV export from step 3).
GO_CACHE = "data/step3_gene_go.tsv"

# Dual-frequency GO term filter (real mode only — mock pipeline bypasses).
# Drops terms annotated to fewer than GO_MIN_GENES genes (statistical noise)
# and terms covering more than GO_MAX_GENE_FRACTION of annotated genes
# (non-discriminative). Applied in step 3 right after parsing.
GO_MIN_GENES         = 20
GO_MAX_GENE_FRACTION = 0.30

# ── Step 4: feature extraction ──────────────────────────────────────────────
FEATURE_MATRIX      = "data/step4_feature_matrix.npz"
MOCK_FEATURE_MATRIX = "data/mock_step4_feature_matrix.npz"

# ── Step 5: dataset construction (train/test split) ─────────────────────────
LABELS          = "data/step5_labels.npy"
TRAIN_DATA      = "data/step5_train.npz"
TEST_DATA       = "data/step5_test.npz"
MOCK_TRAIN_DATA = "data/mock_step5_train.npz"
MOCK_TEST_DATA  = "data/mock_step5_test.npz"

# ── Step 6: XGBoost training ────────────────────────────────────────────────
MODEL_PATH      = "results/step6_pathway_model.json"
MOCK_MODEL_PATH = "results/mock/step6_pathway_model.json"

# ── Step 8: candidate scoring ───────────────────────────────────────────────
CANDIDATE_SCORES      = "results/step8_candidate_scores.csv"
MOCK_CANDIDATE_SCORES = "results/mock/step8_candidate_scores.csv"

# ── Step 9: filter, validate & final report + supplementary tables ──────────
FINAL_REPORT      = "results/step9_final_candidate_pathways.csv"
MOCK_FINAL_REPORT = "results/mock/step9_final_candidate_pathways.csv"

# Long-format batch SHAP summary written by step 9 (one row per candidate × top
# feature, top-5 features per candidate by |SHAP|).
SHAP_BATCH_SUMMARY      = "results/shap/step9_batch_summary.csv"
MOCK_SHAP_BATCH_SUMMARY = "results/mock/shap/step9_batch_summary.csv"

# Per-candidate full GO enrichment results (long format, all enriched terms,
# not just the top-3 written into the final report).
GO_ENRICHMENT_PER_CANDIDATE      = "results/step9_go_enrichment_per_candidate.csv"
MOCK_GO_ENRICHMENT_PER_CANDIDATE = "results/mock/step9_go_enrichment_per_candidate.csv"

# Single-row aggregated metrics CSV written at end of full pipeline runs.
METRICS_SUMMARY      = "results/step9_metrics_summary.csv"
MOCK_METRICS_SUMMARY = "results/mock/step9_metrics_summary.csv"

# Number of top-scoring candidates to generate local SHAP waterfall plots for.
SHAP_LOCAL_TOP_K = 10

# ── Output directories (no rename — directory containers, not files) ────────
RESULTS_DIR      = "results/"
MOCK_RESULTS_DIR = "results/mock/"
SHAP_DIR         = "results/shap/"

# Negative-to-positive sampling ratio for binary pathway classification.
NEG_POS_RATIO = 2

# Hold-out split proportion for model evaluation.
TEST_SIZE = 0.2

# Global random seed shared across data splitting and model training.
RANDOM_STATE = 42

# Number of folds for cross-validation based checks.
CV_FOLDS = 5

# Minimum AUROC expected from the synthetic/mock pipeline.
AUROC_THRESHOLD = 0.85

# Candidate acceptance threshold used by the scoring step.
SCORE_THRESHOLD = 0.75

# Similarity filter threshold for post-scoring pathway deduplication.
JACCARD_THRESHOLD = 0.5

# Default XGBoost configuration for the first baseline model.
# These values are intentionally conservative: enough capacity to learn
# the mock signal while staying fast on a laptop-scale dataset.
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "scale_pos_weight": 2,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "auc",
    "random_state": 42,
    "tree_method": "hist",
}
