# Centralized path configuration used by later pipeline stages.
# Keeping these values in one module makes it easier to switch between
# real-data runs and future mock/test runs without rewriting scripts.

# Parsed KEGG pathway cache generated from the KEGG raw files / API.
KEGG_CACHE = "data/kegg_pathways.json"

# Parsed AraCyc cache written by the AraCyc preprocessing step.
ARACYC_CACHE = "data/aracyc_pathways.json"

# Actual raw AraCyc input available in this repository.
ARACYC_RAW = "data/aracyc_pathways.20251021"

# Merged pathway universe combining KEGG and AraCyc.
ALL_PATHWAYS = "data/all_pathways.json"

# TAIR GO annotation source file used to build gene -> GO mappings.
GO_ANNOTATION = "data/ATH_GO_GOSLIM.txt"

# Serialized gene -> GO mapping cache.
GO_CACHE = "data/gene_go.json"

# Feature matrix and labels saved after dataset construction.
FEATURE_MATRIX = "data/feature_matrix.npz"
LABELS = "data/labels.npy"

# Trained model output and SHAP analysis directory.
MODEL_PATH = "results/pathway_model.json"
SHAP_DIR = "results/shap/"
RESULTS_DIR = "results/"

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
