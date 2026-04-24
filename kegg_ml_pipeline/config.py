# Centralized path configuration used by later pipeline stages.
# Keeping these values in one module makes it easier to switch between
# real-data runs and future mock/test runs without rewriting scripts.

# Parsed KEGG pathway cache generated from the KEGG raw files / API.
# Stored as TSV so each pathway remains easy to inspect and diff.
KEGG_CACHE = "data/kegg_pathways.tsv"

# Parsed AraCyc cache written by the AraCyc preprocessing step.
# Stored in the same TSV layout as the KEGG cache and merged table.
ARACYC_CACHE = "data/aracyc_pathways.tsv"

# Actual raw AraCyc input available in this repository.
ARACYC_RAW = "data/aracyc_pathways.20251021"

# Final merged pathway table combining KEGG and AraCyc.
# Stored as TSV so the output is easy to inspect in spreadsheets or shell tools.
ALL_PATHWAYS = "data/all_pathways.tsv"

# TAIR GO annotation source file used to build gene -> GO mappings.
GO_ANNOTATION = "data/ATH_GO_GOSLIM.txt"

# Serialized gene -> GO mapping cache.
# Reserved for a future TSV export in the GO-processing step.
GO_CACHE = "data/gene_go.tsv"

# Feature matrix and labels saved after dataset construction.
FEATURE_MATRIX = "data/feature_matrix.npz"
LABELS = "data/labels.npy"
TRAIN_DATA = "data/train.npz"
TEST_DATA  = "data/test.npz"
MOCK_TRAIN_DATA = "data/mock_train.npz"
MOCK_TEST_DATA  = "data/mock_test.npz"
MOCK_MODEL_PATH = "results/mock/pathway_model.json"
MOCK_RESULTS_DIR = "results/mock/"
CANDIDATE_SCORES      = "results/candidate_scores.csv"
MOCK_CANDIDATE_SCORES = "results/mock/candidate_scores.csv"
FINAL_REPORT          = "results/final_candidate_pathways.csv"
MOCK_FINAL_REPORT     = "results/mock/final_candidate_pathways.csv"

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
