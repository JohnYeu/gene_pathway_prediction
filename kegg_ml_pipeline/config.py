# Data paths
KEGG_CACHE = "data/kegg_pathways.json"
ARACYC_CACHE = "data/aracyc_pathways.json"
ARACYC_RAW = "data/aracyc_pathways.20251021"
ALL_PATHWAYS = "data/all_pathways.json"
GO_ANNOTATION = "data/ATH_GO_GOSLIM.txt"
GO_CACHE = "data/gene_go.json"
FEATURE_MATRIX = "data/feature_matrix.npz"
LABELS = "data/labels.npy"
MODEL_PATH = "results/pathway_model.json"
SHAP_DIR = "results/shap/"
RESULTS_DIR = "results/"

# Model hyperparameters
NEG_POS_RATIO = 2
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
AUROC_THRESHOLD = 0.85
SCORE_THRESHOLD = 0.75
JACCARD_THRESHOLD = 0.5

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
