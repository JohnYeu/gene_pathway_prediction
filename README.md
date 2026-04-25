# gene_pathway_prediction

Machine-learning pipeline for Arabidopsis thaliana pathway discovery: given a list of differentially expressed genes (DEGs), predicts which KEGG / AraCyc pathways they are associated with.

---

## Installation

```bash
cd kegg_ml_pipeline
pip install -r requirements.txt
```

Python ≥ 3.10 is required.

---

## Data preparation

| Data | Source | Required? |
|------|--------|-----------|
| KEGG *Arabidopsis thaliana* pathways | Fetched automatically by step 2 (internet required) | No (use `--mock` for offline testing) |
| AraCyc pathway file | Included in `data/aracyc_pathways.20251021` | Yes (already in repo) |
| TAIR GO annotation (`ATH_GO_GOSLIM.txt`) | Download from https://www.arabidopsis.org/download/go | Yes for real runs |

Place `ATH_GO_GOSLIM.txt` in `kegg_ml_pipeline/data/` before running the real pipeline.

---

## Quick start

```bash
git clone <repo-url> gene_pathway_prediction
cd gene_pathway_prediction/kegg_ml_pipeline
pip install -r requirements.txt
python run_pipeline.py --mock
```

`--mock` uses a built-in synthetic dataset (300 genes × 150 GO terms × 35 pathways) and requires no internet access or real data files. Output is written to `results/mock/`.

For a real run (after downloading `ATH_GO_GOSLIM.txt`):

```bash
python run_pipeline.py
```

To score a specific DEG list against known pathways:

```bash
python run_pipeline.py --deg my_degs.txt
```

---

## Pipeline steps

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1 | `step1_setup.py` | — | Checks Python packages |
| 2 | `step2a_kegg.py` + `step2b_aracyc.py` | KEGG REST API, `data/aracyc_pathways.20251021` | `data/all_pathways.tsv` |
| 3 | `step3_go_annotation.py` | `data/ATH_GO_GOSLIM.txt` | In-memory gene→GO map |
| 4 | `step4_feature_extraction.py` | Pathway gene sets, gene→GO map | Feature vectors (smoke check in `run_pipeline`; full `data/feature_matrix.npz` via standalone script) |
| 5 | `step5_build_dataset.py` | Pathways, gene→GO map | `data/train.npz`, `data/test.npz` |
| 6 | `step6_train_xgboost.py` | Train/test NPZ | `results/pathway_model.json`, `results/cv_scores.png`, `results/roc_curve.png` |
| 7 | `step7_shap_analysis.py` | Model, training features | `results/shap/global_summary.png`, `results/shap/global_importance.csv` |
| 8 | `step8_score_candidates.py` | Model, pathways, optional DEG list | `results/candidate_scores.csv` |
| 9 | `step9_filter_validate.py` | Candidate scores, pathways, GO annotations | `results/final_candidate_pathways.csv` |

---

## Partial and targeted runs

```bash
# Run only step 6
python run_pipeline.py --step 6

# Resume from step 4 onwards
python run_pipeline.py --from-step 4

# --step and --from-step are mutually exclusive
python run_pipeline.py --step 5 --from-step 3   # error
```

**Mock partial runs:** `--mock --step 2` and `--mock --step 4` are smoke checks that do not write any disk artifacts (step 2 skips the TSV save; step 4 only validates feature dimensions). To run the full mock sequence, use `--mock` or `--mock --from-step 2`.

---

## Output files

| File | Description |
|------|-------------|
| `results/pathway_model.json` | Trained XGBoost model (XGBoost native JSON format) |
| `results/cv_scores.png` | Per-fold CV AUROC bar chart |
| `results/roc_curve.png` | ROC curve on the held-out test set |
| `results/shap/global_summary.png` | SHAP beeswarm plot (top-30 features) |
| `results/shap/global_importance.csv` | Mean \|SHAP\| per feature, sorted descending |
| `results/candidate_scores.csv` | XGBoost scores for all scored candidates; includes DEG coverage columns in DEG mode |
| `results/final_candidate_pathways.csv` | Final report: Jaccard annotation, top GO enrichment terms, top SHAP features per candidate |

Mock-mode outputs use the same names under `results/mock/`.

### Columns in `final_candidate_pathways.csv`

| Column | Description |
|--------|-------------|
| `pathway_id` | Pathway identifier (KEGG or AraCyc) |
| `pathway_name` | Human-readable pathway name |
| `source` | `KEGG` or `AraCyc` |
| `scored_gene_count` | Number of genes scored |
| `score` | XGBoost probability that the gene set belongs to a real pathway |
| `is_candidate` | `True` when score ≥ `SCORE_THRESHOLD` (default 0.75) |
| `max_jaccard` | Highest Jaccard similarity to any known pathway |
| `closest_pathway` | ID of the most similar known pathway |
| `overlap_class` | `high_overlap` or `low_overlap` (threshold: `JACCARD_THRESHOLD` = 0.5) |
| `top_enriched_go` | Top 3 enriched GO terms (pipe-separated) |
| `min_go_pvalue` | Minimum Bonferroni-adjusted GO enrichment p-value |
| `top_shap_features` | Top 3 SHAP-contributing features (pipe-separated) |

DEG mode also includes: `overlap_count`, `pathway_gene_count`, `deg_gene_count`, `pathway_coverage`, `deg_coverage`.

---

## Configuration

Key constants in `config.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `SCORE_THRESHOLD` | 0.75 | Minimum score for `is_candidate=True` |
| `JACCARD_THRESHOLD` | 0.5 | Threshold for `high_overlap` / `low_overlap` |
| `AUROC_THRESHOLD` | 0.85 | Minimum CV AUROC for a passing model |
| `NEG_POS_RATIO` | 2 | Negative-to-positive sampling ratio |
| `XGB_PARAMS` | see file | XGBoost hyperparameters |

---

## FAQ

**The CV AUROC is below the 0.85 threshold — what should I do?**
Increase `n_estimators` (e.g. 500–1000), lower `learning_rate` (e.g. 0.01–0.03), or adjust `max_depth` in `XGB_PARAMS` in `config.py`. For automated search, consider Optuna.

**I get a `UnicodeDecodeError` when reading data files.**
All file readers in the pipeline use `errors="ignore"` for robustness against non-UTF-8 bytes. If you see this error from a custom file, ensure it is UTF-8 or Latin-1 encoded.

**The KEGG API returns 403 / rate-limit errors.**
Step 2a uses automatic exponential-backoff retry (up to 3 attempts). If KEGG is unavailable, use `--mock` to verify the rest of the pipeline.

**`run_pipeline --mock --step 7` produces only a global beeswarm plot, not per-sample waterfall plots.**
`run_pipeline.py` calls only `global_shap_analysis()` (the fast path). For local waterfall plots and a batch SHAP summary, run `python step7_shap_analysis.py --mock` directly.

**`run_pipeline --mock --step 4` does not write `data/feature_matrix.npz`.**
Correct — step 4 in `run_pipeline` is a smoke check only. To produce `feature_matrix.npz`, run `python step4_feature_extraction.py` directly.

**How do I produce `data/feature_matrix.npz` for external inspection?**
```bash
python step4_feature_extraction.py        # real data
python step4_feature_extraction.py --mock # mock data
```
`run_pipeline.py` does not write this file; step 5 re-derives the features from scratch and does not depend on it.
