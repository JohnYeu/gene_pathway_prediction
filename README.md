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
| TAIR GO annotation (`ATH_GO_GOSLIM.txt`) | Download from https://www.arabidopsis.org/download/go | Yes (or supply `tair.gaf.gz`) |
| TAIR GAF archive (`tair.gaf.gz`) | Download from https://current.geneontology.org/products/pages/downloads.html | Optional — used to derive `ATH_GO_GOSLIM.txt` automatically when it is missing |

Place either `ATH_GO_GOSLIM.txt` **or** `tair.gaf.gz` in `kegg_ml_pipeline/data/` before running the real pipeline. When only the GAF archive is present, step 3 will derive a project-compatible 9-column `ATH_GO_GOSLIM.txt` from it on first run (one-shot; not regenerated on subsequent runs because file mtimes after `unzip` / `git checkout` / `cp` are unrelated to the database release date).

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
| 1 | `step1_setup.py` | — | (environment check; no files) |
| 2 | `step2a_kegg.py` + `step2b_aracyc.py` | KEGG REST API, `data/aracyc_pathways.20251021` | `data/step2_kegg_pathways.tsv`, `data/step2_aracyc_pathways.tsv`, `data/step2_all_pathways.tsv` |
| 3 | `step3_go_annotation.py` | `data/ATH_GO_GOSLIM.txt` (or `data/tair.gaf.gz` to derive it) | `data/step3_gene_go.tsv` (with GO filter parameters in JSON header) |
| 4 | `step4_feature_extraction.py` | Pathway gene sets, gene→GO map | `data/step4_feature_matrix.npz` (full per-pathway feature matrix + aligned `pathway_ids`) |
| 5 | `step5_build_dataset.py` | Pathways, gene→GO map | `data/step5_train.npz`, `data/step5_test.npz` |
| 6 | `step6_train_xgboost.py` | Train/test NPZ | `results/step6_pathway_model.json`, `results/step6_cv_scores.png`, `results/step6_roc_curve.png` |
| 7 | `step7_shap_analysis.py` | Model, training features | `results/shap/step7_global_summary.png`, `results/shap/step7_global_importance.csv` (standalone also writes `step7_local_<n>.png`) |
| 8 | `step8_score_candidates.py` | Model, pathways, optional DEG list | `results/step8_candidate_scores.csv` |
| 9 | `step9_filter_validate.py` | Candidate scores, pathways, GO annotations | `results/step9_final_candidate_pathways.csv` + supplementary outputs (see "Output files") |

> **File-naming convention.** Every generated intermediate carries a `stepN_` prefix (e.g. `step4_feature_matrix.npz`, `step9_batch_summary.csv`) so the file itself records which pipeline step produced it — useful for paper supplementary indexing. Inputs (`ATH_GO_GOSLIM.txt`, `aracyc_pathways.20251021`, raw KEGG dumps) are not renamed. Mock-mode files use `mock_stepN_*` in `data/` (file-prefix) and `results/mock/stepN_*` in `results/` (subdirectory).

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

**Mock partial runs:** `--mock --step 2` is a smoke check that does not write any disk artifacts (step 2 skips the TSV save to avoid overwriting real-data caches). All other mock partial runs work normally and write their outputs under `results/mock/` and `data/mock_*.npz`.

---

## Output files

### Data caches and intermediate matrices (`data/`)

| File | Description |
|------|-------------|
| `data/step2_kegg_pathways.tsv` | Raw KEGG pathway cache (pathway_id → genes) |
| `data/step2_aracyc_pathways.tsv` | Raw AraCyc pathway cache |
| `data/step2_all_pathways.tsv` | Merged KEGG + AraCyc pathway table (single source of truth) |
| `data/step3_gene_go.tsv` | Parsed gene → GO term mapping cache |
| `data/step4_feature_matrix.npz` | Full per-pathway feature matrix (`X`, `all_go_terms`, `pathway_ids`) — paper supplementary input |
| `data/step5_train.npz`, `data/step5_test.npz` | Stratified train/test splits with aligned `all_go_terms` |

### Model + evaluation (`results/`)

| File | Description |
|------|-------------|
| `results/step6_pathway_model.json` | Trained XGBoost model (XGBoost native JSON format) |
| `results/step6_cv_scores.png` | Per-fold CV AUROC bar chart |
| `results/step6_roc_curve.png` | ROC curve on the held-out test set |
| `results/step9_metrics_summary.csv` | One-row aggregate of CV mean/std AUROC + test AUROC/AUPRC/accuracy/F1 + candidate counts (only written when this run includes both step 6 and step 9) |

### SHAP analyses (`results/shap/`)

| File | Description |
|------|-------------|
| `results/shap/step7_global_summary.png` | SHAP beeswarm plot (top-30 features) |
| `results/shap/step7_global_importance.csv` | Mean \|SHAP\| per feature, sorted descending |
| `results/shap/step7_local_<n>.png` | Per-training-sample waterfall (only when `step7_shap_analysis.py` runs standalone) |
| `results/shap/step9_local_candidate_<pid>.png` | Per-candidate SHAP waterfall (top-10 candidates by score; `<pid>` is the safe-filename pathway ID) |
| `results/shap/step9_batch_summary.csv` | Long-format top-5 SHAP features per candidate (columns: `candidate_id, rank, feature_name, shap_value`) |

### Candidate scoring + final report (`results/`)

| File | Description |
|------|-------------|
| `results/step8_candidate_scores.csv` | XGBoost scores for all scored candidates; includes DEG coverage columns in DEG mode |
| `results/step9_final_candidate_pathways.csv` | Final report: Jaccard annotation, top-3 GO enrichment terms, top-3 SHAP features per candidate |
| `results/step9_go_enrichment_per_candidate.csv` | Long-format **full** GO enrichment table per candidate (no top-N truncation) — paper supplementary table |

Mock-mode outputs follow the same naming with mock prefixes/subdirs:
- Data: `data/mock_step4_feature_matrix.npz`, `data/mock_step5_{train,test}.npz`
- Results: `results/mock/step6_*`, `results/mock/step8_*`, `results/mock/step9_*`, `results/mock/shap/step{7,9}_*`

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
| `GO_MIN_GENES` | 20 | Lower bound (inclusive) on the number of genes a GO term must annotate to be retained |
| `GO_MAX_GENE_FRACTION` | 0.30 | Upper bound on the fraction of annotated genes a GO term may cover |
| `XGB_PARAMS` | see file | XGBoost hyperparameters |

---

## GO term frequency filter

After parsing TAIR GO annotations, step 3 applies a dual-frequency filter: GO terms annotated to fewer than `GO_MIN_GENES` (default 20) genes are dropped as statistically underpowered, and terms covering more than `GO_MAX_GENE_FRACTION` (default 0.30) of annotated genes are dropped as non-discriminative. Genes whose annotations are entirely removed by this filter are excluded from the filtered GO background. Each pathway gene set is therefore represented by `len(filtered_GO_terms) + 6` features: the GO-frequency vector, four pairwise GO Jaccard statistics, and two gene-set size / annotation-density features.

The filtered cache (`data/step3_gene_go.tsv`) records the filter parameters and source SHA256 in a JSON header line. Source files (`tair.gaf.gz` or `ATH_GO_GOSLIM.txt`) always take precedence over the cache, so changing the filter parameters and re-running step 3 always re-applies the filter rather than silently reusing a stale cache.

A sensitivity sweep over `min_genes ∈ {5, 10, 15, 20}` × `max_fraction ∈ {0.20, 0.30, 0.50}` is available via:

```bash
python kegg_ml_pipeline/analysis/go_filter_sensitivity.py
```

This writes `results/supp_go_filter_sweep/sweep.csv` (12 rows, one per grid cell) and a `readme.txt`. The script is independent of `step6_train_xgboost.train_with_cv` and does not overwrite official step-6 outputs.

---

## FAQ

**The CV AUROC is below the 0.85 threshold — what should I do?**
Increase `n_estimators` (e.g. 500–1000), lower `learning_rate` (e.g. 0.01–0.03), or adjust `max_depth` in `XGB_PARAMS` in `config.py`. For automated search, consider Optuna.

**I get a `UnicodeDecodeError` when reading data files.**
All file readers in the pipeline use `errors="ignore"` for robustness against non-UTF-8 bytes. If you see this error from a custom file, ensure it is UTF-8 or Latin-1 encoded.

**The KEGG API returns 403 / rate-limit errors.**
Step 2a uses automatic exponential-backoff retry (up to 3 attempts). If KEGG is unavailable, use `--mock` to verify the rest of the pipeline.

**`run_pipeline.py` step 7 only produces the global beeswarm; per-candidate waterfalls are produced by step 9.**
This is intentional: step 7 runs before candidates have been scored, so per-candidate waterfalls would not yet make sense. Step 9 generates `results/shap/step9_local_candidate_<pid>.png` for the top-10 candidates *after* scoring. Standalone `python step7_shap_analysis.py` additionally produces `step7_local_0.png` / `step7_local_1.png` / `step7_local_2.png` for the first three training samples — useful for debugging but not for the paper.

**Which files do I need to copy to my paper's supplementary materials?**
For a complete supplementary bundle (all named with their producing-step prefix):
- Pathway tables: `data/step2_kegg_pathways.tsv`, `data/step2_aracyc_pathways.tsv`, `data/step2_all_pathways.tsv`
- Gene–GO map: `data/step3_gene_go.tsv`
- Feature matrix: `data/step4_feature_matrix.npz`
- Train/test splits: `data/step5_train.npz`, `data/step5_test.npz`
- Model: `results/step6_pathway_model.json`
- Metrics: `results/step9_metrics_summary.csv`, `results/step6_cv_scores.png`, `results/step6_roc_curve.png`
- SHAP: `results/shap/step7_global_summary.png`, `step7_global_importance.csv`, `step9_batch_summary.csv`, `step9_local_candidate_*.png`
- Pathway candidates: `results/step8_candidate_scores.csv`, `results/step9_final_candidate_pathways.csv`, `results/step9_go_enrichment_per_candidate.csv`

**The `step9_metrics_summary.csv` was not updated after my run.**
`step9_metrics_summary.csv` is only written when a single invocation executed both step 6 (training) and step 9 (final report). Partial runs such as `--step 9` or `--from-step 7` leave the existing file untouched to avoid mixing stale CV metrics with a fresh report. Re-run with `--from-step 6` (or full `python run_pipeline.py`) to refresh it.

**I have old un-prefixed files in `data/` and `results/` from a previous run.**
The pipeline used to write files without a `stepN_` prefix (e.g. `data/all_pathways.tsv`, `results/pathway_model.json`). After upgrading, those legacy files are no longer touched. You can safely delete them — the new pipeline will only read and write the new prefixed names.
