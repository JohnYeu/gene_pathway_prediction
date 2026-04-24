from __future__ import annotations

import os
import random
import sys
from itertools import combinations

import numpy as np
from tqdm import tqdm

import config
from step3_go_annotation import get_all_go_terms, load_go_annotation
from utils.io_utils import load_pathways_tsv
from utils.mock_data import generate_mock_universe


def _normalize_gene_list(gene_list: list[str]) -> list[str]:
    """Deduplicate and sort a gene list so all feature functions see a stable input.

    Deduplication happens here and nowhere else. This ensures training and
    inference treat repeated gene IDs identically regardless of how the caller
    constructed the list.
    """
    return sorted(set(gene_list))


def go_frequency_vector(
    gene_list: list[str],
    gene_go: dict[str, set[str]],
    all_go_terms: list[str],
) -> np.ndarray:
    """Compute a normalised GO-term frequency vector for a set of genes.

    Each element represents the fraction of genes in `gene_list` that are
    annotated with the corresponding GO term. The output dimension is fixed
    by `all_go_terms`, which must be the same sorted list used during
    training and inference.

    Args:
        gene_list: Gene identifiers whose GO annotations will be aggregated.
        gene_go: Mapping from gene ID to the set of its GO terms.
        all_go_terms: Stable-sorted list of all GO terms in the universe.

    Returns:
        Float32 array of shape ``(len(all_go_terms),)``.
    """
    genes = _normalize_gene_list(gene_list)
    n = len(genes)
    vec = np.zeros(len(all_go_terms), dtype=np.float32)
    if n == 0:
        return vec

    go_to_idx = {go: idx for idx, go in enumerate(all_go_terms)}
    for gene in genes:
        for go_term in gene_go.get(gene, set()):
            idx = go_to_idx.get(go_term)
            if idx is not None:
                vec[idx] += 1.0

    vec /= n
    return vec


def jaccard_similarity_stats(
    gene_list: list[str],
    gene_go: dict[str, set[str]],
) -> np.ndarray:
    """Compute pairwise Jaccard similarity statistics for a set of genes.

    The Jaccard similarity between two genes is defined over their GO-term
    sets: ``|A ∩ B| / |A ∪ B|``. When both sets are empty the similarity is
    defined as 0.0.

    For large gene sets the full O(n²) computation is too expensive, so when
    the deduplicated gene count exceeds 50 we sample 50 unique pairs using a
    seeded random instance that does not affect global state.

    Args:
        gene_list: Gene identifiers to compare pairwise.
        gene_go: Mapping from gene ID to the set of its GO terms.

    Returns:
        Float32 array ``[mean, min, max, std]`` of shape ``(4,)``.
    """
    genes = _normalize_gene_list(gene_list)
    n = len(genes)
    if n < 2:
        return np.zeros(4, dtype=np.float32)

    if n <= 50:
        pairs = list(combinations(range(n), 2))
    else:
        rng = random.Random(config.RANDOM_STATE)
        seen: set[tuple[int, int]] = set()
        pairs = []
        while len(pairs) < 50:
            i, j = sorted(rng.sample(range(n), 2))
            if (i, j) not in seen:
                seen.add((i, j))
                pairs.append((i, j))

    scores: list[float] = []
    for i, j in pairs:
        set_a = gene_go.get(genes[i], set())
        set_b = gene_go.get(genes[j], set())
        union_size = len(set_a | set_b)
        jaccard = len(set_a & set_b) / union_size if union_size > 0 else 0.0
        scores.append(jaccard)

    arr = np.array(scores, dtype=np.float32)
    return np.array([arr.mean(), arr.min(), arr.max(), arr.std()], dtype=np.float32)


def size_features(
    gene_list: list[str],
    gene_go: dict[str, set[str]],
) -> np.ndarray:
    """Compute size and annotation-density features for a gene set.

    The two dimensions capture how large the gene set is and how densely it
    is annotated. Both use the deduplicated gene count so duplicates in the
    input do not inflate the values.

    Args:
        gene_list: Gene identifiers in the pathway or candidate set.
        gene_go: Mapping from gene ID to the set of its GO terms.

    Returns:
        Float32 array ``[log(size+1), mean_go_terms_per_annotated_gene]``
        of shape ``(2,)``.
    """
    genes = _normalize_gene_list(gene_list)
    log_size = np.log(len(genes) + 1)

    annotated_counts = [len(gene_go[g]) for g in genes if g in gene_go and gene_go[g]]
    mean_go = float(np.mean(annotated_counts)) if annotated_counts else 0.0

    return np.array([log_size, mean_go], dtype=np.float32)


def extract_features(
    gene_list: list[str],
    gene_go: dict[str, set[str]],
    all_go_terms: list[str],
) -> np.ndarray:
    """Extract the full fixed-length feature vector for one gene set.

    The vector is the concatenation of three layers:
    - GO frequency vector  (len(all_go_terms) dimensions)
    - Jaccard stats        (4 dimensions)
    - Size features        (2 dimensions)

    Total length is always ``len(all_go_terms) + 6``.

    Args:
        gene_list: Gene identifiers whose features will be extracted.
        gene_go: Mapping from gene ID to the set of its GO terms.
        all_go_terms: Stable-sorted list of all GO terms in the universe.

    Returns:
        Float32 array of shape ``(len(all_go_terms) + 6,)``.
    """
    vec = np.concatenate([
        go_frequency_vector(gene_list, gene_go, all_go_terms),
        jaccard_similarity_stats(gene_list, gene_go),
        size_features(gene_list, gene_go),
    ])
    return vec.astype(np.float32, copy=False)


def build_feature_matrix(
    pathway_gene_lists: list[list[str]],
    gene_go: dict[str, set[str]],
    all_go_terms: list[str],
) -> np.ndarray:
    """Build a feature matrix by extracting features for each gene list.

    This function does not filter pathways. Callers are responsible for
    removing pathways with too few genes before passing the lists here.

    Args:
        pathway_gene_lists: One list of gene IDs per pathway or candidate.
        gene_go: Mapping from gene ID to the set of its GO terms.
        all_go_terms: Stable-sorted list of all GO terms in the universe.

    Returns:
        Float32 array of shape ``(len(pathway_gene_lists), len(all_go_terms)+6)``.
        Returns a zero-row matrix of the correct width when the input is empty.
    """
    n_features = len(all_go_terms) + 6
    if not pathway_gene_lists:
        return np.zeros((0, n_features), dtype=np.float32)

    rows = [
        extract_features(gene_list, gene_go, all_go_terms)
        for gene_list in tqdm(pathway_gene_lists, desc="Extracting features")
    ]
    return np.vstack(rows).astype(np.float32)


def save_feature_matrix(
    X: np.ndarray,
    all_go_terms: list[str],
    pathway_ids: list[str],
    path: str = config.FEATURE_MATRIX,
) -> None:
    """Save the feature matrix and associated metadata to a NumPy archive.

    The archive contains three arrays:
    - ``X``: the feature matrix itself
    - ``all_go_terms``: GO term names aligned to the first (d-6) feature columns
    - ``pathway_ids``: pathway identifiers aligned to the row axis

    Step 7 uses ``all_go_terms`` directly for SHAP feature naming.

    Args:
        X: Feature matrix of shape ``(n, d)``.
        all_go_terms: Sorted list of GO term identifiers.
        pathway_ids: List of pathway identifiers, one per row of ``X``.
        path: Output file path (default: ``config.FEATURE_MATRIX``).
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    np.savez(
        path,
        X=X,
        all_go_terms=np.array(all_go_terms),
        pathway_ids=np.array(pathway_ids),
    )
    print(f"Saved → {path}  shape={X.shape}")


if __name__ == "__main__":
    mock = "--mock" in sys.argv

    gene_go = load_go_annotation(config.GO_ANNOTATION, mock=mock)
    all_go_terms = get_all_go_terms(gene_go)

    if mock:
        all_pathways = generate_mock_universe()["pathways"]
    else:
        all_pathways = load_pathways_tsv(config.ALL_PATHWAYS)

    valid = {pid: p for pid, p in all_pathways.items() if len(p["genes"]) >= 3}
    pathway_ids = list(valid.keys())
    gene_lists = [list(p["genes"]) for p in valid.values()]

    X = build_feature_matrix(gene_lists, gene_go, all_go_terms)
    save_feature_matrix(X, all_go_terms, pathway_ids)
    print(f"Feature dim: {X.shape[1]}  (GO:{len(all_go_terms)} + jaccard:4 + size:2)")
