from __future__ import annotations

import os
import random
import sys

import numpy as np
from tqdm import tqdm

import config
from step3_go_annotation import get_all_go_terms, load_go_annotation
from step4_feature_extraction import build_feature_matrix
from utils.io_utils import load_pathways_tsv
from utils.mock_data import generate_mock_universe


def _collect_gene_pool(pathways: dict[str, dict]) -> list[str]:
    """Return a stable-sorted list of all unique genes across the given pathways.

    Only genes from the supplied pathways are included. Callers should pass
    already-filtered valid_pathways so that genes from tiny (<3-gene) pathways
    do not pollute the negative-sample pool.

    Args:
        pathways: Pathway dict in the standard pipeline format
                  ``{pathway_id: {"genes": set[str], ...}}``.

    Returns:
        Sorted list of unique gene identifiers.
    """
    pool: set[str] = set()
    for p in pathways.values():
        pool.update(p.get("genes", set()))
    return sorted(pool)


def build_dataset(
    pathways: dict[str, dict],
    gene_go: dict[str, set[str]],
    all_go_terms: list[str],
    neg_pos_ratio: int = config.NEG_POS_RATIO,
    random_state: int = config.RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a labelled feature matrix for binary pathway classification.

    Positive samples are extracted from real pathway gene sets (size >= 3).
    Negative samples are random gene sets drawn from the same gene pool, each
    with the exact same size as its corresponding positive pathway.

    Args:
        pathways: All pathways in the standard pipeline format.
        gene_go: Mapping from gene ID to the set of its GO terms.
        all_go_terms: Stable-sorted list of all GO terms (fixes feature width).
        neg_pos_ratio: Number of negative samples per positive pathway.
        random_state: Seed for the local RNG used for negative sampling.

    Returns:
        Tuple ``(X, y)`` where ``X`` is a float32 feature matrix of shape
        ``(n_pos + n_neg, len(all_go_terms)+6)`` and ``y`` is an int32 label
        vector of the same length.

    Raises:
        ValueError: If ``neg_pos_ratio < 1`` or any pathway is larger than the
                    gene pool.
    """
    if neg_pos_ratio < 1:
        raise ValueError(f"neg_pos_ratio must be >= 1, got {neg_pos_ratio}")

    valid_pathways = {
        pid: p for pid, p in pathways.items() if len(p.get("genes", set())) >= 3
    }

    d = len(all_go_terms) + 6
    if not valid_pathways:
        return np.zeros((0, d), dtype=np.float32), np.zeros(0, dtype=np.int32)

    # Positive samples
    positive_gene_lists = [sorted(set(p["genes"])) for p in valid_pathways.values()]
    X_pos = build_feature_matrix(positive_gene_lists, gene_go, all_go_terms)
    y_pos = np.ones(len(positive_gene_lists), dtype=np.int32)

    # Negative samples
    gene_pool = _collect_gene_pool(valid_pathways)
    rng = random.Random(random_state)
    negative_gene_lists: list[list[str]] = []
    for pos_genes in positive_gene_lists:
        k = len(pos_genes)
        if k > len(gene_pool):
            raise ValueError(
                f"Pathway size {k} exceeds gene pool size {len(gene_pool)}; "
                "cannot draw a negative sample without replacement"
            )
        for _ in range(neg_pos_ratio):
            negative_gene_lists.append(rng.sample(gene_pool, k))

    print(f"Building {len(negative_gene_lists)} negative samples...", flush=True)
    X_neg = build_feature_matrix(negative_gene_lists, gene_go, all_go_terms)
    y_neg = np.zeros(len(negative_gene_lists), dtype=np.int32)

    X = np.vstack([X_pos, X_neg]).astype(np.float32)
    y = np.concatenate([y_pos, y_neg]).astype(np.int32)
    print(
        f"Dataset: {len(X)} samples  "
        f"pos={int(y.sum())}  neg={int((y == 0).sum())}  "
        f"features={X.shape[1]}"
    )
    return X, y


def split_and_save(
    X: np.ndarray,
    y: np.ndarray,
    all_go_terms: list[str],
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split and save to disk as NumPy archives.

    Each output archive contains three arrays: ``X`` (feature matrix), ``y``
    (labels), and ``all_go_terms`` (GO term names aligned to the first d-6
    feature columns). Step 6 can load either file and immediately verify that
    the feature space matches the one used during training.

    Args:
        X: Feature matrix of shape ``(n, d)``.
        y: Integer label vector of length ``n``.
        all_go_terms: Sorted list of GO term identifiers used to build ``X``.
        test_size: Fraction of samples reserved for the test set.
        random_state: Seed passed to ``train_test_split``.

    Returns:
        Tuple ``(X_train, y_train, X_test, y_test)``.

    Raises:
        ValueError: If ``len(X) != len(y)``, the dataset is empty, or ``y``
                    does not contain exactly the labels ``{0, 1}``.
    """
    if len(X) != len(y):
        raise ValueError(f"X rows ({len(X)}) != y length ({len(y)})")
    if len(X) == 0:
        raise ValueError("Cannot split an empty dataset")
    unique_labels = set(np.unique(y).tolist())
    if unique_labels != {0, 1}:
        raise ValueError(f"y must contain exactly {{0, 1}}, got {unique_labels}")

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    os.makedirs(os.path.dirname(config.TRAIN_DATA), exist_ok=True)
    os.makedirs(os.path.dirname(config.TEST_DATA), exist_ok=True)

    np.savez(
        config.TRAIN_DATA,
        X=X_train,
        y=y_train,
        all_go_terms=np.array(all_go_terms),
    )
    np.savez(
        config.TEST_DATA,
        X=X_test,
        y=y_test,
        all_go_terms=np.array(all_go_terms),
    )

    print(
        f"Train: {len(X_train)} "
        f"(pos={int(y_train.sum())}  neg={int((y_train == 0).sum())})"
    )
    print(
        f"Test : {len(X_test)}  "
        f"(pos={int(y_test.sum())}  neg={int((y_test == 0).sum())})"
    )
    print(f"Saved -> {config.TRAIN_DATA}")
    print(f"Saved -> {config.TEST_DATA}")
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    mock = "--mock" in sys.argv

    gene_go = load_go_annotation(config.GO_ANNOTATION, mock=mock)
    all_go_terms = get_all_go_terms(gene_go)

    if mock:
        all_pathways = generate_mock_universe()["pathways"]
    else:
        all_pathways = load_pathways_tsv(config.ALL_PATHWAYS)

    X, y = build_dataset(all_pathways, gene_go, all_go_terms)
    split_and_save(X, y, all_go_terms)
