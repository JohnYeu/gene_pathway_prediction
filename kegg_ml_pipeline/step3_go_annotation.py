from __future__ import annotations

import collections
import gzip
import hashlib
import math
import os
import re

import config
from utils.io_utils import (
    load_gene_go_tsv_with_meta,
    save_gene_go_tsv,
)
from utils.mock_data import generate_mock_universe


_GO_PATTERN = re.compile(r"^GO:\d{7}$")


def derive_ath_from_gaf(
    gaf_path: str = config.GO_GAF,
    ath_path: str = config.GO_ANNOTATION,
) -> bool:
    """Derive a 9-col ATH_GO_GOSLIM.txt-compatible file from a GAF 2.2 archive.

    Trigger condition: `ath_path` is missing. mtime is intentionally NOT used
    because unzip / git checkout / cp all reset mtime to a value unrelated to
    the database release date. When both files exist, the existing ATH file
    is preserved and a hint is printed.

    The output uses the same simplified 9-column layout as the project's
    current ATH_GO_GOSLIM.txt (NOT the official 15-column TAIR/Zenodo schema):
    col 0=gene, col 2=symbol, col 3=relation, col 4=GO, col 5=GO (repeated),
    col 8=evidence; col 1/6/7 left empty. step 3's parser only reads col 0
    and col 5.

    Returns True if a regeneration happened, False otherwise.
    """
    if os.path.exists(ath_path):
        print(
            f"[GO] Both {gaf_path} and {ath_path} exist; using existing ATH "
            "(GAF→ATH derivation only triggers when ATH is missing)."
        )
        return False
    if not os.path.exists(gaf_path):
        return False

    print(f"[GO] Deriving {ath_path} from {gaf_path} ...")
    parent = os.path.dirname(ath_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = ath_path + ".tmp"

    n_in = 0
    n_out = 0
    with gzip.open(gaf_path, "rt", encoding="utf-8", errors="ignore") as src, \
         open(tmp_path, "w", encoding="utf-8", newline="") as dst:
        for line in src:
            if line.startswith("!") or not line.strip():
                continue
            n_in += 1
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 7:
                continue
            # Python 0-based; explicit unpacking avoids index confusion.
            db, gene_id, symbol, qualifier, go_id, reference, evidence = cols[:7]
            if not gene_id.startswith("AT"):
                continue
            # 9-col layout matching the project's current ATH_GO_GOSLIM.txt.
            out_row = [
                gene_id,    # 0  gene (parser reads this)
                "",         # 1
                symbol,     # 2
                qualifier,  # 3  relation
                go_id,      # 4  GO id (display)
                go_id,      # 5  GO id (parser reads this)
                "",         # 6  empty (no aspect)
                "",         # 7
                evidence,   # 8
            ]
            dst.write("\t".join(out_row) + "\n")
            n_out += 1

    os.replace(tmp_path, ath_path)
    print(f"[GO] Derived {n_out} records from {n_in} GAF rows → {ath_path}")
    return True


def parse_go_annotation_source(go_file: str) -> dict[str, set[str]]:
    """Parse ATH_GO_GOSLIM.txt → raw gene→GO mapping.

    Pure parser: does NOT read cache, does NOT apply filter, does NOT save.
    Used by `load_go_annotation()` and by the sensitivity sweep script.

    Args:
        go_file: Path to the 9-column ATH_GO_GOSLIM.txt-compatible file.

    Returns:
        `{gene_id: set[go_term]}` with all parsed annotations.
    """
    gene_go: dict[str, set[str]] = {}
    total_records = 0

    with open(go_file, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("!") or line.startswith("#"):
                continue
            columns = line.rstrip("\n").split("\t")
            if len(columns) <= 5:
                continue
            gene = columns[0].strip()
            go_id = columns[5].strip()
            if not gene or not _GO_PATTERN.match(go_id):
                continue
            gene_go.setdefault(gene, set()).add(go_id)
            total_records += 1

    n_terms = len({t for terms in gene_go.values() for t in terms})
    print(
        f"[GO] Parsed {len(gene_go)} genes, {n_terms} unique GO terms, "
        f"{total_records} records (raw, unfiltered)"
    )
    return gene_go


def filter_go_terms(
    gene_go: dict[str, set[str]],
    min_genes: int = config.GO_MIN_GENES,
    max_fraction: float = config.GO_MAX_GENE_FRACTION,
) -> tuple[dict[str, set[str]], dict[str, int | float]]:
    """Drop low-frequency and ubiquitous GO terms; drop genes left empty.

    A term is kept iff `min_genes <= count <= floor(max_fraction * n_genes)`,
    where `n_genes` is the number of annotated genes before filtering.
    Genes whose entire GO set is removed by the filter are dropped from
    the returned mapping (keeps step 9 hypergeometric math consistent —
    `M = len(filtered_gene_go)` then equals "genes with at least one
    surviving annotation").

    Args:
        gene_go: Raw mapping returned by `parse_go_annotation_source`.
        min_genes: Lower count threshold (inclusive).
        max_fraction: Upper count fraction; threshold is `floor(max_fraction * N)`.

    Returns:
        `(filtered_gene_go, stats)` where stats has keys:
        n_terms_before, n_terms_after,
        n_terms_dropped_low, n_terms_dropped_high,
        n_genes_before, n_genes_after,
        max_count_threshold (= floor(max_fraction * n_genes_before)).
    """
    n_annotated = len(gene_go)
    upper = math.floor(max_fraction * n_annotated)

    term_counts: collections.Counter[str] = collections.Counter()
    for terms in gene_go.values():
        term_counts.update(terms)

    n_terms_before = len(term_counts)
    kept_terms: set[str] = set()
    n_dropped_low = 0
    n_dropped_high = 0
    for term, count in term_counts.items():
        if count < min_genes:
            n_dropped_low += 1
        elif count > upper:
            n_dropped_high += 1
        else:
            kept_terms.add(term)

    filtered: dict[str, set[str]] = {}
    for gene, terms in gene_go.items():
        kept = terms & kept_terms
        if kept:
            filtered[gene] = kept

    stats: dict[str, int | float] = {
        "n_terms_before":       n_terms_before,
        "n_terms_after":        len(kept_terms),
        "n_terms_dropped_low":  n_dropped_low,
        "n_terms_dropped_high": n_dropped_high,
        "n_genes_before":       n_annotated,
        "n_genes_after":        len(filtered),
        "max_count_threshold":  upper,
    }
    print(
        f"[GO] Filter (min_genes={min_genes}, max_fraction={max_fraction}): "
        f"{n_terms_before} → {len(kept_terms)} terms "
        f"(dropped {n_dropped_low} rare, {n_dropped_high} ubiquitous); "
        f"{n_annotated} → {len(filtered)} genes"
    )
    return filtered, stats


def _sha256_file(path: str) -> str:
    """Compute the SHA256 hex digest of a file (full content)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def current_go_source_sha256() -> str:
    """Return the SHA256 of the GO snapshot currently in use, or ``""``.

    Mirrors :func:`load_go_annotation`'s real-mode resolution order so the
    returned value matches whichever source the pipeline is actually reading:

    1. ``config.GO_ANNOTATION`` exists → SHA256 of that file.
    2. ``config.GO_ANNOTATION`` missing but ``config.GO_CACHE`` exists →
       ``source_sha256`` from the cache JSON meta header (recorded by
       :func:`load_go_annotation` when the cache was last written).
    3. Neither exists → ``""``.

    The cache fallback matters for users who keep only the derived
    ``step3_gene_go.tsv`` on disk after deleting the raw ATH/GAF; without it,
    ``run_pipeline._load_state`` would crash trying to hash a missing file.
    """
    if os.path.exists(config.GO_ANNOTATION):
        return _sha256_file(config.GO_ANNOTATION)
    if os.path.exists(config.GO_CACHE):
        _, meta = load_gene_go_tsv_with_meta(config.GO_CACHE)
        if meta:
            return str(meta.get("source_sha256", "") or "")
    return ""


def load_go_annotation(go_file: str, mock: bool = False) -> dict[str, set[str]]:
    """Load the gene -> GO term mapping for the pipeline.

    Resolution order (real mode):

    1. mock=True: return synthetic data from generate_mock_universe(), no I/O.
    2. real:
       a. If GO_ANNOTATION is missing but GO_GAF exists, derive the ATH file
          from the GAF archive (one-shot, see `derive_ath_from_gaf`).
       b. Parse the ATH file into a raw mapping.
       c. Apply `filter_go_terms` with the current config thresholds.
       d. Save GO_CACHE with a JSON metadata header recording the filter
          parameters, stats, and source SHA256.
       e. Return the filtered mapping.
    3. Fallback: if both ATH and GAF are absent, allow loading from cache
       only when its meta matches the current filter parameters; otherwise
       raise SystemExit.

    Source (GAF or ATH) always takes precedence over cache, so changing
    `GO_MIN_GENES` / `GO_MAX_GENE_FRACTION` always re-runs the filter.
    """
    if mock:
        universe = generate_mock_universe()
        gene_go = {gene: set(terms) for gene, terms in universe["gene_go"].items()}
        n_terms = len({t for terms in gene_go.values() for t in terms})
        print(f"[GO] Using mock data ({len(gene_go)} genes, {n_terms} GO terms)")
        return gene_go

    if not os.path.exists(go_file) and os.path.exists(config.GO_GAF):
        derive_ath_from_gaf(config.GO_GAF, go_file)

    if os.path.exists(go_file):
        raw = parse_go_annotation_source(go_file)
        filtered, stats = filter_go_terms(raw)

        meta = {
            "go_filter": {
                "min_genes":    config.GO_MIN_GENES,
                "max_fraction": config.GO_MAX_GENE_FRACTION,
            },
            "stats":         stats,
            "source_sha256": _sha256_file(go_file),
        }
        save_gene_go_tsv(config.GO_CACHE, filtered, meta=meta)
        return filtered

    if os.path.exists(config.GO_CACHE):
        gene_go, meta = load_gene_go_tsv_with_meta(config.GO_CACHE)
        cache_filter = (meta or {}).get("go_filter") or {}
        current_filter = {
            "min_genes":    config.GO_MIN_GENES,
            "max_fraction": config.GO_MAX_GENE_FRACTION,
        }
        if cache_filter != current_filter:
            raise SystemExit(
                f"[GO] Cannot fall back to {config.GO_CACHE!r}: cache filter "
                f"params {cache_filter!r} != current config {current_filter!r}, "
                "and source files are missing. Restore data/tair.gaf.gz or "
                "data/ATH_GO_GOSLIM.txt and re-run."
            )
        n_terms = len({t for terms in gene_go.values() for t in terms})
        print(
            f"[GO] Loaded cache ({len(gene_go)} genes, {n_terms} terms) — "
            "source files missing; cache filter params match config."
        )
        return gene_go

    raise SystemExit(
        f"[GO] No GO annotation data available in real mode. Tried:\n"
        f"  - {go_file} (missing)\n"
        f"  - {config.GO_GAF} (missing)\n"
        f"  - {config.GO_CACHE} (missing or filter params don't match config)\n"
        f"Place data/tair.gaf.gz or data/ATH_GO_GOSLIM.txt under "
        f"kegg_ml_pipeline/data/, or pass --mock explicitly to use "
        f"synthetic data. Falling back to mock without --mock would "
        f"silently mix mock GO annotations with real pathway data."
    )


def get_all_go_terms(gene_go: dict[str, set[str]]) -> list[str]:
    """Return a stable-sorted list of every unique GO term in the mapping.

    The sort order is fixed so the feature vector dimension is identical
    across all pipeline runs and between training and inference. Step 4, 7,
    and 8 all depend on this ordering being consistent.

    Args:
        gene_go: Mapping returned by load_go_annotation().

    Returns:
        Sorted list of GO term identifiers, e.g. ["GO:0000001", "GO:0000002", ...]
    """
    all_terms = sorted({go for terms in gene_go.values() for go in terms})
    print(f"[GO] Total unique GO terms: {len(all_terms)}")
    return all_terms


if __name__ == "__main__":
    gene_go = load_go_annotation(config.GO_ANNOTATION, mock=False)
    all_go_terms = get_all_go_terms(gene_go)
    print(f"Feature dimension from GO: {len(all_go_terms)}")
