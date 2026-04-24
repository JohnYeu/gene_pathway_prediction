from __future__ import annotations

import os
import re

import config
from utils.io_utils import load_gene_go_tsv, save_gene_go_tsv
from utils.mock_data import generate_mock_universe


def load_go_annotation(go_file: str, mock: bool = False) -> dict[str, set[str]]:
    """Load the gene -> GO term mapping from mock data, cache, or source file.

    Priority order:
    1. mock=True  -> return synthetic data from generate_mock_universe(), no I/O
    2. GO_CACHE exists -> load data/gene_go.tsv and return
    3. go_file missing -> warn and fall back to mock
    4. go_file exists  -> parse ATH_GO_GOSLIM.txt, write cache, return

    Args:
        go_file: Path to the TAIR GO annotation source file (ATH_GO_GOSLIM.txt).
        mock: When True, skip all file I/O and return synthetic data.

    Returns:
        Mapping of gene ID to the set of GO terms annotated for that gene.
        Example: {"AT1G01010": {"GO:0006355", "GO:0003700"}}
    """
    if mock:
        universe = generate_mock_universe()
        gene_go = {gene: set(terms) for gene, terms in universe["gene_go"].items()}
        n_terms = len({t for terms in gene_go.values() for t in terms})
        print(f"[GO] Using mock data ({len(gene_go)} genes, {n_terms} GO terms)")
        return gene_go

    if os.path.exists(config.GO_CACHE):
        gene_go = load_gene_go_tsv(config.GO_CACHE)
        n_terms = len({t for terms in gene_go.values() for t in terms})
        print(f"[GO] Loaded cache: {len(gene_go)} genes, {n_terms} unique GO terms")
        return gene_go

    if not os.path.exists(go_file):
        print(f"[WARNING] {go_file} not found, using mock data")
        return load_go_annotation(go_file, mock=True)

    # Parse ATH_GO_GOSLIM.txt
    # Column layout (tab-separated, no header):
    #   col[0] = locus (gene ID)   col[5] = go_id
    gene_go: dict[str, set[str]] = {}
    total_records = 0
    go_pattern = re.compile(r"^GO:\d{7}$")

    with open(go_file, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("!") or line.startswith("#"):
                continue
            columns = line.rstrip("\n").split("\t")
            if len(columns) <= 5:
                continue
            gene = columns[0].strip()
            go_id = columns[5].strip()
            if not gene or not go_pattern.match(go_id):
                continue
            gene_go.setdefault(gene, set()).add(go_id)
            total_records += 1

    save_gene_go_tsv(config.GO_CACHE, gene_go)

    n_terms = len({t for terms in gene_go.values() for t in terms})
    print(
        f"[GO] Parsed {len(gene_go)} genes, {n_terms} unique GO terms, "
        f"{total_records} records"
    )
    return gene_go


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
