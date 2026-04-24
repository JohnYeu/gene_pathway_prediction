from __future__ import annotations

import csv
import json
import os
from typing import Any


def _to_jsonable(value: Any) -> Any:
    """Recursively convert Python containers into JSON-serializable values.

    The main special case is `set`, which JSON cannot encode natively.
    We sort sets before writing so cache files are deterministic across runs.
    """
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(value)
    return value


def save_json(path: str, data: dict) -> None:
    """Write a dictionary to disk as formatted JSON.

    This helper is used by multiple pipeline steps, so it also ensures the
    parent directory exists before writing the file.
    """
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(data), handle, ensure_ascii=False, indent=2)

    print(f"Saved → {path}")


def load_json(path: str) -> dict:
    """Load a JSON file and return the raw decoded dictionary."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def pathways_to_sets(pathways: dict) -> dict:
    """Restore each pathway's gene list back into a set in place.

    Pathway caches are written as JSON, so gene sets become lists on disk.
    Downstream code expects fast membership tests and de-duplication, so we
    convert the `genes` field back to `set` after loading.
    """
    for pathway in pathways.values():
        pathway["genes"] = set(pathway.get("genes", []))
    return pathways


def gene_go_to_sets(gene_go: dict) -> dict:
    """Return a new gene -> GO mapping with GO lists converted to sets."""
    return {gene: set(go_terms) for gene, go_terms in gene_go.items()}


def save_pathways_tsv(path: str, pathways: dict) -> None:
    """Write pathway dictionaries to a human-readable TSV table.

    The TSV uses separate `kegg` and `aracyc` ID columns so the source-specific
    identifier is always explicit in the file itself. This keeps the KEGG cache,
    AraCyc cache, and merged pathway table in one consistent format.
    """
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["kegg", "aracyc", "name", "source", "gene_count", "genes"])

        for pathway_id, pathway in sorted(pathways.items()):
            is_kegg = pathway.get("source") == "KEGG"
            genes = sorted(pathway.get("genes", []))
            writer.writerow(
                [
                    pathway_id if is_kegg else "",
                    "" if is_kegg else pathway_id,
                    pathway.get("name", ""),
                    pathway.get("source", ""),
                    len(genes),
                    ",".join(genes),
                ]
            )

    print(f"Saved → {path}")


def load_pathways_tsv(path: str) -> dict[str, dict]:
    """Load a pathway TSV created by `save_pathways_tsv`.

    Returned format matches the in-memory pathway structure used throughout the
    pipeline: `{pathway_id: {name, genes: set[str], source}}`.
    """
    pathways: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            pathway_id = (row.get("kegg") or row.get("aracyc") or "").strip()
            if not pathway_id:
                continue

            source = (row.get("source") or "").strip()
            if not source:
                source = "KEGG" if row.get("kegg") else "AraCyc"

            genes_field = (row.get("genes") or "").strip()
            genes = {gene for gene in genes_field.split(",") if gene}
            pathways[pathway_id] = {
                "name": (row.get("name") or "").strip(),
                "genes": genes,
                "source": source,
            }

    return pathways


def save_gene_go_tsv(path: str, gene_go: dict[str, set[str]]) -> None:
    """Write the gene -> GO mapping to a TSV file.

    Each row represents one gene. The `go_terms` column holds a sorted,
    comma-separated list of GO identifiers so the file is human-readable
    and diffs cleanly in version control.

    Format: gene | go_count | go_terms
    """
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["gene", "go_count", "go_terms"])

        for gene in sorted(gene_go):
            sorted_terms = sorted(gene_go[gene])
            writer.writerow([gene, len(sorted_terms), ",".join(sorted_terms)])

    print(f"Saved → {path}")


def load_gene_go_tsv(path: str) -> dict[str, set[str]]:
    """Load a gene -> GO mapping from a TSV created by `save_gene_go_tsv`.

    Returns `{gene_id: set[go_term]}` with the same structure as the
    in-memory representation used throughout the pipeline.
    """
    gene_go: dict[str, set[str]] = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            gene = (row.get("gene") or "").strip()
            if not gene:
                continue
            go_field = (row.get("go_terms") or "").strip()
            gene_go[gene] = {term for term in go_field.split(",") if term}

    return gene_go
