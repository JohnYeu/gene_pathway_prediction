from __future__ import annotations

import os

import config
from step2a_kegg import get_ath_pathways
from utils.io_utils import save_pathways_tsv
from utils.mock_data import generate_mock_universe


def load_aracyc_mock() -> dict[str, dict]:
    """Return the AraCyc slice of the shared mock universe."""
    universe = generate_mock_universe()
    aracyc_pathways = {
        pathway_id: {
            "name": pathway["name"],
            "genes": set(pathway["genes"]),
            "source": pathway["source"],
        }
        for pathway_id, pathway in universe["pathways"].items()
        if pathway["source"] == "AraCyc"
    }
    print(f"[AraCyc] Using mock data ({len(aracyc_pathways)} pathways)")
    return aracyc_pathways


def parse_aracyc_pathways(dat_file: str, mock: bool = False) -> dict[str, dict]:
    """Parse AraCyc pathway membership from the repository's TSV export.

    Return format:
    {
        "aracyc:PWY-XXXX": {
            "name": str,
            "genes": set[str],
            "source": "AraCyc",
        }
    }
    """
    if mock:
        return load_aracyc_mock()

    if not os.path.exists(dat_file):
        print(f"[WARNING] {dat_file} not found, using mock data")
        return load_aracyc_mock()

    # The actual file in this repository is TSV, not PGDB flat-file format.
    # We parse the three columns we need and ignore the reaction/protein fields.
    pathways: dict[str, dict] = {}
    with open(dat_file, "r", encoding="utf-8", errors="ignore") as handle:
        next(handle, None)  # Skip the header row.

        for line in handle:
            if not line.strip():
                continue

            columns = line.rstrip("\n").split("\t")
            if len(columns) < 7:
                continue

            pathway_raw_id = columns[0].strip()
            pathway_name = columns[1].strip()
            gene_id = columns[6].strip().upper()

            # Keep only Arabidopsis locus-style identifiers and drop sentinel
            # values such as NIL that do not represent real genes.
            if not gene_id.startswith("AT") or gene_id == "NIL":
                continue

            pathway_id = f"aracyc:{pathway_raw_id}"
            pathways.setdefault(
                pathway_id,
                {
                    "name": pathway_name,
                    "genes": set(),
                    "source": "AraCyc",
                },
            )
            pathways[pathway_id]["genes"].add(gene_id)

    # Cache the parsed result in the same TSV structure as the KEGG cache.
    save_pathways_tsv(config.ARACYC_CACHE, pathways)

    pathways_with_three_or_more_genes = sum(
        1 for pathway in pathways.values() if len(pathway["genes"]) >= 3
    )
    print(
        f"[AraCyc] Parsed {len(pathways)} pathways "
        f"({pathways_with_three_or_more_genes} with ≥3 genes)"
    )
    return pathways


def merge_and_save_all_pathways(
    kegg: dict,
    aracyc: dict,
    save: bool = True,
) -> dict:
    """Merge KEGG and AraCyc pathways and optionally export them as TSV.

    The TSV includes separate `kegg` and `aracyc` identifier columns so each
    row makes the source-specific ID explicit:
    - KEGG rows populate `kegg` and leave `aracyc` empty
    - AraCyc rows populate `aracyc` and leave `kegg` empty

    Additional columns keep the file useful for downstream analysis rather than
    reducing it to IDs only.
    """
    merged = {**kegg, **aracyc}

    if save:
        # The merged export shares the exact same TSV schema as the individual
        # caches so every pathway table in the project looks consistent.
        save_pathways_tsv(config.ALL_PATHWAYS, merged)

    print(f"[Merge] KEGG={len(kegg)}, AraCyc={len(aracyc)}, Total={len(merged)}")
    return merged


if __name__ == "__main__":
    kegg = get_ath_pathways(mock=False)
    aracyc = parse_aracyc_pathways(config.ARACYC_RAW, mock=False)
    all_pw = merge_and_save_all_pathways(kegg, aracyc)
    print(f"[Done] Combined pathway library size: {len(all_pw)}")
