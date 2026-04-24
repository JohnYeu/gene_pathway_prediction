from __future__ import annotations

import os
import re
import time

import requests

import config
from utils.io_utils import (
    load_json,
    load_pathways_tsv,
    pathways_to_sets,
    save_pathways_tsv,
)
from utils.mock_data import generate_mock_universe


def _fetch_with_retry(url: str, max_retries: int = 3) -> str:
    """Fetch text data from a URL with exponential-backoff retries.

    Retry timing follows the requested 1s -> 2s -> 4s schedule. The function
    performs one initial attempt plus up to `max_retries` retry attempts.
    """
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            last_error = exc
            if attempt == max_retries:
                break

            retry_number = attempt + 1
            wait_seconds = 2**attempt
            print(f"[Retry {retry_number}/{max_retries}] {url}")
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Failed to fetch {url} after {max_retries} retries"
    ) from last_error


def get_ath_pathways(mock: bool = False) -> dict[str, dict]:
    """Load Arabidopsis KEGG pathways from mock data, cache, or KEGG REST.

    Return format:
    {
        pathway_id: {
            "name": str,
            "genes": set[str],
            "source": "KEGG",
        }
    }
    """
    if mock:
        # Mock mode is used by tests and later dry-run pipeline stages.
        # It should be side-effect free, so we do not read or write caches here.
        universe = generate_mock_universe()
        kegg_pathways = {
            pathway_id: {
                "name": pathway["name"],
                "genes": set(pathway["genes"]),
                "source": pathway["source"],
            }
            for pathway_id, pathway in universe["pathways"].items()
            if pathway["source"] == "KEGG"
        }
        print(f"[KEGG] Using mock data ({len(kegg_pathways)} pathways)")
        return kegg_pathways

    if os.path.exists(config.KEGG_CACHE):
        # The preferred cache format is TSV. Loading it keeps the pipeline
        # entirely in human-readable tables rather than JSON blobs.
        cached_pathways = load_pathways_tsv(config.KEGG_CACHE)
        print(f"[KEGG] Loaded {len(cached_pathways)} pathways from cache")
        return cached_pathways

    legacy_json_cache = os.path.splitext(config.KEGG_CACHE)[0] + ".json"
    if os.path.exists(legacy_json_cache):
        # Migrate older JSON caches in place so existing local data does not
        # force a network fetch the first time the TSV-based code runs.
        print(f"[KEGG] Migrating legacy cache {legacy_json_cache} -> {config.KEGG_CACHE}")
        cached_pathways = pathways_to_sets(load_json(legacy_json_cache))
        save_pathways_tsv(config.KEGG_CACHE, cached_pathways)
        return cached_pathways

    # Fall back to live KEGG fetching only when no local cache exists.
    pathway_list_url = "https://rest.kegg.jp/list/pathway/ath"
    pathway_link_url = "https://rest.kegg.jp/link/ath/pathway"

    pathway_text = _fetch_with_retry(pathway_list_url)
    pathways: dict[str, dict] = {}
    for line in pathway_text.splitlines():
        if not line.strip():
            continue

        pathway_id, pathway_name = line.split("\t", maxsplit=1)
        clean_name = re.sub(r" - Arabidopsis thaliana.*", "", pathway_name)
        pathways[pathway_id] = {
            "name": clean_name,
            "genes": set(),
            "source": "KEGG",
        }

    link_text = _fetch_with_retry(pathway_link_url)
    for line in link_text.splitlines():
        if not line.strip():
            continue

        parts = line.split("\t")
        if len(parts) != 2:
            continue

        pathway_id = parts[0].replace("path:", "")
        gene_id = parts[1].replace("ath:", "").upper()

        # If KEGG returns a pathway in the link table that was missing from the
        # list table, keep the association rather than dropping data silently.
        pathways.setdefault(
            pathway_id,
            {"name": pathway_id, "genes": set(), "source": "KEGG"},
        )
        pathways[pathway_id]["genes"].add(gene_id)

    # Save the fetched data immediately in TSV so later runs stay offline and
    # the cache remains easy to inspect with spreadsheet or shell tools.
    save_pathways_tsv(config.KEGG_CACHE, pathways)
    return pathways


if __name__ == "__main__":
    loaded_pathways = get_ath_pathways(mock=False)
    for pathway_id, pathway in list(loaded_pathways.items())[:3]:
        print(
            f"{pathway_id}\t{pathway['name']}\tgenes={len(pathway['genes'])}"
        )
