from __future__ import annotations

import json
import os
from typing import Any


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(value)
    return value


def save_json(path: str, data: dict) -> None:
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(data), handle, ensure_ascii=False, indent=2)

    print(f"Saved → {path}")


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def pathways_to_sets(pathways: dict) -> dict:
    for pathway in pathways.values():
        pathway["genes"] = set(pathway.get("genes", []))
    return pathways


def gene_go_to_sets(gene_go: dict) -> dict:
    return {gene: set(go_terms) for gene, go_terms in gene_go.items()}
