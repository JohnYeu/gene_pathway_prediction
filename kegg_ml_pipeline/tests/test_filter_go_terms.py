"""Unit tests for step3_go_annotation.filter_go_terms.

Run from kegg_ml_pipeline/ root:
    python -m unittest discover tests
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from step3_go_annotation import filter_go_terms  # noqa: E402


class FilterGoTermsTests(unittest.TestCase):

    def test_drops_terms_below_min(self):
        gene_go = {
            "g1": {"GO:rare", "GO:common"},
            "g2": {"GO:common"},
            "g3": {"GO:common"},
        }
        out, stats = filter_go_terms(gene_go, min_genes=2, max_fraction=1.0)
        all_kept = {t for ts in out.values() for t in ts}
        self.assertNotIn("GO:rare", all_kept)
        self.assertIn("GO:common", all_kept)
        self.assertEqual(stats["n_terms_dropped_low"], 1)

    def test_drops_terms_above_max_fraction(self):
        # N=4, max_fraction=0.5 → upper = floor(0.5*4) = 2
        # GO:everywhere appears in 4 genes (4 > 2) → drop
        # GO:kept       appears in 2 genes (2 == upper) → keep
        gene_go = {
            "g0": {"GO:everywhere"},
            "g1": {"GO:everywhere", "GO:kept"},
            "g2": {"GO:everywhere", "GO:kept"},
            "g3": {"GO:everywhere"},
        }
        out, stats = filter_go_terms(gene_go, min_genes=1, max_fraction=0.5)
        all_kept = {t for ts in out.values() for t in ts}
        self.assertNotIn("GO:everywhere", all_kept)
        self.assertIn("GO:kept", all_kept)
        self.assertEqual(stats["n_terms_dropped_high"], 1)

    def test_drops_genes_left_empty(self):
        gene_go = {
            "g1": {"GO:rare"},
            "g2": {"GO:common"},
            "g3": {"GO:common"},
        }
        out, _ = filter_go_terms(gene_go, min_genes=2, max_fraction=1.0)
        self.assertNotIn("g1", out)
        self.assertEqual(set(out.keys()), {"g2", "g3"})

    def test_min_count_boundary_inclusive(self):
        gene_go = {f"g{i}": {"GO:edge"} for i in range(2)}
        out, _ = filter_go_terms(gene_go, min_genes=2, max_fraction=1.0)
        self.assertIn("GO:edge", {t for ts in out.values() for t in ts})

    def test_max_count_boundary_inclusive(self):
        # N=4, max_fraction=0.5 → upper = 2; GO:half count = 2 → keep
        gene_go = {f"g{i}": {"GO:half"} for i in range(2)}
        gene_go["g2"] = {"GO:other"}
        gene_go["g3"] = {"GO:other"}
        out, _ = filter_go_terms(gene_go, min_genes=1, max_fraction=0.5)
        self.assertIn("GO:half", {t for ts in out.values() for t in ts})


if __name__ == "__main__":
    unittest.main()
