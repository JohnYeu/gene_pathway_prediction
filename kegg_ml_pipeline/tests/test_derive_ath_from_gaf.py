"""Unit tests for step3_go_annotation.derive_ath_from_gaf.

GAF parsing is the most error-prone part of step 3 (the 1-based
documentation vs 0-based Python indexing trap). These tests use a
synthetic gzipped GAF to verify column mapping, AT filtering, no
deduplication, and no ID-suffix stripping.

Run from kegg_ml_pipeline/ root:
    python -m unittest discover tests
"""
from __future__ import annotations

import gzip
import os
import sys
import tempfile
import unittest
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from step3_go_annotation import derive_ath_from_gaf  # noqa: E402


GAF_HEADER = "!gaf-version: 2.2\n"

GAF_BODY = (
    # db        | gene_id     | symbol | qualifier  | go_id      | ref     | evid | with | aspect
    "AGI_LocusCode\tAT1G01010\tNAC001\tenables\tGO:0003700\tPMID:1\tISS\t\tF\n"
    "AGI_LocusCode\tAT1G01010\tNAC001\tinvolved_in\tGO:0006355\tPMID:2\tIEA\t\tP\n"
    # duplicate (AT1G01010, GO:0003700) — must be retained, not deduped
    "AGI_LocusCode\tAT1G01010\tNAC001\tenables\tGO:0003700\tPMID:3\tIEA\t\tF\n"
    # transcript with .1 — must NOT be stripped
    "AGI_LocusCode\tAT5G46315.1\tncRNA\tinvolved_in\tGO:0000353\tPMID:4\tIEA\t\tP\n"
    # non-AT — must be skipped
    "UniProt\tQ9XYZ1\tFOO\tenables\tGO:9999999\tPMID:5\tIEA\t\tF\n"
)


class DeriveAthFromGafTests(unittest.TestCase):

    def _write_fake_gaf(self, dir_: str) -> str:
        path = os.path.join(dir_, "tair.gaf.gz")
        with gzip.open(path, "wt") as f:
            f.write(GAF_HEADER)
            f.write(GAF_BODY)
        return path

    def test_derives_when_ath_missing(self):
        with tempfile.TemporaryDirectory() as d:
            gaf = self._write_fake_gaf(d)
            ath = os.path.join(d, "ATH_GO_GOSLIM.txt")
            self.assertTrue(derive_ath_from_gaf(gaf, ath))

            with open(ath) as f:
                lines = [line.rstrip("\n").split("\t") for line in f]

            # 4 records retained (5 input - 1 non-AT = 4), duplicates kept.
            self.assertEqual(len(lines), 4)

            row = lines[0]
            self.assertEqual(len(row), 9)
            self.assertEqual(row[0], "AT1G01010")    # gene at col 0
            self.assertEqual(row[1], "")
            self.assertEqual(row[2], "NAC001")       # symbol at col 2
            self.assertEqual(row[3], "enables")      # relation at col 3
            self.assertEqual(row[4], "GO:0003700")
            self.assertEqual(row[5], "GO:0003700")   # GO at col 5
            self.assertEqual(row[6], "")
            self.assertEqual(row[7], "")
            self.assertEqual(row[8], "ISS")          # evidence at col 8

    def test_skips_non_at_genes(self):
        with tempfile.TemporaryDirectory() as d:
            gaf = self._write_fake_gaf(d)
            ath = os.path.join(d, "ATH_GO_GOSLIM.txt")
            derive_ath_from_gaf(gaf, ath)
            with open(ath) as f:
                gene_ids = {line.split("\t")[0] for line in f}
            self.assertNotIn("Q9XYZ1", gene_ids)

    def test_does_not_strip_transcript_suffix(self):
        with tempfile.TemporaryDirectory() as d:
            gaf = self._write_fake_gaf(d)
            ath = os.path.join(d, "ATH_GO_GOSLIM.txt")
            derive_ath_from_gaf(gaf, ath)
            with open(ath) as f:
                gene_ids = {line.split("\t")[0] for line in f}
            self.assertIn("AT5G46315.1", gene_ids)

    def test_does_not_dedupe_gene_go_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            gaf = self._write_fake_gaf(d)
            ath = os.path.join(d, "ATH_GO_GOSLIM.txt")
            derive_ath_from_gaf(gaf, ath)
            with open(ath) as f:
                rows = [line.rstrip("\n").split("\t") for line in f]
            pair_count = sum(
                1 for r in rows
                if r[0] == "AT1G01010" and r[5] == "GO:0003700"
            )
            self.assertEqual(pair_count, 2)

    def test_skips_when_ath_already_present(self):
        with tempfile.TemporaryDirectory() as d:
            gaf = self._write_fake_gaf(d)
            ath = os.path.join(d, "ATH_GO_GOSLIM.txt")
            with open(ath, "w") as f:
                f.write("preexisting\n")
            self.assertFalse(derive_ath_from_gaf(gaf, ath))
            with open(ath) as f:
                self.assertEqual(f.read(), "preexisting\n")


if __name__ == "__main__":
    unittest.main()
