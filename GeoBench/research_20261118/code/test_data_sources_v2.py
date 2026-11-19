"""Contract tests for geological contact extraction, before real-data processing."""
import unittest
import data_sources_v2 as ds

class ContactExtractionTests(unittest.TestCase):
    def setUp(self):
        self.meta = {"B1": {"x": 1., "y": 2., "surface": 10., "hole_depth": 30.}}
    def row(self, top, bottom, unit, row):
        return {"NR": "B1", "TOP": str(top), "BASIS": str(bottom), "STRAT": unit, "_row": row}
    def test_adjacent_same_unit_is_not_a_contact(self):
        rows, _ = ds.rvg_contacts([self.row(0, 5, "A", 2), self.row(5, 10, "A", 3), self.row(10, 30, "B", 4)], self.meta)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["depth"], 10.)
        self.assertTrue(rows[0]["eligible"])
        self.assertEqual(rows[0]["source_rows"], "2|3")
        self.assertEqual(rows[1]["reason"], "terminal")
    def test_disconnected_repeated_unit_is_ambiguous(self):
        rows, _ = ds.rvg_contacts([self.row(0, 5, "A", 2), self.row(5, 10, "B", 3), self.row(10, 20, "A", 4), self.row(20, 30, "C", 5)], self.meta)
        self.assertTrue(all(not r["eligible"] for r in rows if r["unit"] == "A"))
        self.assertTrue(next(r for r in rows if r["unit"] == "B")["eligible"])
    def test_gap_does_not_create_a_verified_basal_contact(self):
        rows, _ = ds.rvg_contacts([self.row(0, 5, "A", 2), self.row(7, 30, "B", 3)], self.meta)
        self.assertIn("gap_after", rows[0]["reason"])
        self.assertFalse(rows[0]["eligible"])
    def test_end_elevation_converts_to_depth(self):
        meta = ds.rvg_metadata([{"nr":"B1","x":"1","y":"2","mv":"-2.8","end":"-22.1"}])
        self.assertAlmostEqual(meta["B1"]["hole_depth"], 19.3)
    def test_overlap_and_beyond_terminus_are_not_truth(self):
        rows, _ = ds.rvg_contacts([self.row(0, 10, "A", 2), self.row(8, 31, "B", 3)], self.meta)
        self.assertFalse(any(r["eligible"] for r in rows))
        self.assertIn("overlap_after", rows[0]["reason"])
        self.assertIn("beyond_hole_depth", rows[1]["reason"])
    def test_cross_source_elevation_conflict_is_retained_and_excluded(self):
        row = self.row(0, 5, "A", 2)
        row.update({"X":"1", "Y":"2", "MV":"12"})
        rows, audit = ds.rvg_contacts([row, self.row(5, 30, "B", 3)], self.meta)
        self.assertFalse(any(r["eligible"] for r in rows))
        self.assertIn("metadata_conflict", rows[0]["reason"])
        self.assertEqual(audit["metadata_conflicts"]["B1"]["MV"]["stratigraphy"], 12.)
    def test_unknown_stratigraphy_is_not_a_geological_target(self):
        rows, _ = ds.rvg_contacts([self.row(0, 5, "NN", 2), self.row(5, 10, "A", 3), self.row(10, 30, "B", 4)], self.meta)
        self.assertFalse(rows[0]["eligible"])
        self.assertIn("unknown_unit", rows[0]["reason"])
        self.assertTrue(rows[1]["eligible"])
    def test_boundary_into_unknown_is_not_exact(self):
        rows, _ = ds.rvg_contacts([self.row(0, 5, "A", 2), self.row(5, 10, "NN", 3), self.row(10, 20, "B", 4), self.row(20, 30, "C", 5)], self.meta)
        self.assertFalse(rows[0]["eligible"])
        self.assertIn("unknown_next_unit", rows[0]["reason"])
        self.assertTrue(rows[2]["eligible"])
    def test_parent_member_transitions_collapse_before_scoring(self):
        raw = [self.row(0, 5, "BX", 2), self.row(5, 10, "BXLM", 3), self.row(10, 30, "BE", 4)]
        rows, _ = ds.rvg_contacts(raw, self.meta, unit_map={"BXLM":"BX"})
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["depth"], 10.)
        self.assertEqual(rows[0]["original_units"], "BX|BXLM")
    def test_inconsistent_metadata_fails(self):
        with self.assertRaises(ValueError):
            ds.rvg_metadata([{"nr":"B1","x":"1","y":"2","mv":"1","end":"-20"},
                             {"nr":"B1","x":"9","y":"2","mv":"1","end":"-20"}])

if __name__ == "__main__":
    unittest.main()

