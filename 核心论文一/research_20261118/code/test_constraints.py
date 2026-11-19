import unittest
import numpy as np
from constraint_study import WeightedProjector

class WeightedProjectionTests(unittest.TestCase):
    def test_two_node_closed_form(self):
        projector=WeightedProjector(["A","B"],[("A","B")],{"A":9.,"B":1.})
        result,cert=projector.project({"A":10.,"B":0.})
        self.assertAlmostEqual(result["A"],9.,places=7)
        self.assertAlmostEqual(result["B"],9.,places=7)
        self.assertLess(max(cert.values()),1e-8)
    def test_complete_nodes_and_incomparable_value_retained(self):
        projector=WeightedProjector(["A","B","C"],[("A","B")],{"A":1.,"B":1.,"C":1.})
        result,_=projector.project({"A":10.,"B":0.,"C":42.})
        self.assertEqual(set(result),{"A","B","C"})
        self.assertAlmostEqual(result["C"],42.)
    def test_incomplete_input_rejected(self):
        p=WeightedProjector(["A","B"],[("A","B")],{"A":1.,"B":1.})
        with self.assertRaises(ValueError):p.project({"A":1.})
    def test_cyclic_graph_rejected(self):
        with self.assertRaises(ValueError):
            WeightedProjector(["A","B"],[("A","B"),("B","A")],{"A":1.,"B":1.})
    def test_nonpositive_weights_rejected(self):
        with self.assertRaises(ValueError):WeightedProjector(["A"],[],{"A":0.})
if __name__=="__main__":unittest.main()

