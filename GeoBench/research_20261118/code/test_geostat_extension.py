import unittest
import numpy as np
import geostat_extension as gs
class GeostatTests(unittest.TestCase):
    def test_constant_sparse_fallback(self):
        p=gs.kriging(np.array([[0.,0.],[1.,1.]]),np.array([3.,3.]),np.array([[2.,2.]]))
        self.assertEqual(p[0],3.)
    def test_exact_training_location(self):
        x=np.array([[0.,0.],[100.,0.],[0.,100.],[100.,100.],[50.,30.],[70.,90.]])
        y=np.array([1.,3.,5.,8.,6.,7.])
        p=gs.kriging(x,y,x[:1])
        self.assertAlmostEqual(p[0],1.,places=6)
    def test_weighted_quantile_balances_holes(self):
        rows=[{"borehole":"a","score":1.}]*10+[{"borehole":"b","score":9.}]
        self.assertEqual(gs.marginal_radius(rows),9.)
if __name__=="__main__":unittest.main()

