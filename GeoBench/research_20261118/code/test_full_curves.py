import unittest
import full_curves as fc

class CurveTests(unittest.TestCase):
    def test_ecdf_balances_holes_and_keeps_tail(self):
        result=fc.hole_ecdf([("a",1.),("a",1.),("b",10.)])
        self.assertEqual(result,[(1.,.5),(10.,1.)])
    def test_marginal_quantile_uses_equal_hole_mass(self):
        rows=[{"borehole":"a","score":1.}]*9+[{"borehole":"b","score":10.}]
        self.assertEqual(fc.residual_quantile(rows,.9,"marginal"),10.)
    def test_simultaneous_too_few_holes_is_unbounded(self):
        rows=[{"borehole":"a","score":1.},{"borehole":"b","score":2.}]
        self.assertEqual(fc.residual_quantile(rows,.9,"simultaneous"),float("inf"))
    def test_simultaneous_quantile_uses_hole_maximum(self):
        rows=[{"borehole":str(i),"score":float(i)} for i in range(1,10)]
        rows.append({"borehole":"1","score":0.})
        self.assertEqual(fc.residual_quantile(rows,.9,"simultaneous"),9.)
if __name__=="__main__":unittest.main()

