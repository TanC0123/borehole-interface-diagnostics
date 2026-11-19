import unittest
import numpy as np
import study_core_v2 as sc

class StudyContracts(unittest.TestCase):
    def data(self):
        return [{"borehole":str(i),"unit":"A","x":float(i),"y":float(i%3),
                 "depth":float(2*i+3),"target_id":str(i)} for i in range(15)]
    def test_query_truth_is_never_consumed(self):
        train=self.data()
        q=[{"unit":"A","x":7.5,"y":0.5,"depth":9000,"hole_depth":8000}]
        other=[{**q[0],"depth":-999,"hole_depth":-999}]
        for method in sc.METHODS:
            np.testing.assert_array_equal(sc.predict(train,q,method)[0], sc.predict(train,other,method)[0])
    def test_unsupported_unit_is_explicit(self):
        pred,scale,distance=sc.predict(self.data(),[{"unit":"Z","x":0,"y":0}],"idw2")
        self.assertTrue(np.isnan(pred[0]))
        self.assertTrue(np.isnan(scale[0]))
    def test_idw_exact_coordinate_averages_donors(self):
        rows=[{"unit":"A","x":0.,"y":0.,"depth":2.},{"unit":"A","x":0.,"y":0.,"depth":4.}]
        pred,_,dist=sc.predict(rows,[{"unit":"A","x":0.,"y":0.}],"idw2")
        self.assertEqual(pred[0],3.)
        self.assertEqual(dist[0],0.)
    def test_colocated_holes_never_cross_partitions(self):
        rows=self.data()+[{**self.data()[0],"borehole":"other","target_id":"other"}]
        for kind in ("random","spatial"):
            split=sc.assign_folds(rows,3,kind)
            self.assertEqual(split["0"],split["other"])
            self.assertEqual(set(split.values()),{0,1,2})
    def test_hole_balancing_is_not_row_balancing(self):
        rows=[{"borehole":"a","depth":0.},{"borehole":"a","depth":0.},{"borehole":"b","depth":0.}]
        self.assertEqual(sc.hole_rmse(rows,np.array([0.,0.,10.])),5.)
    def test_partial_support_is_not_complete_hole_coverage(self):
        base={"truth":1.,"prediction":1.,"lower":0.,"upper":2.,"supported":True}
        rows=[dict(base,borehole="a"),dict(base,borehole="a",supported=False),dict(base,borehole="b")]
        result=sc.summarize_predictions(rows)
        self.assertEqual(result["holes_total"],2)
        self.assertEqual(result["holes_partial_support"],1)
        self.assertEqual(result["holes_complete_support"],1)
        self.assertEqual(result["complete_hole_all_covered"],1.)
    def test_infinite_interval_is_not_counted_as_finite_coverage(self):
        rows=[{"borehole":"a","truth":5.,"prediction":1.,"lower":0.,"upper":float("inf"),"supported":True}]
        result=sc.summarize_predictions(rows)
        self.assertEqual(result["finite_interval_fraction"],0.)
        self.assertIsNone(result["finite_interval_coverage"])
    def test_radius_uses_hole_maxima_and_finite_sample_rank(self):
        scores=[{"borehole":str(i),"score":float(i)} for i in range(1,10)]
        scores.append({"borehole":"1","score":0.})
        self.assertEqual(sc.radius(scores),9.)
        self.assertTrue(np.isinf(sc.radius(scores[:5])))

if __name__=="__main__":
    unittest.main()

