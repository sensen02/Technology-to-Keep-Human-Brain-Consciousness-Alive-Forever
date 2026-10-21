import unittest
import copy
from tools_s4_observation_holdout import fit_train,evaluate


def row(a,y,angle=5):return dict(analyze=1,animal_id=float(a),calcium=float(y),L1C_flex=float(angle),roi='population')


class HoldoutTests(unittest.TestCase):
    def test_held_values_cannot_fit_model(self):
        rows=[row(1,0),row(2,10),row(3,2)];m=fit_train(rows,[3]);rows[-1]['calcium']=10000
        self.assertEqual(m,fit_train(rows,[3]))

    def test_animal_equal_fit(self):
        m=fit_train([row(1,0)]*10+[row(2,10)],[3]);self.assertEqual(m['population']['constant'],5);self.assertEqual(m['population']['lookup']['0'],5)

    def test_same_coverage_and_known_errors(self):
        rows=[row(1,0),row(2,10),row(3,7),row(3,100,50)];m=fit_train(rows,[3]);before=copy.deepcopy(m)
        r=evaluate(rows,[3],m)[0];self.assertEqual(r['predicted'],1);self.assertEqual(r['uncovered'],1)
        self.assertEqual(r['lookup_mae'],2);self.assertEqual(r['constant_rmse'],2);self.assertEqual(m,before)

    def test_empty_training_rejected(self):
        with self.assertRaises(ValueError):fit_train([row(3,2)],[3])

    def test_invalid_rejected(self):
        for y in (float('nan'),True):
            r=row(1,0);r['calcium']=y
            with self.assertRaises(ValueError):fit_train([r],[])


if __name__=='__main__':unittest.main()
