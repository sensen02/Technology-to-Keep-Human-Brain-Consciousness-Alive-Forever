import unittest
from tools_s4_descriptive_baseline import position_bin,summarize


class DescriptiveTests(unittest.TestCase):
    def test_bins(self):
        for a,b in [(0,0),(9.9,0),(10,1),(200,19),(-1,None),(201,None)]:self.assertEqual(position_bin(a),b)

    def test_equal_animals_not_equal_samples(self):
        rows=[dict(analyze=1,animal_id=1.,calcium=0.,L1C_flex=5.,L1C_flex_vel=-1.,roi='population') for _ in range(10)]
        rows.append(dict(rows[0],animal_id=2.,calcium=10.))
        r=summarize(rows,[3,4]);cell=next(c for c in r['cells'] if c['kind']=='position')
        self.assertEqual(cell['mean'],5);self.assertEqual(cell['sample_count'],11);self.assertEqual(cell['animal_count'],2)

    def test_direction_split_and_no_single_animal_sd(self):
        row=dict(analyze=1,animal_id=3.,calcium=2.,L1C_flex=201.,L1C_flex_vel=0.,roi='population')
        r=summarize([row],[3,4]);self.assertEqual(r['out_of_angle_range_samples'],1)
        self.assertEqual(r['cells'][0]['split'],'holdout');self.assertEqual(r['cells'][0]['bin'],'zero');self.assertIsNone(r['cells'][0]['animal_sd'])

    def test_nonfinite_reject(self):
        with self.assertRaises(ValueError):summarize([dict(analyze=1,animal_id=1.,calcium=float('nan'),L1C_flex=5.,L1C_flex_vel=0.,roi='x')],[])


if __name__=='__main__':unittest.main()
