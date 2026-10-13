import csv,gzip,tempfile,unittest
from pathlib import Path
import numpy as np
from engine.embodied.feco_identity import resolve_feco

class IdentityTests(unittest.TestCase):
    def fixture(self,path,mutate=None):
        rows=[]
        for i,s in enumerate(('claw','hook','club')):
            rows.append({'Root ID':str(100+i),'Class':'chordotonal_organ_neuron',
                'Sub Class':'front_leg_'+s+'_chordotonal_organ_neuron',
                'Primary Cell Type':'shared_type','Nerve':'left_prothoracic_leg_nerve','Soma side':'left'})
        if mutate:mutate(rows)
        with gzip.open(path,'wt',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        return {'data':{'root_ids':np.array([100,101]),'ann_primary_type':np.array(['shared_type']*2),
                        'ann_nerve':np.array(['left_prothoracic_leg_nerve']*2)},'local_of_global':np.array([0,-1])}
    def test_exact_and_missing(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'x.gz';b=self.fixture(p);g=resolve_feco(p,b)
            self.assertEqual(g['claw'][0]['local_index'],0)
            self.assertEqual(g['hook'][0]['missing_reason'],'absent_tier')
            self.assertEqual(g['club'][0]['missing_reason'],'absent_cache')
            self.assertIsNone(g['claw'][0]['direction'])
    def test_duplicate_refused(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'x.gz';b=self.fixture(p,lambda r:r.append(dict(r[0])))
            with self.assertRaises(ValueError):resolve_feco(p,b)
    def test_primary_disagreement_refused(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'x.gz';b=self.fixture(p);b['data']['ann_primary_type'][0]='bad'
            with self.assertRaises(ValueError):resolve_feco(p,b)
    def test_primary_not_subtype(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'x.gz';b=self.fixture(p);g=resolve_feco(p,b)
            self.assertEqual(set(g),{'claw','hook','club'})
            self.assertTrue(all(not x['current_authorized'] for group in g.values() for x in group))

if __name__=='__main__':unittest.main()
