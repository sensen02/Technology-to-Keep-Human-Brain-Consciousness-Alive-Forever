"""Lightweight regression tests against saved bounded tradeoff evidence."""
import unittest
import json
import numpy as np
import run_support_tradeoff as t

class Tests(unittest.TestCase):
    def test_scaling(self):
        q=np.arange(48,dtype=float); j=np.zeros((3,48)); j[:,:3]=np.eye(3); j[:,3:]=.01
        # Test dimensionless residual generalization directly with explicit L units.
        qm=q.copy(); qm[:3]*=.001
        np.testing.assert_allclose(t.s.dimensionless(q,9801,1),t.s.dimensionless(qm,9.801,.001))
        for w in (0,1,10,100):
            r=t.residual(q,j,9801,w)
            self.assertAlmostEqual(r@r,np.sum(t.s.dimensionless(q,9801)**2)+w*((j@q)[2]/9801)**2)
    def test_saved_evidence(self):
        d=json.loads(t.OUT.read_text()); self.assertEqual(len(d['rows']),16)
        source=json.loads(t.audit.SOURCE.read_text())
        for r in d['rows']:
            self.assertLessEqual(r['objective'],r['seed_objective']+1e-12)
            src=next(x for x in source['rows'] if x['label']==r['pose'])
            self.assertEqual(r['force_audit']['qpos'],src['qpos'])
            self.assertTrue(all(c['within_tolerance'] for c in r['force_audit']['checks'].values()))
            if r['condition']=='all90':
                other=next(x for x in d['rows'] if x['pose']==r['pose'] and x['weight']==r['weight'] and x['condition']=='covered74')
                self.assertLessEqual(r['objective'],other['objective']+1e-12)
        for item in d['integrations']:
            for name in ('baseline','candidate'):
                a=item[name]; self.assertTrue(a['finite']); self.assertFalse(a['automatic_reset_detected']); self.assertEqual(a['warning_count'],0)
        self.assertEqual(d['provenance']['xml_before'],t.s.cp.CANONICAL_XML_SHA256)
        self.assertEqual(d['provenance']['xml_after'],t.s.cp.sha256_file(t.s.cp.CANONICAL_XML))

if __name__=='__main__': unittest.main(verbosity=2)
