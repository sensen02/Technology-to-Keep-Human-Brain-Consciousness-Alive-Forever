import unittest
import json
import numpy as np
import run_pose_command_cross as x

class Tests(unittest.TestCase):
    def test_contrast_arithmetic(self):
        d=x.contrasts([[1,3],[5,9]])
        self.assertEqual(d['interaction'],2)
        self.assertEqual(d['pose_effect_at_command_A'],4)
        self.assertEqual(d['command_effect_at_pose_B'],4)
    def test_factorial_identity_and_health(self):
        d=json.loads(x.OUT.read_text()); self.assertEqual(len(d['trajectories']),4)
        rows=x.dyn.select_rows(json.loads(x.dyn.SOURCE.read_text()))
        for t in d['trajectories']:
            self.assertEqual(t['initial_qpos'],rows[t['pose_index']]['qpos'])
            self.assertEqual(t['initial_activation'],rows[t['command_index']]['activation'])
            self.assertTrue(t['observation_identity']); self.assertTrue(t['finite'])
            self.assertEqual(t['steps'],500); self.assertEqual(t['warning_count'],0)
            self.assertTrue(t['command_unchanged']); self.assertEqual(t['activation_initialization_writes'],1)
            if t['pose_index']==t['command_index']:self.assertTrue(t['prior_diagonal_identity'])
            for r in t['samples']:
                self.assertTrue(all(c['within_tolerance'] for c in r['checks'].values()))
        self.assertEqual(d['provenance']['xml_after'],x.cp.sha256_file(x.cp.CANONICAL_XML))
        for metric, e in d['effects'].items():
            recomputed=x.contrasts(e['matrix'])
            for key,value in recomputed.items():self.assertAlmostEqual(value,e[key])

if __name__=='__main__':unittest.main(verbosity=2)
