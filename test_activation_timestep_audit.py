"""Small scalar regressions plus saved real-model diagnostic artifact validation."""
import json
import unittest
import numpy as np
import mujoco
import run_activation_timestep_audit as audit

class ActivationAuditTests(unittest.TestCase):
    def test_rhs_matches_installed_native_including_out_of_bounds(self):
        prm=np.array([.0001,.0004,0.])
        for ctrl in (-1.,0.,.5,1.,2.):
            for act in (-.2,0.,.25,1.,1.5,2.):
                self.assertAlmostEqual(audit.rhs(act,ctrl,prm),mujoco.mju_muscleDynamics(ctrl,act,prm),places=9)
    def test_local_clamp_is_not_state_clamp(self):
        prm=np.array([.0001,.0004,0.])
        self.assertEqual(mujoco.mju_muscleDynamics(1.,0.,prm),20000.)
        self.assertEqual(mujoco.mju_muscleDynamics(1.,2.,prm),-5000.)
        self.assertEqual(.0001*mujoco.mju_muscleDynamics(1.,0.,prm),2.)
    def test_continuum_bound_and_monotonicity(self):
        sol=audit.reference(np.array([.0001,.0004,0.]))
        vals=sol.sol(np.linspace(0,.005,501))[0]
        self.assertTrue(np.all(vals>=0) and np.all(vals<=1+1e-12))
        self.assertTrue(np.all(np.diff(vals)>=-1e-12))
    def test_saved_native_cases(self):
        result=json.loads((audit.OUT/'activation_timestep_results.json').read_text())
        self.assertEqual(result['label'],'DIRECTCTRL NOT AUTONOMOUS')
        cases=result['cases']; self.assertEqual(len(cases),3)
        self.assertEqual([c['steps'] for c in cases],[50,100,200])
        self.assertEqual(cases[0]['activation_max'],2.)
        self.assertTrue(all(c['finite'] and c['warning_free'] for c in cases))
        self.assertGreater(cases[0]['max_abs_error'],cases[1]['max_abs_error'])
        self.assertGreater(cases[1]['max_abs_error'],cases[2]['max_abs_error'])
        with np.load(audit.OUT/'activation_timestep_traces.npz') as z:
            for i,c in enumerate(cases):
                self.assertEqual(z[f'case{i}_act'].shape,(c['steps']+1,90))
                self.assertAlmostEqual(z[f'case{i}_time_s'][-1],.005,places=14)
                self.assertTrue(np.all(z[f'case{i}_ctrl']==1))
                self.assertTrue(np.all(z[f'case{i}_act'][0]==0))
                self.assertTrue(np.array_equal(z[f'case{i}_qpos'][0],z['case0_qpos'][0]))
                self.assertEqual(float(z[f'case{i}_act'].max()),c['activation_max'])
        for p,h in result['provenance']['source_hashes'].items(): self.assertEqual(audit.sha(p),h)
        for p,h in result['provenance']['source_archive_hashes'].items(): self.assertEqual(audit.sha(p),h)
        self.assertEqual(audit.sha(audit.OUT/'activation_timestep_traces.npz'),result['provenance']['trace_sha256'])

if __name__=='__main__': unittest.main()
