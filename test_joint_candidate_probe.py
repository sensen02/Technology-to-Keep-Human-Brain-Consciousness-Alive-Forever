"""Inert S127 verifier tests: synthetic arrays, no saved physics execution."""
import copy
import unittest
import numpy as np
from verify_com_contact_probe import Checks
from verify_joint_candidate_probe import later_or_absent, admission_table, validate_weights, require_integrity


def formal_fixture():
    metrics=dict(extrema={k:{side:value} for k,side,value in [('limit_excess','max',.1),('angle_rad','max',.2),('nonfoot_depth','max',0.),('height_ratio','min',1.)]},first_failures_ms=dict(height=None,angle=None,legs=None,load=1.9,nonfoot=None,limit=2.6),initial_load_ratio=1.,load_deficit_integral0_1p7_ratio_ms=0.,all_gates_support_ms=1.8,longest_all_gates_true_endpoint_span_ms=1.7,max_downward_root_velocity=0.,max_downward_COM_velocity=.5)
    return dict(benefit_exact_bounds=copy.deepcopy(metrics),safety_exact_bounds=copy.deepcopy(metrics),rules='exact conjunction'),metrics


class JointCandidateVerifierTests(unittest.TestCase):
    def test_censored_null_semantics(self):
        self.assertTrue(later_or_absent(None,None))
        self.assertTrue(later_or_absent(None,1.9))
        self.assertFalse(later_or_absent(2.,None))
        self.assertFalse(later_or_absent(1.8,1.9))
        self.assertTrue(later_or_absent(1.9,1.9))

    def test_exact_equal_bounds_pass(self):
        formal,metrics=formal_fixture(); rows,d=admission_table(formal,metrics,True)
        self.assertEqual(len(rows),17)
        self.assertTrue(all(r['passed'] for r in rows)); self.assertFalse(d['reject'])

    def test_every_gate_independently_rejects(self):
        formal,base=formal_fixture()
        mutations=[lambda m:m.update(initial_load_ratio=.99),lambda m:m.update(load_deficit_integral0_1p7_ratio_ms=1e-30),lambda m:m.update(all_gates_support_ms=1.7),lambda m:m.update(longest_all_gates_true_endpoint_span_ms=1.6),lambda m:m['first_failures_ms'].update(load=1.8),lambda m:m.update(max_downward_root_velocity=1e-30),lambda m:m.update(max_downward_COM_velocity=.6)]
        for key in ('limit_excess','angle_rad','nonfoot_depth'):
            mutations.append(lambda m,k=key:m['extrema'][k].update(max=np.nextafter(m['extrema'][k]['max'],np.inf)))
        mutations.append(lambda m:m['extrema']['height_ratio'].update(min=np.nextafter(1.,0.)))
        for gate in ('height','angle','legs','nonfoot','limit'):
            mutations.append(lambda m,k=gate:m['first_failures_ms'].update({k:0.}))
        for mutate in mutations:
            metrics=copy.deepcopy(base); mutate(metrics)
            rows,d=admission_table(formal,metrics,True)
            self.assertFalse(all(r['passed'] for r in rows)); self.assertTrue(d['reject'])
        rows,d=admission_table(formal,base,False)
        self.assertFalse(rows[-1]['passed']); self.assertTrue(d['reject'])

    def test_weight_scope_and_initialization(self):
        w=np.ones((66,5)); candidate=w.copy(); candidate[3,0]+=.001
        old=dict(weights=w,actuator_pool=np.arange(90)); new=dict(weights=candidate.copy(),actuator_pool=np.arange(90)); init=dict(weights=candidate.copy())
        checks=Checks(); validate_weights(checks,old,new,init,candidate); self.assertTrue(checks.passed)
        for index in ((36,0),(2,1)):
            bad=candidate.copy(); bad[index]+=.001; checks=Checks(); validate_weights(checks,old,new,init,bad); self.assertFalse(checks.passed)
        init['weights']=w; checks=Checks(); validate_weights(checks,old,new,init,candidate); self.assertFalse(checks.passed)

    def test_closure_failure_stops_admission(self):
        checks=Checks(); checks.add('J.T force',False)
        with self.assertRaisesRegex(ValueError,'admission STOPPED'): require_integrity(checks,'closure')


if __name__=='__main__': unittest.main()
