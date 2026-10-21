import copy
import unittest
import numpy as np
from engine.embodied.leg_gain_search import LEGS, EXPECTED_COUNTS, LegGainSearch, pool_leg_mapping, expand_leg_gains


def manifest():
    rows=[]
    for leg,n in zip(LEGS,EXPECTED_COUNTS):
        for _ in range(n):
            i=len(rows);rows.append(dict(actuator=str(i),actuator_index=i,local_indices=[i],leg=leg))
    rows.append(dict(actuator='shared',actuator_index=66,local_indices=[0],leg='LF'))
    rows.append(dict(actuator='empty',actuator_index=67,local_indices=[],leg='unknown'))
    return dict(n_neurons=66,n_actuators=68,rows=rows)


def metric(t=4453,f=.9374045964130864):
    return dict(prefix_good_ticks=t,foot_mean_while_good=f)


class Tests(unittest.TestCase):
    def test_mapping(self):
        m=pool_leg_mapping(manifest());self.assertEqual(tuple(np.bincount(m)),EXPECTED_COUNTS)
        g=np.arange(6)/4+.5
        self.assertTrue(np.array_equal(expand_leg_gains(g,m),g[m]))
    def test_crossleg_and_unknown(self):
        for index in (66,0):
            m=manifest();m['rows'][index]['leg']='RH' if index==66 else 'unknown'
            with self.assertRaises(ValueError):pool_leg_mapping(m)
    def test_duplicate_actuator_index(self):
        m=manifest();m['rows'][1]['actuator_index']=0
        with self.assertRaises(ValueError):pool_leg_mapping(m)
    def test_pair_ties_and_budget(self):
        s=LegGainSearch(metric(),{},pool_leg_mapping(manifest()))
        for i in range(6):
            p,n=s.probes();self.assertEqual(p[i],1.25);self.assertEqual(n[i],.75)
            center=s.center.copy()
            self.assertTrue(np.array_equal(p[np.arange(6)!=i],center[np.arange(6)!=i]))
            s.remember(p,f'{i}plus',metric(),{});s.remember(n,f'{i}minus',metric(),{})
            self.assertFalse(s.update()['changed'])
        self.assertEqual(s.label,'original_cached_baseline')
        with self.assertRaises(ValueError):s.probes()
    def test_strict_plus_priority_and_transaction(self):
        s=LegGainSearch(metric(),{},pool_leg_mapping(manifest()));p,n=s.probes()
        s.remember(p,'plus',metric(4454),{})
        with self.assertRaises(ValueError):s.update()
        self.assertEqual(s.coordinate,0)
        with self.assertRaises(ValueError):s.remember(p,'duplicate',metric(),{})
        s.remember(n,'minus',metric(4454),{});s.update()
        self.assertEqual(s.label,'plus');self.assertEqual(s.center[0],1.25)
        self.assertTrue(np.array_equal(s.weights_for(s.center),expand_leg_gains(s.center,s.mapping)[:,None]*np.array([.4,.25,.2,.1,.05])))

if __name__=='__main__':unittest.main()
