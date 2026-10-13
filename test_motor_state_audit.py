"""Small state/units tests and read-only exact source acceptance; no heavy test runs."""
import json
from pathlib import Path
import unittest
import numpy as np
import run_motor_state_audit as a
from engine.embodied.audit_provenance import inspect_audit


class StateTests(unittest.TestCase):
    def test_drive_units_balance_and_postreset_identity(self):
        net=a.clo.ConductanceNetwork(2,dt_ms=.5)
        net.v[:]=[-55,-60]; net.ge[:]=[.002,.001]; net.gi[:]=[.003,.004]
        net.adaptation[:]=[.004,.006]
        v=net.v.copy(); adapt=net.adaptation.copy()
        s,row,x=a.observed_step(net,np.array([0.,4.]),np.arange(2))
        np.testing.assert_allclose(x['exc_drive_nA'],[.11,.06])
        np.testing.assert_allclose(x['inh_drive_nA'],[-.045,-.04])
        np.testing.assert_allclose(x['leak_drive_nA'],[-.01,-.005])
        np.testing.assert_allclose(x['total_drive_nA'],x['exc_drive_nA']+x['inh_drive_nA']+x['leak_drive_nA']+x['direct_current_nA']-adapt)
        expected=v+x['total_drive_nA']/(.02/.5+.001+x['ge_pre_uS']+x['gi_pre_uS'])
        expected[s]=-65
        np.testing.assert_allclose(x['v_postreset_mV'],expected)
        np.testing.assert_allclose(net.adaptation,adapt*np.exp(-.5/100)+s*.002)
        self.assertTrue(s[1]); self.assertEqual(row['refractory_end_count'],1)
        self.assertEqual(row['time_start_ms'],0); self.assertEqual(row['time_end_ms'],.5)

    def test_arrivals_before_drive_and_refractory_clamp(self):
        net=a.clo.ConductanceNetwork(2,dt_ms=.5)
        net.set_connectivity(np.array([[0,.002],[0,0]]),np.zeros((2,2)))
        net._ring[net._ring_i,1]=True
        net.refractory_until_ms[0]=2
        net.v[0]=-60
        s,row,x=a.observed_step(net,np.zeros(2),np.arange(2))
        self.assertAlmostEqual(x['exc_drive_nA'][0],.12)
        self.assertEqual(row['refractory_start_count'],1)
        self.assertEqual(net.v[0],-65); self.assertFalse(s[0])
        self.assertAlmostEqual(net.ge[0],.002*np.exp(-.5/3))

    def test_current_signed_sums_and_original_overlap_order(self):
        ch=(*a.PROPRIO,'campaniform_strain','bristle_touch')
        groups={'lf':{c:[0] for c in ch}}
        values=[-1,2,-3,4,1]
        record={'channel_record':{'scale_nA_per_unit':.3,'per_leg':{'lf':{c:dict(value=v,n_cells=1) for c,v in zip(ch,values)}}},'external_lf_touch_stimulus':True}
        current,detail=a.reconstruct_current(record,groups,2)
        np.testing.assert_allclose(current,[1.2,0])
        self.assertAlmostEqual(detail['groups']['proprio']['negative_sum_nA'],-.6)
        self.assertEqual(a.current_stats(np.array([-2,0,3])),dict(L1_nA=5.,n_nonzero=2,positive_sum_nA=3.,negative_sum_nA=-2.))


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=json.loads((a.OUT/'motor_state_audit.json').read_text())
        with np.load(cls.d['traces']['path'],allow_pickle=False) as z: cls.z={k:z[k] for k in z.files}

    def test_source_hashes_and_full_counts(self):
        d=self.d
        evidence=inspect_audit(Path(__file__).resolve().parent, 'motor_state_audit')
        bundle=evidence['source_bundle']
        self.assertEqual(evidence['source_snapshot'], 'unavailable')
        self.assertEqual(evidence['run_sources'][str(Path(__file__).resolve().parent/'run_neural_muscle_closure.py')]['status'], 'drift')
        self.assertEqual(bundle['run_id'],a.EXPECTED_RUN)
        self.assertEqual(bundle['manifest']['files'],d['source_bundle']['files'])
        self.assertEqual(a.sha(Path(bundle['path'])/'manifest.json'),d['source_bundle']['manifest_sha256'])
        self.assertEqual(a.sha(d['traces']['path']),d['traces']['sha256'])
        for p,h in d['provenance']['inputs'].items(): self.assertEqual(a.sha(p),h)
        with np.load(Path(bundle['path'])/'closure_traces.npz',allow_pickle=False) as src:
            np.testing.assert_array_equal(self.z['interval_spike_counts'],src['normal_spike_counts'])
        self.assertEqual(self.z['interval_spike_counts'].shape,(10,12000))

    def test_snapshot_identities_and_source_timestamps(self):
        d,z=self.d,self.z
        self.assertEqual(len(d['motor_snapshots']),100)
        self.assertEqual(z['spikes'].shape,(100,202))
        np.testing.assert_array_equal(z['time_end_ms'],np.arange(1,101)*.5)
        np.testing.assert_allclose(z['exc_drive_nA'],z['ge_pre_uS']*(0-z['v_pre_mV']))
        np.testing.assert_allclose(z['inh_drive_nA'],z['gi_pre_uS']*(-70-z['v_pre_mV']))
        np.testing.assert_allclose(z['total_drive_nA'],z['exc_drive_nA']+z['inh_drive_nA']+z['leak_drive_nA']-z['adaptation_pre_nA'])
        np.testing.assert_array_equal(z['v_postreset_mV'][z['spikes']],-65)
        self.assertEqual(np.count_nonzero(z['direct_current_nA']),0)
        motor=z['motor_local_indices']
        np.testing.assert_array_equal(z['spikes'].reshape(10,10,202).sum(axis=1),z['interval_spike_counts'][:,motor])
        result=json.loads((Path(d['source_bundle']['path'])/'closure_results.json').read_text())
        ep=result['measured']['episodes']['normal']
        lookup={i:j for j,i in enumerate(motor)}
        expected=[]
        for k,r in enumerate(ep['records']):
            for step,col in r['probe_raw_spike_events_step_column']:
                i=ep['probe_local_indices'][col]
                if i in lookup: expected.append((k*10+step,lookup[i]))
        self.assertEqual(sorted(expected),list(map(tuple,np.argwhere(z['spikes']))))
        for j,row in enumerate(d['motor_snapshots']):
            self.assertEqual(row['time_end_ms'],(j+1)*.5)
            self.assertEqual(row['actual_spikes'],int(z['spikes'][j].sum()))
            self.assertEqual(row['v_postreset_mV'],a.stats(z['v_postreset_mV'][j]))

    def test_channel_current_and_no_fabricated_causation(self):
        d=self.d
        source=Path(d['source_bundle']['path'])
        groups=json.loads((source/'neural_identity.json').read_text())['sensory_groups']
        records=json.loads((source/'closure_results.json').read_text())['measured']['episodes']['normal']['records']
        for k,r in enumerate(records):
            current,details=a.reconstruct_current(r,groups,12000)
            self.assertEqual(details['groups'],d['channel_current_intervals'][k]['groups'])
            self.assertEqual(np.count_nonzero(current[self.z['motor_local_indices']]),0)
        self.assertEqual(d['observations']['causal_attribution'],'not_identified_by_observational_replay')
        self.assertEqual(d['observations']['motor_spikes_per_5ms'],[0,11,66,154,103,50,7,2,6,6])
        self.assertFalse(d['replay_contract']['body_constructed'])
        self.assertEqual(d['replay_contract']['parameters']['adaptation_increment_nA'],.002)
        self.assertLess(d['wall_s'],60)


if __name__=='__main__': unittest.main()
