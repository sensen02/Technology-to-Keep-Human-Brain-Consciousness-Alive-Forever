"""Bounded synthetic tests only: no simulator imports, no historical writes."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import final_native_audit as a

class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='final-audit-fixture-')
        self.p=Path(self.temp.name)
        self.old=a.WINDOWS
        a.WINDOWS={'auxiliary':(0.,.001),'autonomous':(.001,.002)}
    def tearDown(self):
        a.WINDOWS=self.old
        self.temp.cleanup()
    def fixture(self, nonfoot=0., external=0., corrupt=False):
        n=160;dt=12.5e-6;mass=.002494271478;weight=mass*9801.
        before=np.arange(n)*dt;after=(np.arange(n)+1)*dt
        xyz=np.tile([0.,0.,1.25],(n,1));vel=np.zeros((n,3))
        # Four carrying legs at .25W; two unloaded. Impulse exposes signed vs abs.
        loads=np.tile([.25,.25,.25,.25,0.,0.],(n,1))*weight
        body=np.zeros((n,2,6));body[:,0,2]=external
        general=np.zeros((n,48));ext=np.tile([0.,0.,external],(n,1))
        data={'time_before_s':before,'time_after_s':after,'com_before':xyz.copy(),'com_after':xyz.copy(),
            'com_velocity_before':vel.copy(),'com_velocity_after':vel.copy(),'loads_native':loads,
            'nonfoot_force_native':np.full(n,nonfoot),'contact_speed_native_per_s':np.zeros((n,6)),
            'ground_force_native':np.tile([0.,0.,weight+nonfoot],(n,1)),
            'external_total_force_native':ext,'external_body_wrench_native':body,
            'external_generalized_force_native':general,'adhesion_body_wrench_native':np.zeros((n,2,6)),
            'adhesion_enabled':np.zeros(n,dtype=bool),'adhesion_application_body_leg_point_force_native':np.empty((0,8)),
            'contact_offsets':np.zeros(n+1,dtype=int),'contact_geom_leg_ground':np.empty((0,4),dtype=int),
            'qpos_after':np.zeros((n,49)),'qvel_after':np.zeros((n,48)),
            'action':np.zeros((n,42)),'target_rad':np.zeros((n,42)),'commanded_torque_native':np.zeros((n,42))}
        native=self.p/'native';native.mkdir()
        entries=[]
        for k,sl in enumerate((slice(0,80),slice(80,160))):
            fields={key:(v if key.startswith('adhesion_application_') or key=='contact_geom_leg_ground'
                          else np.zeros(81,dtype=int) if key=='contact_offsets' else v[sl]) for key,v in data.items()}
            path=native/f'native_{k:06d}.npz';np.savez_compressed(path,**fields)
            entries.append({'path':str(path),'rows':80,'sha256':a.sha(path)})
        trace={'time':after[[79,159]],'qpos':data['qpos_after'][[79,159]],'qvel':data['qvel_after'][[79,159]],
            'com':xyz[[79,159]],'action':data['action'][[79,159]],'target':data['target_rad'][[79,159]],
            'actualctrl':data['commanded_torque_native'][[79,159]]}
        if corrupt:trace['qpos'][1,0]=1.
        tp=self.p/'trace.npz';np.savez_compressed(tp,**trace)
        manifest={'chunks':entries,'pending_rows':0,'native_samples':n}
        return manifest,tp,weight
    def test_d7_and_filter_streaming_and_native_sync(self):
        manifest,tp,w=self.fixture()
        report=a.extra_native(manifest,w,tp)
        self.assertEqual(report['native_trace_endpoints_bitexact'],2)
        self.assertTrue(report['all_actual_extra_forces_zero'])
        for window in report['windows'].values():
            self.assertEqual([r['median'] for r in window['D7_sensitivity']],[4.]*5)
            self.assertEqual(len(window['D6_sensitivity']),5)
            self.assertTrue(all(abs(r['signed_p95_W'])<1e-12 for r in window['D6_sensitivity']))
    def test_extra_force_accounting(self):
        manifest,tp,w=self.fixture(external=2.)
        report=a.extra_native(manifest,w,tp)
        self.assertFalse(report['all_actual_extra_forces_zero'])
        rows=report['windows']['autonomous']['D6_sensitivity']
        self.assertTrue(all(abs(r['external_accounted_signed_p95_W']+2/w)<1e-12 for r in rows))
    def test_dynamic_filter_equivalence_and_sensitivity(self):
        manifest,tp,w=self.fixture()
        for j,entry in enumerate(manifest['chunks']):
            path=Path(entry['path'])
            with np.load(path,allow_pickle=False) as z:fields={k:a.bounded(z,k) for k in z.files}
            # Force impulse crosses chunk/window boundary; velocity remains continuous.
            scale=np.ones(80)
            if j==0:scale[65:]=3.
            fields['loads_native']*=scale[:,None]
            fields['ground_force_native'][:,2]=fields['loads_native'].sum(axis=1)
            np.savez_compressed(path,**fields);entry['sha256']=a.sha(path)
        extra=a.extra_native(manifest,w,tp)
        frozen=a.load_frozen().evaluate_directory(self.p/'native',autonomous_window=(.001,.002),auxiliary_window=(0.,.001),
            mass_native=w/9801.,source_manifest=manifest)
        self.assertEqual(frozen['status'],'OK')
        for key in a.WINDOWS:
            rows=extra['windows'][key]['D6_sensitivity']
            primary=next(r for r in rows if r['tau_ms']==1.)
            self.assertAlmostEqual(primary['signed_p95_W'],frozen['windows'][key]['original_D6_signed_p95_W'],places=12)
            self.assertAlmostEqual(primary['absolute_p95_W'],frozen['windows'][key]['original_D6_absolute_p95_W'],places=12)
        r=extra['windows']['autonomous']['D6_sensitivity']
        self.assertGreater(max(v['absolute_p95_W'] for v in r),.15)
        self.assertGreater(len(set(round(v['absolute_p95_W'],8) for v in r)),2)

    def test_last_interval_force_spike_survives_recorded_endpoint_roundoff(self):
        manifest,tp,w=self.fixture()
        scale=1.+2e-7
        for j,entry in enumerate(manifest['chunks']):
            path=Path(entry['path'])
            with np.load(path,allow_pickle=False) as z:fields={k:a.bounded(z,k) for k in z.files}
            fields['time_before_s']*=scale;fields['time_after_s']*=scale
            if j==1:
                fields['loads_native'][-1]=0.
                fields['nonfoot_force_native'][-1]=2*w
                fields['ground_force_native'][-1,2]=2*w
            np.savez_compressed(path,**fields);entry['sha256']=a.sha(path)
        with np.load(tp,allow_pickle=False) as z:trace={k:a.bounded(z,k) for k in z.files}
        trace['time']*=scale;np.savez_compressed(tp,**trace)
        actual={'auxiliary':(0.,.001*scale),'autonomous':(.001*scale,.002*scale)}
        nominal=a.extra_native(manifest,w,tp)
        fixed=a.extra_native(manifest,w,tp,windows=actual)
        self.assertEqual(nominal['windows']['autonomous']['D6_sensitivity'][0]['samples'],79)
        self.assertEqual(fixed['windows']['autonomous']['D6_sensitivity'][0]['samples'],80)
        self.assertEqual(fixed['windows']['autonomous']['D7_sensitivity'][-1]['histogram'][0],1)
        frozen=a.load_frozen().evaluate_directory(self.p/'native',autonomous_window=actual['autonomous'],
            auxiliary_window=actual['auxiliary'],mass_native=w/9801.,source_manifest=manifest)
        self.assertEqual(frozen['status'],'OK')
        self.assertTrue(frozen['windows']['autonomous']['valid'])
        self.assertEqual(frozen['criteria']['D1']['status'],'FAIL')
        self.assertEqual(frozen['criteria']['D1']['value'],2*w)

    def test_rejects_native_trace_mismatch(self):
        manifest,tp,w=self.fixture(corrupt=True)
        with self.assertRaisesRegex(ValueError,'native/trace mismatch qpos'):a.extra_native(manifest,w,tp)
    def test_frozen_legacy_unknown_and_strict_D1(self):
        manifest,tp,w=self.fixture(nonfoot=1e-12)
        f=a.load_frozen()
        report=f.evaluate_directory(self.p/'native',autonomous_window=(.001,.002),auxiliary_window=(0.,.001),
            mass_native=w/9801.,source_manifest=manifest)
        self.assertEqual(report['status'],'OK')
        self.assertEqual(report['criteria']['D1']['status'],'FAIL')
        self.assertEqual(report['criteria']['D7']['status'],'UNKNOWN')
        self.assertEqual(report['windows']['autonomous']['loaded_leg_count_median'],4.)
    def test_rejects_hash_tamper(self):
        manifest,tp,w=self.fixture();manifest['chunks'][0]['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'chunk changed'):a.extra_native(manifest,w,tp)
    def test_exact_replay_rejects_small_change(self):
        x=self.p/'a.npz';y=self.p/'b.npz'
        np.savez_compressed(x,x=np.array([1.]));np.savez_compressed(y,x=np.array([1.+1e-12]))
        self.assertFalse(a.compare_npz(x,y)['all_exact'])
    def test_header_cap_before_allocation(self):
        path=self.p/'large.npz';np.savez_compressed(path,x=np.zeros(a.CAP+1,dtype=np.uint8))
        with np.load(path,allow_pickle=False) as z:
            with self.assertRaisesRegex(ValueError,'scalar cap'):a.bounded(z,'x')
    def test_even_hist_median(self):
        self.assertEqual(a.median_hist(np.array([0,0,1,1,0,0,0])),2.5)
    def test_neural_synchronized_forward_and_tamper(self):
        import torch
        class FakeModel:
            def act(self,x,v,state,deterministic):
                zeros=torch.zeros((1,42),dtype=torch.float64)
                return dict(state=state,action=zeros,mean=zeros,latent=zeros,logstd=zeros,
                    log_prob=torch.zeros(1,dtype=torch.float64),value=torch.zeros(1,dtype=torch.float64))
        source=np.r_[np.full(960,2),np.full(1920,3)]
        n={key:np.zeros(shape) for key,shape in {'X238':(2880,238),'V255':(2880,255),
            'hidden_before':(2880,128),'hidden_after':(2880,128),'policy_action':(2880,42),
            'mean':(2880,42),'latent':(2880,42),'logstd':(2880,42),'value':(2880,),'log_prob':(2880,),
            'actually_applied_action':(2880,42)}.items()}
        n.update(time=1.+np.arange(2880)*.005,control_source=source,actually_applied_action_is_policy=source==3)
        p=self.p/'neural_decisions.npz';np.savez_compressed(p,**n)
        np.savez_compressed(self.p/'decisions_5ms.npz',time=n['time'],X238=n['X238'],V255=n['V255'],action=n['actually_applied_action'],source=source)
        np.savez_compressed(self.p/'trace_1ms.npz',time=np.arange(1,15401)*.001,action=np.zeros((15400,42)),
            control_source=np.r_[np.zeros(1000),np.full(4800,2),np.full(9600,3)],
            external_body_force_max=np.zeros(15400),external_generalized_force_max=np.zeros(15400))
        result={'neural':{'sha256':a.sha(p)}}
        self.assertEqual(a.neural_check(self.p,result,FakeModel(),torch)['status'],'PASS')
        n['hidden_before'][100,0]=1.;np.savez_compressed(p,**n);result['neural']['sha256']=a.sha(p)
        with self.assertRaisesRegex(ValueError,'hidden chain'):a.neural_check(self.p,result,FakeModel(),torch)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(AuditTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    output={'tests_run':result.testsRun,'successful':result.wasSuccessful(),
        'failures':[(str(t),s) for t,s in result.failures], 'errors':[(str(t),s) for t,s in result.errors],
        'physical_steps':0,'fixtures':'temporary bounded synthetic; automatically removed'}
    with (Path(__file__).parent/'TEST_RESULTS.json').open('w') as f:json.dump(output,f,indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 1)
