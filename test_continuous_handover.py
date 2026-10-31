"""Important auxiliary: isolated API contract tests, NOT fly walking evidence.

Fake physics and fake runtime exercise control scheduling/state ownership without
running another fly rollout or initializing a biological network.
"""
from types import SimpleNamespace
from unittest.mock import patch
import ast
from pathlib import Path
import unittest
import numpy as np
from engine.embodied.continuous_handover import run_neural_continuous


class TestContinuousHandover(unittest.TestCase):
    def setup_case(self):
        model = SimpleNamespace(nq=1, nu=1, opt=SimpleNamespace(timestep=.0000125))
        data = SimpleNamespace(time=.32, qpos=np.array([1.2]), qvel=np.array([.7]),
                               act=np.array([.3]), ctrl=np.array([.8]),
                               qacc_warmstart=np.array([.4]), qfrc_applied=np.array([12.]),
                               xfrc_applied=np.ones((1, 6)), warning=[])
        net = SimpleNamespace(t_ms=0.)
        currents = []
        def step(current_nA):
            currents.append(current_nA.copy()); net.t_ms += .5
            return np.array([True])
        net.step = step
        encoder = SimpleNamespace(encode=lambda q: (np.array([q['deviation']]), {}))
        decoder = SimpleNamespace(elapsed=0., calls=0)
        def consume(spikes, *, spike_dt_s, start_s, end_s):
            self.assertAlmostEqual(start_s, decoder.elapsed)
            self.assertEqual(spikes.shape, (10, 1))
            decoder.elapsed=end_s; decoder.calls+=1
            return np.array([.1*decoder.calls]), {'fake': True}
        decoder.consume = consume
        seen = []
        def mj_step(m, d):
            self.assertFalse(d.qfrc_applied.any()); self.assertFalse(d.xfrc_applied.any())
            seen.append(float(d.ctrl[0])); d.time += m.opt.timestep
            d.qpos[:] += m.opt.timestep*d.qvel
            d.act[:] += m.opt.timestep*.01
        physics = SimpleNamespace(mj_step=mj_step, mj_forward=lambda m,d: None)
        def quantities(m,d,ref,j,g):
            np.testing.assert_array_equal(ref, [0.5])
            return dict(deviation=float(d.qpos[0]-ref[0]))
        legacy = SimpleNamespace(observe_quantities=quantities)
        built = {'n':1}
        manifest = {'n_neurons':1,'n_actuators':1,'rows':[{'local_indices':[0]}]}
        return model,data,net,encoder,decoder,physics,legacy,built,manifest,seen,currents

    def run_case(self, c, **kw):
        m,d,n,e,dec,physics,legacy,built,manifest,*_ = c
        with patch.dict('sys.modules', {'mujoco':physics,'run_neural_muscle_closure':legacy}), patch(
                'engine.embodied.continuous_handover._make_runtime',return_value=(n,e,dec)):
            return run_neural_continuous(m,d,built,manifest,{},.01,
                                        sensory_reference_qpos=np.array([.5]),joints={},geom_leg={},**kw)

    def test_continuity_causal_commands_offset_and_callback(self):
        c=self.setup_case(); callbacks=[]
        r=self.run_case(c,body_observer=lambda m,d: callbacks.append(d.time), retain_currents=True)
        before=r['handover_before'];after=r['handover_after_withdrawal']
        for key in ('qpos','qvel','act','qacc_warmstart','time'):
            np.testing.assert_array_equal(before[key],after[key])
        self.assertEqual(before['qvel'][0],.7); self.assertEqual(before['act'][0],.3)
        self.assertFalse(after['qfrc_applied'].any()); self.assertFalse(after['xfrc_applied'].any())
        self.assertEqual(r['counts'],dict(network_instances=1,neural_steps=20,body_steps=800,forward=801))
        self.assertEqual(len(callbacks),801)
        np.testing.assert_allclose(c[-2][:400],0);np.testing.assert_allclose(c[-2][400:],.1)
        self.assertEqual([x['applied_source_interval'] for x in r['records']],[-1,0])
        self.assertAlmostEqual(r['final_absolute_time_s'],.33)
        self.assertAlmostEqual(c[1].qpos[0],1.207)
        self.assertAlmostEqual(r['records'][0]['absolute_start_s'],.32)
        self.assertAlmostEqual(r['records'][1]['relative_start_s'],.005)
        # Caller/reference ownership: supplied reference is not handover qpos.
        self.assertAlmostEqual(c[-1][0][0],.7)
        self.assertAlmostEqual(r['handover_before']['qpos'][0],1.2)

    def test_motor_cut_preserves_network_and_decoded_proposals(self):
        c=self.setup_case();r=self.run_case(c,motor_output_connected=False)
        np.testing.assert_array_equal(c[-2],np.zeros(800))
        self.assertEqual(r['counts']['neural_steps'],20)
        self.assertGreater(sum(x['motor_spikes'] for x in r['records']),0)
        self.assertTrue(all(not x['motor_output_connected'] for x in r['records']))
        self.assertGreater(r['records'][0]['decoded_next_command'][0],0)
        self.assertGreater(r['records'][1]['decoded_next_command'][0],0)
        self.assertTrue(all(not x['applied_command'].any() for x in r['records']))

    def test_sensory_weights_passed_to_runtime(self):
        c=self.setup_case();m,d,n,e,dec,physics,legacy,built,manifest,*_=c
        weights={'bristle_touch':0.,'campaniform_strain':0.}
        with patch.dict('sys.modules',{'mujoco':physics,'run_neural_muscle_closure':legacy}), patch('engine.embodied.continuous_handover._make_runtime',return_value=(n,e,dec)) as factory:
            run_neural_continuous(m,d,built,manifest,{},.01,sensory_reference_qpos=np.array([.5]),joints={},geom_leg={},sensory_channel_weights=weights)
            self.assertIs(factory.call_args.args[-1],weights)

    def test_real_encoder_channel_weight_cut(self):
        from engine.embodied.sensory_channels import ChannelSensoryEncoder,ChannelConfig,CHANNEL_SPEC,ChannelQuantities
        import inspect
        self.assertIn('per_channel_weight',inspect.signature(ChannelSensoryEncoder).parameters)
        groups={leg:{ch:[] for ch in CHANNEL_SPEC} for leg in ('lf','lm','lh','rf','rm','rh')}
        groups['lf']['bristle_touch']=[0];groups['lf']['hook_direction']=[1]
        e=ChannelSensoryEncoder(groups,2,ChannelConfig(),per_channel_weight={'bristle_touch':0.})
        # Encode actual quantity dataclass with no synthetic baseline current.
        q=ChannelQuantities(tibia_joint_dev_rad=0.,tibia_joint_vel_rad_s=1.,multi_joint_dev_rad=0.,contact_force_norm=1.,foot_contact=1.)
        current,record=e.encode({'lf':q})
        self.assertEqual(current[0],0.)
        self.assertGreater(current[1],0.)
        self.assertEqual(record['per_leg']['lf']['bristle_touch']['weight'],0.)
        self.assertEqual(record['per_leg']['lf']['hook_direction']['weight'],1.)

    def test_mutating_observer_rejected(self):
        c=self.setup_case()
        def wrong(m,d):d.qvel[:]=0
        with self.assertRaisesRegex(AssertionError,'observer mutated'):
            self.run_case(c,body_observer=wrong)

    def test_timestep_failure_leaves_assistance_untouched(self):
        c=self.setup_case();c[0].opt.timestep=.0001
        with self.assertRaisesRegex(ValueError,'timestep'):
            self.run_case(c)
        np.testing.assert_array_equal(c[1].qfrc_applied,[12.])
        self.assertEqual(c[2].t_ms,0.)

    def test_no_pose_reset_or_data_allocation_in_api(self):
        import engine.embodied.continuous_handover as api
        tree=ast.parse(Path(api.__file__).read_text())
        for n in ast.walk(tree):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):
                self.assertNotIn(n.func.attr,('MjData','mj_resetData','mj_resetDataKeyframe'))
            if isinstance(n,(ast.Assign,ast.AugAssign)):
                targets=n.targets if isinstance(n,ast.Assign) else [n.target]
                for target in targets:
                    v=target.value if isinstance(target,ast.Subscript) else target
                    if isinstance(v,ast.Attribute) and isinstance(v.value,ast.Name) and v.value.id=='data':
                        self.assertNotIn(v.attr,('qpos','qvel','act','time','qacc_warmstart'))


if __name__=='__main__':unittest.main(verbosity=2)
