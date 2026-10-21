"""Exercise real episode entry with fake MuJoCo/network; no physical integration."""
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

import run_neural_muscle_closure as c


class FakeData:
    def __init__(self, model):
        self.qpos=np.zeros(3); self.qvel=np.zeros(2); self.act=np.zeros(1)
        self.ctrl=np.zeros(1); self.xpos=np.array([[0.,0.,2.]])
        self.xfrc_applied=np.zeros((1,6)); self.actuator_force=np.zeros(1)
        self.time=0.; self.warning=[types.SimpleNamespace(number=0)]


class FakeModel:
    nu=1
    opt=types.SimpleNamespace(timestep=.0001)
    actuator_ctrlrange=np.array([[0.,1.]])
    actuator_ctrllimited=np.array([False])
    def body(self, name):
        return types.SimpleNamespace(id=0)


class FakeNetwork:
    def __init__(self, n, params, dt_ms):
        self.t_ms=0.; self.currents=[]
    def set_connectivity(self, *args):
        pass
    def step(self, current_nA):
        self.currents.append(current_nA.copy())
        self.t_ms+=.5
        return np.array([len(self.currents)==1], dtype=bool)


class FakeEncoder:
    instances=[]
    def __init__(self, *args):
        self.calls=0; self.instances.append(self)
    def encode(self, quantities):
        self.calls+=1
        return np.array([float(self.calls)]), {}


class FakeDecoder:
    def __init__(self):
        self.calls=[]
    def consume(self, spikes, *, spike_dt_s, start_s, end_s, connected=True):
        self.calls.append((spikes.copy(),start_s,end_s,connected))
        return np.array([0. if not connected else float(spikes.sum())]), {}


class MotorCadenceEntryTests(unittest.TestCase):
    def run_entry(self, motor_dt_s, arm='normal', *, silent=False, replay=None):
        model=FakeModel(); state={}; body_steps=[]
        network=FakeNetwork(1,None,.5); decoder=FakeDecoder()
        FakeEncoder.instances=[]
        fake_mujoco=types.ModuleType('mujoco')
        fake_mujoco.MjData=FakeData
        fake_mujoco.mj_forward=lambda m,d: None
        def step(m,d):
            body_steps.append((d.time,float(d.ctrl[0])))
            d.time+=.0001
        fake_mujoco.mj_step=step
        if silent:
            network.step=lambda current_nA: (network.currents.append(current_nA.copy()),
                                              setattr(network,'t_ms',network.t_ms+.5),
                                              np.zeros(1,dtype=bool))[-1]
        groups={leg:{'bristle_touch':[]} for leg in c.LEGS}
        built={'n':1,'edges':{'We':1.,'Wi':1.}}
        manifest={'rows':[{'local_indices':[0]}]}
        with patch.dict(sys.modules,{'mujoco':fake_mujoco}), \
             patch.object(c,'ChannelSensoryEncoder',FakeEncoder), \
             patch.object(c,'ConductanceNetwork',lambda *a,**kw:network), \
             patch.object(c,'make_motor_decoder',lambda *a,**kw:decoder), \
             patch.object(c,'observe_quantities',lambda *a:None):
            result=c.episode(model, np.zeros(3), {}, {}, built, manifest, groups,
                             arm,.02, replay=replay, external_touch=False,motor_dt_s=motor_dt_s,
                             evidence=state)
        return state, network, decoder, FakeEncoder.instances[0], result, body_steps

    def test_split_entry_held_sensory_and_completed_motor_windows(self):
        state,network,decoder,encoder,result,body_steps=self.run_entry(.001)
        self.assertEqual(encoder.calls,4)
        self.assertEqual(state['sensory_sample_start_s'].tolist(),[0.,.005,.01,.015])
        np.testing.assert_array_equal(state['sensory_current_nA'][:,0],[1.,2.,3.,4.])
        self.assertEqual(state['motor_sensory_index'].tolist(),[i for i in range(4) for _ in range(5)])
        self.assertEqual(state['motor_event_complete_ticks'][0].tolist(),[1])
        self.assertTrue(all(not len(x) for x in state['motor_event_complete_ticks'][1:]))
        self.assertEqual(state['motor_observation_end_s'].shape,(20,))
        self.assertEqual(state['motor_activation_end'].shape,(20,1))
        self.assertEqual(state['motor_height_end_model'].shape,(20,))
        np.testing.assert_array_equal(state['motor_decoded_next_controls'][0],[1.])
        self.assertEqual(len(network.currents),40)
        self.assertEqual([float(x[0]) for x in network.currents], [i for i in range(1,5) for _ in range(10)])
        self.assertEqual(len(decoder.calls),20)
        self.assertEqual([(round(a,6),round(b,6)) for _,a,b,_ in decoder.calls],
                         [(round(i*.001,6),round((i+1)*.001,6)) for i in range(20)])
        self.assertTrue(np.all(state['motor_applied_controls'][0]==0))
        self.assertEqual(state['motor_applied_controls'][1,0],1.)
        self.assertEqual(state['motor_source_end_tick'].tolist(),[-1]+list(range(2,40,2)))
        self.assertEqual(len(body_steps),200)
        self.assertEqual([int(round(t/.0001)) for t,ctrl in body_steps[:10] if ctrl==0.],list(range(10)))
        self.assertTrue(all(ctrl==1. for _,ctrl in body_steps[10:20]))
        self.assertEqual(result[2].shape,(20,1))

    def test_old_entry_5ms_interval_compatibility(self):
        state,network,decoder,encoder,result,body_steps=self.run_entry(.005)
        self.assertEqual(encoder.calls,4)
        self.assertEqual(len(network.currents),40)
        self.assertEqual(len(decoder.calls),4)
        self.assertEqual([round(b-a,6) for _,a,b,_ in decoder.calls],[.005]*4)
        self.assertEqual(result[2].shape,(4,1))
        self.assertEqual(state['dt_command_s'],.005)
        self.assertEqual(state['applied_source_interval'].tolist(),[-1,0,1,2])
        self.assertTrue(np.all(result[2][0]==0))
        self.assertEqual(float(result[2][1,0]),1.)

    def test_split_silent_spikes_and_unsupported_replay_fail_closed(self):
        state,_,_,_,_,_=self.run_entry(.001,silent=True)
        np.testing.assert_array_equal(state['motor_applied_controls'],
                                      np.zeros((20,1)))
        with self.assertRaisesRegex(ValueError,'replay/push contract'):
            self.run_entry(.001,arm='replay',replay=np.zeros((4,1)))

    def test_split_motor_cut_no_stale_command(self):
        state,_,decoder,_,_,_=self.run_entry(.001,'motor_cut')
        starts=state['motor_start_s']
        self.assertTrue(np.all(state['motor_applied_controls'][starts>=.015]==0))
        self.assertFalse(decoder.calls[14][3])
        self.assertFalse(decoder.calls[15][3])


if __name__=='__main__':
    unittest.main()
