import unittest
import numpy as np
from engine.embodied.neural_command_audit import translate_motor_spikes,audit_interval

class OriginTests(unittest.TestCase):
    def sample(self):
        s=np.zeros((10,4),bool);s[0,2:]=1
        rates,ctrl=translate_motor_spikes(s,[[2,3],[]],.01)
        return dict(spikes=s,groups=[[2,3],[]],window_s=.01,reference_hz=100.,actual_ctrl=ctrl,
                    sensory_current=np.array([.01,0,0,0]),sensory_indices=[0,1],background_current=np.zeros(4))
    def test_valid(self):self.assertEqual(audit_interval(**self.sample())['status'],'PASS')
    def test_no_spikes_no_drive(self):
        s=np.zeros((10,4),bool)
        _,ctrl=translate_motor_spikes(s,[[2,3],[]],.01)
        np.testing.assert_array_equal(ctrl,[0,0])
    def test_group_size(self):
        s=np.ones((2,2),bool)
        r,_=translate_motor_spikes(s,[[0],[0,1]],.01)
        np.testing.assert_array_equal(r,[200,200])
    def test_injected_bypasses_fail(self):
        for case in ('motor_current','background','hidden_control'):
            d=self.sample()
            if case=='motor_current':d['sensory_current'][2]=.001
            if case=='background':d['background_current'][3]=.001
            if case=='hidden_control':d['actual_ctrl'][1]=.1
            self.assertEqual(audit_interval(**d)['status'],'FAIL',case)
    def test_cut_cannot_leave_control(self):
        d=self.sample();d['readout_connected']=False
        self.assertEqual(audit_interval(**d)['status'],'FAIL')
        d['actual_ctrl'][:]=0
        self.assertEqual(audit_interval(**d)['status'],'PASS')
    def test_bad_mapping(self):
        for g in ([[-1]],[[.1]],[[0,0]]):
            with self.assertRaises(ValueError):translate_motor_spikes(np.zeros((2,4)),g,.01)

if __name__=='__main__':unittest.main()
