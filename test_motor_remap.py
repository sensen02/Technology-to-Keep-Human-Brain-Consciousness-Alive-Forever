import unittest
import numpy as np
from engine.embodied.motor_output import build_motor_manifest
from engine.embodied.motor_remap import remap_frozen_pool_weights
from engine.embodied.motor_translation import FiniteMotorTranslator

class RemapTests(unittest.TestCase):
    def manifests(self):
        src={'measured':{'resolved':[dict(actuator='a',leg='lf',side='left',muscle_target='x',pool_root_ids=[11]),dict(actuator='b',leg='rf',side='right',muscle_target='y',pool_root_ids=[22])],'unresolved':[]}}
        return (build_motor_manifest(src,[11,22],['a','b']),build_motor_manifest(src,[22,11],['a','b']))
    def test_reorder_preserves_commands(self):
        old,new=self.manifests();w=np.array([[1,0,0,0,0],[2,0,0,0,0.]])
        mapped=remap_frozen_pool_weights(old,new,w)
        np.testing.assert_array_equal(mapped,w[::-1])
        a=FiniteMotorTranslator(old,w);b=FiniteMotorTranslator(new,mapped)
        spikes=np.zeros((10,2));spikes[0,0]=1
        x,_=a.consume(spikes,spike_dt_s=.0005,start_s=0,end_s=.005)
        y,_=b.consume(spikes[:,::-1],spike_dt_s=.0005,start_s=0,end_s=.005)
        np.testing.assert_array_equal(x,y)
    def test_changed_membership_rejected(self):
        old,new=self.manifests();new['rows'][0]['pool_root_ids']=[11,33]
        with self.assertRaises(ValueError):remap_frozen_pool_weights(old,new,np.ones((2,5)))
    def test_inconsistent_identity_rejected(self):
        old,new=self.manifests();new['rows'][0]['tier_root_ids']=[]
        with self.assertRaises(ValueError):remap_frozen_pool_weights(old,new,np.ones((2,5)))
if __name__=='__main__':unittest.main()
