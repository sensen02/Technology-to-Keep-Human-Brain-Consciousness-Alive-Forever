import unittest
import numpy as np
from engine.embodied.whole_body_sensing import canonical_region, WholeBodySensing

class RegionTests(unittest.TestCase):
    def test_regions(self):
        for name,region in [('Thorax_geom','thorax'),('Head_geom','head'),('A3_geom','abdomen'),('LWing_geom','wing'),('RHTibia_geom','rh'),('floor',None)]:
            self.assertEqual(canonical_region(name)[0],region)

    def test_regional_annotation_routing(self):
        import mujoco
        from scipy.sparse import csr_matrix
        from unittest.mock import patch
        from engine import canonical_physics as cp
        from engine.embodied import sensory_channels as sc
        m=cp.load_model();d=mujoco.MjData(m);q=m.qpos0.copy()
        groups={leg:{ch:[] for ch in sc.CHANNEL_SPEC} for leg in sc.LEGS}
        built={'n':2,'tier':{'root_ids':[]},'global_index':[0,1],
               'data':{'ann_body_part':['thorax','abdomen'], 'ann_function':['tactile','tactile'],
                       'ann_class':['bristle_neuron','bristle_neuron'],'ann_flow':['afferent','afferent']},
               'edges':{'We':csr_matrix((2,2)), 'Wi':csr_matrix((2,2))}}
        with patch('engine.embodied.whole_body_sensing.channel_index_map',return_value=(groups,{})):
            s=WholeBodySensing(m,built,q)
        quantities={leg:sc.ChannelQuantities() for leg in sc.LEGS}
        observation={'proprio':quantities,'regions':{'thorax':{'contacts':1},'abdomen':{'contacts':0},'head':{'contacts':1}},'contacts':[]}
        with patch.object(s,'observe',return_value=observation):cur,record=s.encode(d)
        self.assertEqual(cur.tolist(),[.3,0.]);self.assertEqual(record['unmapped_contact_regions'],['head'])
        self.assertEqual(record['regional_channels']['thorax']['modeled_outgoing_edges'],[0])
        self.assertFalse(record['regional_channels']['thorax']['propagation_established'])

    def test_real_model_contacts_and_proprio(self):
        import mujoco
        from engine.embodied import sensory_channels as sc
        from unittest.mock import patch
        from engine import canonical_physics as cp
        m=cp.load_model();d=mujoco.MjData(m)
        q=m.qpos0.copy();q[:7]=[0,0,1.5,1,0,0,0];d.qpos[:]=q
        groups={leg:{ch:[] for ch in sc.CHANNEL_SPEC} for leg in sc.LEGS}
        for i,ch in enumerate(['bristle_touch','claw_position','hook_direction']):groups['lf'][ch]=[i]
        with patch('engine.embodied.whole_body_sensing.channel_index_map',return_value=(groups,{})):
            s=WholeBodySensing(m,{'n':3,'tier':{'root_ids':[]}},q)
        j=m.joint('joint_LFTibia_pitch').id;d.qpos[m.jnt_qposadr[j]]+=.1;d.qvel[m.jnt_dofadr[j]]=2
        mujoco.mj_forward(m,d);cur,rec=s.encode(d)
        self.assertGreater(cur[1],0);self.assertGreater(cur[2],0)
        self.assertEqual(rec['whole_body']['air_hair']['neural_current'],0)
        d.qpos[2]=-1;mujoco.mj_forward(m,d);obs=s.observe(d)
        self.assertTrue(any(r['region'] not in sc.LEGS for r in obs['contacts']))
        self.assertTrue(all(np.isfinite(r['force_body']).all() for r in obs['contacts']))
        with patch('engine.embodied.whole_body_sensing.channel_index_map',return_value=(groups,{})):
            zero=WholeBodySensing(m,{'n':3,'tier':{'root_ids':[]}},q,gain=0)
        self.assertFalse(zero.encode(d)[0].any())

if __name__=='__main__':unittest.main()
