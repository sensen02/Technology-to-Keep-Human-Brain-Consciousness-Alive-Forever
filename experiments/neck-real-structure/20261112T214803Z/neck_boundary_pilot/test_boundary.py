"""Synthetic toy queue tests only; never actual BANC transport or biological admission."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest
from boundary import BoundaryTransport, BoundaryError, STATUS, checked_sources, sha256

class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='synthetic_neck_transport_');self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.source=self.root/'TOY_NOT_BIOLOGY.txt';self.source.write_text('only synthetic fixture evidence')
        self.bindings={'schema':'SOURCE_BOUND_TWO_FIBRE_TRANSPORT_V1','status':STATUS,'synthetic_fixture':True,'full_neck_training_allowed':False,'electrophysiology_verified':False,'electrode_coupling_verified':False,'engineering_config':{'dt_s':.001,'delay_s':.002,'delay_steps':2,'activity_units':'DIMENSIONLESS_ENGINEERING_ACTIVITY','activity_min':-1.,'activity_max':1.,'gain':1.},'files':[{'path':str(self.source),'sha256':sha256(self.source)}],'fibres':[{'direction':d,'root_id':'TOY_'+d,'transport_kind':'SAME_AXON_IDENTITY_NO_SYNAPSE'} for d in ('DN','AN')]}
        self.path=self.root/'TOY_BINDINGS.json';self.path.write_text(json.dumps(self.bindings));self.sha=sha256(self.path);self.transport=BoundaryTransport(self.path,self.sha,_allow_synthetic_fixture=True)
    def packet(self,direction='DN',value=.5):
        return {'from':'brain' if direction=='DN' else 'other','to':'neck','direction':direction,'root_ids':['TOY_'+direction],'values':[value],'capture_step':self.transport.step,'source_bindings_sha256':self.sha}
    def test_two_ticks_identity_and_directions(self):
        self.transport.send(self.packet());self.transport.send(self.packet('AN',-.25));self.assertEqual(self.transport.advance(),[]);out=self.transport.advance();self.assertEqual([(x['root_id'],x['value'],x['target_part']) for x in out],[('TOY_DN',.5,'other'),('TOY_AN',-.25,'brain')]);self.assertTrue(all(x['capture_step']==0 and x['release_step']==2 and x['consume_step']==2 for x in out))
    def test_cut_drops_inflight_and_restore_no_reset(self):
        self.transport.send(self.packet());self.transport.send(self.packet('AN'));self.transport.advance();self.transport.set_cut('DN',True);out=self.transport.advance();self.assertEqual([x['direction'] for x in out],['AN']);self.assertEqual(self.transport.step,2)
        with self.assertRaises(BoundaryError):self.transport.send(self.packet())
        self.transport.set_cut('DN',False);self.transport.send(self.packet(value=.75));self.assertEqual(self.transport.advance(),[]);out=self.transport.advance();self.assertEqual([x['value'] for x in out],[.75]);self.assertEqual(self.transport.step,4);self.assertTrue(any(x['event']=='drop_inflight_on_cut' for x in self.transport.journal()))
    def test_no_reset_api(self):self.assertFalse(hasattr(self.transport,'reset'))
    def test_fixture_not_real_binding(self):
        with self.assertRaises(BoundaryError):BoundaryTransport(self.path,self.sha)
        self.assertFalse(self.transport.qualification['source_qualified_two_fibres'])
    def test_extra_action_target_fields(self):
        for field in ('joint_actions','target_position','brain_hidden'):
            p=self.packet();p[field]=[1.]
            with self.assertRaises(BoundaryError):self.transport.send(p)
    def test_original_ID_order_required(self):
        for ids in (['TOY_AN'],['TOY_DN','TOY_AN'],[]):
            p=self.packet();p['root_ids']=ids
            with self.assertRaises(BoundaryError):self.transport.send(p)
    def test_packet_wrong_direction_or_bypass(self):
        p=self.packet();p['to']='other'
        with self.assertRaises(BoundaryError):self.transport.send(p)
    def test_no_past_or_future(self):
        for step in (-1,1,0.0,False):
            p=self.packet();p['capture_step']=step
            with self.assertRaises(BoundaryError):self.transport.send(p)
    def test_nonfinite_or_out_of_bounds(self):
        for value in (float('nan'),float('inf'),2.,True):
            p=self.packet(value=value)
            with self.assertRaises(BoundaryError):self.transport.send(p)
    def test_duplicate_sample(self):
        self.transport.send(self.packet())
        with self.assertRaises(BoundaryError):self.transport.send(self.packet())
    def test_binding_mutation_failbefore_clock(self):
        self.path.write_text(self.path.read_text()+' ')
        with self.assertRaises(BoundaryError):self.transport.advance()
        self.assertEqual(self.transport.step,0)
    def test_source_mutation_rejected(self):
        self.source.write_text('changed')
        for action in (lambda:self.transport.send(self.packet()),self.transport.advance,lambda:self.transport.set_cut('DN',True)):
            with self.assertRaises(BoundaryError):action()
        self.assertEqual(self.transport.step,0)
    def test_not_a_dense_neck_RNN(self):
        c=copy.deepcopy(self.bindings);c['fibres'][0]['transport_kind']='DENSE_RNN'
        with self.assertRaises(BoundaryError):checked_sources(c,allow_fixture=True)
    def test_no_full_training_admission(self):
        c=copy.deepcopy(self.bindings);c['full_neck_training_allowed']=True
        with self.assertRaises(BoundaryError):checked_sources(c,allow_fixture=True)
    def test_timing_config_immutable(self):
        c=copy.deepcopy(self.bindings);c['engineering_config']['delay_steps']=0
        with self.assertRaises(BoundaryError):checked_sources(c,allow_fixture=True)
    def test_no_heavy_modules(self):self.assertFalse(any(k in sys.modules for k in ('numpy','torch','mujoco','physics','model')))
if __name__=='__main__':unittest.main(verbosity=2)
