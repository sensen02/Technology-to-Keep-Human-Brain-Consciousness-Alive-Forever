"""Only stdlib toy boundary tests; no fixture is Drosophila anatomy or neural dynamics."""
from pathlib import Path
import copy
import json
import math
import sys
import tempfile
import unittest
from validator import validate, validate_packet, ContractError, sha256
HERE=Path(__file__).resolve().parent

class InterfaceContractTests(unittest.TestCase):
    def setUp(self):
        self.pending=json.loads((HERE/'MODEL_CONTRACT.json').read_text())
        self.temp=tempfile.TemporaryDirectory(prefix='three_part_contract_toy_')
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'SYNTHETIC_TOY_NOT_ANATOMY.json'
        self.path.write_text(json.dumps({'fictional_ports':['toy_dn','toy_an'],'fictional_fibres':True}))
        self.c=copy.deepcopy(self.pending)
        self.c['status']='SYNTHETIC_FIXTURE_ONLY'
        source=dict(id='TOY',path=str(self.path),sha256=sha256(self.path),dataset='SYNTHETIC_TOY',version='TEST_V1',sample='NO_BIOLOGICAL_SAMPLE',species='NO_REAL_SPECIES',role='FICTIONAL_TEST_BOUNDARY')
        ref={'source_id':'TOY','record':'FICTIONAL_TEST_RECORD'}
        self.c['source_bundle']=dict(synthetic_fixture=True,files=[source],crossing_definition_evidence=ref,topology_source_id='TOY',topology_sha256=source['sha256'])
        self.c['ports']=[dict(id='toy_'+d.lower(),direction=d,source_neuron_id='fictional_'+d,identity_evidence=ref,crossing_evidence=ref,signal_units='TEST_ACTIVITY_ONLY',minimum=-1.,maximum=1.) for d in ('DN','AN')]
        self.c['declared_dimensions']={'DN':1,'AN':1}
        self.c['neck_links']=[dict(port_id=p['id'],kind='same_fibre_transmission',evidence=ref,topology_trainable=False,source_neuron_id=p['source_neuron_id']) for p in self.c['ports']]
        self.c['boundary_routes']=[dict(zip(('from','to','direction','port_id'),r),signal='PORT_ACTIVITY') for r in [('brain','neck','DN','toy_dn'),('neck','other','DN','toy_dn'),('other','neck','AN','toy_an'),('neck','brain','AN','toy_an')]]
        self.packet=dict(zip(('from','to','direction'),('brain','neck','DN')),port_ids=['toy_dn'],values=[.2],capture_time_s=0.,release_time_s=.01,consume_time_s=.02,topology_sha256=source['sha256'])
    def reject(self,c=None):
        with self.assertRaises(ContractError):validate(self.c if c is None else c,fixture=True)
    def test_pending_is_valid_but_not_admitted(self):
        r=validate(self.pending);self.assertTrue(r['valid_contract']);self.assertFalse(r['admitted']);self.assertEqual(r['status'],'NOT_ADMITTED_NECK_SOURCE_PENDING')
    def test_pending_refuses_neuron_dimensions(self):
        self.pending['declared_dimensions']['DN']=1316;self.reject(self.pending)
    def test_pending_refuses_runtime_packet(self):
        with self.assertRaises(ContractError):validate_packet(self.pending,self.packet,fixture=True)
    def test_toy_graph_never_biological_admission(self):
        r=validate(self.c,fixture=True);self.assertFalse(r['admitted']);self.assertTrue(r['fixture'])
        with self.assertRaises(ContractError):validate(self.c)
    def test_claiming_nontoy_without_reviews_is_rejected(self):
        self.c['source_bundle']['synthetic_fixture']=False
        self.reject()
    def test_toy_neural_activity_packet(self):
        self.assertTrue(validate_packet(self.c,self.packet,fixture=True))
    def test_duplicate_port_id(self):
        self.c['ports'].append(copy.deepcopy(self.c['ports'][0]));self.reject()
    def test_missing_source_digest(self):
        del self.c['source_bundle']['files'][0]['sha256'];self.reject()
    def test_changed_source_bytes(self):
        self.path.write_text('changed');self.reject()
    def test_topology_not_bound_to_file(self):
        self.c['source_bundle']['topology_sha256']='0'*64;self.reject()
    def test_neck_dense_rnn_disallowed(self):
        self.c['parts']['neck']['kind']='DENSE_RNN';self.reject()
    def test_trainable_topology_disallowed(self):
        self.c['neck_links'][0]['topology_trainable']=True;self.reject()
    def test_brain_to_joint_bypass(self):
        self.c['boundary_routes'].append({'from':'brain','to':'other','direction':'DN','port_id':'toy_dn','signal':'JOINT_ACTION'});self.reject()
    def test_wrong_direction(self):
        self.c['boundary_routes'][0]['direction']='AN';self.reject()
    def test_unmapped_fibre(self):
        self.c['neck_links'][0]['source_neuron_id']='another_fibre';self.reject()
    def test_missing_crossing_evidence(self):
        del self.c['ports'][0]['crossing_evidence'];self.reject()
    def test_no_feedback_route(self):
        self.c['boundary_routes'].pop();self.reject()
    def test_direct_brain_sensory_needs_anatomy(self):
        self.c['brain_side_sensory']=[dict(id='toy_eye',target_part='brain',anatomy_crosses_neck=False,kind='BRAIN_SIDE_SENSORY_NOT_BODY_MOTOR_OR_FEEDBACK')];self.reject()
    def test_anatomically_declared_direct_sensory_allowed(self):
        self.c['brain_side_sensory']=[dict(id='toy_eye',target_part='brain',anatomy_crosses_neck=False,kind='BRAIN_SIDE_SENSORY_NOT_BODY_MOTOR_OR_FEEDBACK',anatomy_evidence={'source_id':'TOY','record':'FICTIONAL_EYE'})];self.assertTrue(validate(self.c,fixture=True)['valid_contract'])
    def test_runtime_hidden_joint_field(self):
        self.packet['joint_action']=[1.]*42
        with self.assertRaises(ContractError):validate_packet(self.c,self.packet,fixture=True)
    def test_runtime_future_feedback(self):
        self.packet['release_time_s']=1.
        with self.assertRaises(ContractError):validate_packet(self.c,self.packet,fixture=True)
    def test_runtime_order_and_nonfinite(self):
        for field,value in [('port_ids',['wrong']),('values',[float('nan')]),('values',[2.])]:
            p=copy.deepcopy(self.packet);p[field]=value
            with self.assertRaises(ContractError):validate_packet(self.c,p,fixture=True)
    def test_no_scientific_claim_in_pending(self):
        self.assertFalse(any(self.pending['claims'].values()))
    def test_no_heavy_import(self):
        self.assertFalse(any(x in sys.modules for x in ('numpy','torch','mujoco','physics','engine.neural')))

if __name__=='__main__':unittest.main(verbosity=2)
