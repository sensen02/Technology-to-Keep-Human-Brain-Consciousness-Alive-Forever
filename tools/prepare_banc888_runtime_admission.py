"""Create hash-bound snapshot888 interface payloads for an explicit body experiment.
No guard patch, deployment, neural simulation or physical evaluation here.
"""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[1]
C=Path('/home/sensen/Desktop/engine修复验收/s4-banc888-interface-candidates/20261101T212622Z')
H=Path('/home/sensen/Desktop/engine修复验收/s4-banc888-full-prewarm-replay/20261101T214958Z')
A=Path('/home/sensen/Desktop/engine修复验收/s4-neural-boundary-replay/20261101T185129Z')
O=Path('/home/sensen/Desktop/engine修复验收/s3-banc888-runtime-admission/20261101T220018Z')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
def fingerprint(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()
def main():
    O.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),C/'sensory_channel_lookup_root888.candidate.json',C/'mn_pools_root888.candidate.json',C/'INDEPENDENT_VERIFICATION.json',H/'OLD_PREWARM_GATE.json',H/'INDEPENDENT_VERIFICATION.json',H/'RESULT.json',R/'engine/embodied/runtime_pool_guard.py',R/'check_pool_version_contract.py']
    dump(O/'PREREGISTRATION.json',dict(task='S3/S5 explicit snapshot888 interface admission for single candidate physical comparison',source_sha256={str(p):sha(p) for p in sources},basis='Annotation root888 identity correction and independently verified full neural prewarm, not behavior or physiological certification',policy='Old default/historical payload kept unchanged. Newly permitted payload requires exact pool hash AND explicit exact reviewed contract fingerprint pinned in code, all input hashes and namespaces verified. Candidate flags alone grant no admission.',experiment='Same actual3470fullbody, same1.3s assisted reference warmup+.7s live neural release; only new interface tables selected, no tuning/training/body parameter change. Body experiment handled separately.'))
    for source,name in [(R/'engine/embodied/runtime_pool_guard.py','BEFORE_runtime_pool_guard.py'),(R/'check_pool_version_contract.py','BEFORE_check_pool_version_contract.py')]: (O/name).write_bytes(source.read_bytes())
    assert json.loads((C/'INDEPENDENT_VERIFICATION.json').read_text())['passed']
    assert json.loads((H/'INDEPENDENT_VERIFICATION.json').read_text())['passed']
    # These immutable files have already passed identity and full-history replay gates.
    pool=json.loads((C/'mn_pools_root888.candidate.json').read_text());sense=json.loads((C/'sensory_channel_lookup_root888.candidate.json').read_text())
    assert pool['status']=='CANDIDATE_NOT_RUNTIME' and pool['runtime_allowed'] is False
    for payload in (pool,sense):
        payload['status']='PASS_SNAPSHOT888_IDENTITY';payload['runtime_allowed']=True
        payload['contract']=dict(schema='banc_snapshot888_interface_v1',annotation_namespace='BANC/materialization888/root_888',connection_namespace='BANC/materialization888/root_888',runtime_allowed=True,scope='Identity-verified interface used in declared candidate experiments; not autonomous walking certification',source_candidate_sha256=sha(C/('mn_pools_root888.candidate.json' if payload is pool else 'sensory_channel_lookup_root888.candidate.json')))
    poolpath=O/'MN_POOLS_SNAPSHOT888.json';sensepath=O/'SENSORY_LOOKUP_SNAPSHOT888.json';dump(poolpath,pool);dump(sensepath,sense)
    required=[poolpath,sensepath,R/'data/banc888/banc_888_meta.feather',R/'data/flywire/banc_connectome.npz',A/'IDENTITIES.npz',A/'ACTUAL_We.npz',A/'ACTUAL_Wi.npz',C/'INDEPENDENT_VERIFICATION.json',H/'OLD_PREWARM_GATE.json',H/'INDEPENDENT_VERIFICATION.json',H/'RESULT.json',Path('/home/sensen/Desktop/engine修复验收/s5-detailed-handover-measurement/20261028T121000Z/MODEL.xml')]
    contract=dict(schema='banc_snapshot888_interface_v1',authoritative_namespaces=dict(annotation='BANC/materialization888/root_888',connection='BANC/materialization888/root_888'),input_sha256={str(p):sha(p) for p in required},pool_sha256=sha(poolpath),sensory_lookup_sha256=sha(sensepath),hash_bound_namespace_evidence=[dict(kind='annotation_identity',file=str(C/'INDEPENDENT_VERIFICATION.json'),sha256=sha(C/'INDEPENDENT_VERIFICATION.json')),dict(kind='full_prewarm_neural_replay',file=str(H/'INDEPENDENT_VERIFICATION.json'),sha256=sha(H/'INDEPENDENT_VERIFICATION.json'))],scope=dict(neurons=12000,actuators=90,mapped_actuators=74,unmapped_actuators=16,pool_groups=66,physical_behavior='NOT_ESTABLISHED',physiology='NOT_ESTABLISHED'))
    dump(O/'VERSION_CONTRACT.json',contract)
    dump(O/'REGISTRATION_VALUES.json',dict(reviewed_contract_fingerprint=fingerprint(contract),pool_sha256=sha(poolpath),sensory_sha256=sha(sensepath),version_contract_file=str(O/'VERSION_CONTRACT.json'),counts=dict(model_load=0,neural_step=0,body_step=0)))
    print((O/'REGISTRATION_VALUES.json').read_text())
if __name__=='__main__':main()
