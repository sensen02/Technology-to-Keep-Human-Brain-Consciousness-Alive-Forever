"""Explicit snapshot-ID candidate exports; never overwrite runtime interface tables.
Candidate motor pool must remain rejected by the production admission guard.
"""
from pathlib import Path
import sys,json,hashlib,importlib.util
import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO));sys.path.append(str(REPO/'venv/lib/python3.12/site-packages'))
import pandas as pd
O=Path('/home/sensen/Desktop/engine修复验收/s4-banc888-interface-candidates/20261101T212622Z')
FEATHER=REPO/'data/banc888/banc_888_meta.feather'
SENSORY=REPO/'outputs/neural_rebuild/sensory_channel_lookup.json'
MOTOR=REPO/'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json'
TIER=Path('/home/sensen/Desktop/engine修复验收/s4-neural-boundary-replay/20261101T185129Z/IDENTITIES.npz')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
    O.mkdir(parents=True,exist_ok=False);files=[Path(__file__),FEATHER,SENSORY,MOTOR,TIER,REPO/'tools_export_sensory_channel_lookup.py',REPO/'engine/embodied/runtime_pool_guard.py'];hashes={str(p):sha(p) for p in files}
    dump(O/'PREREGISTRATION.json',dict(source_sha256=hashes,scope='S3/S4 explicit BANC materialization888 identity repair candidates only, no runtime deployment',rules='Read root_888 as unique snapshot identity, reconstruct exactly same subtype/leg/side and motor target/body/side annotation rules; preserve aliases and16unresolved; no new neurons, synapses, physiological claims or model activity.',checks='Rebuild root_id sensory dictionary exactly equal historical lookup before candidate. Validate all selected pool annotation rows against historical root_id membership. All current files unchanged. Candidate motor must be rejected by existing guard.',root_column='root_888',runtime_allowed=False))
    spec=importlib.util.spec_from_file_location('sensory_export_reference',REPO/'tools_export_sensory_channel_lookup.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    d=pd.read_feather(FEATHER)
    assert not d['root_888'].isna().any() and not d['root_888'].duplicated().any()
    table888=d['root_888'].astype('int64').to_numpy();old=d['root_id'].astype('int64').to_numpy();tier=set(map(int,np.load(TIER)['root_ids']))
    oldlookup=json.loads(SENSORY.read_text());assert oldlookup['source']['sha256']==sha(FEATHER)
    historic={};candidate={};changes=[]
    for i in range(len(d)):
        sub=str(d['cell_sub_class'].iloc[i]);side=str(d['side'].iloc[i]);leg=None
        for body,code in ref.LEG_CODE.items():
            if sub.startswith(body):
                pre=ref.SIDE_PREFIX.get(side)
                if pre:leg=(pre+code).lower()
                break
        if leg is None:continue
        channel=next((name for name,term in ref.CHANNEL_SPEC.items() if term in sub),None)
        if channel is None:continue
        pair=[leg,channel];a=str(int(old[i]));b=str(int(table888[i]));historic[a]=pair
        assert b not in candidate;candidate[b]=pair
        if a!=b:changes.append(dict(row=i,root_id=a,root_888=b,leg=leg,channel=channel,old_in_tier=int(a) in tier,new_in_tier=int(b) in tier))
    assert historic==oldlookup['leg_channel_by_root']
    outlookup=json.loads(SENSORY.read_text());outlookup.update(status='CANDIDATE_NOT_RUNTIME',runtime_allowed=False,leg_channel_by_root=candidate,n_cells_with_a_channel=len(candidate))
    outlookup['source'].update(root_id_column='root_888',materialization=888)
    dump(O/'sensory_channel_lookup_root888.candidate.json',outlookup)
    pool=json.loads(MOTOR.read_text());resolved=pool['measured']['resolved'];mdiffs=[]
    for row in resolved:
        body={'F':'front_leg','M':'middle_leg','H':'hind_leg'}[row['leg'][1]]
        mask=(d['flow'].astype(str)=='efferent')&(d['peripheral_target_type'].astype(str)==row['muscle_target'])&(d['body_part_effector'].astype(str)==body)&(d['side'].astype(str)==row['side'])
        idx=np.flatnonzero(mask.to_numpy());expected=[int(old[i]) for i in idx]
        assert set(expected)==set(row['pool_root_ids']) and len(expected)==row['n_pool']
        roots=[int(table888[i]) for i in idx]
        if roots!=row['pool_root_ids']:mdiffs.append(dict(actuator=row['actuator'],old=row['pool_root_ids'],candidate=roots))
        row['pool_root_ids']=roots
    pool['status']='CANDIDATE_NOT_RUNTIME';pool['runtime_allowed']=False
    pool['measured']['source'].update(root_id_column='root_888',materialization=888,sha256=sha(FEATHER))
    pool['measured']['columns_used']=['root_888' if c=='root_id' else c for c in pool['measured']['columns_used']]
    dump(O/'mn_pools_root888.candidate.json',pool)
    from engine.embodied.runtime_pool_guard import load_runtime_pool,RuntimePoolRejected
    try:load_runtime_pool(O/'mn_pools_root888.candidate.json')
    except RuntimePoolRejected as e:assert e.status=='CANDIDATE_NOT_RUNTIME';guard_rejected=True
    else:raise AssertionError('Production guard admitted a new candidate unexpectedly')
    previous=json.loads(MOTOR.read_text());oldmotor=set(r for row in previous['measured']['resolved'] for r in row['pool_root_ids']);newmotor=set(r for row in resolved for r in row['pool_root_ids'])
    result=dict(source_sha256=hashes,identity_column='root_888',total_annotation_rows=len(d),sensory=dict(historical_keys=len(historic),candidate_keys=len(candidate),changed_source_rows=changes,historical_in_tier=len(set(map(int,historic))&tier),candidate_in_tier=len(set(map(int,candidate))&tier)),motor=dict(changed_actuators=mdiffs,old_roots=len(oldmotor),candidate_roots=len(newmotor),historical_in_tier=len(oldmotor&tier),candidate_in_tier=len(newmotor&tier),unresolved=len(pool['measured']['unresolved'])),guard_rejected_candidate=guard_rejected,source_unchanged=hashes=={str(p):sha(p) for p in files},calls=dict(model_load=0,network_build=0,neural_step=0,body_step=0,training=0),meaning='Identity correction only, not validated new dynamics, restored navigation or anatomical muscle calibration.')
    assert result['source_unchanged'] and result['motor']['unresolved']==16;dump(O/'RESULT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','sensory','motor')},indent=2));print('sensory tier',result['sensory']['historical_in_tier'],result['sensory']['candidate_in_tier'],'motor',result['motor']['historical_in_tier'],result['motor']['candidate_in_tier'])
if __name__=='__main__':main()
