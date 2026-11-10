"""Independent saved-array FK reconstruction and crosswalk validation, no simulator."""
import json,csv,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
load=lambda n:json.loads((P/n).read_text())
def rot(q):
    w,x,y,z=np.asarray(q)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def axisrot(axis,theta):
    v=np.asarray(axis);x,y,z=v;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+np.sin(theta)*K+(1-np.cos(theta))*(K@K)
result={'method':'Independent NumPy serial-tree rigid transforms and Rodrigues rotations reconstructed from exported compiled body/joint arrays; no MuJoCo imported','models':{},'new_forward_calls':0,'new_step_calls':0}
for tag in ['official','current']:
    m=load(tag+'_model.json');nb=m['counts']['nbody'];nj=m['counts']['njnt'];errors=[]
    for pname,pose in m['poses'].items():
        q=np.array(pose['qpos']);pos=np.zeros((nb,3));R=np.repeat(np.eye(3)[None],nb,axis=0);axis=np.zeros((nj,3));anchor=np.zeros((nj,3))
        for b in m['bodys'][1:]:
            i=b['id'];pa=b['parentid'];pos[i]=pos[pa]+R[pa]@b['pos'];R[i]=R[pa]@rot(b['quat'])
            for ji in range(b['jntadr'],b['jntadr']+b['jntnum']):
                j=m['joints'][ji];qa=j['qposadr']
                if j['type']==0:
                    pos[i]=q[qa:qa+3];R[i]=rot(q[qa+3:qa+7]);continue
                assert j['type']==3
                anchor[ji]=pos[i]+R[i]@j['pos'];axis[ji]=R[i]@j['axis'];R[i]=R[i]@axisrot(j['axis'],q[qa]-j['qpos0']);pos[i]=anchor[ji]-R[i]@j['pos']
        th=next(b['id'] for b in m['bodys'] if b['name']==('walker/thorax' if tag=='official' else 'Thorax'))
        relpos=(pos-pos[th])@R[th];relR=np.einsum('ij,bjk->bik',R[th].T,R)
        hinge=[j['id'] for j in m['joints'] if j['type']==3]
        e=dict(pose=pname,body_position_max_error=float(np.max(abs(relpos-np.array(pose['body_positions_thorax'])))),body_rotation_max_error=float(np.max(abs(relR-np.array(pose['body_rotations_thorax'])))),hinge_axis_max_error=float(np.max(abs((axis@R[th])[hinge]-np.array(pose['joint_axes_thorax'])[hinge]))),hinge_anchor_max_error=float(np.max(abs(((anchor-pos[th])@R[th])[hinge]-np.array(pose['joint_anchors_thorax'])[hinge]))))
        assert max(v for k,v in e.items() if k!='pose')<1e-12,e
        errors.append(e)
    result['models'][tag]=errors
with (P/'official59_actions.csv').open() as f:official=list(csv.DictReader(f))
with (P/'current42_candidates.csv').open() as f:current=list(csv.DictReader(f))
A=load('official_model.json');B=load('current_model.json');ad={a['id']:a for a in A['actuators']}
identity=json.loads((Path(A['runtime']['model_source']).parent/'MODEL_IDENTITY.json').read_text())
assert identity['action_names']==A['action_order']
for a in A['actuators']:
    i=a['action_index'];assert a['ctrlrange']==[identity['action_min'][i],identity['action_max'][i]]
for key in ('nq','nv','nu','na','neq','nmesh'):assert identity[key]==A['counts'][key]
result['archived_model_identity_checks']={'counts_match':True,'all_59_action_names_and_bounds_match':True}
assert len(official)==59 and len(current)==42
assert {int(a['policy_action_index']) for a in official}==set(range(59))
assert {int(a['ctrl_index']) for a in official}==set(range(59))
for r in official:
    a=ad[int(r['ctrl_index'])];assert r['official_actuator']==a['name'];assert json.loads(r['ctrlrange'])==a['ctrlrange']
    assert A['action_order'][int(r['policy_action_index'])]==a['name'].split('/')[-1]
    if r['transmission']=='tendon':
        t=A['tendons'][a['trnid'][0]];assert json.loads(r['target_joints'])==[w['name'] for w in t['wraps']]
        assert json.loads(r['tendon_coefficients'])==[w['parameter'] for w in t['wraps']]
for i,r in enumerate(current):
    assert int(r['current_action_index'])==i and r['current_joint']==B['actuators'][i]['target_name']
    assert r['verified_scalar_map']=='False'
assert sum(not json.loads(r['candidate_current_joints']) for r in official)==23
result['csv_checks']={'official_action_rows':59,'current_joint_rows':42,'permutation_bijective':True,'tendon_targets_and_coefficients_match':True,'candidate_only_labels_preserved':True}
hashes=load('SOURCE_HASHES.json');changed=[];mutable=[]
for p,h in hashes.items():
    currenthash=hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if currenthash!=h:
        if Path(p).name in ('AGENTS.md','GOAL_ROADMAP.zh-CN.md','GOAL_TRACKER.zh-CN.md'):mutable.append(p)
        else:changed.append(p)
assert not changed,changed
result['source_recheck']={'fixed_sources_unchanged':True,'mutable_parent_project_docs_changed_after_snapshot':mutable,'sources_checked':len(hashes)}
result['passed']=True
(P/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
