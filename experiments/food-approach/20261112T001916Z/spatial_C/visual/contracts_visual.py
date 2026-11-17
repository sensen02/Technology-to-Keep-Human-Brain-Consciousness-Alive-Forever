"""Spatial C exact saved-output reader; no heavy imports until arrays() call."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXP=ROOT.parent.parent
BASE=EXP.parent/'banc-walking-recovery/20261110T152714Z'
HEAD=EXP/'20261110T191449Z/odor/HEAD_GEOMETRY_AUDIT.json'
IDS=('C_left_near','C_right_near','C_left_far','C_right_far','C_front','C_noodor')
MOTOR='789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd'
BEHAVIOR='66f423df4f00d801b2f0a194f539baa63ce84973765694e3d3111fbeb62aa034'
REQUIRED=('full288_hidden_chain','actual2880_motor_outputs_exact','actual2880_behavior_outputs_exact','causal_receptor_filter_exact','actual_sensor_geometry_exact','metrics_cycles_recomputed','source_anchor_actual_head_exact','source_fixed_world_after_handover','prime_source_invalid_not_futureleak','all1ms_distance_inside_exact','arrival_and_distances_recomputed','actual5ms_sensor_sites','Cdevelopment_not_arrivalqualification')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(v,m):
 if not v:raise ValueError(m)
def save(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def unchanged(hashes):
 for p,h in hashes.items():require(sha(p)==h,'visual source/input changed '+p)
def metadata(name,r,a,h,folder):
 require(name in IDS and r['condition']==a['condition']==name and r['stage']==a['stage']=='C','spatial scope mismatch')
 require(r['kind']=='STIMULUS_AB_288_V1' and r['functional_pass'] is None,'C schema/development scope changed')
 require(a['integrity_all_pass'] is True and bool(a['checks']) and all(v is True for v in a['checks'].values()),'independent audit incomplete')
 for key in REQUIRED:require(a['checks'].get(key) is True,'missing spatial288 audit '+key)
 require(Path(a['run']).resolve()==folder and a['result_sha256']==sha(folder/'RESULT.json'),'run/result audit binding')
 require(r['checkpoint_sha256']==a['checkpoint_sha256']==MOTOR and r['behavior_checkpoint_sha256']==a['behavior_checkpoint_sha256']==BEHAVIOR,'policy identities differ')
 require(r['state_total']==a['state_total']==288 and r['state_layout']==a['state_layout']==[128,64,64,32],'288 layout required')
 require(r['transition']==h and h['all_exact'] and h['no_reset'] and h['state_dims']==[128,64,64,32],'handover mismatch')
 require(r['teacher_calls_after_handover']==0 and r['extra_forces'] is False,'post teacher/extra forces')
 require(r['native']['actual_native_steps']==1232000 and r['native']['post_native_steps']==768000,'incomplete native counts')
 require(r['engineering_pass']==a['engineering_pass'],'original5 audit differs')
def arrays(name):
 import numpy as np
 folder=ROOT/'trials'/name;r=read(folder/'RESULT.json');a=read(folder/'VERIFY.json');h=read(folder/'HANDOVER.json');ident=read(folder/'IDENTITY.json');metadata(name,r,a,h,folder)
 p=read(ROOT/'PREREG_PROPOSAL.json');require(p['kind']=='SPATIAL_C6_DEVELOPMENT_V1' and tuple(c['id'] for c in p['layouts'])==IDS,'fixed C layouts changed')
 require(r['identity_sha256']==ident['identity_sha256'] and sha(r['checkpoint'])==MOTOR and sha(r['behavior_checkpoint'])==BEHAVIOR,'identity/weight bytes differ')
 paths=[folder/n for n in ('RESULT.json','VERIFY.json','HANDOVER.json','IDENTITY.json','trace_1ms.npz','geometry_1ms.npz','neural_decisions.npz','runtime_geometry.npz')]+[ROOT/'PREREG_PROPOSAL.json',ROOT/'FROZEN.json',ROOT/'INPUT_BINDINGS_C.json',HEAD,HERE/'contracts_visual.py',HERE/'PREREG.json',Path(r['checkpoint']),Path(r['behavior_checkpoint'])]
 hashes={str(p):sha(p) for p in paths}
 for path,digest in r['source_hashes'].items():require(sha(path)==digest,'actual source drift '+path);hashes[path]=digest
 for path,digest in read(ROOT/'FROZEN.json')['source_hashes'].items():require(sha(path)==digest,'C source/input drift '+path);hashes[path]=digest
 def load(n):
  with np.load(folder/n,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
 for key,n in [('trace','trace_1ms.npz'),('geometry','geometry_1ms.npz'),('neural_decisions','neural_decisions.npz'),('runtime_geometry','runtime_geometry.npz')]:require(sha(folder/n)==r[key]['sha256'],'artifact drift '+n)
 t=load('trace_1ms.npz');g=load('geometry_1ms.npz');d=load('neural_decisions.npz')
 require(t['qpos'].shape==(15400,49) and t['qvel'].shape==(15400,48),'full body trajectory missing')
 require(np.max(np.abs(t['time']-np.arange(1,15401)*.001))<1e-8 and np.array_equal(g['time'],t['time']),'actual trace clock')
 require(np.max(np.abs(d['time']-(1+np.arange(2880)*.005)))<1e-8,'decision clock')
 require(all(np.isfinite(v).all() for group in (t,g,d) for v in group.values()),'nonfinite arrays')
 require(d['full_hidden_before'].shape==d['full_hidden_after'].shape==(2880,288) and np.array_equal(d['full_hidden_before'][1:],d['full_hidden_after'][:-1]),'full288 chain')
 for side in ('before','after'):require(np.array_equal(d['full_hidden_'+side],np.concatenate((d['hidden_'+side],d['behavior_hidden_'+side]),1)),'hidden components')
 require(np.array_equal(t['action'],t['actual_requested_action']) and np.array_equal(t['action'][1000:],np.repeat(d['actually_applied_action'],5,axis=0)) and np.array_equal(d['policy_action'][960:],d['actually_applied_action'][960:]),'applied ANN actions differ')
 center=g['source_center_xy'][5799].copy();radius=float(p['source']['patch_radius']);require(np.array_equal(g['source_center_xy'][5799:],np.broadcast_to(center,(9601,2))),'source not fixedworld')
 require(not np.any(g['source_anchor_valid'][:5799]) and np.all(g['source_anchor_valid'][5799:]),'source anchor timing')
 dist=np.linalg.norm(g['sensor_midpoint_xyz'][5799:,:2]-center,axis=1)
 require(np.allclose(dist,g['center_distance'][5799:],rtol=0,atol=1e-14) and np.array_equal(dist<=radius,g['inside_patch'][5799:]),'distance/patch mismatch')
 yaw=np.unwrap(g['thorax_yaw']);require(abs(yaw[-1]-yaw[5799]-r['yaw_change_rad'])<1e-12,'yaw metric')
 return dict(run=folder,result=r,audit=a,identity=ident,trace=t,geometry=g,decisions=d,source=center,radius=radius,source_config=p['source'],yaw=yaw-yaw[5799],hashes=hashes)
def summary(data):
 path=ROOT/'C_DEVELOPMENT_SUMMARY.json';s=read(path);require(s['status']=='C6_DEVELOPMENT_COMPLETE_NOT_D_CONFIRMATION' and s['all_integrity_pass'] is True,'complete spatial summary required')
 require(tuple(row['condition'] for row in s['rows'])==IDS,'summary six layout order')
 for p,h in s['source_hashes'].items():require(sha(p)==h,'spatial summary source drift');data['hashes'][p]=h
 row=next(row for row in s['rows'] if row['condition']==data['result']['condition']);require(row['development_metrics']['arrived_endpoint_proxy']==data['result']['spatial_metrics']['arrived_endpoint_proxy'] and row['original_engineering_pass']==data['result']['engineering_pass'],'summary result mismatch')
 data['hashes'][str(path)]=sha(path);data['summary_row']=row;return data
