"""Stdlib admission until explicitload. Actualnew22stage/10case/304state identities only."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent;SCOPE=HERE.parent;ROOT=SCOPE/'visual_body';CAL=SCOPE/'visual_bridge_calibration';PROJECT=Path('/run/media/sensen/Data2/cell_wound_prototype');VIS=PROJECT/'experiments/visual-orientation/20261112T045316Z';BASE=PROJECT/'experiments/banc-walking-recovery/20261110T152714Z'
IDS=('V_empty','V_center','V_left_pulse','V_right_pulse','V_left_to_right','V_right_to_left','V_left_occluded','V_flip_fixedframe','V_left_DN_cut','V_left_AN_cut');VIDEO_IDS=('V_empty','V_left_to_right','V_left_DN_cut','V_left_occluded')
BRAIN_SHA='2123eeadaffbdcf722e361ad1da26b1816833279918e9231773839ed1cfb924c';VISION_SHA='996980806ac107137d6d5f4743de207fb3350e84a8a145dc5f52b8807398bb87';NECK_SHA='9e88a04dd91aa6f9c1c1c68e0562325205e0f36f5cdd9ae41d1dd46c8171dc23';MOTOR_SHA='789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd';SUMMARY_KIND='ACTUAL_RGB_SOURCE_NECK_VISUAL_BODY_SUMMARY_V1'
OLD_RENDER=VIS/'visual_body_v2/visual/render.py';NEURAL_RENDER=PROJECT/'experiments/neck-real-structure/20261113T000915Z/body_visual_v2/render.py';NEURAL_CORE=PROJECT/'experiments/neck-real-structure/20261112T225331Z/body_visual/render.py'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(v,msg):
 if not v:raise ValueError(msg)
def fresh(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def unchanged(hashes):
 for p,h in hashes.items():require(sha(p)==h,'Source changed '+p)
def arrays(p):
 import numpy as np
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def own_sources():return [HERE/n for n in ('PREREG.json','contracts.py','render.py','plot_sync.py','check_codeonly.py','CODEONLY_CHECK.json')]+[OLD_RENDER,NEURAL_RENDER,NEURAL_CORE,VIS/'camera.py']
def admission():
 c=read(HERE/'CODEONLY_CHECK.json');require(c['all_passed'] and all(c['checks'].values()),'Code source check required');unchanged(c['source_hashes']);jobs=read(ROOT/'JOBS.json');require(len(jobs)==22 and all(j['exit_code']==0 for j in jobs),'Allactual22stagesmustfinish')
 s=read(ROOT/'NEURAL_VISUAL_BODY_SUMMARY.json');v=read(ROOT/'PARENT_READY_REVIEW.json');b=read(ROOT/'INPUT_BINDINGS.json');require(s['kind']==SUMMARY_KIND and s['unique_physical_trials']==10 and s['all10_integrity_verified'],'Actualnew10casesummaryschema');require(v['integrity_all_pass'] and all(v['checks'].values()) and v['summary_sha256']==sha(ROOT/'NEURAL_VISUAL_BODY_SUMMARY.json'),'Independentactualfixedframereview')
 require(b['brain_sha256']==BRAIN_SHA and b['brain_final_step']==300 and b['brain_format']=='SOURCE_NECK_VISUAL_SCALAR_CALIBRATION_V1','Actualfinal300identity');require(s['brain_sha256']==BRAIN_SHA and s['visual_sha256']==b['visual_sha256']==VISION_SHA and s['neck_binding_sha256']==b['neck_sha256']==NECK_SHA and s['motor_sha256']==b['motor_sha256']==MOTOR_SHA,'Source/threepolicyidentities')
 require(s['state_total']==304 and s['neural_records']==3080 and s['visual_brain_scalar_only'] and s['labels_not_actor_inputs'] and not s['whole_neck_or_native_function_verified'],'Actual304no-bypassscope')
 ledger=dict(s['source_hashes']);unchanged(ledger)
 for condition in IDS:
  f=ROOT/'trials'/condition;r=read(f/'RESULT.json');a=read(f/'VERIFY.json');require(a['integrity_all_pass'] and all(a['checks'].values()) and a['result_sha256']==sha(f/'RESULT.json'),'Actualtrial independentaudit '+condition);require(r['kind']=='ACTUAL_RGB_SOURCE_NECK_VISUAL_BODY_V1' and r['condition']==a['condition']==condition,'Actualcasekind')
  require(r['brain_checkpoint_sha256']==a['brain_checkpoint_sha256']==BRAIN_SHA and r['visual_checkpoint_sha256']==a['visual_checkpoint_sha256']==VISION_SHA and r['checkpoint_sha256']==a['checkpoint_sha256']==MOTOR_SHA and r['neck_source_binding_sha256']==a['neck_source_binding_sha256']==NECK_SHA,'Actualtrialsource identity')
  for key in ('complete304chain3080body15400','components256plus32plus16','all_source_queue_consumptions_exact','original_metrics_exact','cycles_recomputed_exact','native_full_duration','continuous_handover'):require(a['checks'].get(key) is True,'Requiredactualaudit '+key)
  require(a['saved_actual_RGB_source_queue_brain_motor_replay'] and r['native']['actual_native_steps']==r['physical_steps']==1232000 and r['native']['post_native_steps']==768000 and r['state_total']==304 and r['neural_records']==3080,'Fullactualbodyduration/replay')
  unchanged(r['source_hashes']);ledger.update(r['source_hashes'])
  for name,key in [('trace_1ms.npz','trace'),('neural_decisions.npz','neural_decisions'),('geometry_1ms.npz','geometry'),('runtime_geometry.npz','runtime_geometry'),('capture_rgb.npz','capture_rgb'),('CAPTURES.json','captures_metadata'),('INITIAL_STATE.json','initial_state'),('SOURCE_QUEUE.json','source_queue')]:require(sha(f/name)==r[key]['sha256'],'ActualartifactSHA '+name);ledger[str(f/name)]=sha(f/name)
  if condition=='V_left_occluded':q=Path(r['occlusion_counterfactual']['path']);require(q.parent==f and sha(q)==r['occlusion_counterfactual']['sha256'],'NewROOTocclusioncounterartifact');ledger[str(q)]=sha(q)
  for name in ('RESULT.json','VERIFY.json','IDENTITY.json','HANDOVER.json'):ledger[str(f/name)]=sha(f/name)
 for p in own_sources()+[ROOT/n for n in ('NEURAL_VISUAL_BODY_SUMMARY.json','PARENT_READY_REVIEW.json','FIXED_FRAME_CAUSAL_COMPARISON.json','JOBS.json','INPUT_BINDINGS.json','FROZEN.json','PREFLIGHT.json')]+[CAL/'EVALUATION/RESULT.json',CAL/'EVALUATION/VERIFY.json',CAL/'TRAIN/VERIFY_000300.json']:
  require(p.is_file(),'Source proof missing '+str(p));ledger[str(p)]=sha(p)
 return s,v,ledger

def load(condition):
 import numpy as np
 require(condition in IDS,'Registeredcondition');s,review,ledger=admission();f=ROOT/'trials'/condition;r=read(f/'RESULT.json');a=read(f/'VERIFY.json');ident=read(f/'IDENTITY.json');hand=read(f/'HANDOVER.json');t=arrays(f/'trace_1ms.npz');d=arrays(f/'neural_decisions.npz');g=arrays(f/'geometry_1ms.npz');rgb=arrays(f/'capture_rgb.npz')['rgb'];cap=read(f/'CAPTURES.json');compiled=arrays(f/'runtime_geometry.npz')
 require(t['qpos'].shape==(15400,49) and t['qvel'].shape==(15400,48) and d['full_hidden_before'].shape==(3080,304) and d['X238'].shape==(3080,238),'New304bodyrecordschema');require(rgb.shape==(770,2,32,32,3) and rgb.dtype==np.uint8 and len(cap)==770,'TruebinocularRGBschema');require(np.max(abs(t['time']-np.arange(1,15401)*.001))<1e-8 and np.max(abs(d['time']-np.arange(3080)*.005))<1e-8 and np.array_equal(t['time'],g['time']),'Synchronizedactualclocks')
 require(np.array_equal(d['full_hidden_before'][1:],d['full_hidden_after'][:-1]) and hand==r['transition'] and hand['all_exact'] and hand['no_reset'] and hand['source_queue_retained'] and hand['camera_queue_retained'],'Actualcontinuous304andqueues');require(np.array_equal(d['visual_raw_turn'],d['visual_brain_input']),'Onlyactualneuralvisualscalar')
 require(not d['image_valid'][:4].any() and np.all(d['image_capture_id'][:4]==-1) and d['image_valid'][4:].all(),'First20msinvalidnotfakeimage');valid=d['image_valid'];ids=d['image_capture_id'][valid].astype(int);require(np.all((ids>=0)&(ids<770)) and np.all(d['image_release_time_s'][valid]<=d['time'][valid]+1e-8),'NonfutureactorimageIDs')
 for i,c in enumerate(cap):require(c['capture_id']==i and c['capture_time_s']==i*.02 and c['release_time_s']==i*.02+.02 and hashlib.sha256(rgb[i].tobytes()).hexdigest()==c['rgb_sha256'],'Actualpixelandcapturemetadata')
 require(np.sum(compiled['geom_type']==7)==71 and np.all(compiled['geom_rgba'][compiled['geom_type']==7,3]==1),'71opaque')
 for direction in ('DN','AN'):
  known=d[direction+'_capture_step']>=0;require(np.all(d[direction+'_release_step'][known]==d[direction+'_capture_step'][known]+2) and np.all(d[direction+'_release_step'][known]<=d['physical_step_ms'][known]),'True neckclock '+direction)
 if condition in ('V_left_DN_cut','V_left_AN_cut'):row=next(x for x in s['cuts'] if x['condition']==condition);functional=None;iscontrol=True;flags={};metrics=row
 elif condition in ('V_left_occluded','V_flip_fixedframe'):row=next(x for x in s['control2'] if x['condition']==condition);functional=row['control_pass'];iscontrol=True;flags=row['flags'];metrics=row['metrics']
 else:row=next(x for x in s['normal6'] if x['condition']==condition);functional=row['functional_pass'];iscontrol=False;flags=row['functional_flags'];metrics=row['metrics']
 yaw=np.unwrap(g['thorax_yaw']);yaw=yaw-yaw[5799]
 return dict(condition=condition,run=f,result=r,audit=a,identity=ident,trace=t,decisions=d,geometry=g,rgb=rgb,captures=cap,summary=s,review=review,yaw=yaw,iscontrol=iscontrol,functional_pass=functional,functional_flags=flags,functional_metrics=metrics,source_hashes=ledger)
