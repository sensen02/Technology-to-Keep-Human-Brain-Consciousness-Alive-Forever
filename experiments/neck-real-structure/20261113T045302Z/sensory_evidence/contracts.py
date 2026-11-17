"""Actual14case/30stage multisensory evidence gate; stdlib until explicitarray load."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'sensory_body';PROJECT=Path('/run/media/sensen/Data2/cell_wound_prototype');VIS=PROJECT/'experiments/visual-orientation/20261112T045316Z';BASE=PROJECT/'experiments/banc-walking-recovery/20261110T152714Z';OLD=PROJECT/'experiments/neck-real-structure/20261113T033622Z/visual_evidence'
IDS=('V_empty','V_center','V_left_pulse','V_right_pulse','V_left_to_right','V_right_to_left','O_left','O_right','O_noodor','F_left_agree','F_right_agree','F_left_DN_cut','O_left_odorcut','O_left_swap');VIDEO_IDS=('V_left_to_right','O_left','F_left_agree','O_left_odorcut');SUMMARY_KIND='SOURCE_NECK_MULTISENSORY_BODY_SUMMARY_V1'
BRAIN_SHA='db2510d46900644c7f8c7b556b4b892dad2a88e3e37ce13792811916a928d4c6';VISION_SHA='996980806ac107137d6d5f4743de207fb3350e84a8a145dc5f52b8807398bb87';ODOR_SHA='66f423df4f00d801b2f0a194f539baa63ce84973765694e3d3111fbeb62aa034';NECK_SHA='9e88a04dd91aa6f9c1c1c68e0562325205e0f36f5cdd9ae41d1dd46c8171dc23';MOTOR_SHA='789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(ok,msg):
 if not ok:raise ValueError(msg)
def fresh(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def unchanged(hashes):
 for p,h in hashes.items():require(sha(p)==h,'Source changed '+p)
def arrays(p):
 import numpy as np
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def own_sources():return [HERE/n for n in ('PREREG.json','contracts.py','adapter.py','render.py','plot_sync.py','check_codeonly.py','CODEONLY_CHECK.json')]+[OLD/n for n in ('render.py','contracts.py','PREREG.json')]+[VIS/'camera.py',ROOT/'runtime.py',ROOT/'runner.py',ROOT/'audit.py',ROOT/'summary.py',ROOT/'review.py']
def admission():
 code=read(HERE/'CODEONLY_CHECK.json');require(code['all_passed'] and all(code['checks'].values()),'Actualcodecheck');unchanged(code['source_hashes']);jobs=read(ROOT/'JOBS.json');require(len(jobs)==30 and all(x['exit_code']==0 for x in jobs),'All30actualstagesmustcomplete')
 s=read(ROOT/'SENSORY_BODY_SUMMARY.json');v=read(ROOT/'PARENT_READY_REVIEW.json');b=read(ROOT/'INPUT_BINDINGS.json');require(s['kind']==SUMMARY_KIND and s['unique_physical_trials']==14 and s['all14_integrity_verified'] and s['state_total']==336,'Actual14/336summary');require(v['integrity_all_pass'] and all(v['checks'].values()) and v['summary_sha256']==sha(ROOT/'SENSORY_BODY_SUMMARY.json'),'Actualindependentpairedreview')
 require(s['brain_sha256']==b['brain_sha256']==BRAIN_SHA and s['visual_sha256']==VISION_SHA and s['odor_sha256']==ODOR_SHA and s['neck_binding_sha256']==NECK_SHA and s['motor_sha256']==MOTOR_SHA,'Allfivepolicy/sourceidentities');require(s['fusion_same_side_not_same_source'] and not s['full_neck_or_native_function_verified'] and not s['food_arrival_or_parking_claim'],'Engineeringfusionnotarrival/fullneck')
 ledger=dict(s['source_hashes']);unchanged(ledger)
 for name in IDS:
  folder=ROOT/'trials'/name;r=read(folder/'RESULT.json');a=read(folder/'VERIFY.json');require(a['integrity_all_pass'] and all(a['checks'].values()) and a['result_sha256']==sha(folder/'RESULT.json'),'Everyactualaudit '+name);require(r['kind']=='SOURCE_NECK_MULTISENSORY_BODY_V1' and r['condition']==a['condition']==name,'Actualcasekind')
  for key,h in [('brain_checkpoint_sha256',BRAIN_SHA),('visual_checkpoint_sha256',VISION_SHA),('odor_checkpoint_sha256',ODOR_SHA),('neck_source_binding_sha256',NECK_SHA),('checkpoint_sha256',MOTOR_SHA)]:require(r[key]==h and (key not in a or a[key]==h),'Actualcasepolicy '+key)
  for key in ('fullshape336_3080_15400_770','336statechain','336components_before','336components_after','sourcequeueall_events_exact','filter_endstate_exact','actualproxies_all1ms','fixedworldodoranchor','originalmetrics_exact','cyclesexact','nativefull_duration','handovercontinuous'):require(a['checks'].get(key) is True,'Requiredactualaudit '+key)
  require(a['same_actualRGB_receptors_X_sourcequeue_full336_replay'] and a['environment_headfield_exact'] and r['state_total']==336 and r['neural_records']==3080 and r['native']['actual_native_steps']==r['physical_steps']==1232000 and r['native']['post_native_steps']==768000,'Full336savedreplayandduration');require(r['modalities_enter_only_brain'] and r['no_label_or_source_actor_inputs'] and r['fusion_same_side_not_same_source'] and not r['food_arrival_or_parking_claim'],'Inputboundaryandphysicalscope');unchanged(r['source_hashes']);ledger.update(r['source_hashes'])
  for filename,key in [('trace_1ms.npz','trace'),('neural_decisions.npz','neural_decisions'),('geometry_1ms.npz','geometry'),('runtime_geometry.npz','runtime_geometry'),('capture_rgb.npz','capture_rgb'),('CAPTURES.json','captures_metadata'),('INITIAL_STATE.json','initial_state'),('SOURCE_QUEUE.json','source_queue')]:require(sha(folder/filename)==r[key]['sha256'],'Actualartifact '+filename);ledger[str(folder/filename)]=sha(folder/filename)
  for filename in ('RESULT.json','VERIFY.json','IDENTITY.json','HANDOVER.json'):ledger[str(folder/filename)]=sha(folder/filename)
 for p in own_sources()+[ROOT/n for n in ('SENSORY_BODY_SUMMARY.json','PARENT_READY_REVIEW.json','JOBS.json','FROZEN.json','INPUT_BINDINGS.json','PREFLIGHT.json')]:require(p.is_file(),'Evidencefile '+str(p));ledger[str(p)]=sha(p)
 return s,v,ledger

def load(condition):
 import numpy as np
 require(condition in IDS,'Registeredcondition');s,review,ledger=admission();folder=ROOT/'trials'/condition;r=read(folder/'RESULT.json');a=read(folder/'VERIFY.json');ident=read(folder/'IDENTITY.json');hand=read(folder/'HANDOVER.json');t=arrays(folder/'trace_1ms.npz');d=arrays(folder/'neural_decisions.npz');g=arrays(folder/'geometry_1ms.npz');rgb=arrays(folder/'capture_rgb.npz')['rgb'];caps=read(folder/'CAPTURES.json');compiled=arrays(folder/'runtime_geometry.npz')
 require(t['qpos'].shape==(15400,49) and t['qvel'].shape==(15400,48) and d['X238'].shape==(3080,238) and d['full_hidden_before'].shape==d['full_hidden_after'].shape==(3080,336),'Actual336/3080schema');require(rgb.shape==(770,2,32,32,3) and rgb.dtype==np.uint8 and len(caps)==770,'Actual770RGBschema');require(np.max(abs(t['time']-np.arange(1,15401)*.001))<1e-8 and np.max(abs(d['time']-np.arange(3080)*.005))<1e-8 and np.array_equal(t['time'],g['time']),'Actualsynchronizedclocks');require(np.array_equal(d['full_hidden_before'][1:],d['full_hidden_after'][:-1]),'Actual336chain')
 require(hand==r['transition'] and hand['no_reset'] and hand['all_exact'] and hand['before_sha256']==hand['after_sha256'],'Actualhandover');require(g['sensor_xyz'].shape==(15400,2,3) and g['odor_source_xy'].shape==(15400,2),'Actualproxy/sourcegeometry');require(np.array_equal(d['visual_brain_input'],d['visual_raw_turn']) and np.array_equal(d['odor_brain_input'],d['odor_raw_turn']),'Onlyactualneuralmodalitiesenterbrain')
 for key in ('actor_receptors','receptors_used','concentration','filter_before','filter_after','pending_before','pending_after'):require(d[key].shape==(3080,2),'Actualbilateralfield '+key)
 require(d['odor_hidden_after'].shape==(3080,32) and d['vision_hidden_after'].shape==(3080,32) and d['brain_hidden_after'].shape==(3080,16),'336components');require(not d['image_valid'][:4].any() and np.all(d['image_capture_id'][:4]==-1) and d['image_valid'][4:].all(),'First20msnotgenuineRGB');valid=d['image_valid'];ids=d['image_capture_id'][valid].astype(int);require(np.all((ids>=0)&(ids<770)) and np.all(d['image_release_time_s'][valid]<=d['time'][valid]+1e-8),'NonfutureactorimageIDs')
 for i,c in enumerate(caps):require(c['capture_id']==i and c['capture_time_s']==i*.02 and c['release_time_s']==i*.02+.02 and hashlib.sha256(rgb[i].tobytes()).hexdigest()==c['rgb_sha256'],'Realpixelcapturemetadata')
 require(np.all(d['odor_consume_step']==d['physical_step_ms']) and np.all(d['odor_capture_step']==d['physical_step_ms']-5),'Actual5msolfactorypending');require(np.sum(compiled['geom_type']==7)==71 and np.all(compiled['geom_rgba'][compiled['geom_type']==7,3]==1),'71opaque')
 row=next(x for x in s['rows'] if x['condition']==condition)
 if condition.startswith('V_'):
  visualrow=next(x for x in s['visual_normal6']['rows'] if x['condition']==condition);functional=visualrow['functional_pass'];flags=visualrow['functional_flags'];metrics=visualrow['metrics']
 else:functional=row['orientation_engineering_pass'];flags=row['orientation_engineering_checks'];metrics=row['metrics']
 yaw=np.unwrap(g['thorax_yaw']);yaw=yaw-yaw[5799]
 return dict(condition=condition,run=folder,result=r,audit=a,identity=ident,trace=t,decisions=d,geometry=g,rgb=rgb,captures=caps,summary=s,review=review,yaw=yaw,iscontrol=condition in ('O_left_odorcut','O_left_swap','F_left_DN_cut'),functional_pass=functional,functional_flags=flags,functional_metrics=metrics,summaryrow=row,source_hashes=ledger)
