"""Stdlib until explicit actualarray load. Strict six-complete-trial source identity admission."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent;SCOPE=HERE.parent;ROOT=SCOPE/'body_pilot';PROJECT=SCOPE.parents[2]
BASE=PROJECT/'experiments/banc-walking-recovery/20261110T152714Z'
ORIGINAL_RENDER=PROJECT/'experiments/visual-orientation/20261112T045316Z/visual_body_v2/visual/render.py'
IDS=('P_empty','P_left_pulse','P_right_pulse','P_left_DN_cut','P_left_AN_cut','P_left_cuezero')
BRAIN_SHA='eed1ad3cc61a1b59086ff2324175f6bf3b5da5aa197165e6c4b009c17027c95e';NECK_SHA='9e88a04dd91aa6f9c1c1c68e0562325205e0f36f5cdd9ae41d1dd46c8171dc23';MOTOR_SHA='789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(v,message):
 if not v:raise ValueError(message)
def fresh(p,value):
 with Path(p).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def unchanged(hashes):
 for p,h in hashes.items():require(sha(p)==h,'source drift '+p)
def arrays(p):
 import numpy as np
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def own_sources():return [HERE/n for n in ('PREREG.json','contracts.py','render.py','plot_sync.py','codecheck.py','CODEONLY_CHECK.json')]+[ORIGINAL_RENDER]
def admission():
 checked=read(HERE/'CODEONLY_CHECK.json');require(checked['all_passed'] and all(checked['checks'].values()),'codeonlysourcecheckrequired');unchanged(checked['source_hashes'])
 s=read(ROOT/'BODY_SUMMARY.json');jobs=read(ROOT/'JOBS.json');require(len(jobs)==13 and all(j['exit_code']==0 for j in jobs),'all6physical/all6audit/summary stages must complete beforevisuals')
 require(s['kind']=='ARTIFICIAL_BRAIN_SOURCE_NECK_SIX_CASE_BODY_SUMMARY' and s['unique_physical_trials']==6 and s['all6_integrity_verified'] is True,'actualsixcase summaryschema')
 require([x['condition'] for x in s['rows']]==list(IDS),'summary actualconditionorder');require(s['brain_sha256']==BRAIN_SHA and s['neck_binding_sha256']==NECK_SHA and s['motor_sha256']==MOTOR_SHA,'actualANN/source/motoridentities');require(s['whole_neck_or_native_tuning_verified'] is False and s['lowlevel_basewalking_not_brain_startstop'] is True and s['no_visual_input'] is True,'engineeringonly scope/no fakevisuals')
 unchanged(s['source_hashes']);ledger=dict(s['source_hashes'])
 for condition in IDS:
  f=ROOT/'trials'/condition;r=read(f/'RESULT.json');a=read(f/'VERIFY.json');require(a['integrity_all_pass'] and bool(a['checks']) and all(a['checks'].values()) and a['result_sha256']==sha(f/'RESULT.json'),'every actualtrial must pass integrity')
  require(a['saved_source_queue_bridge_motor_replay'] and a['condition']==r['condition']==condition,'independent actualsourcebridge+motor replay')
  require(r['brain_checkpoint_sha256']==a['brain_checkpoint_sha256']==BRAIN_SHA and r['checkpoint_sha256']==a['checkpoint_sha256']==MOTOR_SHA and r['neck_source_binding_sha256']==a['neck_source_binding_sha256']==NECK_SHA,'actualtrialweight/sourcebinding mismatch')
  for key in ('272_chain','brain16_motor256_components','complete_body15400_neural3080','all_source_queue_and_consumption_exact','original_metrics_exact','cycles_recomputed_exact','native_full_duration','continuous_handover'):require(a['checks'].get(key) is True,'missing independent audit '+key)
  require(r['native']['actual_native_steps']==r['physical_steps']==1232000 and r['native']['post_native_steps']==768000,'fullnativephysicsduration');require(r['state_total']==272 and r['state_layout']==[128,64,64,16] and r['neural_records']==3080,'newbrain16/motor256 memories separateportbuffers');require(r['visual_or_odor_input'] is False,'no cameraorodor pipeline here')
  unchanged(r['source_hashes']);ledger.update(r['source_hashes'])
  for name,key in [('trace_1ms.npz','trace'),('neural_decisions.npz','neural_decisions'),('geometry_1ms.npz','geometry'),('runtime_geometry.npz','runtime_geometry'),('INITIAL_STATE.json','initial_state'),('SOURCE_QUEUE.json','source_queue')]:require(sha(f/name)==r[key]['sha256'],'actualartifactsha '+name);ledger[str(f/name)]=sha(f/name)
  for name in ('RESULT.json','VERIFY.json','IDENTITY.json','HANDOVER.json'):ledger[str(f/name)]=sha(f/name)
 for p in own_sources()+[ROOT/'BODY_SUMMARY.json',ROOT/'JOBS.json',ROOT/'INPUT_BINDINGS.json',ROOT/'FROZEN.json',ROOT/'PREFLIGHT.json']:
  require(p.is_file(),'required source '+str(p));ledger[str(p)]=sha(p)
 return s,ledger

def load(condition):
 import numpy as np
 require(condition in IDS,'registeredcondition');s,ledger=admission();f=ROOT/'trials'/condition;r=read(f/'RESULT.json');a=read(f/'VERIFY.json');ident=read(f/'IDENTITY.json');hand=read(f/'HANDOVER.json');t=arrays(f/'trace_1ms.npz');d=arrays(f/'neural_decisions.npz');g=arrays(f/'geometry_1ms.npz');compiled=arrays(f/'runtime_geometry.npz')
 require(t['qpos'].shape==(15400,49) and t['qvel'].shape==(15400,48) and d['X238'].shape==(3080,238) and d['full_hidden_before'].shape==(3080,272),'actualfullbodyshapes')
 require(np.max(abs(t['time']-np.arange(1,15401)*.001))<1e-8 and np.max(abs(d['time']-np.arange(3080)*.005))<1e-8 and np.array_equal(g['time'],t['time']),'actualsynchronizedclocks')
 require(np.array_equal(d['full_hidden_before'][1:],d['full_hidden_after'][:-1]),'actual272chain');require(hand==r['transition'] and hand['all_exact'] and hand['no_reset'] and hand['queue_retained'],'continuousactualhandover');require(r['identity_sha256']==ident['identity_sha256'],'actualphysicalidentity')
 for key,shape in [('AN_held',(3080,1)),('DN_held',(3080,2)),('brain_hidden_after',(3080,16)),('AN_encoded',(3080,1)),('DN_encoded',(3080,2))]:require(d[key].shape==shape,'actualportbrainshape '+key)
 for direction in ('AN','DN'):
  mask=d[direction+'_capture_step']>=0;require(np.all(d[direction+'_release_step'][mask]==d[direction+'_capture_step'][mask]+2) and np.all(d[direction+'_release_step'][mask]<=d['physical_step_ms'][mask]),'truecapture/release/consumeclock '+direction)
 require(np.sum(compiled['geom_type']==7)==71 and np.all(compiled['geom_rgba'][compiled['geom_type']==7,3]==1),'full71opaque')
 row=next(x for x in s['rows'] if x['condition']==condition);yaw=np.unwrap(g['thorax_yaw']);yaw=yaw-yaw[5799]
 return dict(condition=condition,run=f,result=r,audit=a,identity=ident,trace=t,decisions=d,geometry=g,yaw=yaw,summaryrow=row,summary=s,source_hashes=ledger)
