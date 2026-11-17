"""Stdlib actualmatchedteacher evidence gates; no numericalimports before explicitarray load."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent;SCOPE=HERE.parent;PROJECT=Path('/run/media/sensen/Data2/cell_wound_prototype');BASE=PROJECT/'experiments/banc-walking-recovery/20261110T152714Z'
CASES=('stop','left_pivot','right_pivot');STOP=SCOPE/'stop_shadow_control';PIVOT=SCOPE/'pivot_fixed_shadow_v2';OLDCORE=PROJECT/'experiments/neck-real-structure/20261112T225331Z/body_visual/render.py'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(x,msg):
 if not x:raise ValueError(msg)
def fresh(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def unchanged(h):
 for p,v in h.items():require(sha(p)==v,'Source drift '+p)
def arrays(p):
 import numpy as np
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sources():return [HERE/n for n in ('PREREG.json','ACTUAL_ARCHIVE_HEADERS.json','contracts.py','render.py','plot_sync.py','check_codeonly.py','CODEONLY_CHECK.json')]+[OLDCORE]
def admission():
 checked=read(HERE/'CODEONLY_CHECK.json');require(checked['all_passed'] and all(checked['checks'].values()),'Code qualification');unchanged(checked['source_hashes']);ledger={};summaries={}
 for root,count in ((STOP,2),(PIVOT,4)):
  jobs=read(root/'JOBS.json');require(len(jobs)==count and all(x['exit_code']==0 for x in jobs),'Actualmatchedteacherstagescomplete');pre=read(root/'PREFLIGHT.json');require(pre['all_passed'] and all(pre['checks'].values()) and pre['physical_integration_steps']==0,'Actualzerointegrationpreflight');summaries[root.name]=read(root/'SUMMARY.json')
  for p in (root/'JOBS.json',root/'PREFLIGHT.json',root/'SUMMARY.json',root/'PREREG.json'):ledger[str(p)]=sha(p)
 require(summaries[STOP.name]['new_support_full_window_qualified'] and summaries[STOP.name]['stop_speed_PASS'] is None,'Stopsupportonly_nospeedgate');unchanged(summaries[STOP.name]['source_hashes'])
 for case in CASES:
  root=STOP if case=='stop' else PIVOT;folder=root/'trials'/case;r=read(folder/'RESULT.json');a=read(folder/'VERIFY.json');require(r['kind']==('STOP_FIXED_SHADOW_MATCHED_CONTROL_V1' if case=='stop' else 'FIXED_PRIVATE_SHADOW_PIVOT_TEACHER_V2') and r['condition']==case,'Actualteacherkind');require(r['control']=='ALGORITHM_TEACHER_ONLY_NOT_ANN' and r['no_neural_models'] and r['no_optimizer'],'TeacheronlynotANN');require(r['full_duration'] and r['actual_ms']==15400 and r['actual_native_steps']==1232000 and r['early_stop'] is None,'Complete15.4sactualteacher');require(a['integrity_all_pass'] and all(a['checks'].values()) and a['result_sha256']==sha(folder/'RESULT.json'),'Independentactualsaved26checks')
  require(all(r['candidate_metrics']['gates'].values()) and r['native']['post_nonfoot_peak_native']==0 and r['native']['extra_force_max']==0 and r['native']['post_native_steps']==768000,'Actualsupportpivotgates_nativeNFextra');require((r['candidate_metrics']['task_pass_candidate'] is None) if case=='stop' else (r['candidate_metrics']['task_pass_candidate'] is True),'No fabricatedstopPASS');unchanged(r['source_hashes']);ledger.update(r['source_hashes'])
  matched=root/('MATCHED_COMPARISON.json' if case=='stop' else 'MATCHED_'+case+'.json');m=read(matched);require(m['integrity_all_pass'] and all(m['checks'].values()),'ActualmatchedprefixPD_RNGfirstaction');ledger[str(matched)]=sha(matched)
  for key in ('trace','geometry','teacher_samples','diagnostics'):item=r[key];require(sha(item['path'])==item['sha256'],'Actualteacherartifact '+key);ledger[item['path']]=item['sha256']
  for name in ('RESULT.json','VERIFY.json','IDENTITY.json','HANDOVER.json','BODY_LENGTH.json','INITIAL_STATE.json','runtime_geometry.npz'):ledger[str(folder/name)]=sha(folder/name)
 for p in sources():require(p.is_file(),'Evidencecoremissing '+str(p));ledger[str(p)]=sha(p)
 return ledger

def load(case):
 import numpy as np
 require(case in CASES,'Registeredthreecases');ledger=admission();root=STOP if case=='stop' else PIVOT;folder=root/'trials'/case;r=read(folder/'RESULT.json');a=read(folder/'VERIFY.json');ident=read(folder/'IDENTITY.json');body=read(folder/'BODY_LENGTH.json');hand=read(folder/'HANDOVER.json');t=arrays(folder/'trace_1ms.npz');g=arrays(folder/'geometry_1ms.npz');d=arrays(folder/'teacher_samples_5ms.npz');compiled=arrays(folder/'runtime_geometry.npz');diagnostics=[json.loads(line) for line in (folder/'teacher_diagnostics.jsonl').read_text().splitlines() if line]
 require(t['qpos'].shape==(15400,49) and t['qvel'].shape==(15400,48) and g['thorax_yaw'].shape==(14400,) and g['head_rotation'].shape==(14400,3,3) and d['Y42'].shape==(2880,42),'Actualteacherschema');require(np.max(abs(t['time']-np.arange(1,15401)*.001))<1e-8 and np.max(abs(g['time']-np.arange(1001,15401)*.001))<1e-8 and np.max(abs(d['time']-(1+np.arange(2880)*.005)))<1e-8,'Causalteacherclock');require(np.array_equal(t['control_source'],np.r_[np.zeros(1000),np.full(4800,2),np.full(9600,6)]) and np.array_equal(d['control_source'],np.r_[np.full(960,2),np.full(1920,6)]),'Actualteacherphases');require(len(diagnostics)==2880 and all(x['physical_ms']==1000+5*i for i,x in enumerate(diagnostics)),'Actualdiagnosticsclock');require(np.all(np.isfinite(t['qpos'])) and np.all(np.isfinite(t['qvel'])),'Actualfiniteposes');require(np.sum(compiled['geom_type']==7)==71 and np.all(compiled['geom_rgba'][compiled['geom_type']==7,3]==1),'71opaque');require(body['length_native']==r['body_length']['length_native'] and body['absolute_SI']=='NOT_ESTABLISHED','Compiledbodylength_scale');require(hand['actual_state_exact'] and hand['no_reset'] and hand['no_neural_state_exists'],'ActualstatecontinuitynoteacherANN');require(np.allclose(t['com'][5799],r['candidate_start']['com'],atol=0,rtol=0),'CandidateboundaryCOM')
 q=t['qpos'][:,3:7];yaw=np.unwrap(np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2)));delta=yaw-yaw[5799];drift=np.linalg.norm(t['com'][:,:2]-t['com'][5799,:2],axis=1)/body['length_native'];events=r['candidate_metrics']['pivot_cycle_events'];counts=np.asarray([[sum(e['leg_index']==leg and e['time']<=tt+1e-10 for e in events) for leg in range(6)] for tt in t['time']],dtype=int);require(np.array_equal(counts[-1],r['candidate_metrics']['pivot_cycle_counts']),'Actualeventtimecyclecounts');require(abs(float(delta[-1])-r['candidate_metrics']['yaw_change_rad'])<1e-10 and abs(float(drift[5800:].max())-r['candidate_metrics']['max_COM_xy_drift_bodylength'])<1e-10,'Independentactualyawdrift')
 return dict(case=case,folder=folder,result=r,audit=a,identity=ident,body_length=body,trace=t,geometry=g,decisions=d,diagnostics=diagnostics,yaw=delta,drift=drift,cycle_counts=counts,source_hashes=ledger)
