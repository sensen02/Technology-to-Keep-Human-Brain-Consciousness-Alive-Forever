"""Actualderivedscene/sourceNN qualification; stdlib until explicitarray load."""
from pathlib import Path
import json,hashlib,importlib.util,sys
HERE=Path(__file__).resolve().parent;SCOPE=HERE.parent;PROJECT=Path('/run/media/sensen/Data2/cell_wound_prototype');OLD=SCOPE/'terrain_neural_baseline';REPAIR=SCOPE/'terrain_start_recovery_v1';CASES=('plane_control','flat_box_control','mild_blocks_seed17_start_repair_v1');ROUGH=CASES[2]
BRAIN='92dc0d243dad42b1df284ab797fa3dcaa13ca2d9ebfbb4cb434cebba7c9fe490';ODOR='e21dfb68e38760a6c74a871cfa3686cb79d1512b1ca60b5432397a456e473bfc';VISION='996980806ac107137d6d5f4743de207fb3350e84a8a145dc5f52b8807398bb87';MOTOR='789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd';NECK='9e88a04dd91aa6f9c1c1c68e0562325205e0f36f5cdd9ae41d1dd46c8171dc23'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(v,msg):
 if not v:raise ValueError(msg)
def fresh(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def unchanged(h):
 for p,d in h.items():require(sha(p)==d,'Source drift '+p)
def module(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def arrays(p):
 import numpy as np
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def root(case):return REPAIR if case==ROUGH else OLD
def own_sources():return [HERE/n for n in ('PREREG.json','ACTUAL_ARCHIVE_HEADERS.json','contracts.py','render.py','plot_sync.py','check_codeonly.py','CODEONLY_CHECK.json')]+[OLD/'runtime.py',REPAIR/'runtime.py',SCOPE/'terrain_body_baseline/terrain_core.py']
def admission():
 c=read(HERE/'CODEONLY_CHECK.json');require(c['all_passed'] and all(c['checks'].values()),'CODEONLYproof');unchanged(c['source_hashes']);ledger={}
 oldjobs=read(OLD/'JOBS.json');newjobs=read(REPAIR/'JOBS.json');require(len(oldjobs)>=5 and [x['exit_code'] for x in oldjobs[:4]]==[0,0,0,0] and oldjobs[4]['exit_code']!=0,'Archivedtwoqualifiedcontrolsplusretainedguardfailure');require(len(newjobs)==2 and all(x['exit_code']==0 for x in newjobs),'Newroughphysicalandauditbothcomplete')
 for p in (OLD/'JOBS.json',REPAIR/'JOBS.json',OLD/'PREFLIGHT.json',REPAIR/'PREFLIGHT.json',REPAIR/'SUMMARY.json',REPAIR/'PREPARED/MODEL_LEDGER.json',OLD.parent/'terrain_body_baseline/PREPARED/MODEL_LEDGER.json'):ledger[str(p)]=sha(p)
 for case in CASES:
  f=root(case)/'trials'/case;r=read(f/'RESULT.json');a=read(f/'VERIFY.json');i=read(f/'IDENTITY.json');require(r['kind']==('SOURCE_NECK_ROUGH_TERRAIN_START_REPAIR_V1' if case==ROUGH else 'SOURCE_NECK_ROUGH_TERRAIN_BASELINE_V1') and r['terrain_case']==case,'Actualcasekind');require(a['integrity_all_pass'] and all(a['checks'].values()) and a['result_sha256']==sha(f/'RESULT.json'),'ActualfreshNNterrainaudit')
  for k,h in [('brain_checkpoint_sha256',BRAIN),('odor_checkpoint_sha256',ODOR),('visual_checkpoint_sha256',VISION),('checkpoint_sha256',MOTOR),('neck_source_binding_sha256',NECK)]:require(r[k]==h,'Actualsourcepolicyidentity '+k)
  require(r['state_total']==336 and r['neural_records']==3080 and r['physical_steps']==1232000 and r['native']['actual_native_steps']==1232000 and r['native']['post_native_steps']==768000,'Full336bodyduration');require(a['terrain_geometry_and_summary_replayed'] and a['checks']['warm_flat_checked_before_ANN'] and a['checks']['actual_terrain_case'] and a['checks']['identity_case_and_actual_XML_tied_to_ledger'],'Actualderivedgeometryandwarmproof');require(r['real_terrain_RGB_used_by_vision'] and not r['terrain_visual_input_forced_zero'] and not r['mechanical_hook_or_pulvilli_claim'],'RealterrainRGB/nohookclaim');require(r['actual_xml']==i['xml'] and sha(r['actual_xml'])==r['actual_xml_sha256']==i['xml_sha256'],'ActualXMLidentity');unchanged(r['source_hashes']);ledger.update(r['source_hashes']);unchanged(i['source_hashes']);ledger.update(i['source_hashes'])
  for filename,key in [('trace_1ms.npz','trace'),('neural_decisions.npz','neural_decisions'),('geometry_1ms.npz','geometry'),('runtime_geometry.npz','runtime_geometry'),('capture_rgb.npz','capture_rgb'),('CAPTURES.json','captures_metadata'),('terrain_1ms.npz','terrain_trace'),('INITIAL_TERRAIN.json','initial_terrain'),('WARM_FLAT_CHECK.json','warm_flat_proof')]:require(sha(f/filename)==r[key]['sha256'],'ArtifactSHA '+filename);ledger[str(f/filename)]=sha(f/filename)
  for name in ('RESULT.json','VERIFY.json','IDENTITY.json','HANDOVER.json','INITIAL_STATE.json'):ledger[str(f/name)]=sha(f/name)
 for p in own_sources():require(p.is_file(),'Evidencefilemissing '+str(p));ledger[str(p)]=sha(p)
 return ledger

def load(case):
 import numpy as np
 require(case in CASES,'Registeredcase');ledger=admission();folder=root(case)/'trials'/case;r=read(folder/'RESULT.json');a=read(folder/'VERIFY.json');i=read(folder/'IDENTITY.json');t=arrays(folder/'trace_1ms.npz');d=arrays(folder/'neural_decisions.npz');g=arrays(folder/'geometry_1ms.npz');terrain=arrays(folder/'terrain_1ms.npz');rgb=arrays(folder/'capture_rgb.npz')['rgb'];caps=read(folder/'CAPTURES.json');compiled=arrays(folder/'runtime_geometry.npz')
 require(t['qpos'].shape==(15400,49) and t['qvel'].shape==(15400,48) and d['full_hidden_before'].shape==d['full_hidden_after'].shape==(3080,336) and rgb.shape==(770,2,32,32,3),'Actual336fullarchiveschema');require(np.max(abs(t['time']-np.arange(1,15401)*.001))<1e-8 and np.max(abs(d['time']-np.arange(3080)*.005))<1e-8 and np.array_equal(g['time'],t['time']) and np.array_equal(terrain['time'],t['time']),'Actualclocks');require(not d['image_valid'][:4].any() and np.all(d['image_capture_id'][:4]==-1) and np.all(d['image_valid'][4:]),'First20msinvalid');require(np.allclose(terrain['com_world_xyz'],t['com'],atol=0,rtol=0),'TerrainCOMtiedtophysical');require(np.sum(compiled['geom_type']==7)==71 and np.all(compiled['geom_rgba'][compiled['geom_type']==7,3]==1),'Full71opaque')
 for index,c in enumerate(caps):require(c['capture_id']==index and c['capture_time_s']==index*.02 and c['release_time_s']==index*.02+.02 and hashlib.sha256(rgb[index].tobytes()).hexdigest()==c['rgb_sha256'],'Actualpixelmetadata')
 yaw=np.unwrap(g['thorax_yaw']);yaw-=yaw[5799]
 return dict(case=case,folder=folder,result=r,audit=a,identity=i,trace=t,decisions=d,geometry=g,terrain=terrain,rgb=rgb,captures=caps,yaw=yaw,source_hashes=ledger)

def physical_context(case):
 # No policyload. Use exactderivedruntime context/pin so actualsceneXML cannot disappear.
 if case==ROUGH:runtime=module('_terrainvideo_exact_repair_runtime',REPAIR/'runtime.py');ns,state=runtime.context(audit=True)
 else:runtime=module('_terrainvideo_exact_baseline_runtime',OLD/'runtime.py');ns,state=runtime.context(case,audit=True)
 sw,identity=ns['pin']();return sw,identity,ns,state
