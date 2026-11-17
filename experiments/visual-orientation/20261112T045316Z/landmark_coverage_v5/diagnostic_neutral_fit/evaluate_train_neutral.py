"""Parent CPU1 exacttrain240neutral-history diagnosis only; no training or physical backend."""
from pathlib import Path
import argparse,hashlib,json,sys,time,os,importlib.util,resource
HERE=Path(__file__).resolve().parent;SCOPE=HERE.parent;POLICY=SCOPE/'policy';VIS=SCOPE.parent

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def fresh(p,obj):
 with Path(p).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--parent-resource-released',action='store_true');args=ap.parse_args()
 if not args.parent_resource_released:print('CODE_ONLY no model/array execution');return
 for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
  if os.environ.get(k)!='1':raise ValueError(k+'=1 required')
 if any((HERE/n).exists() for n in ('INPUTS_BEFORE.json','RESULT.json','OUTPUTS.npz')):raise FileExistsError('inspect existing diagnostic before any rerun')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});start=time.monotonic();pr=read(HERE/'PREREG.json');checkpoint=Path(pr['checkpoint'])
 if sha(checkpoint)!=pr['checkpoint_sha256']:raise ValueError('fixeddiagnosticcheckpointSHA')
 val=read(POLICY/'VALIDATION/RESULT.json');va=read(POLICY/'VALIDATION/VERIFY.json');saved=read(POLICY/'TRAIN/VERIFY_001000.json')
 if not va['integrity_all_pass'] or not all(va['checks'].values()) or va['result_sha256']!=sha(POLICY/'VALIDATION/RESULT.json') or val['checkpoint_sha256']!=pr['checkpoint_sha256'] or saved['sha256']!=pr['checkpoint_sha256'] or not saved['all_passed'] or not all(saved['checks'].values()):raise ValueError('actualfailed-validation and savedreplay identity required')
 frozen=read(POLICY/'FROZEN.json');hashes=dict(frozen['source_hashes'])
 for p in (HERE/'PREREG.json',HERE/'evaluate_train_neutral.py',checkpoint,POLICY/'FROZEN.json',POLICY/'PREFLIGHT.json',POLICY/'TRAIN/RESULT.json',POLICY/'TRAIN/VERIFY_001000.json',POLICY/'VALIDATION/RESULT.json',POLICY/'VALIDATION/VERIFY.json',POLICY/'DATASET.json',POLICY/'train.npz'):hashes[str(p)]=sha(p)
 for p,h in hashes.items():
  if sha(p)!=h:raise ValueError('diagnostic input drift '+p)
 fresh(HERE/'INPUTS_BEFORE.json',hashes)
 # Restore basenamealiases after exact loader/data imports; no physicsimports anywhere.
 sentinel=object();priorcommon=sys.modules.get('common',sentinel);priormodel=sys.modules.get('model',sentinel)
 try:
  original=module('_neutralfit_original_visualmodel',VIS/'visual_policy_v1/model.py');sys.modules['model']=original
  io=module('_neutralfit_coverage_common',POLICY/'common.py');sys.modules['common']=io
  data=module('_neutralfit_coverage_data',POLICY/'data.py');io.cpu1();model,payload=io.load_checked(checkpoint,pr['checkpoint_sha256'],require_final=True)
 finally:
  for name,previous in (('common',priorcommon),('model',priormodel)):
   if previous is sentinel:sys.modules.pop(name,None)
   else:sys.modules[name]=previous
 import numpy as np
 import torch
 model.requires_grad_(False);model.eval();io.guard()
 def tensor_sha(m):
  return {name:hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest() for name,value in m.state_dict().items()}
 before=tensor_sha(model);rgb,manifest=io.library();train=data.arrays('train');meta=read(POLICY/'DATASET.json')['splits']['train']['rows'];indices=[j for j,r in enumerate(meta) if r['motif'] in ('empty','center','occluder_only')]
 if len(indices)!=240 or len({(meta[j]['config_id'],meta[j]['pose_id']) for j in indices})!=80:raise ValueError('240exactneutraltrainpopulation')
 table=data.pair_table();pairid={episode:j for j,pair in enumerate(table) for episode in pair};coverage=read(POLICY/'TRAIN/RESULT.json')['pair_coverage']['pair_selection_counts'];timegrid=train['decision_time'];rows=[];turns=[]
 if len(coverage)!=360:raise ValueError('actual360paircoverage')
 with torch.no_grad():
  for number,j in enumerate(indices):
   cfg=meta[j];x,target,weights=data.batch(rgb,train,[j]);turn,latent,_=original.production(model,x);y=turn[:,0].numpy().copy();labels=target[:,0].numpy()
   if np.max(np.abs(labels))!=0:raise ValueError('neutraldiagnosticrequiresactualzerotarget')
   active_scene={'empty':'off','center':'center','occluder_only':'occluder_only'}[cfg['motif']];ids=train['frame_ids'][train['used_capture_index'],j];active=np.asarray([manifest['rows'][int(fid)]['stimulus']==active_scene for fid in ids],dtype=bool)
   if not active.any():raise ValueError('originalsceneactivewindow empty')
   threshold=.02 if cfg['motif']=='empty' else .05;rms=float(np.sqrt(np.mean(y*y)));objective=float(original.objective(turn,latent,target,weights));pair=pairid[j]
   row=dict(source_train_episode_index=j,config=cfg,pair_id=pair,pair_times_selected=int(coverage[pair]),original_gate_threshold=threshold,original_gate_pass=rms<=threshold,full9_6s_RMS=rms,signed_mean=float(y.mean()),std=float(y.std()),peak_abs_turn=float(np.abs(y).max()),final0_5s_RMS=float(np.sqrt(np.mean(y[timegrid>=9.1]**2))),active_scene=active_scene,active_scene_samples=int(active.sum()),active_scene_RMS=float(np.sqrt(np.mean(y[active]**2))),active_scene_signed_mean=float(y[active].mean()),original_weighted_objective=objective,exact_original_training_frame_schedule=True,batch_size=1,no_training=True)
   rows.append(row);turns.append(y)
   if number==0 or (number+1)%20==0:print(json.dumps(dict(completed=number+1,total=240,motif=cfg['motif'],latest_RMS=rms,wall_s=time.monotonic()-start)),flush=True)
 after=tensor_sha(model)
 if before!=after or any(q.requires_grad for q in model.parameters()) or model.training:raise ValueError('diagnostic unexpectedly modified model')
 for p,h in hashes.items():
  if sha(p)!=h:raise ValueError('input sourcechanged during diagnostic '+p)
 io.guard();aggregates={};comparison={}
 for motif in ('empty','center','occluder_only'):
  rr=[r for r in rows if r['config']['motif']==motif];values=np.array([r['full9_6s_RMS'] for r in rr]);activevals=np.array([r['active_scene_RMS'] for r in rr]);oldrows=[r for r in val['rows'] if r['config']['motif']==motif and r['functional_required']];oldrms=[r['metrics']['turn_RMS'] for r in oldrows]
  aggregates[motif]=dict(train_configpose_count=len(rr),passing=sum(r['original_gate_pass'] for r in rr),failing=sum(not r['original_gate_pass'] for r in rr),RMS_min=float(values.min()),RMS_median=float(np.median(values)),RMS_p90=float(np.quantile(values,.9)),RMS_max=float(values.max()),active_RMS_median=float(np.median(activevals)),active_RMS_max=float(activevals.max()),unselected_pair_rows=sum(r['pair_times_selected']==0 for r in rr))
  comparison[motif]=dict(existing_failed_development_validation_count=len(oldrows),existing_validation_RMS=oldrms,existing_validation_pass_count=sum(bool(r['all_targets_pass']) for r in oldrows),not_fresh_confirmation=True,validation_not_rerun=True)
 output=HERE/'OUTPUTS.npz';np.savez_compressed(output,turn=np.stack(turns,1),time=timegrid,source_train_episode_indices=np.asarray(indices,dtype=np.int64))
 result=dict(kind=pr['kind'],status='DIAGNOSTIC_COMPLETE_NO_TRAINING',checkpoint=str(checkpoint),checkpoint_sha256=pr['checkpoint_sha256'],rows=rows,aggregate=aggregates,existing_development_validation_comparison=comparison,source_hashes=hashes,outputs=dict(path=str(output),sha256=sha(output)),parameter_tensor_sha_before=before,parameter_tensor_sha_after=after,parameter_unchanged=True,whole_history_training_neutral_error_exists=any(not r['original_gate_pass'] for r in rows),model_eval=True,all_parameters_requires_grad_false=True,formal_training_updates=0,optimizer_steps=0,Core_constructions=0,motor_model_loads=0,physics_steps=0,actual_sequences=240,sequence_seconds=9.6,actual_production_batch=1,seconds=time.monotonic()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,sameunderlyingtrajectory_not240independenttrajectories=True,interpretation='Training failures rule out validation-only gap as sole account; passing training and failed heldout supports gap but not unique causal mechanism. Pair counts correlations descriptive; no补训/nothresholdchange.')
 fresh(HERE/'RESULT.json',result);print(json.dumps(dict(status=result['status'],aggregate=aggregates,training_neutral_error_exists=result['whole_history_training_neutral_error_exists'],parameter_unchanged=True,training_updates=0,seconds=result['seconds'],peak_RSS_MiB=result['peak_RSS_MiB']),indent=2),flush=True)
if __name__=='__main__':main()
