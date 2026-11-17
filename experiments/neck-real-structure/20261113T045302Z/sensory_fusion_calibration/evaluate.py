"""Final400 archive-dependent fusion metrics, independent audit, not new multisensory body proof."""
from common import *
def metrics(np,pred,target,current,weights,dn,zero):
 a=np.abs(target);neutral=a<=.02;active=a>.1
 def error(mask):return None if not mask.any() else float(np.sqrt(np.mean((pred[mask]-target[mask])**2)))
 return dict(causal_target_RMSE=float(np.sqrt(np.mean((pred-target)**2))),weighted_causal_MSE=float(np.mean(np.sum(weights*(pred-target)**2,axis=0))),against_current_desired_RMSE_descriptive=float(np.sqrt(np.mean((pred-current)**2))),neutral_samples=int(neutral.sum()),neutral_output_RMS=None if not neutral.any() else float(np.sqrt(np.mean(pred[neutral]**2))),neutral_target_RMSE=error(neutral),active_samples=int(active.sum()),active_target_RMSE=error(active),active_sign_fraction=None if not active.any() else float(np.mean(np.sign(pred[active])==np.sign(target[active]))),DN_activity_RMS=float(np.sqrt(np.mean(dn**2))),silent_DN_decoder_output=float(zero),no_new_acceptance_threshold=True)
def main():
 torch,np=backend();f=frozen();out=HERE/'EVALUATION';out.mkdir(exist_ok=False);path=HERE/'TRAIN/checkpoint_000400.pt';saved=read(HERE/'TRAIN/VERIFY_000400.json')
 if not saved['all_passed'] or saved['sha256']!=sha(path):raise ValueError('Actualfinal400savedreplay required')
 m,p=load_checkpoint(path,sha(path));from data import arrays,tensors
 from model import python_rollout
 rows=[];artifacts={};bp,b,digest=binding();ds=read(HERE/'DATASET.json');fixed=frozen_tensors(m)
 with torch.no_grad():
  silent=float(m.decoder(torch.zeros((1,2),dtype=torch.float64))[0,0])
  for split,names in (('train',TRAIN),('development',DEV)):
   d=arrays(split);x,u,y,w=tensors(d)
   for j,name in enumerate(names):
    actual=python_rollout(m,x[:,j:j+1],u[:,j:j+1]);score=metrics(np,actual[0].numpy(),y[:,j:j+1].numpy(),d['current_desired'][:,j:j+1],w[:,j:j+1].numpy(),actual[1].numpy(),silent);pathcase=out/(name+'.npz');np.savez_compressed(pathcase,turn=actual[0].numpy(),DN=actual[1].numpy(),AN=actual[2].numpy(),reconstruction=actual[3].numpy(),brain_hidden=actual[4].numpy(),teacher=y[:,j:j+1].numpy(),modalities=u[:,j:j+1].numpy(),current_desired=d['current_desired'][:,j:j+1],weights=w[:,j:j+1].numpy());artifacts[str(pathcase)]=sha(pathcase)
    if name in ('VIS_V_empty','B_left_pulse','C_left_near','PAIR_V_left_pulse') or split=='development':
     queue,proof=real_queue_rollout(m,x[:,j:j+1],u[:,j:j+1],bp,digest)
     if any(float((l-r).abs().max())>1e-10 for l,r in zip(actual,queue)):raise ValueError('Independentactualsourcequeue '+name)
    meta=next(v for v in ds['rows'] if v['split']==split and v['condition']==name);rows.append(dict(split=split,condition=name,metrics=score,artifact=str(pathcase),input_source_kind=meta['input_source_kind'],actual_samples=1920,new_physical_trial=False,fresh_holdout=False,simultaneous_multisensory_physics=False))
 check_frozen(m,fixed);verify(f['source_hashes']);verify(f['data_hashes']);save(out/'RESULT.json',dict(status='FINAL400_SENSORY_FUSION_ENGINEERING_METRICS_NOT_BODY',checkpoint=str(path),checkpoint_sha256=sha(path),parent_sha256=PARENT_SHA,visual_sha256=VISION_SHA,odor_sha256=ODOR_SHA,source_binding_sha256=digest,identity=m.identity(),rows=rows,output_hashes=artifacts,source_hashes=f['source_hashes'],data_hashes=f['data_hashes'],training_updates=400,extra_training_updates=0,all_frozen_feedback_tensors_exact=True,silent_DN_decoder_output=silent,new_body_physics_steps=0,full_neck_training_allowed=False,native_sensory_preference_established=False,old_D_delayedcut_failure_retained=True));print(json.dumps(dict(status='FINAL400_SENSORY_FUSION_METRICS',rows=rows),indent=2))
def audit():
 torch,np=backend();f=frozen();rp=HERE/'EVALUATION/RESULT.json';r=read(rp);m,p=load_checkpoint(r['checkpoint'],r['checkpoint_sha256']);from data import arrays,tensors
 from model import make_scan
 checks={};errors={};scan=make_scan(m);fixed=frozen_tensors(m)
 with torch.no_grad():
  silent=float(m.decoder(torch.zeros((1,2),dtype=torch.float64))[0,0])
  for split,names in (('train',TRAIN),('development',DEV)):
   d=arrays(split);x,u,y,w=tensors(d)
   for j,name in enumerate(names):
    row=next(v for v in r['rows'] if v['split']==split and v['condition']==name);artifact=Path(row['artifact']);checks[name+'_artifactSHA']=sha(artifact)==r['output_hashes'][str(artifact)]
    with np.load(artifact,allow_pickle=False) as z:
     actual=scan(x[:,j:j+1],u[:,j:j+1]);errs=[float(np.max(abs(value.numpy()-z[k]))) for k,value in zip(('turn','DN','AN','reconstruction','brain_hidden'),actual)]
    errors[name]=max(errs);checks[name+'_independent_scan_outputs']=max(errs)<=1e-10;score=metrics(np,actual[0].numpy(),y[:,j:j+1].numpy(),d['current_desired'][:,j:j+1],w[:,j:j+1].numpy(),actual[1].numpy(),silent);checks[name+'_independent_metrics']=all(score[k]==row['metrics'][k] or (type(score[k]) is float and type(row['metrics'][k]) is float and abs(score[k]-row['metrics'][k])<=1e-10) for k in score)
 if not all(checks.values()):raise ValueError('Independentfusionfinalmetricsaudit '+str(checks))
 check_frozen(m,fixed);verify(f['source_hashes']);verify(f['data_hashes']);save(HERE/'EVALUATION/VERIFY.json',dict(integrity_all_pass=True,checks=checks,max_errors=errors,result_sha256=sha(rp),checkpoint_sha256=r['checkpoint_sha256'],parent_sha256=PARENT_SHA,visual_sha256=VISION_SHA,odor_sha256=ODOR_SHA,source_binding_sha256=NECK_SHA,source_hashes=f['source_hashes'],data_hashes=f['data_hashes'],all_frozen_tensors_exact=True,additional_training_updates=0,new_physics_steps=0));print(json.dumps(dict(integrity_all_pass=True,checks=len(checks),max_error=max(errors.values()))))
if __name__=='__main__':raise SystemExit('Use explicitflag run.py')
