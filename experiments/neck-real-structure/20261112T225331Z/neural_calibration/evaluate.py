"""Final200 engineering train/development metrics; archived data are not new body trials."""
from common import *
def metrics(np,pred,truth,recon,normalized,dn):
 active=np.abs(truth)>.1;neutral=np.abs(truth)<=.02
 return dict(turn_RMSE=float(np.sqrt(np.mean((pred-truth)**2))),active_teacher_samples=int(active.sum()),active_sign_fraction=None if not active.any() else float(np.mean(np.sign(pred[active])==np.sign(truth[active]))),teacher_neutral_samples=int(neutral.sum()),neutral_turn_RMS=None if not neutral.any() else float(np.sqrt(np.mean(pred[neutral]**2))),feedback_normalized_RMSE=float(np.sqrt(np.mean((recon-normalized)**2))),DN_activity_RMS=float(np.sqrt(np.mean(dn**2))),no_acceptance_threshold_invented=True)
def main():
 torch,np=backend();f=frozen();out=HERE/'EVALUATION';out.mkdir(exist_ok=False);path=HERE/'TRAIN/checkpoint_000200.pt';saved=read(HERE/'TRAIN/VERIFY_000200.json')
 if not saved['all_passed'] or saved['sha256']!=sha(path):raise ValueError('Saved final200 independent replay required')
 m,p=load_checkpoint(path,sha(path));from data import arrays,tensors
 from model import python_rollout
 rows=[];outputs={};ds=read(HERE/'DATASET.json');bp,b,digest=binding()
 with torch.no_grad():
  for split,names in [('train',TRAIN),('development',DEV)]:
   d=arrays(split);x,cue,y=tensors(d)
   for j,name in enumerate(names):
    actual=python_rollout(m,x[:,j:j+1],cue[:,j:j+1]);normalized=m.normalize(x[:,j:j+1]);metric=metrics(np,actual[0].numpy(),y[:,j:j+1].numpy(),actual[3].numpy(),normalized.numpy(),actual[1].numpy());artifact=out/(name+'.npz');np.savez_compressed(artifact,turn=actual[0].numpy(),DN=actual[1].numpy(),AN=actual[2].numpy(),reconstruction=actual[3].numpy(),brain_hidden=actual[4].numpy(),teacher=y[:,j:j+1].numpy(),cue=cue[:,j:j+1].numpy());outputs[str(artifact)]=sha(artifact)
    # Development check against actualsource FIFO in independentpass, not feedback on new physics.
    if split=='development':
     queue,proof=real_queue_rollout(m,x[:,j:j+1],cue[:,j:j+1],bp,digest)
     if any(float((a-b).abs().max())>1e-10 for a,b in zip(actual,queue)):raise ValueError('Development actual sourcequeue parity failed')
    rows.append(dict(split=split,condition=name,metrics=metric,artifact=str(artifact),actual_samples=1920,archived_body_observations_are_exogenous=True,closed_loop_body_physics=False,fresh_holdout=False))
 verify(f['source_hashes']);verify(f['data_hashes']);save(out/'RESULT.json',dict(status='FINAL200_ENGINEERING_OFFLINE_METRICS_REPORTED_NOT_NEW_WALKING',checkpoint=str(path),checkpoint_sha256=sha(path),identity=m.identity(),source_binding_sha256=digest,rows=rows,output_hashes=outputs,source_hashes=f['source_hashes'],data_hashes=f['data_hashes'],training_updates=200,extra_training_updates=0,motor_model_loads=0,new_physics_steps=0,full_neck_training_allowed=False,native_DN_function_established=False,native_AN_tuning_established=False))
 print(json.dumps(dict(status='FINAL200_ENGINEERING_METRICS_REPORTED',rows=rows),indent=2))
def audit():
 torch,np=backend();f=frozen();rp=HERE/'EVALUATION/RESULT.json';r=read(rp);m,p=load_checkpoint(r['checkpoint'],r['checkpoint_sha256']);from data import arrays,tensors
 from model import make_scan
 checks={};errors={};scan=make_scan(m)
 with torch.no_grad():
  for split,names in [('train',TRAIN),('development',DEV)]:
   d=arrays(split);x,cue,y=tensors(d)
   for j,name in enumerate(names):
    row=next(v for v in r['rows'] if v['condition']==name);artifact=Path(row['artifact']);checks[name+'_artifact_sha']=sha(artifact)==r['output_hashes'][str(artifact)]
    with np.load(artifact,allow_pickle=False) as z:actual=scan(x[:,j:j+1],cue[:,j:j+1]);fields=('turn','DN','AN','reconstruction','brain_hidden');errs=[float(np.max(abs(v.numpy()-z[key]))) for key,v in zip(fields,actual)]
    errors[name]=max(errs);checks[name+'_independent_scan_actualoutputs']=max(errs)<=1e-10;metric=metrics(np,actual[0].numpy(),y[:,j:j+1].numpy(),actual[3].numpy(),m.normalize(x[:,j:j+1]).numpy(),actual[1].numpy());checks[name+'_metrics_exact']=all((metric[k]==row['metrics'][k]) or (type(metric[k]) is float and type(row['metrics'][k]) is float and abs(metric[k]-row['metrics'][k])<=1e-10) for k in metric)
 if not all(checks.values()):raise ValueError('Final evaluation independent audit failed '+str(checks))
 verify(f['source_hashes']);verify(f['data_hashes']);save(HERE/'EVALUATION/VERIFY.json',dict(integrity_all_pass=True,checks=checks,max_errors=errors,result_sha256=sha(rp),checkpoint_sha256=r['checkpoint_sha256'],source_hashes=f['source_hashes'],data_hashes=f['data_hashes'],additional_training_updates=0,new_physics_steps=0));print(json.dumps(dict(integrity_all_pass=True,checks=len(checks),max_error=max(errors.values()))))
if __name__=='__main__':raise SystemExit('Use run.py --execute-parent-calibration')
