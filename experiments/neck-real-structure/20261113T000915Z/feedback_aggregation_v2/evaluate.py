"""Final400 train7/development2 instruction metrics, not new physical validation."""
from common import *
def metrics(np,pred,truth,recon,normalized,dn,cue,weights,old_teacher,decoderzero):
 active=np.abs(cue)>.5;neutral=~active
 return dict(instruction_turn_RMSE=float(np.sqrt(np.mean((pred-truth)**2))),instruction_weighted_turn_MSE=float(np.mean(np.sum(weights*(pred-truth)**2,axis=0))),active_samples=int(active.sum()),active_instruction_sign_fraction=None if not active.any() else float(np.mean(np.sign(pred[active])==np.sign(truth[active]))),active_instruction_RMSE=None if not active.any() else float(np.sqrt(np.mean((pred[active]-truth[active])**2))),neutral_samples=int(neutral.sum()),neutral_turn_RMS=None if not neutral.any() else float(np.sqrt(np.mean(pred[neutral]**2))),feedback_normalized_reconstruction_RMSE=float(np.sqrt(np.mean((recon-normalized)**2))),DN_activity_RMS=float(np.sqrt(np.mean(dn**2))),silent_DN_decoder_output=float(decoderzero),old_retinal_or_actual_teacher_RMSE_descriptive_only=float(np.sqrt(np.mean((pred-old_teacher)**2))),no_acceptance_threshold_invented=True)
def main():
 torch,np=backend();f=frozen();out=HERE/'EVALUATION';out.mkdir(exist_ok=False);path=HERE/'TRAIN/checkpoint_000400.pt';saved=read(HERE/'TRAIN/VERIFY_000400.json')
 if not saved['all_passed'] or saved['sha256']!=sha(path):raise ValueError('Actual final400 savedsourcequeue replay required')
 m,p=load_checkpoint(path,sha(path));from data import arrays,tensors
 from model import python_rollout
 rows=[];outputs={};bp,b,digest=binding()
 with torch.no_grad():
  decoderzero=float(m.decoder(torch.zeros((1,2),dtype=torch.float64))[0,0])
  for split,names in [('train',TRAIN),('development',DEV)]:
   d=arrays(split);x,cue,y,w=tensors(d)
   for j,name in enumerate(names):
    actual=python_rollout(m,x[:,j:j+1],cue[:,j:j+1]);norm=m.normalize(x[:,j:j+1]);metric=metrics(np,actual[0].numpy(),y[:,j:j+1].numpy(),actual[3].numpy(),norm.numpy(),actual[1].numpy(),cue[:,j:j+1].numpy(),w[:,j:j+1].numpy(),d['original_teacher'][:,j:j+1],decoderzero)
    artifact=out/(name+'.npz');np.savez_compressed(artifact,turn=actual[0].numpy(),DN=actual[1].numpy(),AN=actual[2].numpy(),reconstruction=actual[3].numpy(),brain_hidden=actual[4].numpy(),teacher=y[:,j:j+1].numpy(),cue=cue[:,j:j+1].numpy(),weights=w[:,j:j+1].numpy(),original_teacher=d['original_teacher'][:,j:j+1]);outputs[str(artifact)]=sha(artifact)
    if split=='development' or name=='P_empty':
     queue,proof=real_queue_rollout(m,x[:,j:j+1],cue[:,j:j+1],bp,digest)
     if any(float((a-b).abs().max())>1e-10 for a,b in zip(actual,queue)):raise ValueError('Final actualsourcequeue parity failed '+name)
    rows.append(dict(split=split,condition=name,metrics=metric,artifact=str(artifact),actual_samples=1920,input_body_observations_exogenous=True,new_physical_trial=False,fresh_holdout=False))
 verify(f['source_hashes']);verify(f['data_hashes']);save(out/'RESULT.json',dict(status='FINAL400_INSTRUCTION_AGGREGATION_OFFLINE_METRICS_NOT_PHYSICAL',checkpoint=str(path),checkpoint_sha256=sha(path),parent_sha256=PARENT_SHA,identity=m.identity(),source_binding_sha256=digest,rows=rows,output_hashes=outputs,source_hashes=f['source_hashes'],data_hashes=f['data_hashes'],training_updates=400,extra_training_updates=0,silent_DN_decoder_output=decoderzero,full_neck_training_allowed=False,new_body_physics_steps=0,native_DN_or_AN_function_established=False,old_feedback_instability_not_yet_physically_retested=True))
 print(json.dumps(dict(status='FINAL400_OFFLINE_METRICS_REPORTED',rows=rows),indent=2))
def audit():
 torch,np=backend();f=frozen();path=HERE/'EVALUATION/RESULT.json';r=read(path);m,p=load_checkpoint(r['checkpoint'],r['checkpoint_sha256']);from data import arrays,tensors
 from model import make_scan
 checks={};errors={};scan=make_scan(m)
 with torch.no_grad():
  zero=float(m.decoder(torch.zeros((1,2),dtype=torch.float64))[0,0])
  for split,names in [('train',TRAIN),('development',DEV)]:
   d=arrays(split);x,cue,y,w=tensors(d)
   for j,name in enumerate(names):
    row=next(v for v in r['rows'] if v['condition']==name);artifact=Path(row['artifact']);checks[name+'_artifactsha']=sha(artifact)==r['output_hashes'][str(artifact)]
    with np.load(artifact,allow_pickle=False) as z:actual=scan(x[:,j:j+1],cue[:,j:j+1]);errs=[float(np.max(abs(v.numpy()-z[key]))) for key,v in zip(('turn','DN','AN','reconstruction','brain_hidden'),actual)]
    errors[name]=max(errs);checks[name+'_independent_outputs']=max(errs)<=1e-10;metric=metrics(np,actual[0].numpy(),y[:,j:j+1].numpy(),actual[3].numpy(),m.normalize(x[:,j:j+1]).numpy(),actual[1].numpy(),cue[:,j:j+1].numpy(),w[:,j:j+1].numpy(),d['original_teacher'][:,j:j+1],zero);checks[name+'_independent_metrics']=all(metric[k]==row['metrics'][k] or (type(metric[k]) is float and type(row['metrics'][k]) is float and abs(metric[k]-row['metrics'][k])<=1e-10) for k in metric)
 if not all(checks.values()):raise ValueError('Finalmetricsindependentaudit failed '+str(checks))
 verify(f['source_hashes']);verify(f['data_hashes']);save(HERE/'EVALUATION/VERIFY.json',dict(integrity_all_pass=True,checks=checks,max_errors=errors,result_sha256=sha(path),checkpoint_sha256=r['checkpoint_sha256'],parent_sha256=PARENT_SHA,source_hashes=f['source_hashes'],data_hashes=f['data_hashes'],additional_training_updates=0,new_physics_steps=0));print(json.dumps(dict(integrity_all_pass=True,checks=len(checks),max_error=max(errors.values()))))
if __name__=='__main__':raise SystemExit('Use run.py --execute-parent-aggregation')
