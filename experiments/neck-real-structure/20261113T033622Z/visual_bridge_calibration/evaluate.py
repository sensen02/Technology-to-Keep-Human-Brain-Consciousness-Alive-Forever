"""Final300 known-archive visual amplitude/causal-delay metrics, not new body validation."""
from common import *
def metrics(np,pred,target,current,weights,dn,zero):
 a=np.abs(target);neutral=a<=.02;active=a>.1;transition=(a>.02)&(a<=.1)
 def rmse(mask):return None if not mask.any() else float(np.sqrt(np.mean((pred[mask]-target[mask])**2)))
 return dict(causal_target_RMSE=float(np.sqrt(np.mean((pred-target)**2))),weighted_causal_MSE=float(np.mean(np.sum(weights*(pred-target)**2,axis=0))),against_current_visual_RMSE_descriptive=float(np.sqrt(np.mean((pred-current)**2))),neutral_samples=int(neutral.sum()),neutral_output_RMS=None if not neutral.any() else float(np.sqrt(np.mean(pred[neutral]**2))),neutral_target_RMSE=rmse(neutral),active_samples=int(active.sum()),active_target_RMSE=rmse(active),active_sign_fraction=None if not active.any() else float(np.mean(np.sign(pred[active])==np.sign(target[active]))),transition_target_RMSE=rmse(transition),DN_activity_RMS=float(np.sqrt(np.mean(dn**2))),silent_DN_decoder_output=float(zero),no_new_acceptance_threshold=True)
def main():
 torch,np=backend();f=frozen();out=HERE/'EVALUATION';out.mkdir(exist_ok=False);path=HERE/'TRAIN/checkpoint_000300.pt';saved=read(HERE/'TRAIN/VERIFY_000300.json')
 if not saved['all_passed'] or saved['sha256']!=sha(path):raise ValueError('Actualfinal300savedreplay required')
 m,p=load_checkpoint(path,sha(path));from data import arrays,tensors
 from model import python_rollout
 rows=[];outputs={};bp,b,digest=binding();dataset=read(HERE/'DATASET.json');fixed=frozen_tensors(m)
 with torch.no_grad():
  zero=float(m.decoder(torch.zeros((1,2),dtype=torch.float64))[0,0])
  for split,names in [('train',TRAIN),('development',DEV)]:
   d=arrays(split);x,u,y,w=tensors(d)
   for j,name in enumerate(names):
    actual=python_rollout(m,x[:,j:j+1],u[:,j:j+1]);metric=metrics(np,actual[0].numpy(),y[:,j:j+1].numpy(),d['current_visual'][:,j:j+1],w[:,j:j+1].numpy(),actual[1].numpy(),zero);artifact=out/(name+'.npz');np.savez_compressed(artifact,turn=actual[0].numpy(),DN=actual[1].numpy(),AN=actual[2].numpy(),reconstruction=actual[3].numpy(),brain_hidden=actual[4].numpy(),teacher=y[:,j:j+1].numpy(),visual_input=u[:,j:j+1].numpy(),current_visual=d['current_visual'][:,j:j+1],weights=w[:,j:j+1].numpy());outputs[str(artifact)]=sha(artifact)
    if split=='development' or name in ('V_empty','P_empty'):
     queue,proof=real_queue_rollout(m,x[:,j:j+1],u[:,j:j+1],bp,digest)
     if any(float((a-b).abs().max())>1e-10 for a,b in zip(actual,queue)):raise ValueError('Actualsourcequeueparity '+name)
    meta=next(z for z in dataset['rows'] if z['condition']==name);rows.append(dict(split=split,condition=name,metrics=metric,artifact=str(artifact),input_source_kind=meta['input_source_kind'],actual_paired_RGB=meta['actual_paired_RGB'],actual_samples=1920,new_physical_trial=False,fresh_holdout=False,neural_zero_aux_not_visual_behavior=name in AUX))
 check_frozen(m,fixed);verify(f['source_hashes']);verify(f['data_hashes']);save(out/'RESULT.json',dict(status='FINAL300_ACTUALRGB_SCALAR_CALIBRATION_METRICS_NOT_NEW_BODY',checkpoint=str(path),checkpoint_sha256=sha(path),parent_sha256=PARENT_SHA,visual_sha256=VISION_SHA,source_binding_sha256=digest,identity=m.identity(),rows=rows,output_hashes=outputs,source_hashes=f['source_hashes'],data_hashes=f['data_hashes'],training_updates=300,extra_training_updates=0,all_frozen_feedback_tensors_exact=True,silent_DN_decoder_output=zero,new_body_physics_steps=0,full_neck_training_allowed=False,native_visual_or_AN_function_established=False));print(json.dumps(dict(status='FINAL300_VISUAL_CALIBRATION_METRICS',rows=rows),indent=2))
def audit():
 torch,np=backend();f=frozen();rp=HERE/'EVALUATION/RESULT.json';r=read(rp);m,p=load_checkpoint(r['checkpoint'],r['checkpoint_sha256']);from data import arrays,tensors
 from model import make_scan
 checks={};errors={};scan=make_scan(m);fixed=frozen_tensors(m)
 with torch.no_grad():
  zero=float(m.decoder(torch.zeros((1,2),dtype=torch.float64))[0,0])
  for split,names in [('train',TRAIN),('development',DEV)]:
   d=arrays(split);x,u,y,w=tensors(d)
   for j,name in enumerate(names):
    row=next(z for z in r['rows'] if z['condition']==name);artifact=Path(row['artifact']);checks[name+'_artifact_sha']=sha(artifact)==r['output_hashes'][str(artifact)]
    with np.load(artifact,allow_pickle=False) as z:
     actual=scan(x[:,j:j+1],u[:,j:j+1]);errs=[float(np.max(abs(value.numpy()-z[key]))) for key,value in zip(('turn','DN','AN','reconstruction','brain_hidden'),actual)]
    errors[name]=max(errs);checks[name+'_independent_outputs']=max(errs)<=1e-10;metric=metrics(np,actual[0].numpy(),y[:,j:j+1].numpy(),d['current_visual'][:,j:j+1],w[:,j:j+1].numpy(),actual[1].numpy(),zero);checks[name+'_independent_metrics']=all(metric[k]==row['metrics'][k] or (type(metric[k]) is float and type(row['metrics'][k]) is float and abs(metric[k]-row['metrics'][k])<=1e-10) for k in metric)
 if not all(checks.values()):raise ValueError('Independentvisualcalibrationmetric audit failed '+str(checks))
 check_frozen(m,fixed);verify(f['source_hashes']);verify(f['data_hashes']);save(HERE/'EVALUATION/VERIFY.json',dict(integrity_all_pass=True,checks=checks,max_errors=errors,result_sha256=sha(rp),checkpoint_sha256=r['checkpoint_sha256'],parent_sha256=PARENT_SHA,visual_sha256=VISION_SHA,source_binding_sha256=NECK_SHA,source_hashes=f['source_hashes'],data_hashes=f['data_hashes'],all_frozen_tensors_exact=True,additional_training_updates=0,new_physics_steps=0));print(json.dumps(dict(integrity_all_pass=True,checks=len(checks),max_error=max(errors.values()))))
if __name__=='__main__':raise SystemExit('Use explicitflag run.py')
