"""Final400 knowntrajectoryfit and offsetmetrics, not fresh body success."""
from common import *
def metrics(np,pred,y,current,w,dn,zero):
 quiet=np.abs(y)<=.02;active=np.abs(y)>.1
 return dict(causal_target_RMSE=float(np.sqrt(np.mean((pred-y)**2))),weighted_causal_MSE=float(np.mean(np.sum(w*(pred-y)**2,axis=0))),against_current_teacher_RMSE_descriptive=float(np.sqrt(np.mean((pred-current)**2))),neutral_samples=int(quiet.sum()),neutral_turn_RMS=None if not quiet.any() else float(np.sqrt(np.mean(pred[quiet]**2))),active_samples=int(active.sum()),active_target_RMSE=None if not active.any() else float(np.sqrt(np.mean((pred[active]-y[active])**2))),active_sign_fraction=None if not active.any() else float(np.mean(np.sign(pred[active])==np.sign(y[active]))),DN_activity_RMS=float(np.sqrt(np.mean(dn**2))),silent_DN_decoder_output=float(zero),no_body_acceptance_threshold_invented=True)
def main():
 torch,np=backend();f=frozen();out=HERE/'EVALUATION';out.mkdir(exist_ok=False);path=HERE/'TRAIN/checkpoint_000400.pt';saved=read(HERE/'TRAIN/VERIFY_000400.json')
 if not saved['all_passed'] or saved['sha256']!=sha(path):raise ValueError('Savedfinal400independentreplayrequired')
 m,p=load_checkpoint(path,sha(path));from data import arrays,tensors
 from model import python_rollout
 rows=[];outputs={};bp,b,digest=binding();ds=read(HERE/'DATASET.json');fixed=frozen_tensors(m)
 with torch.no_grad():
  zero=float(m.decoder(torch.zeros((1,2),dtype=torch.float64))[0,0])
  for split,names in [('train',TRAIN),('development',DEV)]:
   d=arrays(split);x,u,y,w=tensors(d)
   for j,name in enumerate(names):
    actual=python_rollout(m,x[:,j:j+1],u[:,j:j+1]);metric=metrics(np,actual[0].numpy(),y[:,j:j+1].numpy(),d['current_desired'][:,j:j+1],w[:,j:j+1].numpy(),actual[1].numpy(),zero);artifact=out/(name+'.npz');np.savez_compressed(artifact,turn=actual[0].numpy(),DN=actual[1].numpy(),AN=actual[2].numpy(),reconstruction=actual[3].numpy(),brain_hidden=actual[4].numpy(),teacher=y[:,j:j+1].numpy(),modalities=u[:,j:j+1].numpy(),current_desired=d['current_desired'][:,j:j+1],weights=w[:,j:j+1].numpy());outputs[str(artifact)]=sha(artifact)
    if split=='development' or name in ('LATEST_V_empty','LATEST_F_left_agree','LATEST_O_left_odorcut'):
     queue,proof=real_queue_rollout(m,x[:,j:j+1],u[:,j:j+1],bp,digest)
     if any(float((a-b).abs().max())>1e-10 for a,b in zip(actual,queue)):raise ValueError('Finalactualqueueparity '+name)
    meta=next(z for z in ds['rows'] if z['name']==name);rows.append(dict(split=split,condition=name,metrics=metric,artifact=str(artifact),source_kind=meta['source_kind'],actual_samples=1920,new_physical_trial=False,fresh_holdout=False,explicitbothneuralzero=name.startswith('ZERO_')))
 check_frozen(m,fixed);verify(f['source_hashes']);verify(f['data_hashes']);ob=odor_binding();save(out/'RESULT.json',dict(status='FINAL400_MULTISENSORY_AGGREGATION_KNOWN_OBSERVATION_METRICS_NOT_BODY',checkpoint=str(path),checkpoint_sha256=sha(path),parent_sha256=PARENT_SHA,visual_sha256=VISION_SHA,odor_sha256=ob['sha256'],source_binding_sha256=digest,identity=m.identity(),rows=rows,output_hashes=outputs,source_hashes=f['source_hashes'],data_hashes=f['data_hashes'],training_updates=400,extra_training_updates=0,all_frozen_tensors_exact=True,silent_DN_decoder_output=zero,runtime_fusion_rule=False,new_body_physics_steps=0,full_neck_training_allowed=False,all_prior_physical_failures_retained=True));print(json.dumps(dict(status='FINAL400_MULTISENSORY_AGGREGATION_METRICS',rows=rows),indent=2))
def audit():
 torch,np=backend();f=frozen();path=HERE/'EVALUATION/RESULT.json';r=read(path);m,p=load_checkpoint(r['checkpoint'],r['checkpoint_sha256']);from data import arrays,tensors
 from model import make_scan
 checks={};errors={};scan=make_scan(m);fixed=frozen_tensors(m)
 with torch.no_grad():
  zero=float(m.decoder(torch.zeros((1,2),dtype=torch.float64))[0,0])
  for split,names in [('train',TRAIN),('development',DEV)]:
   d=arrays(split);x,u,y,w=tensors(d)
   for j,name in enumerate(names):
    row=next(z for z in r['rows'] if z['condition']==name);artifact=Path(row['artifact']);checks[name+'_artifactsha']=sha(artifact)==r['output_hashes'][str(artifact)]
    with np.load(artifact,allow_pickle=False) as z:
     actual=scan(x[:,j:j+1],u[:,j:j+1]);errs=[float(np.max(abs(v.numpy()-z[key]))) for key,v in zip(('turn','DN','AN','reconstruction','brain_hidden'),actual)]
    errors[name]=max(errs);checks[name+'_independentoutputs']=max(errs)<=1e-10;metric=metrics(np,actual[0].numpy(),y[:,j:j+1].numpy(),d['current_desired'][:,j:j+1],w[:,j:j+1].numpy(),actual[1].numpy(),zero);checks[name+'_independentmetrics']=all(metric[k]==row['metrics'][k] or (type(metric[k]) is float and type(row['metrics'][k]) is float and abs(metric[k]-row['metrics'][k])<=1e-10) for k in metric)
 if not all(checks.values()):raise ValueError('Finalrepairmetricsauditfailed '+str(checks))
 check_frozen(m,fixed);verify(f['source_hashes']);verify(f['data_hashes']);save(HERE/'EVALUATION/VERIFY.json',dict(integrity_all_pass=True,checks=checks,max_errors=errors,result_sha256=sha(path),checkpoint_sha256=r['checkpoint_sha256'],parent_sha256=PARENT_SHA,visual_sha256=VISION_SHA,odor_sha256=r['odor_sha256'],source_hashes=f['source_hashes'],data_hashes=f['data_hashes'],additional_training_updates=0,new_physics_steps=0));print(json.dumps(dict(integrity_all_pass=True,checks=len(checks),max_error=max(errors.values()))))
if __name__=='__main__':raise SystemExit('Use explicitflag run.py')
