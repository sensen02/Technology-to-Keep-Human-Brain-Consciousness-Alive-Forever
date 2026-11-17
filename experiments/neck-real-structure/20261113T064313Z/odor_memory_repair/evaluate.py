"""Final250 known-input sensory repair metrics; no freshbody or causalqualification claim."""
from common import *
def metrics(np,pred,target,original,weights,row):
 active=np.abs(target)>.1;quiet=np.abs(target)<=.02;out=dict(target_RMSE=float(np.sqrt(np.mean((pred-target)**2))),weighted_target_MSE=float(np.mean(np.sum(weights*(pred-target)**2,axis=0))),original66f_fidelity_RMSE_descriptive=float(np.sqrt(np.mean((pred-original)**2))),quiet_samples=int(quiet.sum()),quiet_output_RMS=None if not quiet.any() else float(np.sqrt(np.mean(pred[quiet]**2))),active_samples=int(active.sum()),active_sign_fraction=None if not active.any() else float(np.mean(np.sign(pred[active])==np.sign(target[active]))),no_new_physical_acceptance_threshold=True)
 cut=row.get('withdrawal_target_cut_s')
 if cut is not None:
  t=np.arange(1920)*.005;mask=(t>=cut+.5-1e-9)&(t<cut+1.-1e-9);out.update(withdrawal_target_cut_s=cut,withdrawal_afterpoint5to1_output_RMS=float(np.sqrt(np.mean(pred[mask]**2))),withdrawal_window_source='knownfixedreceptor input;notnewphysicalrecovery')
 return out
def main():
 torch,np=backend();f=frozen();out=HERE/'EVALUATION';out.mkdir(exist_ok=False);path=HERE/'TRAIN/checkpoint_000250.pt';saved=read(HERE/'TRAIN/VERIFY_000250.json')
 if not saved['all_passed'] or saved['sha256']!=sha(path):raise ValueError('Actualfinal250savedreplay required')
 m,p=load_checkpoint(path,sha(path));from data import arrays,tensors
 from model import production
 d=arrays();z,y,w=tensors(d);dataset=read(HERE/'DATASET.json');rows=[];artifacts={}
 with torch.no_grad():
  for j,name in enumerate(TRAIN):
   actual=production(m,z[:,j:j+1]);row=next(q for q in dataset['rows'] if q['condition']==name);score=metrics(np,actual[0].numpy(),y[:,j:j+1].numpy(),d['original_teacher'][:,j:j+1],w[:,j:j+1].numpy(),row);artifact=out/(name+'.npz');np.savez_compressed(artifact,turn=actual[0].numpy(),latent=actual[1].numpy(),odor_hidden=actual[2].numpy(),receptors=z[:,j:j+1].numpy(),teacher=y[:,j:j+1].numpy(),original66f_teacher=d['original_teacher'][:,j:j+1],weights=w[:,j:j+1].numpy());artifacts[str(artifact)]=sha(artifact);rows.append(dict(condition=name,split='known_train_inputs_descriptive',metrics=score,artifact=str(artifact),input_source_kind=row['source_kind'],new_physical_trial=False,fresh_holdout=False,no_hidden_reset=True))
 verify(f['source_hashes']);verify(f['data_hashes']);save(out/'RESULT.json',dict(status='FINAL250_LEARNED_ODOR_MEMORY_REPAIR_METRICS_NOT_BODY',checkpoint=str(path),checkpoint_sha256=sha(path),parent_sha256=PARENT_SHA,identity=m.identity(),rows=rows,output_hashes=artifacts,source_hashes=f['source_hashes'],data_hashes=f['data_hashes'],training_updates=250,extra_training_updates=0,source_ALPHA=float(m.ALPHA),original_class_identity_unchanged=True,new_body_physics_steps=0,no_runtime_gate_or_hiddenreset=True,old_D_arrival_confirmation_failure_retained=True));print(json.dumps(dict(status='FINAL250_ODOR_REPAIR_METRICS',rows=rows),indent=2))
def audit():
 torch,np=backend();f=frozen();rp=HERE/'EVALUATION/RESULT.json';r=read(rp);m,p=load_checkpoint(r['checkpoint'],r['checkpoint_sha256']);from data import arrays,tensors
 from model import make_scan
 d=arrays();z,y,w=tensors(d);dataset=read(HERE/'DATASET.json');scan=make_scan(m);checks={};errors={}
 with torch.no_grad():
  for j,name in enumerate(TRAIN):
   row=next(q for q in r['rows'] if q['condition']==name);artifact=Path(row['artifact']);checks[name+'_artifactSHA']=sha(artifact)==r['output_hashes'][str(artifact)]
   with np.load(artifact,allow_pickle=False) as q:actual=scan(z[:,j:j+1]);errs=[float(np.max(abs(value.numpy()-q[key]))) for key,value in zip(('turn','latent','odor_hidden'),actual)]
   errors[name]=max(errs);checks[name+'_independent_scan_outputs']=max(errs)<=1e-10;meta=next(q for q in dataset['rows'] if q['condition']==name);score=metrics(np,actual[0].numpy(),y[:,j:j+1].numpy(),d['original_teacher'][:,j:j+1],w[:,j:j+1].numpy(),meta);checks[name+'_independent_metrics']=all(score[k]==row['metrics'][k] or (type(score[k]) is float and type(row['metrics'][k]) is float and abs(score[k]-row['metrics'][k])<=1e-10) for k in score)
 if not all(checks.values()):raise ValueError('Independentodorrepairmetricsaudit '+str(checks))
 verify(f['source_hashes']);verify(f['data_hashes']);save(HERE/'EVALUATION/VERIFY.json',dict(integrity_all_pass=True,checks=checks,max_errors=errors,result_sha256=sha(rp),checkpoint_sha256=r['checkpoint_sha256'],parent_sha256=PARENT_SHA,source_hashes=f['source_hashes'],data_hashes=f['data_hashes'],identity=m.identity(),extra_training_updates=0,new_physics_steps=0));print(json.dumps(dict(integrity_all_pass=True,checks=len(checks),max_error=max(errors.values()))))
if __name__=='__main__':raise SystemExit('Use parent explicitflag run.py')
