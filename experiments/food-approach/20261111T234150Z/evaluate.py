"""Once-only final1000 frozen12episode long-history validation; no physics."""
from common import *
from data import arrays

def main():
 cpu1();freeze_proof();out=HERE/'VALIDATION'
 if out.exists():raise FileExistsError('onlyone final frozen validation; do not pickcheckpoints')
 path=HERE/'TRAIN/checkpoint_001000.pt';audit=read(HERE/'TRAIN/VERIFY_001000.json')
 if not audit['all_passed'] or audit['sha256']!=sha(path) or not all(audit['checks'].values()):raise ValueError('finalsavedindependent audit required')
 m,payload=load_checked(path,sha(path));data=arrays(HERE/'validation.npz');meta=read(HERE/'DATASET.json')['splits']['validation'];time=data['time'];rows=[];outputs=[];latents=[];states=[]
 def region(a,b):return (time>=a-1e-10)&(time<b-1e-10)
 for j,cfg in enumerate(meta['configs']):
  x=torch.from_numpy(data['receptors'][:,j:j+1]);state=m.initial_state(1);pred=[];latent=[];hidden=[]
  with torch.no_grad():
   for z in x:
    v,state=m.mean_step(z,state);pred.append(float(v[0,0].tanh()));latent.append(float(v[0,0]));hidden.append(state[0].numpy().copy())
  y=np.asarray(pred);z=data['receptors'][:,j];target=data['targets'][:,j];flags={};metrics=dict(action_MSE=float(np.mean((y-target)**2)),neutral_period_RMS=float(np.sqrt(np.mean(y[np.abs(target)<.05]**2))) if np.any(np.abs(target)<.05) else None,max_abs_turn=float(np.abs(y).max()),final0_5s_RMS=float(np.sqrt(np.mean(y[region(9.1,9.6)]**2))))
  motif=cfg['motif']
  if motif=='noodor':flags['noodor_RMS']=float(np.sqrt(np.mean(y*y)))<=.02;metrics['noodor_RMS']=float(np.sqrt(np.mean(y*y)))
  elif motif=='equal':flags['equal_RMS']=float(np.sqrt(np.mean(y*y)))<=.05;metrics['equal_RMS']=float(np.sqrt(np.mean(y*y)))
  else:
   sign=1 if motif.startswith('left') else -1
   first_end=cfg['off_s'] if motif.endswith('pulse') else cfg['switch_s']
   q=(np.mean(z,axis=1)>=.05)&(sign*(z[:,0]-z[:,1])>=.001);plateau=q&region(first_end-.5,first_end);n=int(plateau.sum());fraction=float(np.mean(sign*y[plateau]>0)) if n else None
   flags['first_sign_fraction']=n>=10 and fraction>=.9;metrics.update(first_qualified_plateau_samples=n,first_sign_fraction=fraction)
   origin=float(time[np.flatnonzero(q)[0]]) if np.any(q) else None;response=q&(sign*y>0)&(time<origin+2 if origin is not None else np.zeros(1920,dtype=bool));latency=float(time[np.flatnonzero(response)[0]]-origin) if np.any(response) else None
   flags['response_within2s']=latency is not None and latency<=2.;metrics['first_effective_input_s']=origin;metrics['response_latency_s']=latency
   if motif.endswith('pulse'):
    recovery=region(cfg['off_s']+.5,cfg['off_s']+1.);rms=float(np.sqrt(np.mean(y[recovery]**2)));flags['offset_RMS']=rms<=.05;metrics['offset_RMS_offplus0_5to1_0']=rms
    off_end=cfg['off_s']+.2;eligible=region(off_end,9.6);tail=np.abs(y)<=.05;first=None
    # Diagnostic return-and-remain band for .25s, not a new criterion.
    for k in np.flatnonzero(eligible):
     if k+50<=1920 and np.all(tail[k:k+50]):first=float(time[k]-cfg['off_s']);break
    metrics['first_return_band0_05_for0_25s_after_off_s']=first
   else:
    q2=(np.mean(z,axis=1)>=.05)&(-sign*(z[:,0]-z[:,1])>=.001)&region(9.1,9.6);n2=int(q2.sum());fraction2=float(np.mean(-sign*y[q2]>0)) if n2 else None;flags['second_sign_fraction']=n2>=10 and fraction2>=.9;metrics['second_sign_fraction']=fraction2;metrics['second_qualified_plateau_samples']=n2
  rows.append(dict(index=j,config=cfg,metrics=metrics,flags=flags,all_targets_pass=all(flags.values())));outputs.append(y);latents.append(latent);states.append(hidden)
 # Assess all frozen validation rows once; failure retained without retrying another save.
 out.mkdir();replay=out/'OUTPUTS.npz';np.savez_compressed(replay,turn=np.stack(outputs,1),latent=np.asarray(latents).T,hidden=np.asarray(states).transpose(1,0,2),receptors=data['receptors'],target=data['targets'],time=time)
 result=dict(status='ALL_LONG_HISTORY_TARGETS_PASS' if all(r['all_targets_pass'] for r in rows) else 'LONG_HISTORY_VALIDATION_FAILED',all_targets_pass=all(r['all_targets_pass'] for r in rows),checkpoint=str(path),checkpoint_sha256=sha(path),motor_sha256=MOTOR_SHA,architecture_identity=m.identity(),rows=rows,configuration_count=12,distinct_receptor_histories=len({hashlib.sha256(data['receptors'][:,j].tobytes()).hexdigest() for j in range(12)}),replicates_per_configuration=1,identical_noodor_sequences_are_not_independent_replicates=True,source_hashes={str(p):sha(p) for p in (HERE/'evaluate.py',HERE/'PREREG.json',HERE/'DATASET.json',HERE/'validation.npz',HERE/'FROZEN.json',HERE/'TRAIN/VERIFY_001000.json')},outputs=dict(path=str(replay),sha256=sha(replay)),Core_constructions=0,physics_steps=0,motor_model_loads=0,food_success=False,body_response_qualified=False,future_physics='PARENT_COORDINATES_ONLY_IF_THIS_ALLPASS',no_runtime_deadzone_or_gate_or_reset=True)
 save(out/'RESULT.json',result);guard();print(json.dumps(dict(all_targets_pass=result['all_targets_pass'],rows=[dict(index=r['index'],motif=r['config']['motif'],pass_all=r['all_targets_pass'],flags=r['flags'],metrics=r['metrics']) for r in rows]),ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
