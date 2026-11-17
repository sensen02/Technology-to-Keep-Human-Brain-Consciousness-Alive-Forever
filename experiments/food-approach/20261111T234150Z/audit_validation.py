"""Independent final validation production replay and functional flag reconstruction."""
from common import *
from data import arrays,split

def main():
 cpu1();freeze_proof();folder=HERE/'VALIDATION';r=read(folder/'RESULT.json');a=arrays(folder/'OUTPUTS.npz');m,p=load_checked(r['checkpoint'],r['checkpoint_sha256']);checks={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('validationaudit '+k)
 for path,h in r['source_hashes'].items():check('source:'+path,sha(path)==h)
 check('outputs_hash',sha(folder/'OUTPUTS.npz')==r['outputs']['sha256'])
 regen,meta=split('validation');check('actual_inputs_targets_exact',np.array_equal(regen['receptors'],a['receptors']) and np.array_equal(regen['targets'],a['target']))
 turns=[];latents=[];states=[]
 for j in range(12):
  state=m.initial_state(1);turn=[];latent=[];hidden=[]
  with torch.no_grad():
   for row in a['receptors'][:,j]:
    raw,state=m.mean_step(torch.tensor(row[None],dtype=torch.float64),state);turn.append(float(raw[0,0].tanh()));latent.append(float(raw[0,0]));hidden.append(state[0].numpy().copy())
  turns.append(turn);latents.append(latent);states.append(hidden)
 check('actual_validation_turn_bitexact',np.array_equal(np.asarray(turns).T,a['turn']));check('actual_validation_latent_bitexact',np.array_equal(np.asarray(latents).T,a['latent']));check('actual_validation_hidden32_bitexact',np.array_equal(np.asarray(states).transpose(1,0,2),a['hidden']))
 time=a['time']
 def region(x,y):return (time>=x-1e-10)&(time<y-1e-10)
 for j,row in enumerate(r['rows']):
  cfg=row['config'];turn=a['turn'][:,j];z=a['receptors'][:,j];motif=cfg['motif'];flags={}
  if motif=='noodor':flags['noodor_RMS']=float(np.sqrt(np.mean(turn**2)))<=.02
  elif motif=='equal':flags['equal_RMS']=float(np.sqrt(np.mean(turn**2)))<=.05
  else:
   sign=1 if motif.startswith('left') else -1;end=cfg['off_s'] if motif.endswith('pulse') else cfg['switch_s'];q=(z.mean(1)>=.05)&(sign*(z[:,0]-z[:,1])>=.001);p=q&region(end-.5,end);flags['first_sign_fraction']=int(p.sum())>=10 and np.mean(sign*turn[p]>0)>=.9
   origin=float(time[np.flatnonzero(q)[0]]) if np.any(q) else None;response=np.zeros(1920,dtype=bool) if origin is None else q&(sign*turn>0)&(time<origin+2);flags['response_within2s']=bool(np.any(response))
   if motif.endswith('pulse'):flags['offset_RMS']=float(np.sqrt(np.mean(turn[region(cfg['off_s']+.5,cfg['off_s']+1)]**2)))<=.05
   else:
    p=(z.mean(1)>=.05)&(-sign*(z[:,0]-z[:,1])>=.001)&region(9.1,9.6);flags['second_sign_fraction']=int(p.sum())>=10 and np.mean(-sign*turn[p]>0)>=.9
  flags={k:bool(v) for k,v in flags.items()};check('episode'+str(j)+'_flags_recomputed',flags==row['flags'] and all(flags.values())==row['all_targets_pass'])
  check('episode'+str(j)+'_MSE',abs(float(np.mean((turn-a['target'][:,j])**2))-row['metrics']['action_MSE'])<1e-15)
 check('whole_functional_flag',r['all_targets_pass']==all(row['all_targets_pass'] for row in r['rows']));check('motor_SHA_unchanged_not_loaded',sha(MOTOR)==MOTOR_SHA);guard()
 save(folder/'VERIFY.json',dict(integrity_all_pass=True,checks=checks,all_targets_pass=r['all_targets_pass'],checkpoint_sha256=r['checkpoint_sha256'],result_sha256=sha(folder/'RESULT.json'),Core_constructions=0,physics_steps=0,optimizer_updates=0,motor_model_loads=0,source_hashes={str(p):sha(p) for p in (HERE/'audit_validation.py',HERE/'common.py',HERE/'data.py')}))
 print(json.dumps(dict(integrity_all_pass=True,checks=len(checks),all_targets_pass=r['all_targets_pass'],checkpoint_sha256=r['checkpoint_sha256'])))
if __name__=='__main__':main()
