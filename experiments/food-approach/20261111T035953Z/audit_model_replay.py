"""Independent final-model inference identity replay. No learning or new test selection."""
import json
import numpy as np,torch
from common import HERE,load_behavior,digest,save_json,guard,verify_hashes


def main():
 torch.set_num_threads(1);guard()
 ck=HERE/'TRAIN/checkpoint_003000.pt';sha='bfd65a3896d259496114695d7f7f3550272cf4c369895a4ddb34f3ddc3352c06'
 m,p=load_behavior(ck,sha);report=json.loads((HERE/'EVALUATION/RESULT.json').read_text())
 verify_hashes(report['source_hashes']);verify_hashes(report['data_hashes'])
 paths=[ck,HERE/'fresh_test.npz',HERE/'FRESH_TEST.json',HERE/'EVALUATION/PREDICTIONS.npz',HERE/'EVALUATION/RESULT.json']
 before={str(f):digest(f) for f in paths}
 with np.load(HERE/'fresh_test.npz',allow_pickle=False) as z:fresh=z['receptors'].copy()
 with np.load(HERE/'EVALUATION/PREDICTIONS.npz',allow_pickle=False) as z:saved={key:z[key].copy() for key in ('receptors','teacher_turn','learned_turn','swapped_turn','hidden')}
 checks=dict(input_only44x240x2=fresh.shape==(44,240,2),saved_receptors_bitexact=np.array_equal(fresh,saved['receptors']),saved_checkpoint_identity=report['checkpoint_sha256']==sha and report['checkpoint']==str(ck),scope_remains_sensor_only=not p['deployment_qualified'] and p['scope_tag']=='SENSOR_PRETRAIN_ONLY')
 metrics={};normal_mean=None
 with torch.no_grad():
  for name,obs in [('normal',fresh),('swapped',fresh[:,:,::-1].copy())]:
   state=m.initial_state(44);outputs=[];history=[];latents=[];speed=[];direct_states=True;direct_turns=True
   for k in range(240):
    receptors=torch.tensor(obs[:,k].copy(),dtype=torch.float64)
    command,newstate=m.step(receptors,state)
    latent,meanstate=m.mean_step(receptors,state)
    direct_states=direct_states and torch.equal(newstate,meanstate)
    direct_turns=direct_turns and torch.equal(command[:,1],latent[:,0].tanh())
    outputs.append(command[:,1]);history.append(newstate);latents.append(latent[:,0]);speed.append(command[:,0]);state=newstate
   output=torch.stack(outputs).T.numpy();hidden=torch.stack(history).transpose(0,1).numpy();latent=torch.stack(latents).T.numpy()
   target=saved['learned_turn'] if name=='normal' else saved['swapped_turn']
   error=float(np.max(np.abs(output-target)))
   checks[name+'_all10560_turns_bitexact']=np.array_equal(output,target)
   checks[name+'_speed_structuralzero_every_step']=torch.equal(torch.stack(speed),torch.zeros((240,44),dtype=torch.float64))
   checks[name+'_mean_step_and_step_state_exact']=direct_states
   checks[name+'_latent_to_turn_exact']=direct_turns
   metrics[name+'_turn_max_error']=error
   if name=='normal':
    checks['normal_all337920_hidden_values_bitexact']=np.array_equal(hidden,saved['hidden'])
    checks['normal_initial_state_reset_only_episode']=bool(m.initial_state(44).eq(0).all())
    # Earlier artifact stores output+hidden, not latent. Verify the latent against
    # the immutable head applied to its actually saved current state; never invent
    # a preexisting latent array.
    reconstructed=m.head(torch.tensor(saved['hidden'],dtype=torch.float64)).squeeze(-1).numpy()
    latent_error=float(np.max(np.abs(latent-reconstructed)))
    checks['normal10560_latents_match_saved_hidden_readout_tol1e12']=latent_error<=1e-12
    metrics['normal_hidden_max_error']=float(np.max(np.abs(hidden-saved['hidden'])))
    metrics['normal_latent_from_saved_hidden_max_error']=latent_error
    normal_mean=latent
 verify_hashes(report['source_hashes']);verify_hashes(report['data_hashes'])
 checks['all_execution_inputs_unchanged']=all(digest(filename)==h for filename,h in before.items())
 old=HERE.parent/'20261111T034309Z';oldmanifest=json.loads((old/'ARTIFACT_SHA256.json').read_text())
 checks['old_failed29_artifacts_unchanged']=all(digest(filename)==r['sha256'] for filename,r in oldmanifest['files'].items())
 guard()
 result=dict(all_passed=all(checks.values()),checks={key:bool(v) for key,v in checks.items()},metrics=metrics,checkpoint=str(ck),checkpoint_sha256=sha,
  replay=dict(episodes=44,steps_per_episode=240,normal_predictions=10560,swapped_predictions=10560,normal_hidden_values=337920,input_channels=2,hidden_channels=32),
  source_hashes={str(HERE/'audit_model_replay.py'):digest(HERE/'audit_model_replay.py')},bound_inputs=before,physical_steps=0,Core_constructions=0,optimizer_updates=0,
  limitation='This replays existing fixed final-model evidence only. Existing artifact had no latent array; latent consistency uses saved current hidden through immutable head within1e-12. No new evaluation selection, training, physical navigation or deployment qualification.')
 save_json(HERE/'EVALUATION/AUDIT_MODEL_REPLAY.json',result)
 print(json.dumps(dict(all_passed=result['all_passed'],checks=len(checks),metrics=metrics)))
 if not result['all_passed']:raise ValueError('final sensory model evidence replay failed')
if __name__=='__main__':main()
