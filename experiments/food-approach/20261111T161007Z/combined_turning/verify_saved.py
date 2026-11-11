"""Future new-process composed checkpoint/three-state256/reference-stream audit, no Core."""
import argparse,json
import numpy as np,torch
from model_io import *
from prepare import streams

def main():
 a=argparse.ArgumentParser();a.add_argument('--checkpoint',required=True);a.add_argument('--sha256',required=True);args=a.parse_args();torch.set_num_threads(1);m,p=load_checked(args.checkpoint,args.sha256);checks={}
 verify_hashes(p['stream_hashes']);bindings=json.loads((HERE/'COMPONENT_BINDINGS.json').read_text());right,left,current,_=components(bindings['right_option'],bindings['LEFT']['sha256'],bindings['RIGHT']['sha256']);checks['frozen_all']=all(not q.requires_grad for q in m.parameters());checks['no_training_optimizer']=p['optimizer'] is None and p['training_performed'] is False and 'curve' not in p and 'training_updates' not in p
 for condition,data,u,path in streams(bindings):
  x=torch.from_numpy(data['X238'])[:,None]
  with torch.no_grad():means,h=m.sequence(x,u)
  saved=p['replay'][condition];checks[condition+'_productionmeans_bitexact']=torch.equal(means,saved['means']);checks[condition+'_sharedbase_bitexact']=torch.equal(h[0],saved['base_states']);checks[condition+'_RIGHT64_bitexact']=torch.equal(h[1],saved['RIGHT_states']);checks[condition+'_LEFT64_bitexact']=torch.equal(h[2],saved['LEFT_states']);checks[condition+'_all256finite']=torch.isfinite(torch.cat(h,-1)).all().item()
  if condition=='neutral':checks['neutral_originalmeans_exact']=np.array_equal(means[:,0].numpy(),data['mean']);checks['neutral_originalbasehidden_exact']=np.array_equal(h[0][:,0].numpy(),data['hidden_after'])
  else:
   checks[condition+'_component_actualmeans_exact']=np.array_equal(means[:,0].numpy(),data['mean']);checks[condition+'_component_actualbasehidden_exact']=np.array_equal(h[0][:,0].numpy(),data['hidden_after'][:,:128]);checks[condition+'_component_actual_activehidden_exact']=np.array_equal(h[1 if condition=='right' else 2][:,0].numpy(),data['hidden_after'][:,128:])
 result=dict(all_passed=all(checks.values()),checks=checks,checkpoint=args.checkpoint,sha256=args.sha256,state_layout=[128,64,64],state_total=256,Core_constructions=0,physics_integration_steps=0,training_performed=False,scope='constant composition saved replay only; no actualcomposedbodyqualification yet')
 if (HERE/'VERIFY_SAVED.json').exists():raise ValueError('no overwrite composed saved audit')
 (HERE/'VERIFY_SAVED.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
 if not result['all_passed']:raise ValueError('composed saved fixedstream audit failed')
if __name__=='__main__':main()
