"""Independent process RGB-to-CNN/GRU replay and finalvisual flags; zero physics."""
from data import *
from model import production
from evaluate import metrics_flags
def main():
 cpu1();frozen();folder=HERE/'VALIDATION';r=read(folder/'RESULT.json');m,p=load_checked(r['checkpoint'],r['checkpoint_sha256']);rgb,manifest=library();checks={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('visual validation audit '+k)
 check('all_saved_sources_unchanged',True);verify(r['source_hashes']);allrows=[]
 for name in ('validation','controls'):
  d=arrays(name);fresh,metadata=split(name,manifest)
  check(name+'_schedule_regeneration_exact',all(np.array_equal(d[k],fresh[k]) for k in d))
  output=r['outputs'][name];check(name+'_output_hash',sha(output['path'])==output['sha256'])
  with np.load(output['path'],allow_pickle=False) as z:saved={k:z[k].copy() for k in z.files}
  check(name+'_causal_frame_ids_exact',np.array_equal(saved['frame_ids'],d['frame_ids']) and np.array_equal(saved['used_capture_index'],d['used_capture_index']))
  with torch.no_grad():
   for j,meta in enumerate(metadata):
    x,_,_=batch(rgb,d,[j]);t,l,h=production(m,x);st,_,_=production(m,x.flip(2));check(name+str(j)+'_CNN_GRU_turn_latent_memory_exact',np.array_equal(t[:,0].numpy(),saved['turn'][:,j]) and np.array_equal(l[:,0].numpy(),saved['latent'][:,j]) and np.array_equal(h[:,0].numpy(),saved['hidden'][:,j]));check(name+str(j)+'_eyeswap_exact',np.array_equal(st[:,0].numpy(),saved['eyeswap_turn'][:,j]))
    metrics,flags=metrics_flags(t[:,0].numpy(),d['target'][:,j],meta,d['decision_time']);row=next(v for v in r['rows'] if v['split']==name and v['index']==j)
    check(name+str(j)+'_configuration_exact',row['config']==meta);check(name+str(j)+'_all_functional_metrics_exact',all(row['metrics'][k]==v for k,v in metrics.items()) and row['flags']==flags)
    if row['functional_required']:allrows.append(all(flags.values()))
 check('all_targets_conjunction_exact',r['all_targets_pass']==all(allrows));check('scope_not_body_or_navigation',r['body_response_qualified'] is False and r['dynamic_navigation_qualified'] is False and r['physical_steps']==0)
 frozen();save(folder/'VERIFY.json',dict(integrity_all_pass=all(checks.values()),all_targets_pass=r['all_targets_pass'],checks=checks,result_sha256=sha(folder/'RESULT.json'),checkpoint_sha256=r['checkpoint_sha256'],source_hashes={str(HERE/'audit_validation.py'):sha(HERE/'audit_validation.py')},Core_constructions=0,motor_model_loads=0,physics_steps=0));print(json.dumps(dict(integrity_all_pass=True,all_targets_pass=r['all_targets_pass'],checks=len(checks))))
if __name__=='__main__':main()
