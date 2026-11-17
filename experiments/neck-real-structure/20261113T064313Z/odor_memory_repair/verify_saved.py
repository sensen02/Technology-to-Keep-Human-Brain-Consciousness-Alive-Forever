"""Freshprocess0/125/250 savedproduction replay/RNG/optimizer andsourceclass checks;0Core."""
from common import *
def main(step):
 torch,np=backend();path=HERE/'TRAIN'/('checkpoint_'+str(step).zfill(6)+'.pt');m,p=load_checkpoint(path,sha(path),final=False)
 from data import arrays,tensors
 from model import production,make_scan
 checks={};errors={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('Savedodorrepair '+k)
 def close(k,l,r):
  e=float((l-r).abs().max().detach());errors[k]=e;check(k,e<=1e-10)
 d=arrays();z,y,w=tensors(d);j=p['replay_index'];check('actual_receptorcut_replayindex',j==11)
 with torch.no_grad():
  actual=production(m,z[:,j:j+1]);allbatch=make_scan(m)(z)
  for key,l,r in zip(('turn','latent','h32'),actual,p['replay']):close('savedproduction_batch1_'+key,l,r)
  close('savedbatch16turn',allbatch[0],p['train_prediction']);check('actualfull1920continuousmemory',actual[2].shape==(1920,1,32))
 torch.manual_seed(20261113);initial,parent=initialize();check('RNG_exact_strictinitialize',torch.equal(torch.get_rng_state(),p['torch_rng_state']));check('sameoriginalsourceidentity_ALPHA',m.identity()==initial.identity()==parent['identity'] and m.ALPHA==initial.ALPHA and p['repair_lineage']['source_ALPHA']==m.ALPHA);check('eightparameternames',p['parameter_names']==[k for k,v in m.named_parameters()] and len(p['parameter_names'])==8)
 if step==0:check('initial66f_allstate_exact',all(torch.equal(v,p['state_dict'][k]) for k,v in initial.state_dict().items()))
 else:check('actualNNweightsupdated',any(not torch.equal(v,p['state_dict'][k]) for k,v in initial.state_dict().items()))
 check('same_effective16micro8x2',p['effective_batch']==16 and p['microbatch']==8 and p['accumulation']==2 and p['history_steps']==1920);check('runtimeflags_no_gate_reset',p['repair_lineage']['no_hidden_reset'] and p['repair_lineage']['no_runtime_receptor_gate'] and p['deployment_qualified'] is False)
 # Strict loader must restore hostile caller module aliases even during nested source-parent loading.
 import types
 sentinel=object();previous={k:sys.modules.get(k,sentinel) for k in ('common','model','steering_behavior')};fake={k:types.ModuleType('_odor_saved_fake_'+k) for k in previous};paths=list(sys.path)
 try:
  sys.modules.update(fake);reload,again=load_checkpoint(path,sha(path),final=False);check('publicloader_restores_all_aliases',all(sys.modules.get(k) is v for k,v in fake.items()) and sys.path==paths);check('publicloader_nested_sameweights',all(torch.equal(v,p['state_dict'][k]) for k,v in reload.state_dict().items()))
 finally:
  sys.path[:]=paths
  for k,v in previous.items():
   if v is sentinel:sys.modules.pop(k,None)
   else:sys.modules[k]=v
 f=frozen();save(HERE/'TRAIN'/('VERIFY_'+str(step).zfill(6)+'.json'),dict(all_passed=True,checks=checks,numerical_errors=errors,step=step,sha256=sha(path),parent_sha256=PARENT_SHA,identity=m.identity(),source_hashes=f['source_hashes'],data_hashes=f['data_hashes'],actual_full1920_source_receptor_replay=True,production_batch1_and16=True,extra_training_updates=0,new_physics_steps=0));print(json.dumps(dict(step=step,all_passed=True,checks=len(checks),max_error=max(errors.values()))))
if __name__=='__main__':raise SystemExit('Use parent explicitflag run.py')
