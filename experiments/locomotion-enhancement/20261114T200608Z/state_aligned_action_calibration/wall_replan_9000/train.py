"""SameNN restore600Adam/RNG/anchor, TFaction + conditionaltarget separatedgraphs."""
import argparse,copy,json,subprocess,sys,time
from sources import ROOT,read,save,sha,verify

def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-parent-resume600-to900',action='store_true');a=p.parse_args()
 if not a.execute_parent_resume600_to900:p.error('Explicitstatealignedfinite300release required')
 start=time.monotonic()
 from learning import cpu1,rng_snapshot,rng_equal,restore_rng,tf_forward,forward,slice_data,tf_objective,conditional_objective,metrics,resource_gate
 cpu1()
 import torch
 import common
 from scan import make_scan,make_actual_scan
 from data import prepare
 from utils import max_error
 b=read(ROOT/'INPUT_BINDINGS.json');pre=read(ROOT/'PREFLIGHT.json')
 if not pre['all_passed'] or pre['timing_status']!='READY_STATE_ALIGNED600_TO900' or pre['source_hashes']!=b['source_hashes'] or pre['data_hashes']!=b['data_hashes']:raise ValueError('Actualcompletepreflightrequired')
 verify(b['source_hashes']);verify(b['data_hashes']);out=ROOT/'TRAIN';out.mkdir(exist_ok=False)
 m,parent=common.initialize();data=prepare(m);condscan=make_scan(m);actualscan=make_actual_scan(m);anchor=parent['config']['initial_anchor'];opt=torch.optim.Adam(m.adapter_parameters(),lr=3e-4);opt.load_state_dict(copy.deepcopy(parent['optimizer_state_dict']))
 if parent['step']!=600 or len(opt.state)!=8 or any(float(s['step'])!=600 for s in opt.state.values()) or any(g['lr']!=3e-4 for g in opt.param_groups):raise ValueError('Actual600canonical8AdamandLRrequired')
 config=copy.deepcopy(parent['config']);config.update(source_hashes={**b['source_hashes'],str(ROOT/'INPUT_BINDINGS.json'):sha(ROOT/'INPUT_BINDINGS.json'),str(ROOT/'PREFLIGHT.json'):sha(ROOT/'PREFLIGHT.json')},data_hashes=b['data_hashes'],prereg=read(ROOT/'PREREG.json'),steps=900,lineage='STATE_ALIGNED_ACTION_RESUME600_TO900',resume_parent_sha256=b['resume_checkpoint_sha256'])
 curve=copy.deepcopy(parent['curve']);restore_rng(parent['rng']);save(out/'IDENTITY.json',dict(lineage=config['lineage'],resume_step=600,total_updates=900,new_updates=300,parent_checkpoint_sha256=b['resume_checkpoint_sha256'],original_anchor_preserved=True,originalAdamRNGrestored=True,source_hashes=config['source_hashes'],data_hashes=config['data_hashes'],action_supervision='actualXTFonly',conditionalactionlabels_supervised=False))
 def publish(step):
  path=out/f'checkpoint_{step:06d}.pt';rng=rng_snapshot();digest=common.save_checkpoint(path,m,opt,config,curve,step,rng);loaded,payload=common.load_checkpoint(path,digest,final=False)
  if max_error(m.state_dict(),loaded.state_dict())!=0 or max_error(opt.state_dict(),payload['optimizer_state_dict'])!=0 or not rng_equal(rng,payload['rng']):raise ValueError('ExactliveModelAdamRNGroundtripfailed')
  del loaded,payload
  subprocess.run([sys.executable,'-B',str(ROOT/'verify_saved.py'),'--checkpoint',str(path),'--sha256',digest,'--execute-parent-saved-verify'],check=True);resource_gate(start);return path,digest
 path,digest=publish(600)
 with (out/'TRAINING.jsonl').open('x') as log:
  for step in range(601,901):
   resource_gate(start);opt.zero_grad(set_to_none=True);total=0.;terms={}
   for lo in (0,4,8):
    d=slice_data(data,lo,lo+4)
    tf=tf_forward(actualscan,d);loss,parts=tf_objective(tf,d);loss.backward();total+=float(loss.detach())
    for k,v in parts.items():terms[k]=terms.get(k,0.)+v
    del tf,loss,parts
    cond=forward(condscan,d);loss,parts=conditional_objective(cond,d,m,anchor);loss.backward();total+=float(loss.detach())
    for k,v in parts.items():terms[k]=terms.get(k,0.)+v
    del cond,loss,parts,d
   if any(p.grad is None or not torch.isfinite(p.grad).all() for p in m.adapter_parameters()) or any(p.grad is not None or p.requires_grad for p in m.old.parameters()):raise ValueError('Finite8grads/frozenoldrequired')
   opt.step();row=dict(step=step,loss=total,**terms,wall_s=time.monotonic()-start,peak_RSS_MiB=resource_gate(start),lineage=config['lineage']);curve.append(row);log.write(json.dumps(row)+'\n');log.flush()
   if step%25==0:print(json.dumps(row),flush=True)
   if step%50==0:verify(config['source_hashes']);verify(config['data_hashes'])
   if step in (750,900):row['metrics_after_update']=metrics(actualscan,condscan,data);path,digest=publish(step)
 verify(config['source_hashes']);verify(config['data_hashes'])
 save(out/'RESULT.json',dict(kind='STATE_ALIGNED_ACTION900_TRAINED_NOT_BODY_VERIFIED',updates=900,total_updates=900,new_updates=300,already_completed_updates=600,checkpoint=str(path),sha256=digest,final=curve[-1],wall_s=time.monotonic()-start,source_hashes=config['source_hashes'],data_hashes=config['data_hashes'],physical_steps=0,new_neck_deployment=False))
if __name__=='__main__':main()
