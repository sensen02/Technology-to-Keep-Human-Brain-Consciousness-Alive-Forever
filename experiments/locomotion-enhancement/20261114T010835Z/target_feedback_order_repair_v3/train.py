"""RESUME300_ORDER_REPAIR_TO600: restoredAdam/RNG/anchor, exactly300newupdates."""
import argparse,copy,json,subprocess,sys,time
from sources import ROOT,read,save,sha,verify


def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-parent-resume300-to600',action='store_true');a=p.parse_args()
 if not a.execute_parent_resume300_to600:p.error('Explicitresume300onlyrelease required')
 start=time.monotonic()
 from learning import cpu1,rng_snapshot,rng_equal,restore_rng,forward,slice_data,objective,metrics,resource_gate
 cpu1()
 import torch
 import common
 from scan import make_scan
 from data import prepare
 from utils import max_error
 b=read(ROOT/'INPUT_BINDINGS.json');pre=read(ROOT/'PREFLIGHT.json')
 if not pre['all_passed'] or pre['timing_status']!='READY_RESUME300_TO600' or pre['source_hashes']!=b['source_hashes'] or pre['data_hashes']!=b['data_hashes']:raise ValueError('Actualorderrepairpreflightrequired')
 verify(b['source_hashes']);verify(b['data_hashes']);out=ROOT/'TRAIN';out.mkdir(exist_ok=False)
 m,parent=common.initialize();data=prepare(m);sc=make_scan(m);anchor=parent['config']['initial_anchor'];opt=torch.optim.Adam(m.adapter_parameters(),lr=3e-4);opt.load_state_dict(copy.deepcopy(parent['optimizer_state_dict']))
 if parent['step']!=300 or len(opt.state)!=8 or any(float(v['step'])!=300 for v in opt.state.values()):raise ValueError('Canonicalactual300Adamrequired')
 if any(g['lr']!=3e-4 for g in opt.param_groups):raise ValueError('Preserveoriginalactuallearningrate')
 config=copy.deepcopy(parent['config']);config.update(source_hashes={**b['source_hashes'],str(ROOT/'INPUT_BINDINGS.json'):sha(ROOT/'INPUT_BINDINGS.json'),str(ROOT/'PREFLIGHT.json'):sha(ROOT/'PREFLIGHT.json')},data_hashes=b['data_hashes'],prereg=read(ROOT/'PREREG.json'),lineage='RESUME300_ORDER_REPAIR_TO600',resume_parent_sha256=b['resume_checkpoint_sha256'])
 curve=copy.deepcopy(parent['curve']);restore_rng(parent['rng']);save(out/'IDENTITY.json',dict(lineage=config['lineage'],resume_step=300,total_updates=600,new_updates=300,parent_checkpoint_sha256=b['resume_checkpoint_sha256'],original_anchor_preserved=True,originalAdamRNGrestored=True,source_hashes=config['source_hashes'],data_hashes=config['data_hashes']))
 def publish(step):
  path=out/f'checkpoint_{step:06d}.pt';rng=rng_snapshot();digest=common.save_checkpoint(path,m,opt,config,curve,step,rng);restored,payload=common.load_checkpoint(path,digest,final=False)
  if max_error(m.state_dict(),restored.state_dict())!=0 or max_error(opt.state_dict(),payload['optimizer_state_dict'])!=0 or not rng_equal(rng,payload['rng']):raise ValueError('SavedexactmodelAdamRNGfailed')
  del restored,payload
  subprocess.run([sys.executable,'-B',str(ROOT/'verify_saved.py'),'--checkpoint',str(path),'--sha256',digest,'--execute-parent-saved-verify'],check=True);resource_gate(start);return path,digest
 path,digest=publish(300)
 with (out/'TRAINING.jsonl').open('x') as log:
  for step in range(301,601):
   resource_gate(start);opt.zero_grad(set_to_none=True);value=0.;terms={}
   for lo in (0,4,8):
    d=slice_data(data,lo,lo+4);values=forward(sc,d);loss,parts=objective(values,d,m,anchor);loss.backward();value+=float(loss.detach())
    for k,v in parts.items():terms[k]=terms.get(k,0.)+v
    del values,d,loss
   if any(p.grad is None or not torch.isfinite(p.grad).all() for p in m.adapter_parameters()) or any(p.grad is not None for p in m.old.parameters()):raise ValueError('Gradientfinite/frozenoldviolation')
   opt.step();row=dict(step=step,loss=value,**terms,wall_s=time.monotonic()-start,peak_RSS_MiB=resource_gate(start),lineage='RESUME300_ORDER_REPAIR_TO600');curve.append(row);log.write(json.dumps(row)+'\n');log.flush()
   if step%25==0:print(json.dumps(row),flush=True)
   if step%50==0:verify(config['source_hashes']);verify(config['data_hashes'])
   if step in (450,600):row['metrics_after_update']=metrics(sc,data);path,digest=publish(step)
 verify(config['source_hashes']);verify(config['data_hashes'])
 save(out/'RESULT.json',dict(kind='RESUME300_ORDER_REPAIR_TO600_COMPLETED_NOT_BODY_VERIFIED',updates=600,total_updates=600,new_updates=300,already_completed_updates=300,checkpoint=str(path),sha256=digest,final=curve[-1],wall_s=time.monotonic()-start,source_hashes=config['source_hashes'],data_hashes=config['data_hashes'],physical_steps=0,new_neck_deployment=False))


if __name__=='__main__':main()
