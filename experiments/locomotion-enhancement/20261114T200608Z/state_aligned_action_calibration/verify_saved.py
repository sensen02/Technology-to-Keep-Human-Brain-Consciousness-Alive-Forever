"""Freshsaved 20actualX/12conditional replay; strict600/750/900 statealignment."""
import argparse,sys,time
from sources import ROOT,read,save,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--sha256',required=True);p.add_argument('--execute-parent-saved-verify',action='store_true');a=p.parse_args()
 if not a.execute_parent_saved_verify:p.error('Parentfreshsavedverificationrequired')
 start=time.monotonic()
 from learning import cpu1,rng_snapshot,rng_equal,slice_data,forward,tf_forward,resource_gate
 cpu1()
 import torch,numpy as np
 import common,model
 from scan import make_scan,make_actual_scan
 from data import prepare,production_sequence,dev_sources
 from utils import production_feedback,output_errors
 rng=rng_snapshot();m,payload=common.load_checkpoint(a.checkpoint,a.sha256,final=False);weights_before=model.state_digest(m.state_dict());checks={'strictloader_RNGunchanged':rng_equal(rng,rng_snapshot()),'oldweights_frozen':m.verify_old()};data=prepare(m);conditional=make_scan(m);actualscan=make_actual_scan(m);rows=[];saved={}
 with torch.no_grad():
  for i,raw in enumerate(data['raw']+dev_sources(read(ROOT/'INPUT_BINDINGS.json'))):
   prod,hidden=production_sequence(m,raw);tag=raw['entry']['id']+str(i);checks[tag+'_production_finite']=bool(torch.isfinite(prod).all() and torch.isfinite(hidden).all());rows.append(dict(id=raw['entry']['id'],training=i<12,actual_rows=len(raw['x']),labels_available=i<12))
   state=m.initial_state(1)
   for x in raw['warm']:_,state=m.prime_old(x[None],x.new_zeros((1,2)),state)
   oldmean,features,oldstates=m.cache_sequence(raw['x'][:,None],raw['u'][:,None],state[:3]);residual,h=actualscan(features)
   checks[tag+'_actualX_cachedproduction20']=float((oldmean[:,0]+residual[:,0]-prod).abs().max())<=1e-12 and float((h[:,0]-hidden[:,256:]).abs().max())<=1e-12 and torch.equal(hidden[:,:256],torch.cat(tuple(v[:,0] for v in oldstates),-1))
   if i<12:
    one=slice_data(data,i,i+1);tf=tf_forward(actualscan,one);out=forward(conditional,one);full,old=production_feedback(m,raw,data,i);errors=output_errors(tuple(v[:,0] for v in out),full)
    checks[tag+'_TFproduction']=float((tf[0][:,0]-prod).abs().max())<=1e-12 and float((tf[2][:,0]-hidden[:,256:]).abs().max())<=1e-12 and torch.equal(hidden[:,:256],data['oldafter'][:,i])
    checks[tag+'_conditional_fullproduction']=max(errors)<=1e-12 and torch.equal(old,data['oldafter'][:,i]);checks[tag+'_conditional_finite']=all(bool(torch.isfinite(v).all()) for v in out);rows[-1]['conditional_output_errors']=errors
    for key,value in zip(('mean','residual','new96','pretarget','posttarget','action'),out):saved[tag+'_'+key]=value[:,0].numpy().copy()
    saved[tag+'_TFmean']=tf[0][:,0].numpy().copy();saved[tag+'_TFaction']=tf[3][:,0].numpy().copy()
   resource_gate(start)
  checks['canonical8Adamstates']=len(payload['optimizer_state_dict']['state'])==8 and all(float(v['step'])==payload['step'] for v in payload['optimizer_state_dict']['state'].values())
  checks['all_model_weights_unchanged']=model.state_digest(m.state_dict())==weights_before
  checks['no_Core_physics_teacherbackend']=not any(n in sys.modules for n in ('mujoco','physics','flygym','static_teacher','student_wave'))
 path=ROOT/'TRAIN'/f'CONDITIONAL_{payload["step"]:06d}.npz';np.savez_compressed(path,**saved);resource_gate(start)
 save(ROOT/'TRAIN'/f'VERIFY_{payload["step"]:06d}.json',dict(integrity_all_pass=bool(all(checks.values())),checks=checks,step=payload['step'],checkpoint=a.checkpoint,checkpoint_sha256=a.sha256,rows=rows,conditional_arrays=dict(path=str(path),sha256=sha(path)),source_hashes=payload['config']['source_hashes'],data_hashes=payload['config']['data_hashes'],physical_steps=0,conditional_notphysicalcounterfactual=True,training_RSS_limit_MiB=1800))
 if not all(checks.values()):raise ValueError('Freshstatealignedsavedqualificationfailed')
 print('STATE_ALIGNED_FRESH_SAVED_VERIFY_PASS',payload['step'],flush=True)
if __name__=='__main__':main()
