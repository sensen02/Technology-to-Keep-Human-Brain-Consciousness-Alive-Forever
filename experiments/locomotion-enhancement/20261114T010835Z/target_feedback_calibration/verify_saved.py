"""Freshsaved20actualX and12conditionalfeedback fullproduction dualreplay, noCore."""
import argparse,sys
from sources import ROOT,read,save,sha


def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--sha256',required=True);p.add_argument('--execute-parent-saved-verify',action='store_true');a=p.parse_args()
 if not a.execute_parent_saved_verify:p.error('Parentfreshsavedverify required')
 from learning import cpu1,rng_snapshot,rng_equal,slice_data,forward
 cpu1()
 import torch
 import common,model
 from scan import make_scan
 from data import prepare,production_sequence,dev_sources
 from utils import production_feedback,output_errors
 rng=rng_snapshot();m,payload=common.load_checkpoint(a.checkpoint,a.sha256,final=False);checks={'strictload_RNG_unchanged':rng_equal(rng,rng_snapshot()),'oldweights_frozen':m.verify_old()};data=prepare(m);conditional=make_scan(m);actualscan=model.make_scan(m);rows=[];saved_conditional={};weights_before=model.state_digest(m.state_dict())
 with torch.no_grad():
  for i,raw in enumerate(data['raw']+dev_sources(read(ROOT/'INPUT_BINDINGS.json'))):
   state=m.initial_state(1)
   for x in raw['warm']:_,state=m.prime_old(x[None],x.new_zeros((1,2)),state)
   oldmean,features,oldstates=m.cache_sequence(raw['x'][:,None],raw['u'][:,None],state[:3]);res,h=actualscan(features);prod,hidden=production_sequence(m,raw);err=float((oldmean[:,0]+res[:,0]-prod).abs().max());herr=float((h[:,0]-hidden[:,256:]).abs().max());oe=torch.equal(hidden[:,:256],torch.cat(tuple(s[:,0] for s in oldstates),-1));tag=raw['entry']['id']+str(i)
   checks[tag+'_actualX_production_cache']=err<=1e-12 and herr<=1e-12 and oe
   rows.append(dict(id=raw['entry']['id'],training=i<12,actual_rows=len(raw['x']),actualX_mean_error=err,new96_error=herr,old256exact=oe,DEV_labels_available=False if i>=12 else None))
   if i<12:
    one=slice_data(data,i,i+1);out=forward(conditional,one);full,oldfull=production_feedback(m,raw,data,i);errors=output_errors(tuple(v[:,0] for v in out),full);checks[tag+'_conditional_fullproduction']=max(errors)<=1e-12 and torch.equal(oldfull,data['oldafter'][:,i])
    checks[tag+'_conditional_finite']=all(bool(torch.isfinite(v).all()) for v in out);rows[-1]['conditional_output_errors']=errors
    for key,value in zip(('mean','residual','new96','pretarget','posttarget','action'),out):saved_conditional[tag+'_'+key]=value[:,0].numpy().copy()
 checks['no_physical_or_teacherbackend']=not any(n in sys.modules for n in ('mujoco','physics','flygym','static_teacher','student_wave'))
 checks['canonical8Adamstates']=payload['step']==0 and not payload['optimizer_state_dict']['state'] or len(payload['optimizer_state_dict']['state'])==8
 checks['all_model_weights_unchanged']=model.state_digest(m.state_dict())==weights_before
 import numpy as np
 conditional_path=ROOT/'TRAIN'/f'CONDITIONAL_{payload["step"]:06d}.npz';np.savez_compressed(conditional_path,**saved_conditional)
 save(ROOT/'TRAIN'/f'VERIFY_{payload["step"]:06d}.json',dict(integrity_all_pass=bool(all(checks.values())),checks=checks,step=payload['step'],checkpoint=a.checkpoint,checkpoint_sha256=a.sha256,rows=rows,conditional_arrays=dict(path=str(conditional_path),sha256=sha(conditional_path)),source_hashes=payload['config']['source_hashes'],data_hashes=payload['config']['data_hashes'],physical_steps=0,conditional_notphysicalcounterfactual=True))
 if not all(checks.values()):raise ValueError('Freshsavedconditionalqualificationfailed')
 print('TARGETFEEDBACK_FRESH_SAVED_VERIFY_PASS',payload['step'],flush=True)


if __name__=='__main__':main()
