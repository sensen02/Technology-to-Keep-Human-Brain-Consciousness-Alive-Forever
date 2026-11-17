"""Freshprocess0/300/600 strictNNstorage, 12source+8variableDEV productioncache."""
import argparse
import sys
from sources import ROOT,read,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--sha256',required=True);p.add_argument('--execute-parent-saved-verify',action='store_true');a=p.parse_args()
    if not a.execute_parent_saved_verify:p.error('Parentfreshsavedverifyrequired')
    from learning import cpu1,rng_snapshot,rng_equal
    cpu1()
    import torch
    import common,model
    from data import prepare,production_sequence,dev_sources
    before=rng_snapshot();m,payload=common.load_checkpoint(a.checkpoint,a.sha256,final=False);after=rng_snapshot();checks={'strictloader_RNG_unchanged':rng_equal(before,after),'oldweights_frozen':m.verify_old()}
    data=prepare(m);scan=model.make_scan(m);rows=[]
    with torch.no_grad():
        for i,raw in enumerate(data['raw']+dev_sources(read(ROOT/'INPUT_BINDINGS.json'))):
            state=m.initial_state(1)
            for row in raw['warm']:_,state=m.prime_old(row[None],row.new_zeros((1,2)),state)
            oldmean,features,oldstates=m.cache_sequence(raw['x'][:,None],raw['u'][:,None],state[:3]);res,h=scan(features);prod,hidden=production_sequence(m,raw)
            err=float((oldmean[:,0]+res[:,0]-prod).abs().max());herr=float((h[:,0]-hidden[:,256:]).abs().max());oldexact=torch.equal(hidden[:,:256],torch.cat(tuple(v[:,0] for v in oldstates),-1));tag=raw['entry']['id']+str(i)
            checks[tag+'_production_cache']=err<=1e-12 and herr<=1e-12 and oldexact
            checks[tag+'_412directoldmeanfeature']=features.shape[-1]==412 and torch.equal(features[:,:,-42:],oldmean)
            rows.append(dict(id=raw['entry']['id'],training=i<12,labels_available=i<12,actual_rows=len(raw['x']),mean_max_error=err,new96_max_error=herr,old256exact=oldexact))
    checks['no_physical_or_teacherquery']=not any(k in sys.modules for k in ('mujoco','physics','flygym','student_wave'))
    checks['8Adam_moments_bound']=payload['step']==0 and not payload['optimizer_state_dict']['state'] or len(payload['optimizer_state_dict']['state'])==8
    checks['state352']=m.identity()['state_dims']==[128,64,64,96]
    save(ROOT/'TRAIN'/f'VERIFY_{payload["step"]:06d}.json',dict(integrity_all_pass=bool(all(checks.values())),checks=checks,step=payload['step'],checkpoint=a.checkpoint,checkpoint_sha256=a.sha256,rows=rows,physical_steps=0,source_hashes=payload['config']['source_hashes'],data_hashes=payload['config']['data_hashes'],DEV_teacher_labels_available=False))
    if not all(checks.values()):raise ValueError('Saved12plus8variableDEVverificationfailed')
    print('LOCALMEAN_FRESH_SAVED_VERIFY_PASS',payload['step'],flush=True)


if __name__=='__main__':main()
