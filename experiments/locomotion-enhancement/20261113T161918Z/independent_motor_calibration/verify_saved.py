"""Fresh-process strict saved NN checkpoint replay, no physical backend."""
import argparse
import sys
from sources import ROOT,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--sha256',required=True);p.add_argument('--execute-parent-saved-verify',action='store_true');a=p.parse_args()
    if not a.execute_parent_saved_verify:p.error('Parent savedNN verification flag required')
    from learning import cpu1,rng_snapshot,restore_rng
    cpu1()
    import torch,numpy as np,random
    import common,model
    from data import prepare,production_sequence
    torch.manual_seed(20261113);np.random.seed(20261113);random.seed(20261113)
    before=rng_snapshot();m,payload=common.load_checkpoint(a.checkpoint,a.sha256,final=False);after=rng_snapshot();checks={}
    checks['loader_RNG_exact']=torch.equal(before['torch'],after['torch']) and before['python']==after['python'] and before['numpy'][0]==after['numpy'][0] and np.array_equal(before['numpy'][1],after['numpy'][1]) and before['numpy'][2:]==after['numpy'][2:]
    checks['eight_Adamstates_exact_step']=payload['step']==0 and not payload['optimizer_state_dict']['state'] or len(payload['optimizer_state_dict']['state'])==8
    checks['all_old789f_frozen']=all(not p.requires_grad for p in m.old.parameters())
    data=prepare(m);scan=model.make_scan(m);errors=[];state_errors=[]
    with torch.no_grad():
        for i,raw in enumerate(data['raw']):
            residual,h=scan(data['features'][:,i:i+1]);expected=data['oldmean'][:,i]+residual[:,0];prod,hidden=production_sequence(m,raw)
            error=float((expected-prod).abs().max());herr=float((h[:,0]-hidden[:,256:]).abs().max());errors.append(error);state_errors.append(herr)
            checks[raw['entry']['id']+'_savedcache_production_mean']=error<=1e-10
            checks[raw['entry']['id']+'_savedcache_production_new96']=herr<=1e-10
            checks[raw['entry']['id']+'_savedold256_exact']=torch.equal(hidden[:,:256],data['features'][:,i,112:368])
    checks['no_physical_backend_import']=not any(n in sys.modules for n in ('mujoco','physics','flygym','student_wave'))
    checks['step_in0_300_600']=payload['step'] in (0,300,600)
    target=ROOT/'TRAIN'/f'VERIFY_{payload["step"]:06d}.json'
    save(target,dict(integrity_all_pass=bool(all(checks.values())),checks=checks,checkpoint=a.checkpoint,checkpoint_sha256=a.sha256,step=payload['step'],production_cache_mean_errors=errors,production_cache_state_errors=state_errors,physical_steps=0,source_hashes=payload['config']['source_hashes'],data_hashes=payload['config']['data_hashes']))
    if not all(checks.values()):raise ValueError('Saved checkpoint replay failed')
    print('SAVED_NN_VERIFY_PASS',payload['step'],flush=True)


if __name__=='__main__':main()
