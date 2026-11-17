"""Fresh strictnewFORMAT audit: exactprior8 replay plus4unlabeledDEVcache parity."""
from adapter import run
from sources import ROOT,read,save,sha


def main():
    run('verify_saved.py')
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--sha256',required=True);p.add_argument('--execute-parent-saved-verify',action='store_true');a=p.parse_args()
    import torch
    import common,model
    from data import dev_sources,production_sequence
    m,payload=common.load_checkpoint(a.checkpoint,a.sha256,final=False);scan=model.make_scan(m);checks={};rows=[]
    with torch.no_grad():
        for raw in dev_sources(read(ROOT/'INPUT_BINDINGS.json')):
            state=m.initial_state(1)
            for row in raw['warm']:_,state=m.prime_old(row[None],row.new_zeros((1,2)),state)
            oldmean,features,oldstates=m.cache_sequence(raw['x'][:,None],raw['u'][:,None],state[:3]);res,h=scan(features)
            prod,hidden=production_sequence(m,raw);error=float((oldmean[:,0]+res[:,0]-prod).abs().max());herr=float((h[:,0]-hidden[:,256:]).abs().max())
            oldexact=all(torch.equal(oldstates[i][:,0],hidden[:,sum((128,64,64)[:i]):sum((128,64,64)[:i+1])]) for i in range(3))
            checks[raw['entry']['id']+'_unlabeledDEV_productioncache']=error<=1e-12 and herr<=1e-12 and oldexact
            rows.append(dict(id=raw['entry']['id'],mean_max_error=error,new96_max_error=herr,old256_exact=oldexact,teacher_labels_available=False,behavior_PASS=None))
    save(ROOT/'TRAIN'/f'DEV_VERIFY_{payload["step"]:06d}.json',dict(integrity_all_pass=bool(all(checks.values())),checks=checks,step=payload['step'],checkpoint_sha256=a.sha256,rows=rows,physical_steps=0,teacher_labels_available=False))
    if not all(checks.values()):raise ValueError('FreshDEVsavedparityfailed')


if __name__=='__main__':main()
