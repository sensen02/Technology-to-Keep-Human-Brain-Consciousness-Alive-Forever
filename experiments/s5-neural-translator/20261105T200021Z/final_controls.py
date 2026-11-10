"""Paired final controls, chosen from development data only; never retrain."""
from pathlib import Path
import numpy as np
import runtime as rt
from translator import NumpyTranslator
O=rt.O;js=rt.js;dump=rt.dump;sha=rt.sha

def actor(path):
    model=NumpyTranslator(path);return lambda x,stochastic:(model(x),0.,0.)
def key(rs):return(sum(r['validation']['passed'] for r in rs),-max(r['J'] for r in rs),-np.mean([r['J'] for r in rs]))
def main():
    s=js(O/'SUMMARY.json');history=js(O/'VALIDATION_HISTORY.json');groups={}
    for r in history:
        if r['label'].startswith('checkpoint_') or r['label'].startswith('initial_offline_mlp'):
            groups.setdefault(r['model_file'],[]).append(r)
    path,development=max(groups.items(),key=lambda pair:key(pair[1]));zero=O/'MODELS/zero.npz'
    # This merely exhibits the best unsuccessful nonlinear attempt when the incumbent is zero.
    dump(O/'FINAL_CONTROLS_PREREGISTRATION.json',dict(model_file=path,model_sha256=sha(path),selection='Only existing four-phase development results; final heldout data excluded; no deployment selection changes or training.',development=development,tests='frozen trained MLP at representative1.3 and four final phases, pair with zero-residual baseline; reuse exact existing same-model/same-phase results instead of duplicate simulations',budget_remaining=64-s['validation_evaluations'],source=str(Path(__file__)),source_sha256=sha(__file__)))
    t=rt.Runtime();t.load_cache();extra=[]
    unused=O/'COVERAGE.json'
    if unused.exists():
        ledger=js(O/'CLEANUP_LEDGER.json');ledger['deleted'].append(dict(path=str(unused),sha256=sha(unused),bytes=unused.stat().st_size,classification_before_removal='可再生临时',reason='Legacy constructor regenerated the same unused dominant-joint coverage during final controls; reference audit still applies.'));unused.unlink();dump(O/'CLEANUP_LEDGER.json',ledger)
    def get(model,time,label):
        candidates=s['final_heldout']+[s['representative']]
        for r in candidates:
            if r['release_time_s']==time and sha(r['model_file'])==sha(model):return r
        r=t.episode(label,time,actor(model),model,duration=s['stage_reached_s'])[0];extra.append(r);return r
    representative=get(path,1.3,'EXHIBIT_trained_MLP');pairs=[]
    for time in rt.TEST:pairs.append(dict(release_time_s=time,old=get(zero,time,'PAIRED_old_'+str(time)),trained_MLP=get(path,time,'PAIRED_trained_'+str(time))))
    assert s['validation_evaluations']+len(extra)<=64
    dump(O/'FINAL_CONTROLS.json',dict(trained_MLP_file=path,development=development,representative=representative,paired_final=pairs,additional_evaluations=len(extra),deployment_unchanged=True,importance='核心必需'))
    print('PAIRED FINAL CONTROLS COMPLETE',len(extra),flush=True)
if __name__=='__main__':main()
