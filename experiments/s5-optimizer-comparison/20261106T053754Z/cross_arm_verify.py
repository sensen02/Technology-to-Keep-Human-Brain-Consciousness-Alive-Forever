"""Post-freeze independent duplicate-point determinism audit; no stepping."""
from pathlib import Path
import numpy as np
import run as r

def main():
    summary=r.js(r.O/'SUMMARY.json');checks=[];vectors=[]
    for m in summary['candidate_metrics'][1:]:vectors.append(r.js(r.O/'CONFIGS'/m['candidate']/'PARAMETERS.json')['x'])
    for i in range(4):
        a,b=f'diagonal_g0_c{i}',f'de_g0_c{i}'
        assert np.array_equal(np.array(r.js(r.O/'CONFIGS'/a/'PARAMETERS.json')['x']),np.array(r.js(r.O/'CONFIGS'/b/'PARAMETERS.json')['x']))
        wa=r.arrays(r.O/'CONFIGS'/a/'WARM_SPIKES.npz');wb=r.arrays(r.O/'CONFIGS'/b/'WARM_SPIKES.npz');assert all(np.array_equal(wa[k],wb[k]) for k in wa)
        for ph in r.TRAIN+[r.GUARD]:
            ar=next(v for v in summary['results'] if v['candidate']==a and v['release_time_s']==ph and '_train_guard_' in v['label']);br=next(v for v in summary['results'] if v['candidate']==b and v['release_time_s']==ph and '_train_guard_' in v['label']);assert ar['J']==br['J']
            for file in ['BODY.npz','NATIVE.npz','NEURAL.npz']:
                aa=r.arrays(Path(ar['folder'])/file);bb=r.arrays(Path(br['folder'])/file);assert set(aa)==set(bb) and all(np.array_equal(aa[k],bb[k]) for k in aa)
            checks.append(dict(a=a,b=b,phase=ph,J_exact=True,all_physics_neural_arrays_exact=True))
    r.dump(r.O/'CROSS_ARM_VERIFICATION.json',dict(passed=True,paid_candidates=len(vectors),unique_parameter_vectors=len(np.unique(np.asarray(vectors),axis=0)),exact_matching_initial_points=4,paired_releases=len(checks),details=checks,note='same default_rng seed makes first4 ES and DE points equal; each arm still independently charged/evaluated. No extra rollouts.'))
    print('CROSS_ARM_VERIFIED',len(checks),flush=True)
if __name__=='__main__':main()
