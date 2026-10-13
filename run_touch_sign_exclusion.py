"""Read-only source sign-coverage audit, no network build or sign substitution."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
from pathlib import Path
import signal
import time
import numpy as np
from engine import memory_guard as mg

HERE = Path(__file__).resolve().parent
OUT = HERE/'outputs/neural_muscle_closure'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def main():
    start = time.monotonic()
    paths = [OUT/'touch_path_audit_spikes.npz',HERE/'data/flywire/banc_connectome.npz',HERE/'engine/neck_cut_data.py',Path(__file__).resolve()]
    hashes = {str(p):sha(p) for p in paths}
    with np.load(paths[0],allow_pickle=False) as b:
        tier = b['root_ids']; roots450 = tier[b['lf_bristle_indices']]
    with np.load(paths[1],allow_pickle=False) as a:
        roots,pre,post,syn,labels = (a[k] for k in ('root_ids','pre','post','syn','ann_nt_verified'))
        mask = np.isin(roots,roots450); inside = np.isin(roots,tier)
        outgoing = mask[pre]
        lab,count = np.unique(labels[mask],return_counts=True)
        assert mask.sum() == 450 and len(np.unique(roots450)) == 450
        counts = {'lf_bristle_nt_verified_hist':dict(zip(lab.tolist(),count.tolist())), 'cells':int(mask.sum())}
        for name, rows in [('full',outgoing),('intier',outgoing & inside[post]),('outside_tier',outgoing & ~inside[post])]:
            counts['outgoing_'+name+'_rows'] = int(rows.sum())
            counts['outgoing_'+name+'_synapses'] = float(syn[rows].sum())
        assert counts['lf_bristle_nt_verified_hist'] == {'':450}
        assert counts['outgoing_full_rows'] == counts['outgoing_intier_rows']+counts['outgoing_outside_tier_rows']
    assert all(sha(Path(p)) == h for p,h in hashes.items())
    payload = {'status':'MEASURED_SOURCE_SIGN_EXCLUSION_NOT_REPAIR', 'source_sha256':hashes,
        'lf_bristle_root_ids':[int(r) for r in roots450], 'measured':counts,
        'method':'Use exact LF bristle root IDs from verified touch replay NPZ; match global root_ids; select raw rows with pre in this set; partition by post root present in exact replay tier. Count rows and sum syn without weight cap. Histogram ann_nt_verified over all450 roots.',
        'sign_rule_evidence':{'file':str(paths[2]),'lines':'1163-1167;1200',
            'rule':'annotation mode selects inside & pure verified presynaptic transmitter label; blank labels fail pure. Matrix row=post,column=pre.'},
        'conclusion':'All450 LF bristle verified transmitter labels are blank. Raw outgoing connections exist, mostly within tier, but the annotation-only sign rule excludes them from We/Wi. This explains zero weighted outgoing edges measured by touch_path_audit.',
        'limitations':['Source cached annotation completeness is not biological absence of transmitter.',
            'Raw edge rows and synapse sums are not physiological conductance or causal behavioral proof.',
            'No counterfactual acetylcholine assignment, sign replacement, network gain or wiring change performed.',
            'This artifact does not independently validate raw connectome anatomy or recover missing transmitter evidence.'],
        'next_useful_scope':'Resolve transmitter evidence for these exact root IDs and document uncertainty before proposing signed connectivity changes.',
        'wall_s':time.monotonic()-start}
    (OUT/'touch_sign_exclusion.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(json.dumps(counts,indent=2))

if __name__ == '__main__':
    def deadline(*_):
        raise TimeoutError('60-second source-audit bound exceeded')
    signal.signal(signal.SIGALRM,deadline); signal.alarm(60)
    with mg.guarded_run('touch_sign_exclusion',**mg.BUDGETS['neural_muscle_closure']):
        main()
    signal.alarm(0)
