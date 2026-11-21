"""Independent pair-wise checks for collisions, channel loss and retained drift."""
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from evaluate import obs, reference
from sorting import detect


def main(out):
    m = json.loads((out/'result.json').read_text())
    labels = {int(k): v for k,v in m['root_map'].items()}
    ref = reference(11701,'intact')
    data = np.load(out/'predictions.npz')
    pairs = [(a,b) for a,b in combinations(ref,2) if a[2]!=b[2] and min(abs(a[0]-b[0]),abs(a[1]-b[1]))<=1]
    rows = {}
    for method in ('single_best','single_peel','multi','combined'):
        calls = data['intact_'+method]
        names = {(h,t,labels.get(k)) for h,t,k in calls}
        rows[method] = dict(all_pairs=len(pairs),both_named=sum(a in names and b in names for a,b in pairs))
    p = np.load(out/'templates.npz')
    v = obs(11701,'intact')
    def solve(voltage):
        return detect(voltage, p['bank'], p['signed_lag'], p['scale'], multichannel=True, subtract=True)
    assert np.array_equal(solve(v),data['intact_combined'])
    noise = np.random.default_rng(781).normal(0., .08, v.shape)
    controls = dict(noise_calls=len(solve(noise)), head_muted=len(solve(np.column_stack((v[:,:8]*0,v[:,8:])))),
                    thorax_muted=len(solve(np.column_stack((v[:,:8],v[:,8:]*0)))))
    phase = {}
    for mode in ('intact','intact_drift'):
        subset = [r for r in ref if min(r[:2]) >= 12000 and r[0]<r[1]]
        from collections import Counter
        guesses = Counter((h,t) for h,t,_ in data[mode+'_combined'] if h<t and min(h,t)>=12000)
        actual = Counter(r[:2] for r in subset)
        phase[mode] = dict(correct=(guesses & actual).total(),called=guesses.total(),down=len(subset))
    report=dict(collision_pairs=rows,controls=controls,steady_half_gain=phase)
    (out/'checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)


if __name__ == '__main__':
    import sys
    main(Path(sys.argv[1]))
