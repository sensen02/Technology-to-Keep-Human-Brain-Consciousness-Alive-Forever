from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from sorting import learn, detect

PARENT = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261120T043500Z-neck-recording')
sys.path.insert(0, str(PARENT))
from decode import decode
from run import score

ARMS = ('intact','tear_an','block_an','block_dn','dropout_an','dropout_dn')
METHODS = dict(single_best=(False,False), single_peel=(False,True), multi=(True,False), combined=(True,True))


def obs(seed, arm, drift=False):
    path = (PARENT/'persistent'/f'{seed}_{arm}.npz' if drift else
            PARENT/'recordings'/f'{seed}_{arm}_observation.npz')
    with np.load(path) as z:
        return np.column_stack((z['head_uv'], z['thorax_uv']))


def reference(seed, arm):
    params = json.loads(Path(json.loads((PARENT/'recordings/RESULT.json').read_text())
                             ['geometry']['parameters']).read_text())['ports']
    events = []
    with np.load(PARENT/'recordings'/f'{seed}_{arm}_reference.npz') as z:
        arrival, entry = z['arrival'], z['entry']
        for i, (root, direction) in enumerate(zip(z['roots'],z['direction'])):
            end = np.flatnonzero(arrival[:,i])
            start = end - params[root]['delay_ticks']
            assert np.all(start >= 0) and np.all(entry[start,i])
            h,t = (start,end) if direction == 'DN' else (end,start)
            events.extend(zip(h,t,[str(root)]*len(h)))
    return sorted(events)


def associate(cal_predictions, seeds):
    votes = defaultdict(Counter)
    for pred, seed in zip(cal_predictions,seeds):
        labels = defaultdict(set)
        for h,t,r in reference(seed,'intact'):
            labels[h,t].add(r)
        for h,t,k in pred:
            roots = labels[h,t]
            if len(roots) == 1:
                votes[int(k)].update(roots)
    # Scoring-only names for anonymous voltage templates. Ambiguous identities stay unknown.
    out = {}
    for k, counts in votes.items():
        r,n = counts.most_common(1)[0]
        if n >= 10 and n >= .90*counts.total():
            out[k] = r
    return out


def grade(pred, ref, mapping):
    pairs = np.array([(h,t) for h,t,_ in ref],int)
    result = score(pred[:,:2], pairs)
    collisions = []
    sites = [defaultdict(list), defaultdict(list)]
    for h,t,r in ref:
        sites[0][h].append(r)
        sites[1][t].append(r)
    for h,t,r in ref:
        if any(other != r for i,tick in enumerate((h,t)) for j in (tick-1,tick,tick+1)
               for other in sites[i][j]):
            collisions.append((h,t,r))
    found_pairs = Counter(map(tuple,pred[:,:2]))
    overlap = Counter((h,t) for h,t,_ in collisions)
    # A repeated pair from two simultaneous roots counts at most as often as predicted.
    observed = Counter((h,t,mapping.get(k)) for h,t,k in pred if k in mapping)
    result.update(overlap=dict(correct=sum((overlap & found_pairs).values()), total=len(collisions)),
                  root_and_time=dict(correct=sum((observed & Counter(ref)).values()), calls=observed.total(),
                                     total=len(ref), overlap_correct=sum((observed & Counter(collisions)).values())))
    return result


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    seeds = (10901,11311)
    cal = [obs(s,'intact') for s in seeds]
    bank, delays, scale, support = learn(cal)
    np.savez(out/'templates.npz', bank=bank, signed_lag=delays, scale=scale, support=support)
    def predict(v, method):
        if method == 'legacy':
            pairs,_ = decode(v[:,:8],v[:,8:])
            return np.column_stack((pairs,np.full(len(pairs),-1,int)))
        multi, peel = METHODS[method]
        return detect(v,bank,delays,scale,multichannel=multi,subtract=peel)

    # All voltage-only predictions are written before reading reference events.
    keys = [(a, False) for a in ARMS] + [('intact',True),('tear_an',True)]
    archive = {}
    for arm, drift in keys:
        key = arm+('_drift' if drift else '')
        v = obs(11701,arm,drift)
        for m in ('legacy', *METHODS):
            archive[key+'_'+m] = predict(v,m)
    np.savez_compressed(out/'predictions.npz',**archive)
    named = associate([predict(v,'combined') for v in cal], seeds)
    results = {}
    for arm, drift in keys:
        key = arm+('_drift' if drift else '')
        ref = reference(11701,arm)
        results[key] = {m:grade(archive[key+'_'+m],ref,named) for m in ('legacy',*METHODS)}
    (out/'result.json').write_text(json.dumps(dict(templates=len(bank),root_map=named,results=results),indent=2)+'\n')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
