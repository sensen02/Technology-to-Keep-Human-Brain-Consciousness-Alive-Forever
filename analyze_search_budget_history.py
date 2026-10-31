#!/usr/bin/env python3
"""S172/P1-S2: saved-only S169 finite-difference budget reconstruction.
No engine, neural code or optimization import/execution. NumPy + stdlib only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

HISTORY = Path('/home/sensen/Desktop/engine修复验收/s169-near-initial-support/20261024T233428554259Z')
PROJECT = Path('/run/media/sensen/Data2/cell_wound_prototype')
SCIPY_SOURCE = PROJECT / 'venv_body/lib/python3.12/site-packages/scipy/optimize/_numdiff.py'
FROZEN_PRODUCER = HISTORY / 'SOURCE_FREEZE/0366_newrun_near_initial_support.py'
CURRENT_PRODUCER = PROJECT / 'newrun_near_initial_support.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bits(x):
    return int(np.float64(x).view(np.uint64))


def number(b):
    return np.asarray(b, dtype=np.uint64).view(np.float64)[()]


def preimage(q, lower, width):
    """Exact inclusive binary64 preimage of NumPy lower + x*width, x in [0,1].
    Binary-search all nonnegative finite float bit patterns, not an analytic
    inverse with assumed rounding. Positive width makes this map monotone.
    """
    def value(b):
        return np.float64(lower + number(b) * width)
    def first(strict):
        left, right = bits(0.), bits(1.) + 1
        while left < right:
            mid = (left + right) // 2
            if (value(mid) > q if strict else value(mid) >= q):
                right = mid
            else:
                left = mid + 1
        return left
    a, b = first(False), first(True) - 1
    if a > b or a > bits(1.) or value(a) != q or value(b) != q:
        return None
    assert a == 0 or value(a - 1) < q
    assert b == bits(1.) or value(b + 1) > q
    return a, b


def fd(x):
    """Installed SciPy _numdiff binary64 2-point formula, normalized [0,1]."""
    x = np.asarray(x, dtype=np.float64)
    sign = (x >= 0).astype(x.dtype) * 2 - 1
    h = (1e-5 * sign * np.abs(x)).astype(x.dtype)
    h = np.where((x + h) - x == 0,
                 np.sqrt(np.finfo(np.float64).eps) * sign * np.maximum(1., np.abs(x)), h)
    lower_dist, upper_dist = x, 1. - x
    violated = (x + h < 0) | (x + h > 1)
    fitting = np.abs(h) <= np.maximum(lower_dist, upper_dist)
    h = np.where(violated & fitting, -h, h)
    h = np.where((upper_dist >= lower_dist) & ~fitting, upper_dist, h)
    h = np.where((upper_dist < lower_dist) & ~fitting, -lower_dist, h)
    return x + h, h, (x + h) - x


def exact_fd_match(qprobe, lo, width, interval):
    a, b = interval
    # Exhaustively test every possible hidden normalized input when tractable.
    # Otherwise explicitly decline exact classification (never silently relax).
    if b - a + 1 > 100000:
        return dict(status='UNRESOLVED_LARGE_PREIMAGE', possible_x_count=b-a+1)
    xs = np.arange(a, b + 1, dtype=np.uint64).view(np.float64)
    xp, h, dx = fd(xs)
    output = lo + xp * width
    found = np.flatnonzero(output == qprobe)
    if not len(found):
        return dict(status='EXACT_INCOMPATIBLE', possible_x_count=len(xs))
    k = int(found[0])
    return dict(status='EXACT_COMPATIBLE', possible_x_count=len(xs),
                compatible_x_count=len(found), witness_x=float(xs[k]),
                witness_x_hex=float(xs[k]).hex(), witness_x_perturbed=float(xp[k]),
                scipy_requested_adjusted_h=float(h[k]), scipy_actual_denominator=float(dx[k]),
                physical_predicted=float(output[k]), physical_observed=float(qprobe))


def analyze(history):
    z = np.load(history / 'FROZEN_SEARCH.npz', allow_pickle=False)
    idx, lo, hi, q0 = z['indices'], z['lower'], z['upper'], z['qpos0']
    width = hi - lo
    assert np.array_equal(idx, np.r_[2, np.arange(7,49)]) and len(idx) == 43
    assert np.all(width > 0)
    records = json.loads((history / 'EVALUATIONS.json').read_text())
    ids = [int(r['evaluation']) for r in records]
    assert len(ids) == len(set(ids))
    files = [history / f'EVAL_{i:03d}.npz' for i in ids]
    q = np.array([np.load(p, allow_pickle=False)['qpos'] for p in files])
    assert q.shape == (len(ids),49) and np.isfinite(q).all()
    fixed = np.setdiff1d(np.arange(49), idx)
    assert np.all(q[:,fixed] == q0[fixed])
    intervals = [[preimage(row[j],lo[k],width[k]) for k,j in enumerate(idx)] for row in q]
    assert all(v is not None for row in intervals for v in row)
    pair_evidence, unresolved = [], []
    coordinate_index = {int(v):k for k,v in enumerate(idx)}
    # Search all ordered pairs, NOT adjacent IDs or guessed 44-evaluation chunks.
    for a in range(len(q)):
        for b in range(len(q)):
            changed = np.flatnonzero(q[a] != q[b])
            if len(changed) != 1 or int(changed[0]) not in coordinate_index:
                continue
            j = int(changed[0]); k = coordinate_index[j]
            match = exact_fd_match(q[b,j],lo[k],width[k],intervals[a][k])
            detail = dict(anchor_evaluation=ids[a], probe_evaluation=ids[b],
                          normalized_coordinate=k, physical_qpos_coordinate=j,
                          exact_changed_physical_coordinate_count=1,
                          physical_delta=float(q[b,j]-q[a,j]), **match)
            if match['status'] == 'EXACT_COMPATIBLE': pair_evidence.append(detail)
            elif match['status'].startswith('UNRESOLVED'): unresolved.append(detail)
    anchors, probe_to_anchors = [], {}
    for a,eid in enumerate(ids):
        pairs = [p for p in pair_evidence if p['anchor_evaluation'] == eid]
        if not pairs: continue
        coords = sorted({p['normalized_coordinate'] for p in pairs})
        unique = len(coords) == len(pairs)
        complete = coords == list(range(43)) and unique
        xnom = (q[a,idx]-lo)/width
        nomq = lo+xnom*width
        ip = [dict(normalized_coordinate=k, physical_qpos_coordinate=int(idx[k]),
                   x_lower=float(number(v[0])),x_upper=float(number(v[1])),
                   x_lower_hex=float(number(v[0])).hex(),x_upper_hex=float(number(v[1])).hex(),
                   inclusive_float_count=v[1]-v[0]+1) for k,v in enumerate(intervals[a])]
        block = dict(anchor_evaluation=eid, classification='COMPLETE_43_PATTERN' if complete else 'PARTIAL_PATTERN',
                     matched_probe_count=len(pairs), coordinates=coords,
                     missing_coordinates=sorted(set(range(43))-set(coords)),
                     unique_probe_per_coordinate=unique, probes=pairs,
                     nominal_normalized_x=xnom.tolist(),
                     nominal_qpos_roundtrip_exact=bool(np.array_equal(nomq,q[a,idx])),
                     nominal_qpos_roundtrip_max_abs_error=float(np.max(abs(nomq-q[a,idx]))),
                     exact_binary64_normalized_preimages=ip)
        anchors.append(block)
        for p in pairs: probe_to_anchors.setdefault(p['probe_evaluation'],[]).append(eid)
    unique_probes = sorted(p for p,a in probe_to_anchors.items() if len(a)==1)
    ambiguous = {str(p):a for p,a in probe_to_anchors.items() if len(a)!=1}
    anchor_ids = {a['anchor_evaluation'] for a in anchors}
    unknown = sorted(set(ids)-set(unique_probes)-anchor_ids)
    assert not (set(unique_probes) & anchor_ids), 'Overlapping anchor/probe roles require separate uncertainty handling'
    best_path = history / 'BEST_DIAGNOSTIC_NOT_CANDIDATE.json'
    best = json.loads(best_path.read_text())
    bestid = int(best['evaluation'])
    paths = files + [history/'FROZEN_SEARCH.npz',history/'EVALUATIONS.json',best_path,
                     history/'OPTIMIZER_TERMINATION.json',FROZEN_PRODUCER,CURRENT_PRODUCER,SCIPY_SOURCE]
    hashes = {str(p.resolve()):sha(p) for p in paths}
    return dict(scope='S172/P1-S2 saved-only independent historical budget analysis',
        history_root=str(history.resolve()), new_calls=dict(model_load=0,forward=0,step=0,neural=0,optimization=0),
        input_hashes=hashes, producer_current_identical_to_frozen=sha(FROZEN_PRODUCER)==sha(CURRENT_PRODUCER),
        scipy_source=dict(path=str(SCIPY_SOURCE),version_environment_observed='1.18.1',
            historical_scipy_version_proven=False,
            lines=dict(bound_adjustment='14-90',step='147-203',dispatch='580-613',dense_formula='683-711')),
        method=dict(pair_search='all ordered evaluation pairs; exactly one physical coordinate changed bitwise',
            inverse='exact binary64 inclusive preimage via monotone bit-pattern binary search; all x candidates exhaustively tested',
            normalized_bounds=[0.,1.],diff_step=1e-5,
            formula='h=rel_step*sign(x)*abs(x); zero-effective-step fallback sqrt(eps)*sign(x)*max(1,abs(x)); one-sided bound flip/shrink; xp=fl(x+h); dx=fl(xp-x); q=fl(lower+fl(x*width))',
            exact_compatible_not_unique_hidden_optimizer_x=True,
            role_evidence='Exact constitutive pattern and frozen producer; no callback/optimizer internal state trace'),
        counts=dict(saved_evaluations=len(ids),unique_fd_compatible_probes=len(unique_probes),
            complete_fd_base_patterns=sum(a['classification']=='COMPLETE_43_PATTERN' for a in anchors),
            partial_fd_base_patterns=sum(a['classification']=='PARTIAL_PATTERN' for a in anchors),
            base_pattern_total=len(anchors),unknown_role_evaluations=len(unknown),ambiguous_probes=len(ambiguous)),
        anchors=anchors,unique_probe_evaluations=unique_probes,unknown_role_evaluations=unknown,
        ambiguous_probe_anchors=ambiguous,unresolved_pairs=unresolved,
        best_diagnostic=dict(evaluation=bestid,compatible_anchor_evaluations=probe_to_anchors.get(bestid,[]),
                             role='unique finite-difference-compatible probe' if bestid in unique_probes else 'not uniquely classified'),
        role_uncertainty=dict(accepted_optimizer_update_count=None,rejected_trial_count=None,
            callback_trace_saved=False, accepted_field_in_evaluations='physical acceptance gate, NOT optimizer step acceptance',
            statement='Complete/partial FD base pattern counts are not accepted optimizer update counts. Accepted versus trial bookkeeping is unavailable from the saved callback-free log; exact floating-pattern compatibility alone is not an optimizer trace.',
            conservative_noninitial_logged_nonprobe_candidate_count=max(0,len(ids)-len(unique_probes)-1)),
        interpretation_zh='99次姿态评估不能称99次独立优化更新。这里以全部两两物理坐标差与精确浮点前像证明探针模式，不靠相邻编号猜块。模式相容不是回调轨迹：不宣称优化器接受更新次数；也不把EVALUATIONS的accepted物理门当优化器接受。历史失败不等于全域不可行。')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputroot',type=Path,required=True)
    parser.add_argument('--historyroot',type=Path,default=HISTORY)
    args=parser.parse_args()
    out=args.outputroot.resolve(); history=args.historyroot.resolve()
    assert out != history and history not in out.parents, 'Never write into historical frozen root'
    out.mkdir(parents=True,exist_ok=True)
    target=out/'HISTORY_BUDGET_ANALYSIS.json'
    assert not target.exists(), 'Refuse to overwrite prior analysis'
    report=analyze(history)
    assert all(sha(Path(p))==h for p,h in report['input_hashes'].items())
    report['input_hashes_unchanged_after_analysis']=True
    report['analysis_script']=dict(path=str(Path(__file__).resolve()),sha256=sha(Path(__file__)))
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(target),counts=report['counts'],best=report['best_diagnostic']),ensure_ascii=False))


if __name__ == '__main__': main()
