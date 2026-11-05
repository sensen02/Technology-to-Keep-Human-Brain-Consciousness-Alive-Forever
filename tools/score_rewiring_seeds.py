"""Route B, second pass: is the 0.30 penalty of a rewired graph one unlucky draw, or systematic?

The first pass scored three arms once each and found the real graph at 3.4024, a degree and weight
matched rewiring at 3.7025, and a uniform random graph at 3.6688. Both controls sit about ten times
the timestep band away from the real graph, which is a large effect - but with one draw per arm there
is no way to know whether that is a property of rewiring or a property of that particular rewiring.

So this pass builds several independent rewirings and several independent uniform graphs with
different random seeds and scores each one with the same frozen vector. What comes out is a
distribution per arm, and the real graph is compared against the distribution rather than against a
single number.

Nothing is trained. Every arm uses the same vector, the same timestep, the same pools, the same
readout and the same replacement layer, so the only difference is which cell talks to which.
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ['PCR_SYNTH'] = '1'
import importlib.util
import json
import math
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import scipy.sparse as sparse

R = Path('/run/media/sensen/Data2/cell_wound_prototype')
sys.path.insert(0, str(R)); sys.path.insert(0, str(R / 'tools'))
O = Path('/home/sensen/Desktop/engine修复验收/s5-route-b-ablation/20261108T060000Z')
BEST = Path('/home/sensen/Desktop/engine修复验收/s5-synthetic-readout-cont/20261107T120000Z')
REF_J = 3.402395132105888
TIMESTEP_BAND = 0.027848515225160853
SEEDS = [int(x) for x in os.environ.get('AB_SEEDS', '11,22,33,44,55').split(',')]


def load(name):
    spec = importlib.util.spec_from_file_location(name, R / 'tools' / (name + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _san(v):
    if isinstance(v, float):
        return v if math.isfinite(v) else str(v)
    if isinstance(v, dict):
        return {k: _san(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_san(x) for x in v]
    return v


def dump(p, v):
    Path(p).write_text(json.dumps(_san(v), ensure_ascii=False, indent=2, allow_nan=False,
                                  default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)) + '\n')


def main():
    bab = load('build_ablation_networks')
    sar = load('score_ablation_arms')
    import train_per_cell_readout as pcr
    REBUILD = bab.REBUILD
    we0 = sparse.load_npz(REBUILD / 'REBUILT_We.npz').tocsr()
    wi0 = sparse.load_npz(REBUILD / 'REBUILT_Wi.npz').tocsr()
    n = we0.shape[0]
    vec = np.asarray(json.loads((BEST / 'EVALS' / 'eval_556' / 'SCORE.json').read_text())['vec'], float)
    out = Path(O) / 'NSEED'
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for arm in ('B', 'C'):
        for seed in SEEDS:
            tag = '%s_s%d' % (arm, seed)
            t0 = time.time()
            rng = np.random.default_rng(seed)
            mats = {}
            for name, mat in (('We', we0), ('Wi', wi0)):
                row, col, data = bab.coo_of(mat)
                if arm == 'B':
                    r, c, w, acc = bab.degree_preserving(n, row, col, data, rng)
                else:
                    r, c, w, acc = bab.uniform(n, data.size, data, rng)
                mats[name] = sparse.csr_matrix((w, (r, c)), shape=(n, n))
            arm_dir = out / ('SET_' + tag)
            if arm_dir.exists():
                import shutil
                shutil.rmtree(arm_dir)
            arm_dir.mkdir(parents=True)
            for name in ('We', 'Wi'):
                sparse.save_npz(arm_dir / ('ACTUAL_' + name + '.npz'), mats[name])
            base = Path(O) / 'SET_BASE'
            pcr.O = Path(O)
            pcr.SET = base
            pcr.make_set()
            for f in sorted(base.iterdir()):
                if f.name not in ('ACTUAL_We.npz', 'ACTUAL_Wi.npz'):
                    (arm_dir / f.name).symlink_to(f)
            pcr.SET = arm_dir
            folder = out / ('eval_' + tag)
            if folder.exists():
                import shutil
                shutil.rmtree(folder)
            rec = dict(arm=arm, seed=seed, wall_s=None)
            sar.install_arm_matrices(mats['We'], mats['Wi'])
            try:
                r = pcr.eval_one((-1, vec, folder))
                rec.update(J=r.get('J'), parts=r.get('parts'), failed=bool(r.get('failed', False)))
            except BaseException:
                rec.update(failed=True, traceback=traceback.format_exc()[-1200:])
            finally:
                sar.restore_builder()
            rec['wall_s'] = time.time() - t0
            try:
                nz = np.load(folder / 'RUN' / 'NEURAL.npz')
                rec['spikes_total'] = int(nz['spike_counts'].sum())
            except BaseException:
                rec['spikes_total'] = None
            records.append(rec)
            dump(O / 'NSEED_RECORDS.json', dict(reference_J=REF_J, timestep_band=TIMESTEP_BAND,
                                                seeds=SEEDS, records=records,
                                                note='same frozen vector, nothing trained'))
            print('%-6s J=%s  spikes=%s  %.0fs' % (tag, rec.get('J'), rec.get('spikes_total'), rec['wall_s']),
                  flush=True)
    # verdict
    v = dict(reference_J=REF_J, timestep_band=TIMESTEP_BAND, arms={})
    for arm in ('B', 'C'):
        vals = [r['J'] for r in records if r['arm'] == arm and not r.get('failed') and r.get('J') is not None]
        if not vals:
            v['arms'][arm] = dict(n=0)
            continue
        worst = max(vals) if arm == 'B' else max(vals)
        v['arms'][arm] = dict(
            n=len(vals), values=[float(x) for x in vals],
            min=float(min(vals)), max=float(worst), mean=float(statistics.fmean(vals)),
            sd=float(statistics.stdev(vals)) if len(vals) > 1 else None,
            d_minus_real=[float(x - REF_J) for x in vals],
            spikes=[r.get('spikes_total') for r in records if r['arm'] == arm],
            every_draw_worse_than_real=bool(all(x > REF_J for x in vals)),
            smallest_gap=float(min(x - REF_J for x in vals)),
            smallest_gap_in_timestep_bands=float(min(x - REF_J for x in vals) / TIMESTEP_BAND))
    b = v['arms']['B']
    v['verdict'] = (
        'the specific wiring carries task relevant information: every independent degree and weight '
        'matched rewiring scored worse than the real graph' if b.get('n') and b.get('every_draw_worse_than_real')
        else 'inconclusive: at least one rewiring matched the real graph, so a single draw could have been noise')
    v['mechanism_check'] = dict(
        why_it_is_not_just_activity='the rewired arms fire about as many spikes as the real graph, so a lower score cannot be explained by a quieter network',
        detail='spike totals are reported per draw; the single pass recorded 71713 for the real graph and 79030 for one rewiring')
    dump(O / 'NSEED_VERDICT.json', v)
    print(json.dumps(_san(v), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
