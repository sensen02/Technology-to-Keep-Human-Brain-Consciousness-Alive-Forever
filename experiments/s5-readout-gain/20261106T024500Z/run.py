"""Bounded decoder-gain validation on archived handover states.

Only the readout scalar g changes: torque = clip(g * limits/100 * (matrix @ rates), +-limits).
Neurons, connectome, sensory encoder, plant, actuator limits and the warmup teacher are
untouched, and no body-state signal is added to the command path. The warmup is unaffected
by g, so the archived 13 handover states remain valid and no new warmup is needed.

Scored by the unchanged D1-D8 gates and the unchanged loss, with an explicit regression
guard against the current g=1.0 baseline at 1.300 s.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MUJOCO_GL', 'egl')
import sys, json, pickle, time
from pathlib import Path
import numpy as np

R = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = R / 'experiments/s5-neural-translator/20261105T200021Z'
sys.path[:0] = [str(BASE), str(R), str(R / 'tools')]
import runtime as rt

OUT = Path('/home/sensen/Desktop/engine修复验收/s5-readout-gain/20261106T024500Z')
SRC = Path(__file__).resolve().parent
rt.O = OUT
rt.b.O = OUT
js, sha, dump = rt.js, rt.sha, rt.dump
ARCH = Path('/home/sensen/Desktop/engine修复验收/s5-neural-translator/20261105T200021Z')
DEV = [1.225, 1.275, 1.325, 1.375]
GUARD = 1.3
FINAL = [1.209, 1.259, 1.309, 1.359]
GAINS = [1.0, 0.25, 0.1, 0.05]


def arrays(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def prereg():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'MODELS').mkdir()
    z = np.zeros(42, np.float32)
    model = OUT / 'MODELS' / 'zero.npz'
    np.savez_compressed(model, kind=np.array('mlp'), W0=np.zeros((64, 612), np.float32),
                        W1=np.zeros((64, 64), np.float32), W2=np.zeros((42, 64), np.float32))
    ident = js(ARCH / 'DATA/CACHE_IDENTITY.json')
    assert sha(ident['file']) == ident['sha256']
    dump(OUT / 'PREREGISTRATION.json', dict(
        stage='S5 bounded decoder-gain validation', importance='核心必需',
        question='Is the saturated bang-bang readout (0.78 of joint-timesteps at clip in replay) a decoder-scale '
                 'problem rather than a neural one?',
        change='only the scalar g in torque = clip(g*limits/100*(matrix@rates), +-limits); g in {1.0,0.25,0.1,0.05}',
        unchanged='12000-cell SNN, A connectome, sensory encoder, plant, actuator limits, warmup teacher, '
                  'no body-state feedback, no online PD/CPG, no MLP residual (zero branch)',
        handover='archived 13 complete warm states from 20261105T200021Z, hash-checked; g does not affect warmup so no new warmup is run',
        phases=dict(dev=DEV, guard=GUARD, final=FINAL),
        selection='protect 1.300 s: J must not rise, and foot support / minimum COM / first nonfoot must not fall, '
                  'versus the g=1.0 baseline; then full D1-D8 pass count, worst dev J, mean dev J. Final phases are '
                  'untouched until the gain is frozen.',
        units='NOT_ESTABLISHED', intent='walk/stop intent NOT_ESTABLISHED'), )
    dump(OUT / 'INPUT_HASHES_BEFORE.json',
         {str(p): sha(p) for p in [Path(ident['file']), ARCH / 'MANIFEST_SHA256.json', model,
                                   Path(rt.b.AB / 'A' / 'ACTUAL_We.npz'), Path(rt.b.AB / 'A' / 'ACTUAL_Wi.npz'),
                                   Path(rt.__file__), SRC / 'run.py']})
    dump(OUT / 'CLEANUP_LEDGER.json', dict(importance='重要辅助', deleted=[]))


def main():
    mode = sys.argv[1]
    if mode == 'prereg':
        prereg()
        return
    t = rt.Runtime()
    cov = OUT / 'COVERAGE.json'
    if cov.exists():
        d = js(OUT / 'CLEANUP_LEDGER.json')
        d['deleted'].append(dict(path=str(cov), sha256=sha(cov), bytes=cov.stat().st_size,
                                 classification='可再生临时',
                                 reason='Legacy dominant-count table from the inherited constructor; unused here.'))
        cov.unlink()
        dump(OUT / 'CLEANUP_LEDGER.json', d)
    ident = js(ARCH / 'DATA/CACHE_IDENTITY.json')
    assert sha(ident['file']) == ident['sha256']
    with Path(ident['file']).open('rb') as f:
        t.cache = pickle.load(f)
    base_command = t.old_command
    model = OUT / 'MODELS' / 'zero.npz'
    actor = lambda x, stochastic: (np.zeros(42, np.float32), 0., 0.)

    def with_gain(g):
        def command(rates):
            return np.clip(g * base_command(rates), -t.limits, t.limits)
        t.old_command = command
        return command

    results = {}
    for g in GAINS:
        with_gain(g)
        rs = []
        for phase in DEV:
            r, _ = t.episode(f'gain_{g}_dev_{phase}', phase, actor, model)
            rs.append(r)
        rg, _ = t.episode(f'gain_{g}_guard_{GUARD}', GUARD, actor, model)
        results[g] = dict(dev=rs, guard=rg)
        print('GAIN', g, 'guardJ', round(rg['J'], 6), 'sat', round(rg['saturation_fraction'], 3),
              'worst', round(max(x['J'] for x in rs), 6), 'passes', sum(x['validation']['passed'] for x in rs), flush=True)

    def score(rs):
        return (sum(r['validation']['passed'] for r in rs), -max(r['J'] for r in rs), -float(np.mean([r['J'] for r in rs])))

    zero = results[1.0]
    ranking = []
    for g in GAINS:
        r = results[g]['guard']
        b = zero['guard']
        guard = (r['J'] <= b['J'] + 1e-9
                 and r['parts']['foot_support_mean'] >= b['parts']['foot_support_mean'] - 1e-9
                 and r['parts']['com_z_min'] >= b['parts']['com_z_min'] - 1e-9
                 and ((r['parts']['first_nonfoot_s'] is None and b['parts']['first_nonfoot_s'] is None) or
                      (r['parts']['first_nonfoot_s'] is not None and b['parts']['first_nonfoot_s'] is not None
                       and r['parts']['first_nonfoot_s'] >= b['parts']['first_nonfoot_s'] - 1e-9) or
                      (r['parts']['first_nonfoot_s'] is None and b['parts']['first_nonfoot_s'] is not None)))
        ranking.append(dict(gain=g, guard_pass=bool(guard), score=score(results[g]['dev']),
                            worst_dev_J=max(x['J'] for x in results[g]['dev']),
                            mean_dev_J=float(np.mean([x['J'] for x in results[g]['dev']])),
                            dev_passes=sum(x['validation']['passed'] for x in results[g]['dev'])))
    safe = [r for r in ranking if r['guard_pass']]
    selected = max(safe, key=lambda r: r['score']) if safe else ranking[0]
    dump(OUT / 'GAIN_SELECTION.json', dict(ranking=ranking, selected=selected, final_untouched=True))
    g = selected['gain']
    with_gain(g)
    final = []
    for phase in FINAL:
        ro, _ = t.episode(f'final_gain1.0_{phase}', phase, actor, model) if g != 1.0 else (None, None)
        if g != 1.0:
            with_gain(1.0)
            ro, _ = t.episode(f'final_gain1.0_{phase}', phase, actor, model)
            with_gain(g)
        rn, _ = t.episode(f'final_gain{g}_{phase}', phase, actor, model)
        final.append(dict(phase=phase, baseline=ro, selected=rn))
    ra, _ = t.episode(f'repeat_gain{g}_{GUARD}', GUARD, actor, model)
    with_gain(1.0)
    rb, _ = t.episode('repeat_gain1.0', GUARD, actor, model)
    summary = dict(ranking=ranking, selected=selected, final=final,
                   repeat=dict(selected=ra, baseline=rb),
                   selected_gain=g, baseline_gain=1.0,
                   interpretation='Decoder-scale change only; neurons and connectivity unchanged. No behaviour gate is '
                                  'claimed unless D1-D8 pass, and no intent-preservation or biological claim is made.')
    dump(OUT / 'SUMMARY.json', summary)
    after = {p: sha(p) for p in js(OUT / 'INPUT_HASHES_BEFORE.json')}
    dump(OUT / 'INPUT_HASHES_AFTER.json', after)
    print('SELECTED', g, 'ranking', json.dumps(ranking), flush=True)
    for f in final:
        b = f['baseline']
        s = f['selected']
        print('FINAL', f['phase'], 'base', None if b is None else round(b['J'], 6),
              'sel', round(s['J'], 6), 'feet', None if b is None else (round(b['parts']['foot_support_mean'], 4),
                                                                     round(s['parts']['foot_support_mean'], 4)),
              'pass', s['validation']['passed'], flush=True)


if __name__ == '__main__':
    main()
