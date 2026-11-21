"""Engineered interface drift with saved state, on identical archived spike trains."""
import argparse
import json
from pathlib import Path

import numpy as np
from decode import decode
from run import reference_pairs, score


def advance(start_loss, exposed):
    # Exposure accumulates an engineering 0.5 fractional loss over 550 ms.
    loss = np.minimum(.5, start_loss + np.cumsum(exposed, dtype=np.int64) / 11000)
    return (1 - loss), float(loss[-1])


def replay(recordings, output):
    output.mkdir(parents=True, exist_ok=False)
    meta = json.loads((recordings/'RESULT.json').read_text())
    params = json.loads(Path(meta['geometry']['parameters']).read_text())
    results = []
    for trial in meta['trials']:
        if trial['arm'] not in ('intact', 'tear_an'):
            continue
        name = f"{trial['seed']}_{trial['arm']}"
        with np.load(recordings/(name+'_reference.npz')) as ref:
            truth = reference_pairs(ref['entry'], ref['arrival'], ref['roots'], params, ref['direction'])
        with np.load(recordings/(name+'_observation.npz')) as obs:
            head, thorax = obs['head_uv'], obs['thorax_uv']
        noise = np.random.default_rng(trial['seed']).normal(0., .08, (len(head), 16))[:, :8]
        tick = np.arange(len(head))
        exposure = (tick >= 6500) & (tick < 12000)
        gain_ref, _ = advance(0., exposure)
        split = 9250
        gain1, loss = advance(0., exposure[:split])
        checkpoint = output/(name+'_checkpoint.json')
        checkpoint.write_text(json.dumps(dict(next_tick=split, accumulated_loss=loss)))
        restored = json.loads(checkpoint.read_text())
        gain2, _ = advance(restored['accumulated_loss'], exposure[restored['next_tick']:])
        gain = np.r_[gain1, gain2]
        if not np.allclose(gain, gain_ref, rtol=0, atol=2e-16) or not np.all(gain[12000:] == .5):
            raise ValueError('interface state did not persist')
        # Change transduction of the signal only; electronics noise stays at its initial level.
        head_drift = ((head-noise)*gain[:, None] + noise).astype('float32')
        pair, peaks = decode(head_drift, thorax)
        baseline, _ = decode(head, thorax)
        np.savez_compressed(output/(name+'.npz'), head_uv=head_drift, thorax_uv=thorax, head_gain=gain,
                            head_thorax_ticks=pair)
        phases = {}
        for phase, lo, hi in [('before',0,6500), ('exposure',6500,12000), ('retained',12000,len(head))]:
            def window(events):
                return events[(events.min(axis=1)>=lo) & (events.max(axis=1)<hi)]
            phases[phase] = dict(drift=score(window(pair), window(truth)),
                                 fixed=score(window(baseline), window(truth)))
        results.append(dict(seed=trial['seed'], arm=trial['arm'], phases=phases, peaks=peaks))
    (output/'RESULT.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('recordings', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    replay(args.recordings, args.output)
