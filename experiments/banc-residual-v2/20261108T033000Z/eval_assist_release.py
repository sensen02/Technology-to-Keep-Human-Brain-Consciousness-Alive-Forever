"""Assist-then-release evaluation of the v2 residual policy (frozen protocol structure).

The frozen acceptance protocol measured the autonomous phase AFTER a warmup, because a cold start
from a standing keyframe is a fall transient, not a gait. This evaluator does the same: the first
1 s replays measured targets (the declared assist phase, from the training set), then the base
network + trained residual runs autonomously for 3 s and only that phase is measured.

Labels: assist phase = replay of measured targets (not autonomous), release phase = autonomous.
"""
from pathlib import Path
import argparse, json, sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
WEIGHT = 24.446354755878005
LEGS = ['LF', 'LH', 'LM', 'RF', 'RH', 'RM']
TIBIA = [6, 13, 20, 27, 34, 41]
PDT = .005; CAP = 30.
TS = ROOT / 'experiments/banc-locomotion-dataset/20261108T011500Z/TRAINING_SET.npz'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--residual', default=str(HERE / 'MODEL_RESIDUAL_V2.npz'))
    ap.add_argument('--assist-ticks', type=int, default=200)
    ap.add_argument('--release-ticks', type=int, default=600)
    ap.add_argument('--segment', type=int, default=0)
    ap.add_argument('--out', default='EVAL_ASSIST_RELEASE')
    args = ap.parse_args()
    import importlib.util
    spec = importlib.util.spec_from_file_location('tr', str(HERE / 'train_residual_v2.py'))
    tr = importlib.util.module_from_spec(spec); sys.modules['tr'] = tr; spec.loader.exec_module(tr)
    act_mod = tr.load_module('act', tr.ACT); sensory = tr.load_module('sensory', tr.SENSORY)
    base = np.load(tr.BASE); sim = tr.Sim(base, act_mod, sensory)
    z = np.load(args.residual)
    P = dict(w0=z['actor_w0'].astype(np.float64), b0=z['actor_b0'].astype(np.float64),
             w1=z['actor_w1'].astype(np.float64), b1=z['actor_b1'].astype(np.float64),
             w2=z['actor_w2'].astype(np.float64), b2=z['actor_b2'].astype(np.float64))
    seg = np.load(TS)['targets_rad'][np.load(TS)['offsets'][args.segment]:np.load(TS)['offsets'][args.segment + 1]]
    assist = seg[:args.assist_ticks]
    sim.reset()
    if assist.shape[0] >= sim.K:
        sim.buf = assist[:sim.K].copy()
    rec = []
    phase = []
    for k in range(args.assist_ticks + args.release_ticks):
        target = sim.base_target()
        loads, nonfoot, speed_pre = sim.loads_and_speed()
        proprio = sensory.features(sim.shim, sim.d, target, loads)
        obs = np.concatenate([proprio, sim.buf[sim.lags].reshape(-1)])
        h0 = np.tanh(obs @ P['w0'] + P['b0']); h1 = np.tanh(h0 @ P['w1'] + P['b1'])
        mean = h1 @ P['w2'] + P['b2']
        residual_rate = np.zeros(42); residual_rate[TIBIA] = np.tanh(mean) * float(z['delta_scale']) / PDT
        if k < assist.shape[0]:
            rate = np.clip((assist[k] - sim.actuator.target_rad) / PDT, -CAP, CAP)
            phase.append('assist')
        else:
            rate = np.clip((target - sim.actuator.target_rad) / PDT, -CAP, CAP) + residual_rate
            phase.append('release')
        sim.step(np.clip(rate, -CAP, CAP))
        loads, nonfoot, speed = sim.loads_and_speed()
        rec.append((sim.d.time, sim.d.subtree_com[sim.th].copy(), loads, nonfoot, speed))
        sim.buf = np.vstack([sim.buf[1:], target[None]])
    phase = np.array(phase)
    rel = np.flatnonzero(phase == 'release')
    out = dict(controller='base measured-data network + v2 residual (6 tibia, +-0.003 rad/tick)',
               assist_ticks=int(args.assist_ticks), release_ticks=int(args.release_ticks),
               assist_label='replay of measured targets; NOT autonomous',
               release_label='autonomous: base network + learned residual, no teacher, no phase input',
               native_steps=int(sim.steps), residual=str(args.residual), residual_sha256=tr.sha(args.residual))
    for tag, idx in (('assist', np.arange(len(phase))[phase == 'assist']), ('release', rel)):
        times = np.array([rec[i][0] for i in idx]); com = np.array([rec[i][1] for i in idx])
        loads = np.array([rec[i][2] for i in idx]); nonfoot = np.array([rec[i][3] for i in idx])
        speed = np.array([rec[i][4] for i in idx])
        support = loads.sum(1) / WEIGHT
        duration = float(times[-1] - times[0])
        fwd = float(np.linalg.norm((com[-1] - com[0])[:2]) / duration)
        per_leg = loads.__mul__(speed).sum(0) / np.maximum(loads.sum(0), 1e-12)
        ratio = per_leg / max(fwd, 1e-9)
        out[tag] = dict(duration_s=duration, forward_speed_native_per_s=fwd,
                        foot_support_mean=float(support.mean()), foot_support_min=float(support.min()),
                        nonfoot_force_max_W=float(nonfoot.max()), com_height_min=float(com[:, 2].min()),
                        com_height_mean=float(com[:, 2].mean()), free_fall_fraction=float(np.mean(support < .05)),
                        per_leg_slip_ratio=dict(zip(LEGS, map(float, ratio))), slip_ratio_max=float(ratio.max()),
                        slip_ratio_mean=float(ratio.mean()), slip_criterion_passed=bool(ratio.max() <= .5))
    g = out['release']
    out['release_gates'] = dict(foot_support_mean=g['foot_support_mean'] >= .9,
                                nonfoot=g['nonfoot_force_max_W'] <= .2 * WEIGHT,
                                com_height=g['com_height_min'] >= 1.20,
                                slip=g['slip_ratio_max'] <= .5,
                                forward=g['forward_speed_native_per_s'] >= .05)
    out['release_gates_passed'] = all(out['release_gates'].values())
    d = HERE / args.out; d.mkdir(exist_ok=True)
    (d / 'RESULT.json').write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(d / 'TRACE.npz', time=np.array([r[0] for r in rec]), com=np.array([r[1] for r in rec]),
                        loads=np.array([r[2] for r in rec]), nonfoot=np.array([r[3] for r in rec]),
                        speed=np.array([r[4] for r in rec]), phase=phase)
    print(json.dumps(dict(assist={k: round(out['assist'][k], 4) for k in ('forward_speed_native_per_s', 'foot_support_mean', 'slip_ratio_max')},
                          release={k: round(out['release'][k], 4) for k in ('forward_speed_native_per_s', 'foot_support_mean', 'foot_support_min', 'nonfoot_force_max_W', 'com_height_min', 'slip_ratio_max', 'slip_ratio_mean')},
                          gates=out['release_gates'])))


if __name__ == '__main__':
    main()
