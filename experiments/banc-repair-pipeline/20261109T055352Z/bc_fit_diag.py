"""Per-leg and per-phase imitation error after the warm start.

The aggregate warm-start MSE hides whether every leg is fitted equally well. If the network fits
some legs well and others badly, that maps directly onto which legs pass the acceptance test and
explains why the passing set changes when the teacher changes.
"""
from pathlib import Path
import json, sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc
import train_single_leg as T


def collect(ticks_per_leg=600):
    core = rc.Core()
    base = rc.BaseNet()
    demos = T.demo_directions()
    rng = np.random.default_rng(3)
    X, Y, LEG, K = [], [], [], []
    for leg in range(rc.NL):
        core.reset(); base.reset(core.q0)
        for k in range(ticks_per_leg):
            obs = core.obs(T.onehot(leg))
            a = T.teacher_action(core, leg, k % T.T_EP)
            X.append(obs); Y.append(np.arctanh(np.clip(a, -.95, .95))); LEG.append(leg); K.append(k % T.T_EP)
            core.tick(a)
    return np.asarray(X, float), np.asarray(Y, float), np.asarray(LEG), np.asarray(K)


def main():
    X, Y, LEG, K = collect()
    net = T.Net(np.random.default_rng(11))
    for k in ('w0', 'b0', 'w1', 'b1', 'w2', 'b2'):
        net.p[k] = np.load(HERE / 'MODEL_singleleg_demo_repro.npz')['actor_' + k].astype(np.float64)
    out = dict(label='imitation error of the saved network by leg and by phase; offline',
               note='the network was saved after PPO, the warm start is not re-run', per_leg={})
    _, _, mean, _ = net.forward(X)
    err = ((np.tanh(mean) - np.tanh(Y)) ** 2).mean(axis=1)
    for leg in range(rc.NL):
        sel = LEG == leg
        phase = np.where(K[sel] < T.DEMO_LIFT_TICKS, 'raise',
                         np.where(K[sel] < T.DEMO_LIFT_TICKS + T.DEMO_SWING_TICKS, 'swing', 'place'))
        out['per_leg'][rc.LEG_ORDER[leg]] = dict(
            mse=float(err[sel].mean()),
            raise_mse=float(err[sel][phase == 'raise'].mean()) if (phase == 'raise').any() else None,
            swing_mse=float(err[sel][phase == 'swing'].mean()) if (phase == 'swing').any() else None,
            place_mse=float(err[sel][phase == 'place'].mean()) if (phase == 'place').any() else None)
        print(rc.LEG_ORDER[leg], {k: (None if v is None else round(v, 4))
                                  for k, v in out['per_leg'][rc.LEG_ORDER[leg]].items()}, flush=True)
    out['mean_mse'] = float(err.mean())
    (HERE / 'BC_FIT.json').write_text(json.dumps(out, indent=2) + '\n')
    print('mean', round(out['mean_mse'], 4))


if __name__ == '__main__':
    main()
