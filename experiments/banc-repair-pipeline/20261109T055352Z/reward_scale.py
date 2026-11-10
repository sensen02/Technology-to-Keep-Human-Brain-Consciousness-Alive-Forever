"""Measure the magnitude of every reward term under the warm-start policy.

Purpose: the pre-registered stance-slip weight is 2.0 per unit of summed stance excess. If
that term is an order of magnitude larger than the task terms, the run would be dominated by
a signal that is known not to be solvable yet. This script measures the actual scale so the
decision is evidence-based rather than a guess.
"""
from pathlib import Path
import json, sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc
import train_single_leg as T

W = rc.WEIGHT


def main():
    rng = np.random.default_rng(11)
    core = rc.Core()
    net = T.Net(rng)
    T.warm_start(core, net, rng, ticks=600, steps=300)
    rows = []
    for leg in range(rc.NL):
        core.reset()
        prompt = T.onehot(leg)
        st = dict(swing=False, start_x=0., max_lift=0., best_adv=0., placed=False, prev_x=0.)
        for k in range(T.T_EP):
            obs = core.obs(prompt)
            h0, h1, mean, _ = net.forward(obs[None])
            a = np.tanh(mean[0])
            m = core.tick(a)
            loads = m['loads']
            sup = float(loads.sum() / W)
            rel = float(np.clip(1. - loads[leg] / (.30 * W), 0., 1.))
            lift = float(np.clip(float(m['padz'][leg]) / T.THRESH['lift'], 0., 1.))
            fwd = abs(core.forward_body_velocity())
            allow = max(.5 * fwd, .02)
            excess = float(sum(max(0., float(m['speed'][i]) - allow) for i in range(rc.NL) if i != leg))
            rot = core.d.xmat[core.th].reshape(3, 3)
            omega = core.d.qvel[3:6]
            grav = rot.T @ np.array([0., 0., -1.])
            rows.append(dict(unload=T.RW['unload'] * rel, lift=T.RW['lift'] * lift,
                             nonfoot=-T.RW['nonfoot'] * float(m['nonfoot'] / W),
                             support=-T.RW['support'] * max(0., T.SUP_TARGET - sup) ** 2,
                             com=-T.RW['com'] * max(0., T.COM_TARGET - m['com']) ** 2,
                             other_stance=-T.RW['other_stance'] * excess,
                             act=-T.RW['act'] * float(np.mean(a ** 2)),
                             omega=-T.RW['omega'] * float(np.sum(omega ** 2)),
                             grav=-T.RW['grav'] * float(np.sum(grav[:2] ** 2)),
                             raw_excess=excess))
    keys = [k for k in rows[0] if k != 'raw_excess']
    summ = {k: dict(mean=float(np.mean([r[k] for r in rows])),
                    min=float(np.min([r[k] for r in rows])),
                    max=float(np.max([r[k] for r in rows]))) for k in keys}
    summ['raw_excess'] = dict(mean=float(np.mean([r['raw_excess'] for r in rows])),
                              max=float(np.max([r['raw_excess'] for r in rows])))
    total = float(np.mean([sum(r[k] for k in keys) for r in rows]))
    out = dict(label='warm-start policy, 6 prompts x 300 ticks, no PPO update',
               per_term=summ, mean_total_reward=total)
    (HERE / 'REWARD_SCALE.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
