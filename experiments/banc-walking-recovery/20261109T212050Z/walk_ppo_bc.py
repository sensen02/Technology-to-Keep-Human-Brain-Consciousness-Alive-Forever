"""Imitation warm-up from the trunk-stable gait, then PPO for propulsion.

Evidence behind this order: unconstrained PPO from scratch learned to drag its abdomen for speed
(non-foot 22 W at 30 updates) and the harsher gate-aligned version collapsed to a low crawl (0.018
forward, support 0.41 at 80 updates). The kinematic gait, by contrast, holds COM 1.34, support 0.95+
and non-foot 0.00 while failing only to translate. So the network first imitates the gait's actions
from observation only (no phase, no teacher at run time), which gives it balance, and PPO then adds
forward propulsion on top of that.

The gait is a declared algorithm: this file uses it only to fill a data set. The deployed policy is
the network, driven by its own observations.
"""
from pathlib import Path
import json, os, sys, time
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import walk_ppo as P
from walk_ppo import Core, Net, CFG, RW, GATES, T_EP, NJ, OBS, log_prob, gaussian_kl, gae_terminal
from gait_teacher import T, LEGS, NL, MAX_RATE, W, rel_and_world
from gait_world import phase_of


def gait_action(core, rest, k, period=60, stride=0.03, lift=0.08, tau=0.02):
    """The declared warm-up controller's joint-rate command at this tick."""
    a = np.zeros(NJ)
    for leg in range(NL):
        rel, world = rel_and_world(core, leg)
        ids = T.leg_joint_action_index(core, leg); ds = core.a_dof[ids]
        ph = phase_of(leg, k, period)
        cur = np.asarray(core.measure()['foot'][leg], float)
        rv = core.d.qvel.copy(); rv[core.a_dof] = 0.
        if ph < 0.5:
            u = ph / 0.5
            des = np.array([(rest[leg][0] + stride * (0.5 - u) - cur[0]) / tau,
                            (rest[leg][1] - cur[1]) / tau])
            A = rel[:2, ds]
            dq = A.T @ np.linalg.solve(A @ A.T + 1e-2 * np.eye(2), des - rel[:2] @ rv)
        else:
            u = (ph - 0.5) / 0.5
            des = np.array([(rest[leg][0] + stride * (-0.5 + u) - cur[0]) / tau,
                            (rest[leg][1] - cur[1]) / tau,
                            (rest[leg][2] + lift * np.sin(np.pi * u) - cur[2]) / tau])
            A = rel[:, ds]
            dq = A.T @ np.linalg.solve(A @ A.T + 1e-2 * np.eye(3), des - rel @ rv)
        a[ids] = dq / MAX_RATE / (1. + np.abs(dq) / MAX_RATE)
    return np.clip(a, -1., 1.)


def collect(episodes=12, ticks=T_EP, **kw):
    X, Y, ev = [], [], []
    for _ in range(episodes):
        core = Core(); core.reset()
        rest = [np.asarray(core.measure()['foot'][l], float) for l in range(NL)]
        for k in range(ticks):
            X.append(core.obs()); a = gait_action(core, rest, k, **kw); Y.append(a)
            core.tick(a)
        ev.append(core)
    return np.asarray(X), np.asarray(Y)


def bc_fit(net, X, Y, steps=3000, lr=2e-3, minibatch=128, rng=None):
    rng = rng or np.random.default_rng(0)
    Tgt = np.arctanh(np.clip(Y, -.95, .95))
    curve = []
    for s in range(steps):
        ids = rng.integers(0, len(X), minibatch)
        x, y = X[ids], Tgt[ids]
        h0, h1, mean, value, g0, g1 = net.forward(x)
        d = (mean - y) / len(ids)
        grads = net.backward(x, h0, h1, mean, g0, g1, d, np.zeros_like(value),
                             np.zeros(NJ))
        grads = {k: v for k, v in grads.items() if k in
                 ('w0', 'b0', 'w1', 'b1', 'w2', 'b2')}
        net.step(grads, lr)
        if s % 500 == 0:
            mse = float(np.mean((np.tanh(net.forward(X)[2]) - Y) ** 2))
            curve.append(dict(step=s, mse=mse))
            print('bc step %4d mse %.5f' % (s, mse), flush=True)
    curve.append(dict(step=steps, mse=float(np.mean((np.tanh(net.forward(X)[2]) - Y) ** 2))))
    return curve


def main():
    t0 = time.time()
    rng = np.random.default_rng(int(os.environ.get('SEED', '20261111')))
    print('collecting gait data', flush=True)
    X, Y = collect(episodes=int(os.environ.get('BC_EPISODES', '12')))
    print('data', X.shape, Y.shape, flush=True)
    net = Net(rng)
    curve = bc_fit(net, X, Y, steps=int(os.environ.get('BC_STEPS', '3000')), rng=rng)
    ev_bc = P.evaluate(net)
    print(json.dumps({'bc_final_mse': curve[-1]['mse'],
                      'eval_after_bc': {k: round(ev_bc[k], 4) for k in
                                        ('forward_signed_native_per_s', 'foot_support_mean',
                                         'foot_support_min', 'nonfoot_force_max_W',
                                         'com_height_min', 'slip_ratio_max')},
                      'gates_after_bc': ev_bc['gates']}), flush=True)
    log = dict(label='BC warm start from the trunk-stable gait then PPO; replacement ANN',
               bc=curve, eval_after_bc=ev_bc, config=CFG, reward=RW, gates=GATES, updates=[])
    (HERE / 'WALK_BC_LOG.json').write_text(json.dumps(log, indent=2) + '\n')
    # PPO fine-tune
    core = Core()
    best = -1e9
    for u in range(CFG['updates']):
        O, A, LOGP, OM, OS, V, R, TERM = ([] for _ in range(8))
        steps = 0
        while steps < CFG['rollout']:
            core.reset(); prev = float(core.d.subtree_com[core.th][0])
            for k in range(T_EP):
                x, lat, logp, info, m, prev = P.episode_step(core, net, rng, prev)
                O.append(x); A.append(lat); LOGP.append(logp)
                OM.append(info['mean']); OS.append(info['std'])
                V.append(info['value'])
                R.append(RW['fwd'] * float(np.clip(info['vx'], -.5, .5))
                         + RW['sup_hit'] * (1. if info['support'] >= GATES['foot_support_mean_min']
                                            else -RW['sup_miss'])
                         - RW['nonfoot'] * info['nonfoot'] - RW['slip'] * info['slip']
                         - RW['com'] * max(0., GATES['com_height_min'] - info['com']) + RW['alive']
                         + (RW['fall'] if info['fall'] else 0.))
                term = info['fall'] or k == T_EP - 1
                TERM.append(term); steps += 1
                if term:
                    break
        O, A, LOGP = map(np.asarray, (O, A, LOGP))
        OM, OS, V, R, TERM = map(np.asarray, (OM, OS, V, R, TERM))
        adv, ret = gae_terminal(R, V, TERM)
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        opt = P.optimize(net, O, A, LOGP, OM, OS, adv_n, ret, rng, CFG['lr'])
        ev = P.evaluate(net)
        row = dict(update=u, decisions=int(len(R)), reward_mean=float(R.mean()),
                   minibatch_updates=opt['minibatch_updates'], drift=opt['drift_vs_rollout'],
                   **{k: ev[k] for k in ('forward_signed_native_per_s', 'foot_support_mean',
                                         'foot_support_min', 'nonfoot_force_max_W',
                                         'com_height_min', 'slip_ratio_max', 'all_gates')})
        log['updates'].append(row)
        score = row['forward_signed_native_per_s'] - 2 * row['nonfoot_force_max_W'] / 10.
        if ev['all_gates'] or score > best:
            best = score
            np.savez_compressed(HERE / 'WALK_MODEL_best.npz',
                                **{k: v.astype(np.float32) for k, v in net.p.items()})
            log['best_update'] = u
            log['best_eval'] = ev
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}),
              flush=True)
        (HERE / 'WALK_BC_LOG.json').write_text(json.dumps(log, indent=2) + '\n')
    log['wall_s'] = time.time() - t0
    (HERE / 'WALK_BC_LOG.json').write_text(json.dumps(log, indent=2) + '\n')
    print('done', round(log['wall_s'], 1))


if __name__ == '__main__':
    main()
