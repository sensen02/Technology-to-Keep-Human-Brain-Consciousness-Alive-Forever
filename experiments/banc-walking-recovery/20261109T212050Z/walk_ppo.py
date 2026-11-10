"""Whole-body single-network PPO for walking. Replacement ANN, NOT connectome capability.

Why this replaces the kinematic gait: twelve configurations of the declared kinematic tripod gait,
including stance force amplification and trunk velocity feedback, did not produce sustained forward
motion (GAIT_PUSH.json). The single-leg work in this project reached every criterion with PPO where
kinematics failed, so learning is the route with evidence behind it.

Architecture: one actor for all 42 joints. The observation already carries 294 values of command
history, which is the internal memory that lets a rhythm exist without an external clock and without
a scripted leg sequence. There is no gait phase input, no teacher, and no per-leg selection at run
time.

Correctness carried over from the audited trainer fixes: complete episodes, an explicit task deadline
with zero bootstrap, analytic Gaussian KL over the whole rollout, and a rejected step restores
parameters, Adam moments and the step counter.
"""
from pathlib import Path
import json, os, sys, time
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gait_teacher import Core, T, LEGS, NL, NJ, MAX_RATE, W, summarise, GATES

PDT = 0.001
OBS = 495
HID = 128
T_EP = int(os.environ.get('T_EP', '2000'))     # 2 s of simulated walking per episode
CFG = dict(updates=int(os.environ.get('UPDATES', '40')), rollout=int(os.environ.get('ROLLOUT', '2000')),
           epochs=4, minibatch=64, lr=1e-4, lr_min=1e-6, clip=.2, backtracks=6, gamma=float(.5 ** (PDT / 0.5)),
           gae=.95, value_coef=.5, grad_norm=.5, target_kl=.03,
           initial_logstd=-0.5, logstd_bounds=[-3.0, 0.3])
# Gate-aligned reward: the penalties are large enough that belly dragging cannot pay for speed.
RW = dict(fwd=6.0, sup_hit=1.0, sup_miss=2.0, nonfoot=2.0, slip=1.0, alive=0.1,
          com=4.0, fall=-20.0, slip_tol=0.05)


def onehot(i):
    v = np.zeros(NL); v[i] = 1.; return v


class Net:
    """Single shared actor and critic; no per-leg parameters and no phase input."""

    def __init__(self, rng):
        def w(a, b, s):
            return (rng.standard_normal((a, b)) * s).astype(np.float64)
        self.p = dict(w0=w(OBS, HID, 1 / np.sqrt(OBS)), b0=np.zeros(HID),
                      w1=w(HID, HID, 1 / np.sqrt(HID)), b1=np.zeros(HID),
                      w2=w(HID, NJ, 1 / np.sqrt(HID)), b2=np.zeros(NJ),
                      vc0=w(OBS, HID, 1 / np.sqrt(OBS)), vbc0=np.zeros(HID),
                      vc1=w(HID, HID, 1 / np.sqrt(HID)), vbc1=np.zeros(HID),
                      vc=w(HID, 1, 1 / np.sqrt(HID)), vbc=np.zeros(1),
                      logstd=np.full(NJ, CFG['initial_logstd']))
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, x):
        p = self.p
        h0 = np.tanh(x @ p['w0'] + p['b0'])
        h1 = np.tanh(h0 @ p['w1'] + p['b1'])
        mean = h1 @ p['w2'] + p['b2']
        g0 = np.tanh(x @ p['vc0'] + p['vbc0'])
        g1 = np.tanh(g0 @ p['vc1'] + p['vbc1'])
        value = (g1 @ p['vc'] + p['vbc'])[:, 0]
        return h0, h1, mean, value, g0, g1

    def std(self):
        return np.exp(np.clip(self.p['logstd'], *CFG['logstd_bounds']))

    def backward(self, x, h0, h1, mean, g0, g1, dmean, dvalue, dlogstd):
        p = self.p
        grads = {}
        grads['w2'] = h1.T @ dmean; grads['b2'] = dmean.sum(0)
        dh1 = (dmean @ p['w2'].T) * (1 - h1 ** 2)
        grads['w1'] = h0.T @ dh1; grads['b1'] = dh1.sum(0)
        dh0 = (dh1 @ p['w1'].T) * (1 - h0 ** 2)
        grads['w0'] = x.T @ dh0; grads['b0'] = dh0.sum(0)
        dv = dvalue[:, None]
        grads['vc'] = g1.T @ dv; grads['vbc'] = dv.sum(0)
        dg1 = (dv @ p['vc'].T) * (1 - g1 ** 2)
        grads['vc1'] = g0.T @ dg1; grads['vbc1'] = dg1.sum(0)
        dg0 = (dg1 @ p['vc1'].T) * (1 - g0 ** 2)
        grads['vc0'] = x.T @ dg0; grads['vbc0'] = dg0.sum(0)
        grads['logstd'] = dlogstd
        # Every incoming gradient is already a per-sample average, so no second division here.
        return grads

    def snapshot(self):
        return {k: v.copy() for k, v in self.p.items()} | \
               {'_m': {k: v.copy() for k, v in self.m.items()},
                '_v': {k: v.copy() for k, v in self.v.items()}, '_t': self.t}

    def restore(self, s):
        self.p = {k: v.copy() for k, v in s.items() if not k.startswith('_')}
        self.m = {k: v.copy() for k, v in s['_m'].items()}
        self.v = {k: v.copy() for k, v in s['_v'].items()}
        self.t = s['_t']

    def step(self, grads, lr, b1=.9, b2=.999, eps=1e-8):
        self.t += 1
        for k in self.p:
            g = grads.get(k)
            if g is None:
                continue
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] = self.p[k] - lr * mh / (np.sqrt(vh) + eps)


def log_prob(lat, mean, std):
    z = (lat - mean) / std
    return np.sum(-.5 * z ** 2 - np.log(std) - .5 * np.log(2 * np.pi), axis=1), z


def reward_of(m, prev_x, w):
    loads = m['loads']; support = float(loads.sum() / W)
    fwd = float(m['foot'][LEGS[0]][0]) * 0.  # placeholder, replaced below
    return loads, support


def episode_step(core, net, rng, prev_com_x):
    x = core.obs()
    h0, h1, mean, value, g0, g1 = net.forward(x[None])
    std = net.std()
    lat = mean[0] + std * rng.standard_normal(NJ)
    a = np.tanh(lat)
    m = core.tick(a)
    com_x = float(core.d.subtree_com[core.th][0])
    vx = (com_x - prev_com_x) / PDT
    support = float(m['loads'].sum() / W)
    nf = float(m['nonfoot'])
    allow = max(.5 * abs(vx), .02)
    slip = sum(max(0., float(m['speed'][i]) - allow) for i in range(NL))
    com = float(core.d.subtree_com[core.th][2])
    r = (RW['fwd'] * float(np.clip(vx, -0.5, 0.5))
         + RW['sup_hit'] * (1.0 if support >= GATES['foot_support_mean_min'] else -RW['sup_miss'])
         - RW['nonfoot'] * nf - RW['slip'] * slip - RW['com'] * max(0., GATES['com_height_min'] - com)
         + RW['alive'])
    # Termination is aligned with the frozen gates, so an episode cannot end "successfully" while
    # the body is on its abdomen or below the height gate.
    fall = bool(nf > GATES['nonfoot_force_max_W'] or com < GATES['com_height_min'] - .05)
    if fall:
        r += RW['fall']
    info = dict(mean=mean[0].copy(), std=std.copy(), value=float(value[0]), vx=vx,
                support=support, nonfoot=nf, slip=slip, fall=fall, com=com)
    return x, lat, float(log_prob(lat[None], mean, std)[0][0]), info, m, com_x


def gae_terminal(rewards, values, terminal):
    adv = np.zeros(len(rewards)); carry = 0.
    for i in range(len(rewards) - 1, -1, -1):
        nxt = 0. if terminal[i] else (values[i + 1] if i + 1 < len(values) else 0.)
        if terminal[i]:
            carry = 0.
        delta = rewards[i] + CFG['gamma'] * nxt - values[i]
        carry = delta + CFG['gamma'] * CFG['gae'] * carry
        adv[i] = carry
    return adv, adv + np.asarray(values)


def gaussian_kl(om, os_, mean, std):
    lr = np.log(std) - np.log(os_)
    return float(np.mean(np.sum(lr + .5 * np.expm1(-2 * lr) + .5 * ((om - mean) / std) ** 2, axis=1)))


def optimize(net, O, A, LOGP, OM, OS, adv_n, ret, rng, lr):
    if len(O) == 0:
        raise ValueError('empty rollout')
    def kl():
        mean = net.forward(O)[2]
        return gaussian_kl(OM, OS, mean, net.std())
    if kl() > CFG['target_kl']:
        raise ValueError('initial policy exceeds KL budget')
    total, stopped = 0, None
    for ep in range(CFG['epochs']):
        perm = rng.permutation(len(O))
        for s in range(0, len(perm), CFG['minibatch']):
            ids = perm[s:s + CFG['minibatch']]
            x, lat_t, oldlog = O[ids], A[ids], LOGP[ids]
            h0, h1, mean, value, g0, g1 = net.forward(x)
            std = net.std()
            logp, z = log_prob(lat_t, mean, std)
            ratio = np.exp(logp - oldlog)
            advb = adv_n[ids]
            active = (advb * ratio <= advb * np.clip(ratio, 1 - CFG['clip'], 1 + CFG['clip'])).astype(float)
            dlogp = -(active * advb * ratio) / len(ids)
            dvalue = 2 * CFG['value_coef'] * (value - ret[ids]) / len(ids)
            dmean = dlogp[:, None] * (z / std[None, :])
            dlogstd = np.sum(dlogp[:, None] * (z ** 2 - 1), axis=0) / len(ids)
            grads = net.backward(x, h0, h1, mean, g0, g1, dmean, dvalue, dlogstd)
            gn = np.sqrt(sum(float(np.sum(v * v)) for v in grads.values()))
            if gn > CFG['grad_norm']:
                grads = {k: v * (CFG['grad_norm'] / gn) for k, v in grads.items()}
            snap = net.snapshot()
            # Trust region: a candidate step that overshoots the KL budget is undone and retried
            # with a halved learning rate, instead of throwing the whole update away.
            step_lr = lr; cand = None; done = False
            for attempt in range(CFG['backtracks'] + 1):
                net.restore(snap)
                net.step(grads, step_lr)
                cand = kl()
                if np.isfinite(cand) and cand <= CFG['target_kl']:
                    done = True
                    break
                step_lr *= .5
            if not done:
                net.restore(snap)
                stopped = dict(epoch=ep, minibatch=total, kl=cand, rolled_back=True,
                               note='KL budget not reachable within the backtrack budget')
                break
            lr = max(step_lr, CFG['lr_min'])
            total += 1
        if stopped:
            break
    return dict(minibatch_updates=total, stopped_at=stopped, drift_vs_rollout=kl(), lr=lr)


def evaluate(net, ticks=T_EP):
    core = Core(); core.reset()
    rec = []
    for k in range(ticks):
        x = core.obs()
        mean = net.forward(x[None])[2]
        m = core.tick(np.tanh(mean[0]))
        rec.append((float(core.d.time), np.array(core.d.subtree_com[core.th], float),
                    m['loads'].copy(), float(m['nonfoot']), np.array(m['speed'], float)))
    return summarise(rec, ticks * 80)


def main():
    t0 = time.time()
    rng = np.random.default_rng(int(os.environ.get('SEED', '20261109')))
    core = Core(); net = Net(rng)
    log = dict(label='whole-body single-network PPO walking attempt; replacement ANN; NOT walking claim',
               architecture='shared actor 495->128->128->42, shared critic, no phase input',
               config=CFG, reward=RW, gates=GATES, t_ep=T_EP, updates=[])
    for u in range(CFG['updates']):
        O, A, LOGP, OM, OS, V, R, TERM = ([] for _ in range(8))
        steps = 0
        while steps < CFG['rollout']:
            core.reset(); prev = float(core.d.subtree_com[core.th][0])
            for k in range(T_EP):
                x, lat, logp, info, m, prev = episode_step(core, net, rng, prev)
                O.append(x); A.append(lat); LOGP.append(logp)
                OM.append(info['mean']); OS.append(info['std'])
                V.append(info['value']); R.append(0.0)
                deadline = k == T_EP - 1
                term = info['fall'] or deadline
                # reward is recomputed here from the recorded info to keep one definition
                R[-1] = (RW['fwd'] * float(np.clip(info['vx'], -.5, .5))
                         + RW['sup_hit'] * (1. if info['support'] >= GATES['foot_support_mean_min']
                                            else -RW['sup_miss'])
                         - RW['nonfoot'] * info['nonfoot'] - RW['slip'] * info['slip']
                         - RW['com'] * max(0., GATES['com_height_min'] - info['com']) + RW['alive']
                         + (RW['fall'] if info['fall'] else 0.))
                TERM.append(term)
                steps += 1
                if term:
                    break
        O, A, LOGP = map(np.asarray, (O, A, LOGP))
        OM, OS, V, R, TERM = map(np.asarray, (OM, OS, V, R, TERM))
        adv, ret = gae_terminal(R, V, TERM)
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        opt = optimize(net, O, A, LOGP, OM, OS, adv_n, ret, rng, CFG['lr'])
        ev = evaluate(net)
        row = dict(update=u, decisions=int(len(R)), reward_mean=float(R.mean()),
                   terminated_fraction=float(TERM.mean()),
                   minibatch_updates=opt['minibatch_updates'], drift=opt['drift_vs_rollout'],
                   stopped=opt['stopped_at'],
                   **{k: ev[k] for k in ('forward_signed_native_per_s', 'foot_support_mean',
                                         'foot_support_min', 'nonfoot_force_max_W',
                                         'com_height_min', 'slip_ratio_max', 'all_gates')})
        log['updates'].append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                          if k != 'stopped'}), flush=True)
        (HERE / 'WALK_PPO_LOG.json').write_text(json.dumps(log, indent=2) + '\n')
    log['wall_s'] = time.time() - t0
    (HERE / 'WALK_PPO_LOG.json').write_text(json.dumps(log, indent=2) + '\n')
    print('done', round(log['wall_s'], 1), 's')


if __name__ == '__main__':
    main()
