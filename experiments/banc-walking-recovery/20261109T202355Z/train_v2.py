"""Diagnostic trainer fork of the repaired single-leg trainer, NOT a walking policy.

Source: banc-repair-pipeline/20261109T055352Z/train_single_leg.py. The inherited
independent per-leg actor/value architecture cannot coordinate walking. This isolated
S2/S5 training correction uses core_v2's COM-referenced adhesion fix; it does not
establish autonomous walking, original-brain ability, or physical feasibility limits.
BC prompts remain fixed for complete episodes. The task deadline is a true terminal
with zero bootstrap. PPO uses full-rollout analytic Gaussian KL with transactional
Adam rollback. Historical preregistrations remain in OLD; outputs go only to HERE.
The inherited world-x single-leg teacher remains a diagnostic teacher.
"""
from pathlib import Path
import json, os, sys, time
import numpy as np

HERE = Path(__file__).resolve().parent
OLD = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-repair-pipeline/20261109T055352Z')
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(HERE))
import mujoco as mj
from repair_core import WEIGHT, NJ, OBS, NL, PDT, MAX_RATE, LEG_ORDER, BaseNet
import core_v2 as rc

W = WEIGHT
T_EP = int(os.environ.get('T_EP', '300'))
HALFLIFE_S, GAE_LAMBDA = 1.0, .99
GAMMA = float(.5 ** (PDT / HALFLIFE_S))
HORIZON_S = float(-PDT / np.log(GAMMA * GAE_LAMBDA))
RW = dict(unload=.8, lift=1.2, advance_rate=1.0, place=2.0, nonfoot=20., support=30.,
          com=60., other_stance=.05, act=.02, omega=.01, grav=.01, place_hold=.6)
CFG = dict(updates=int(os.environ.get('UPDATES', '48')),
           rollout=int(os.environ.get('ROLLOUT', '1024')),
           epochs=4, minibatch=64, lr=1e-5, lr_min=1e-7, lr_max=3e-4, lr_grow=1.02,
           clip=.2, gamma=GAMMA, gae=GAE_LAMBDA, value_coef=.5, grad_norm=.5,
           target_kl=.03, initial_logstd=-3., logstd_bounds=[-5., -1.5],
           eval_every=int(os.environ.get('EVAL_EVERY', '12')))
WS_STEPS = int(os.environ.get('WS_STEPS', '600'))
WS_TICKS = int(os.environ.get('WS_TICKS', '1800'))
WS_MODE = os.environ.get('WS_MODE', 'base')
TAG = os.environ.get('TAG', '')
FIXED_LEG = os.environ.get('FIXED_LEG', '')
LEG_SEQ = os.environ.get('LEG_SEQ', '')
SEQ_BLOCK = int(os.environ.get('SEQ_BLOCK', '16'))
DEMO_LIFT_TICKS = int(os.environ.get('DEMO_LIFT_TICKS', '60'))
DEMO_SWING_TICKS = int(os.environ.get('DEMO_SWING_TICKS', '90'))
DEMO_HOLD = float(os.environ.get('DEMO_HOLD', '0.5'))
THRESH = dict(unload=0.10, lift=0.015, advance=0.03, place_load=0.10, place_padz=0.004,
              support=0.90, nonfoot=0.20 * W, com=1.00)
SUP_TARGET, COM_TARGET, LOW, HIGH, SWING_PADZ = .90, 1.00, .10 * W, .30 * W, .008
SWING_START, SWING_END = .05 * W, THRESH['place_load'] * W
TRAVEL = MAX_RATE * PDT * T_EP


def onehot(i):
    v = np.zeros(NL)
    v[i] = 1.
    return v


def leg_joint_action_index(core, leg):
    L = LEG_ORDER[leg]
    return [i for i, j in enumerate(core.actuator.joint_ids)
            if core.plant.joint(int(j)).name.startswith('joint_' + L)]


def pad_vertical_jacobian(core, leg):
    rows = []
    for g in core.pads[LEG_ORDER[leg]]:
        r = float(core.plant.geom_size[g][0])
        point = np.asarray(core.d.geom_xpos[g]) - np.array([0., 0., r])
        jacp = np.zeros((3, core.plant.nv))
        mj.mj_jac(core.plant, core.d, jacp, None, point, int(core.plant.geom_bodyid[g]))
        rows.append(jacp[2].copy())
    return np.mean(rows, axis=0)


def descend_direction(core, leg, scale=1.0):
    J = pad_vertical_jacobian(core, leg)
    a = np.zeros(NJ)
    for i in leg_joint_action_index(core, leg):
        j = float(J[int(core.actuator.dof_addresses[i])])
        if abs(j) > 1e-9:
            a[i] = -np.sign(j) * scale
    return a


def foot_horizontal_jacobian(core, leg):
    """Inherited world-x diagnostic teacher; not whole-body walking supervision."""
    rows = []
    for g in core.pads[LEG_ORDER[leg]]:
        jacp = np.zeros((3, core.plant.nv))
        mj.mj_jac(core.plant, core.d, jacp, None, np.asarray(core.d.geom_xpos[g]),
                  int(core.plant.geom_bodyid[g]))
        rows.append(jacp[0].copy())
    return np.mean(rows, axis=0)


def forward_direction(core, leg, scale=1.0):
    J = foot_horizontal_jacobian(core, leg)
    a = np.zeros(NJ)
    for i in leg_joint_action_index(core, leg):
        j = float(J[int(core.actuator.dof_addresses[i])])
        if abs(j) > 1e-9:
            a[i] = np.sign(j) * scale
    return a


def teacher_action(core, leg, k):
    """Declared timed diagnostic teacher; deployed network has no clock."""
    raise_a = -descend_direction(core, leg)
    if k < DEMO_LIFT_TICKS:
        return raise_a
    if k < DEMO_LIFT_TICKS + DEMO_SWING_TICKS:
        return np.clip(forward_direction(core, leg) + DEMO_HOLD * raise_a, -1., 1.)
    return descend_direction(core, leg)


def demo_directions():
    doc = json.loads((OLD / 'LIFT_DEMO.json').read_text())
    core = rc.Core()
    name_to_index = {core.plant.joint(int(j)).name: i
                     for i, j in enumerate(core.actuator.joint_ids)}
    dirs = {}
    for leg, L in enumerate(LEG_ORDER):
        rows = doc['per_leg'][L]
        chosen = doc['chosen'][L]['candidate']
        if chosen is None:
            chosen = max(rows.items(), key=lambda kv: kv[1]['rise'])[0]
        u = np.zeros(NJ)
        for n, v in zip(rows[chosen]['joints'], rows[chosen]['unit_command']):
            u[name_to_index['joint_' + L + n]] = v
        dirs[leg] = dict(unit=u, press=-u, candidate=chosen,
                         screened=bool(doc['chosen'][L]['candidate'] is not None))
    return dirs


def demo_action(demo, load, excursion):
    if load > SWING_END:
        return demo['unit']
    if excursion >= .5:
        return demo['press']
    return demo['unit']


def excursion(core, leg, unit):
    idx = leg_joint_action_index(core, leg)
    q = core.d.qpos[core.actuator.qpos_addresses[idx]]
    q0 = core.pose_q[idx]
    u = unit[idx]
    den = float(np.sum(np.abs(u)))
    return 0. if den < 1e-9 else float(np.sum(np.sign(u) * (q - q0))) / (den * TRAVEL)


class Net:
    """Inherited independent per-leg actors and critics; diagnostic, not walking."""

    def __init__(self, rng):
        self.p = dict(w0=(rng.standard_normal((NL, OBS, 128)) * np.sqrt(1 / OBS)),
                      b0=np.zeros((NL, 128)),
                      w1=(rng.standard_normal((NL, 128, 128)) * np.sqrt(1 / 128)),
                      b1=np.zeros((NL, 128)),
                      w2=(rng.standard_normal((NL, 128, NJ)) * .01), b2=np.zeros((NL, NJ)),
                      wc0=(rng.standard_normal((NL, OBS, 128)) * np.sqrt(1 / OBS)),
                      bc0=np.zeros((NL, 128)),
                      wc=(rng.standard_normal((NL, 128)) * .01), bc=np.zeros(NL),
                      logstd=np.full(NJ, CFG['initial_logstd']))
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0
        self._cache = {}
        self._critic_cache = None

    def std(self):
        return np.exp(np.clip(self.p['logstd'], *CFG['logstd_bounds']))

    @staticmethod
    def head_index(X):
        return np.argmax(X[:, -NL:], axis=1)

    def _value(self, X):
        idx = self.head_index(X)
        out = np.zeros(X.shape[0])
        cache = {}
        for h in range(NL):
            m = idx == h
            if m.any():
                hc = np.tanh(X[m] @ self.p['wc0'][h] + self.p['bc0'][h])
                out[m] = hc @ self.p['wc'][h] + self.p['bc'][h]
                cache[h] = hc
        self._critic_cache = (X, cache, idx)
        return out

    def forward(self, X):
        idx = self.head_index(X)
        out = np.zeros((X.shape[0], NJ))
        value = self._value(X)
        self._cache = {}
        for h in range(NL):
            m = idx == h
            if m.any():
                h0 = np.tanh(X[m] @ self.p['w0'][h] + self.p['b0'][h])
                h1 = np.tanh(h0 @ self.p['w1'][h] + self.p['b1'][h])
                out[m] = h1 @ self.p['w2'][h] + self.p['b2'][h]
                self._cache[h] = (h0, h1)
        return None, None, out, value

    def mean_backward(self, X, _h1_unused, d):
        idx = self.head_index(X)
        g = {k: np.zeros_like(self.p[k]) for k in ('w0', 'b0', 'w1', 'b1', 'w2', 'b2')}
        for h in range(NL):
            m = idx == h
            if not m.any():
                continue
            h0, h1 = self._cache[h]
            dh = d[m]
            g['w2'][h] = h1.T @ dh
            g['b2'][h] = dh.sum(0)
            dh1 = (dh @ self.p['w2'][h].T) * (1 - h1 ** 2)
            g['w1'][h] = h0.T @ dh1
            g['b1'][h] = dh1.sum(0)
            dh0 = (dh1 @ self.p['w1'][h].T) * (1 - h0 ** 2)
            g['w0'][h] = X[m].T @ dh0
            g['b0'][h] = dh0.sum(0)
        return g

    def value_backward(self, dvalue):
        X, cache, idx = self._critic_cache
        g = {k: np.zeros_like(self.p[k]) for k in ('wc0', 'bc0', 'wc', 'bc')}
        for h in range(NL):
            m = idx == h
            if not m.any():
                continue
            hc = cache[h]
            dv = dvalue[m]
            g['wc'][h] = hc.T @ dv
            g['bc'][h] = dv.sum()
            dpre = (dv[:, None] * self.p['wc'][h][None, :]) * (1 - hc ** 2)
            g['wc0'][h] = X[m].T @ dpre
            g['bc0'][h] = dpre.sum(0)
        return g

    def step(self, grads, lr):
        self.t += 1
        norm = np.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
        if norm > CFG['grad_norm']:
            grads = {k: g * (CFG['grad_norm'] / (norm + 1e-12)) for k, g in grads.items()}
        for k, g in grads.items():
            self.m[k] = .9 * self.m[k] + .1 * g
            self.v[k] = .999 * self.v[k] + .001 * g ** 2
            mh = self.m[k] / (1 - .9 ** self.t)
            vh = self.v[k] / (1 - .999 ** self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        self.p['logstd'] = np.clip(self.p['logstd'], *CFG['logstd_bounds'])

    def snapshot(self):
        """All mutable Adam state, including moments of currently inactive heads."""
        return tuple({k: v.copy() for k, v in group.items()}
                     for group in (self.p, self.m, self.v)) + (self.t,)

    def restore(self, state):
        self.p, self.m, self.v = ({k: v.copy() for k, v in group.items()} for group in state[:3])
        self.t = state[3]
        self._cache = {}
        self._critic_cache = None

    def save_to(self, path, extra):
        Z = {('actor_' + k): v.astype(np.float32) for k, v in self.p.items()}
        Z['critic_w'] = self.p['wc'].astype(np.float32)
        Z['critic_b'] = self.p['bc'].astype(np.float32)
        Z['logstd'] = self.p['logstd'].astype(np.float32)
        Z.update(extra)
        np.savez_compressed(path, **Z)


def warm_start(core, net, rng, ticks=None, steps=None, lr=1e-3):
    """BC on complete episodes, with a single fixed leg prompt per episode.

    ticks is a minimum data budget, rounded UP to a multiple of T_EP. A teacher
    schedule is allowed for diagnostic initialisation; the network has no clock.
    """
    ticks = int(os.environ.get('BC_TICKS', '1800')) if ticks is None else ticks
    steps = int(os.environ.get('BC_STEPS', '600')) if steps is None else steps
    if T_EP <= 0 or ticks < 0 or steps < 0:
        raise ValueError('T_EP must be positive; BC ticks/steps must be nonnegative')
    base = BaseNet()
    X, Y = [], []
    episodes = (ticks + T_EP - 1) // T_EP
    for _ in range(episodes):
        leg = LEG_ORDER.index(FIXED_LEG) if FIXED_LEG else int(rng.integers(0, NL))
        prompt = onehot(leg)
        core.reset()
        base.reset(core.q0)
        for k in range(T_EP):
            obs = core.obs(prompt)
            if WS_MODE == 'base':
                base_rate = (base.target() - core.actuator.target_rad) / PDT
                a = np.clip(base_rate / MAX_RATE, -1., 1.)
                core.tick_rate(base_rate)
            elif WS_MODE == 'demo':
                a = teacher_action(core, leg, k)
                core.tick(a)
            else:
                raise ValueError('WS_MODE must be base or demo')
            X.append(obs.copy())
            Y.append(np.arctanh(np.clip(a, -.95, .95)))
    if steps and not X:
        raise ValueError('BC optimisation needs at least one complete episode')
    X, Y = np.asarray(X), np.asarray(Y)
    losses = []
    for s in range(steps):
        ids = rng.integers(0, len(X), CFG['minibatch'])
        x, y = X[ids], Y[ids]
        h0, h1, mean, value = net.forward(x)
        grads = net.mean_backward(x, h1, (mean - y) / len(ids))
        for k in ('wc0', 'bc0', 'wc', 'bc', 'logstd'):
            grads[k] = np.zeros_like(net.p[k])
        net.step(grads, lr)
        if s % 100 == 0:
            mean = net.forward(X)[2]
            losses.append(dict(step=s, mse=float(np.mean((np.tanh(mean) - np.tanh(Y)) ** 2))))
    return dict(samples=int(len(X)), requested_ticks=ticks, episodes=episodes,
                prompt_sampling='once per complete episode', steps=steps, curve=losses,
                teacher=('frozen measured-motion base network' if WS_MODE == 'base'
                         else 'inherited timed single-leg Jacobian diagnostic teacher'))


def log_prob(lat, mean, std):
    z = (lat - mean) / std
    return np.sum(-.5 * z ** 2 - np.log(std) - .5 * np.log(2 * np.pi), axis=1), z


def episode_step(core, net, rng, leg, k, obs, st):
    h0, h1, mean, value = net.forward(obs[None])
    std = net.std()
    lat = mean[0] + std * rng.standard_normal(NJ)
    logp = float(log_prob(lat[None], mean, std)[0][0])
    a = np.tanh(lat)
    m = core.tick(a)
    loads = m['loads']
    sup = float(loads.sum() / W)
    padz = float(m['padz'][leg])
    fx = float(m['foot'][leg][0])
    rel = float(np.clip(1. - loads[leg] / (.30 * W), 0., 1.))
    lift = float(np.clip((padz - st['rest_z']) / .02, 0., 1.))
    fwd_body = abs(core.forward_body_velocity())
    if loads[leg] <= SWING_START and not st['swing']:
        st['swing'] = True; st['start_x'] = fx; st['max_lift'] = 0.; st['best_adv'] = 0.
    adv_rate = 0.
    if st['swing']:
        st['max_lift'] = max(st['max_lift'], padz)
        st['best_adv'] = max(st['best_adv'], fx - st['start_x'])
        adv_rate = float(np.clip((fx - st['prev_x']) / PDT / .5, -1., 1.))
        if loads[leg] >= SWING_END and padz <= THRESH['place_padz']:
            st['swing'] = False
            if (st['max_lift'] >= SWING_PADZ and st['best_adv'] >= THRESH['advance']
                    and not st['placed']):
                st['placed'] = True
                st['place_bonus'] = RW['place']
    st['prev_x'] = fx
    allow = max(.5 * fwd_body, .02)
    other = [i for i in range(NL) if i != leg]
    excess = float(sum(max(0., float(m['speed'][i]) - allow) for i in other))
    rot = core.d.xmat[core.th].reshape(3, 3)
    omega = core.d.qvel[3:6]
    grav = rot.T @ np.array([0., 0., -1.])
    # Inherited achievement-flag potential shaping and placement gates unchanged.
    if rel >= 0.5:
        st['did_unload'] = True
    if st['max_lift'] >= THRESH['lift']:
        st['did_lift'] = True
    phi = (RW['unload'] * (1. if st.get('did_unload') else 0.)
           + RW['lift'] * (1. if st.get('did_lift') else 0.)
           + RW['advance_rate'] * float(np.clip(st['best_adv'] / THRESH['advance'], 0., 1.))
           + RW['place'] * (1. if st['placed'] else 0.))
    shaping = 0. if st.get('phi_prev') is None else CFG['gamma'] * phi - st['phi_prev']
    st['phi_prev'] = phi
    placed_hold = 0.
    if st['placed'] and loads[leg] >= SWING_END and padz <= THRESH['place_padz']:
        placed_hold = RW['place_hold']
    reward = (shaping + st.pop('place_bonus', 0.) + placed_hold
              - RW['nonfoot'] * float(m['nonfoot'] / W)
              - RW['support'] * max(0., SUP_TARGET - sup) ** 2
              - RW['com'] * max(0., COM_TARGET - m['com']) ** 2
              - RW['other_stance'] * excess - RW['act'] * float(np.mean(a ** 2))
              - RW['omega'] * float(np.sum(omega ** 2))
              - RW['grav'] * float(np.sum(grav[:2] ** 2)))
    term = bool(m['com'] < .95 or m['nonfoot'] > .05 * W)
    return a, lat, logp, float(value[0]), reward, term, dict(
        sup=sup, com=m['com'], nonfoot=float(m['nonfoot']), rel=rel, lift=lift,
        adv=float(st['best_adv']), placed=bool(st['placed']),
        old_mean=mean[0].copy(), old_std=std.copy())


def evaluate(core, net, ticks=T_EP):
    """Deterministic inherited per-leg diagnostic criteria, NOT walking acceptance."""
    out = {}
    for leg in range(NL):
        core.reset()
        prompt = onehot(leg)
        st = dict(swing=False, start_x=0., max_lift=0., best_adv=0., prev_x=0.)
        sup, com, nf = [], [], []
        min_load = 1e9
        stance_excess = 0.
        loads = np.zeros(NL)
        padz_leg = 0.
        for k in range(ticks):
            mean = net.forward(core.obs(prompt)[None])[2]
            m = core.tick(np.tanh(mean[0]))
            loads = m['loads']
            padz_leg = float(m['padz'][leg])
            fx = float(m['foot'][leg][0])
            min_load = min(min_load, float(loads[leg]))
            if loads[leg] <= SWING_START and not st['swing']:
                st['swing'] = True; st['start_x'] = fx; st['max_lift'] = 0.; st['best_adv'] = 0.
            if st['swing']:
                st['max_lift'] = max(st['max_lift'], padz_leg)
                st['best_adv'] = max(st['best_adv'], fx - st['start_x'])
                if loads[leg] >= SWING_END and padz_leg <= THRESH['place_padz']:
                    st['swing'] = False
            sup.append(float(loads.sum() / W)); com.append(m['com']); nf.append(float(m['nonfoot']))
            allow = max(.5 * abs(core.forward_body_velocity()), .02)
            stance_excess = max(stance_excess, sum(max(0., float(m['speed'][i]) - allow)
                                                  for i in range(NL) if i != leg))
        m_load = float(loads[leg] / W)
        crit = dict(unload=bool(min_load / W <= THRESH['unload']),
                    lift=bool(st['max_lift'] >= THRESH['lift']),
                    advance=bool(st['best_adv'] >= THRESH['advance']),
                    place=bool(m_load >= THRESH['place_load'] and padz_leg <= THRESH['place_padz']),
                    safe=bool(min(sup) >= THRESH['support'] and max(nf) <= THRESH['nonfoot']
                              and min(com) >= THRESH['com']))
        out[LEG_ORDER[leg]] = dict(
            min_load_over_body_weight=float(min_load / W), max_lift=float(st['max_lift']),
            best_advance=float(st['best_adv']), final_load_over_body_weight=m_load,
            final_padz=padz_leg, support_min=float(min(sup)), com_min=float(min(com)),
            nonfoot_max=float(max(nf)), other_stance_excess_max=float(stance_excess),
            criteria=crit, passed=bool(all(crit.values())))
    out['_summary'] = dict(passed_legs=[k for k in LEG_ORDER if out[k]['passed']],
                           n_passed=int(sum(1 for k in LEG_ORDER if out[k]['passed'])),
                           all_six=bool(all(out[k]['passed'] for k in LEG_ORDER)))
    return out


def terminal_gae(rewards, values, terminated, deadlines):
    """Complete-episode GAE: task deadlines are TRUE terminals, bootstrap = 0.

    The rollout budget is a minimum and never cuts an episode. There are no
    artificial rollout truncations or V(s_t) stand-ins for successor values.
    """
    terminal = np.asarray(terminated, dtype=bool) | np.asarray(deadlines, dtype=bool)
    if len(rewards) == 0 or not terminal[-1]:
        raise ValueError('GAE requires nonempty complete episodes ending in a true terminal')
    adv = np.zeros(len(rewards), dtype=float)
    carry = 0.
    for i in range(len(rewards) - 1, -1, -1):
        if terminal[i]:
            next_value = 0.  # Explicit: failure OR actual task deadline has zero bootstrap.
            carry = 0.
        else:
            next_value = values[i + 1]
        delta = rewards[i] + CFG['gamma'] * next_value - values[i]
        carry = delta + CFG['gamma'] * CFG['gae'] * carry
        adv[i] = carry
    return adv, adv + values


def gaussian_kl(old_mean, old_std, mean, std):
    """Mean KL(old || current), summing independent latent Gaussian action dims.

    The shared invertible tanh transform preserves this KL. Old means AND standard
    deviations are stored at rollout collection, never reconstructed from logp.
    """
    log_ratio = np.log(std) - np.log(old_std)
    terms = (log_ratio + .5 * np.expm1(-2. * log_ratio)
             + .5 * ((old_mean - mean) / std) ** 2)
    return float(np.mean(np.sum(terms, axis=1)))


def optimize(net, O, A, LOGP, OLD_MEAN, OLD_STD, adv_n, ret, rng, lr):
    """Accept Adam steps only inside the full-rollout analytic KL budget.

    Each candidate step is provisional: on a nonfinite or over-budget full-rollout
    KL, restore ALL parameters, Adam moments and t, then halt the entire update.
    The bound is the rollout-average KL, not a pointwise per-observation bound.
    """
    if len(O) == 0:
        raise ValueError('PPO requires a nonempty rollout')
    OLD_MEAN, OLD_STD = np.asarray(OLD_MEAN), np.asarray(OLD_STD)
    if OLD_MEAN.shape != A.shape or OLD_STD.shape != A.shape:
        raise ValueError('Stored rollout means/std must have the same shape as latent actions')
    if not np.all(np.isfinite(OLD_MEAN)) or not np.all(np.isfinite(OLD_STD)) or np.any(OLD_STD <= 0):
        raise ValueError('Invalid stored rollout Gaussian')
    def rollout_kl():
        return gaussian_kl(OLD_MEAN, OLD_STD, net.forward(O)[2], net.std())
    drift = rollout_kl()
    if not np.isfinite(drift) or drift > CFG['target_kl']:
        raise ValueError('Initial policy already violates the rollout KL budget')
    total, kls, stopped_at = 0, [], None
    for ep in range(CFG['epochs']):
        perm = rng.permutation(len(O))
        for start in range(0, len(perm), CFG['minibatch']):
            ids = perm[start:start + CFG['minibatch']]
            x, lat_t, oldlog, advb, retb = O[ids], A[ids], LOGP[ids], adv_n[ids], ret[ids]
            h0, h1, mean, value = net.forward(x)
            std = net.std()
            logp, z = log_prob(lat_t, mean, std)
            ratio = np.exp(logp - oldlog)
            unclipped = advb * ratio
            clipped = advb * np.clip(ratio, 1 - CFG['clip'], 1 + CFG['clip'])
            active = (unclipped <= clipped).astype(float)
            dlogp = -(active * advb * ratio) / len(ids)
            dvalue = 2 * CFG['value_coef'] * (value - retb) / len(ids)
            dmean = dlogp[:, None] * (z / std[None, :])
            dlogstd = np.sum(dlogp[:, None] * (z ** 2 - 1), axis=0)
            grads = net.mean_backward(x, h1, dmean)
            grads.update(net.value_backward(dvalue))
            grads['logstd'] = dlogstd
            previous = net.snapshot()
            net.step(grads, lr)
            candidate_kl = rollout_kl()
            finite_state = all(np.all(np.isfinite(v)) for group in (net.p, net.m, net.v)
                               for v in group.values())
            if not finite_state or not np.isfinite(candidate_kl) or candidate_kl > CFG['target_kl']:
                net.restore(previous)
                stopped_at = dict(epoch=ep, minibatch=total, kl=candidate_kl,
                                  reason='candidate rejected; parameters and Adam state restored',
                                  rolled_back=True)
                break
            drift = candidate_kl
            kls.append(drift)
            total += 1
        if stopped_at is not None:
            break
    drift = rollout_kl()
    return dict(lr=lr, minibatch_updates=total, stopped_at=stopped_at, drift_vs_rollout=drift,
                kl_metric='full-rollout mean analytic KL(old Gaussian || current Gaussian)',
                kl_step_mean=float(np.mean(kls)) if kls else None,
                kl_step_max=float(np.max(kls)) if kls else None,
                nominal_updates=CFG['epochs'] * ((len(O) + CFG['minibatch'] - 1) // CFG['minibatch']))


def main():
    if T_EP <= 0 or CFG['rollout'] <= 0:
        raise ValueError('T_EP and rollout must be positive')
    t0 = time.time()
    rng = np.random.default_rng(int(os.environ.get('SEED', '20261109')))
    core, net = rc.Core(), Net(rng)
    # Historical documents are evidence references, not claims that the fork was preregistered.
    refs = {'prereg': 'PREREG.json', 'prereg_obs_v2': 'PREREG_OBS_V2.json'}
    refs.update({('prereg_amendment' if i == 1 else 'prereg_amendment_%d' % i):
                 'PREREG_AMENDMENT_%d.json' % i for i in range(1, 13)})
    log = {key: json.loads((OLD / name).read_text()) if (OLD / name).exists() else None
           for key, name in refs.items()}
    log.update(source=str(OLD / 'train_single_leg.py'), historical_prereg_directory=str(OLD),
               output_directory=str(HERE), diagnostic_only=True, walking_claim=False,
               seed=int(os.environ.get('SEED', '20261109')),
               architecture='independent per-leg sub-networks and value functions (diagnostic only)',
               bc_ticks=WS_TICKS, bc_steps=WS_STEPS, fixed_leg=FIXED_LEG or None,
               leg_sequence=LEG_SEQ or None, seq_block=SEQ_BLOCK if LEG_SEQ else None,
               config=CFG.copy(), gamma=GAMMA, gae=GAE_LAMBDA, gae_horizon_s=HORIZON_S,
               task_deadline='true terminal at T_EP; zero bootstrap',
               rollout_budget='minimum decisions; collect complete episodes, never truncate',
               reward_weights=RW, thresholds={k: float(v) for k, v in THRESH.items()},
               action_interface='declared engineering target-rate limit +/-2 rad/s; native torque/angle limits remain enforced',
               observation=dict(dim=OBS, proprioception=183, own_command_history=294, prompt=6))
    ws = warm_start(core, net, rng, ticks=WS_TICKS, steps=WS_STEPS)
    log['warm_start'] = ws
    print(json.dumps({'warm_start_mse': ws['curve'][-1]['mse'] if ws['curve'] else None}), flush=True)
    updates = []
    lr_state = CFG['lr']
    for u in range(CFG['updates']):
        O, A, LOGP, OLD_MEAN, OLD_STD, V, R, TERM, DEADLINE = ([] for _ in range(9))
        sup_l, com_l, nf_l, rel_l, lift_l, adv_l, place_l = ([] for _ in range(7))
        steps = 0
        while steps < CFG['rollout']:
            if LEG_SEQ:
                seq = [LEG_ORDER.index(x) for x in LEG_SEQ.split(',')]
                leg = seq[min(len(seq) - 1, u // SEQ_BLOCK)]
            else:
                leg = LEG_ORDER.index(FIXED_LEG) if FIXED_LEG else int(rng.integers(0, NL))
            prompt = onehot(leg)
            core.reset()
            st = dict(swing=False, start_x=0., max_lift=0., best_adv=0., placed=False, prev_x=0.,
                      rest_z=float(core.measure()['padz'][leg]))
            for k in range(T_EP):
                obs = core.obs(prompt)
                a, lat, logp, value, reward, term, info = episode_step(core, net, rng, leg, k, obs, st)
                O.append(obs.copy()); A.append(lat); LOGP.append(logp); V.append(value); R.append(reward)
                OLD_MEAN.append(info['old_mean']); OLD_STD.append(info['old_std'])
                deadline = k == T_EP - 1
                TERM.append(term); DEADLINE.append(deadline)
                sup_l.append(info['sup']); com_l.append(info['com']); nf_l.append(info['nonfoot'])
                rel_l.append(info['rel']); lift_l.append(info['lift']); adv_l.append(info['adv'])
                place_l.append(1. if info['placed'] else 0.)
                steps += 1
                if term or deadline:
                    break
        O, A, LOGP, OLD_MEAN, OLD_STD, V, R, TERM, DEADLINE = map(
            np.asarray, (O, A, LOGP, OLD_MEAN, OLD_STD, V, R, TERM, DEADLINE))
        adv, ret = terminal_gae(R, V, TERM, DEADLINE)
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        opt = optimize(net, O, A, LOGP, OLD_MEAN, OLD_STD, adv_n, ret, rng, lr_state)
        lr_state = opt['lr']
        row = dict(update=u, decisions=int(len(R)), reward_mean=float(R.mean()), reward_sum=float(R.sum()),
                   terminated_fraction=float((TERM | DEADLINE).mean()),
                   safety_terminated_fraction=float(TERM.mean()), deadline_fraction=float(DEADLINE.mean()),
                   truncated_fraction=0., kl_step_mean=opt['kl_step_mean'], kl_step_max=opt['kl_step_max'],
                   drift_vs_rollout=opt['drift_vs_rollout'], kl_metric=opt['kl_metric'],
                   minibatch_updates=opt['minibatch_updates'], nominal_minibatch_updates=opt['nominal_updates'],
                   lr=opt['lr'], stopped_at=opt['stopped_at'], support_mean=float(np.mean(sup_l)),
                   support_min=float(np.min(sup_l)), com_min=float(np.min(com_l)), nonfoot_max=float(np.max(nf_l)),
                   unload_frac=float(np.mean(rel_l)), lift_mean=float(np.mean(lift_l)),
                   advance_max=float(np.max(adv_l)), placed_fraction=float(np.mean(place_l)),
                   logstd=float(np.mean(net.p['logstd'])))
        updates.append(row)
        print(json.dumps(row), flush=True)
        if (u + 1) % CFG['eval_every'] == 0 or u == CFG['updates'] - 1:
            ev = evaluate(core, net)
            net.save_to(HERE / ('MODEL_singleleg_%s%s_u%02d.npz' % (WS_MODE, TAG, u + 1)),
                        dict(scheme=np.array('diagnostic_singleleg_v2'), global_updates=u + 1, obs_dim=OBS))
            (HERE / ('EVAL_%s%s_u%02d.json' % (WS_MODE, TAG, u + 1))).write_text(json.dumps(ev, indent=2) + '\n')
            print(json.dumps({'EVAL': u + 1, 'summary': ev['_summary']}), flush=True)
    net.save_to(HERE / ('MODEL_singleleg_%s%s.npz' % (WS_MODE, TAG)),
                dict(scheme=np.array('diagnostic_singleleg_v2_' + WS_MODE), global_updates=CFG['updates'], obs_dim=OBS))
    log['updates'] = updates
    log['final_eval'] = evaluate(core, net)
    log['wall_s'] = time.time() - t0
    log['physics_ticks'] = int(ws['samples'] + sum(row['decisions'] for row in updates)
                               + T_EP * NL * (1 + sum((u + 1) % CFG['eval_every'] == 0
                                                       or u == CFG['updates'] - 1
                                                       for u in range(CFG['updates']))))
    (HERE / ('TRAIN_LOG_singleleg_%s%s.json' % (WS_MODE, TAG))).write_text(json.dumps(log, indent=2) + '\n')
    print(json.dumps({'DONE': True, 'diagnostic_only': True, 'wall_s': round(log['wall_s'], 1),
                      'eval': log['final_eval']}, indent=2))


if __name__ == '__main__':
    main()
