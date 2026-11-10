"""Single-leg unload-lift-place on the repaired pipeline (see PREREG.json).

Task: a one-hot prompt names a leg; within 300 ms that leg must unload, lift its pads,
move its foot forward and be loaded again at the end, while the body stays supported and
no non-foot part touches the ground. No clock, phase or external gait generator is used:
the prompt is a task instruction input and the only other inputs are current body sensing
and the last seven own commands.
"""
from pathlib import Path
import json, os, sys, time
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mujoco as mj
import repair_core as rc

W = rc.WEIGHT
NJ, OBS, NL = rc.NJ, rc.OBS, rc.NL
T_EP = int(os.environ.get('T_EP', '300'))
PDT = rc.PDT
HALFLIFE_S, GAE_LAMBDA = 1.0, .99
GAMMA = float(.5 ** (PDT / HALFLIFE_S))
HORIZON_S = float(-PDT / np.log(GAMMA * GAE_LAMBDA))
RW = dict(unload=.8, lift=1.2, advance_rate=1.0, place=2.0, nonfoot=20., support=30.,
          com=60., other_stance=.05, act=.02, omega=.01, grav=.01, place_hold=.6)
# other_stance and omega were reduced before the run: see PREREG_AMENDMENT_1.json and
# REWARD_SCALE.json. The measured stance-slip term was 28x the task terms and would have
# dominated a task that is not about slip.
CFG = dict(updates=int(os.environ.get('UPDATES', '48')),
           rollout=int(os.environ.get('ROLLOUT', '1024')),
           epochs=4, minibatch=64, lr=1e-5, lr_min=1e-7, lr_max=3e-4, lr_grow=1.02,
           clip=.2, gamma=GAMMA,
           gae=GAE_LAMBDA, value_coef=.5, grad_norm=.5, target_kl=.03, kl_backtracks=6,
           initial_logstd=-3., logstd_bounds=[-5., -1.5],
           eval_every=int(os.environ.get('EVAL_EVERY', '12')))
WS_STEPS = int(os.environ.get('WS_STEPS', '600'))
WS_TICKS = int(os.environ.get('WS_TICKS', '1800'))
WS_MODE = os.environ.get('WS_MODE', 'base')      # 'base' = frozen base network, 'demo' = lift demo
TAG = os.environ.get('TAG', '')                  # arm suffix so runs never overwrite each other
FIXED_LEG = os.environ.get('FIXED_LEG', '')       # restrict sampling to one leg (diagnostic)
LEG_SEQ = os.environ.get('LEG_SEQ', '')           # e.g. 'LF,LM,LH,RF,RM,RH': block curriculum
SEQ_BLOCK = int(os.environ.get('SEQ_BLOCK', '16'))  # updates per leg block


DEMO_LIFT_TICKS = int(os.environ.get('DEMO_LIFT_TICKS', '60'))
DEMO_SWING_TICKS = int(os.environ.get('DEMO_SWING_TICKS', '90'))
DEMO_HOLD = float(os.environ.get('DEMO_HOLD', '0.5'))


def pad_vertical_jacobian(core, leg):
    """Mean d(pad bottom height)/d(q) over the leg's three pad spheres."""
    rows = []
    for g in core.pads[rc.LEG_ORDER[leg]]:
        r = float(core.plant.geom_size[g][0])
        point = np.asarray(core.d.geom_xpos[g]) - np.array([0., 0., r])
        jacp = np.zeros((3, core.plant.nv))
        mj.mj_jac(core.plant, core.d, jacp, None, point, int(core.plant.geom_bodyid[g]))
        rows.append(jacp[2].copy())
    return np.mean(rows, axis=0)


def descend_direction(core, leg, scale=1.0):
    """Joint rates that lower the leg's pads, computed at the CURRENT configuration.

    Negating the lift direction cannot do this: demo_sweep showed it undoes the lift instead of
    lowering the foot (DEMO_SWEEP.json, DEMO_CYCLE.json).
    """
    J = pad_vertical_jacobian(core, leg)
    a = np.zeros(rc.NJ)
    for i in leg_joint_action_index(core, leg):
        j = float(J[int(core.actuator.dof_addresses[i])])
        if abs(j) > 1e-9:
            a[i] = -np.sign(j) * scale
    return a


def foot_horizontal_jacobian(core, leg):
    """Mean d(foot x position)/d(q) over the leg's pad spheres."""
    rows = []
    for g in core.pads[rc.LEG_ORDER[leg]]:
        point = np.asarray(core.d.geom_xpos[g])
        jacp = np.zeros((3, core.plant.nv))
        mj.mj_jac(core.plant, core.d, jacp, None, point, int(core.plant.geom_bodyid[g]))
        rows.append(jacp[0].copy())
    return np.mean(rows, axis=0)


def forward_direction(core, leg, scale=1.0):
    """Joint rates that move this leg's foot forward (+x) at the current pose."""
    J = foot_horizontal_jacobian(core, leg)
    a = np.zeros(rc.NJ)
    for i in leg_joint_action_index(core, leg):
        j = float(J[int(core.actuator.dof_addresses[i])])
        if abs(j) > 1e-9:
            a[i] = np.sign(j) * scale
    return a


def teacher_action(core, leg, k):
    """Three phase declared demonstration: raise, swing forward, put down.

    Both the raise and the lowering direction come from the vertical Jacobian at the current
    pose; the swing comes from the horizontal Jacobian with half the raise kept in to hold
    height. The teacher has a fixed schedule; the deployed network has no clock
    (PREREG_AMENDMENT_8.json)."""
    raise_a = -descend_direction(core, leg)
    if k < DEMO_LIFT_TICKS:
        return raise_a
    if k < DEMO_LIFT_TICKS + DEMO_SWING_TICKS:
        return np.clip(forward_direction(core, leg) + DEMO_HOLD * raise_a, -1., 1.)
    return descend_direction(core, leg)


def demo_directions():
    """Two-phase state-feedback teacher built from LIFT_DEMO.json (declared teacher).

    This is a DEMONSTRATION generator, not the deployed controller: it is used only to
    initialise the network by supervised regression, and the network is then fine-tuned with
    the teacher absent (PREREG_AMENDMENT_5.json). It contains no clock and no phase variable;
    both transitions are functions of measured state:

      still carrying load           -> command the screened lift direction
      unloaded but not yet lifted   -> command the screened lift direction
      unloaded and lifted to 0.015  -> command the mirrored direction, i.e. put the foot down

    The mirrored direction is the same Jacobian direction with the opposite sign, so the
    teacher is exactly two declared constants per leg plus one measured height threshold.
    """
    doc = json.loads((HERE / 'LIFT_DEMO.json').read_text())
    core = rc.Core()
    name_to_index = {core.plant.joint(int(j)).name: i
                     for i, j in enumerate(core.actuator.joint_ids)}
    dirs = {}
    for leg, L in enumerate(rc.LEG_ORDER):
        rows = doc['per_leg'][L]
        chosen = doc['chosen'][L]['candidate']
        if chosen is None:
            chosen = max(rows.items(), key=lambda kv: kv[1]['rise'])[0]
        u = np.zeros(rc.NJ)
        for n, v in zip(rows[chosen]['joints'], rows[chosen]['unit_command']):
            u[name_to_index['joint_' + L + n]] = v
        dirs[leg] = dict(unit=u, press=-u, candidate=chosen,
                         screened=bool(doc['chosen'][L]['candidate'] is not None))
    return dirs


def demo_action(demo, load, excursion):
    """The teacher's action, a function of measured state only (no clock, no phase variable).

    `excursion` is how far the leg has already travelled along the lift direction, measured from
    its own joint angles and expressed as a fraction of the full commanded travel. It is used
    instead of the pad height because the 183-feature sensory vector contains joint angles,
    tracking error, velocities, targets, foot loads, gravity and body velocities -- but NOT the
    pad height, so a pad-height switch would be unobservable to the network being taught.
    """
    if load > SWING_END:
        return demo['unit']
    if excursion >= .5:
        return demo['press']
    return demo['unit']
THRESH = dict(unload=0.10, lift=0.015, advance=0.03, place_load=0.10, place_padz=0.004,
              support=0.90, nonfoot=0.20 * W, com=1.00)
SUP_TARGET, COM_TARGET, LOW, HIGH, SWING_PADZ = .90, 1.00, .10 * W, .30 * W, .008
# Swing hysteresis and placement gates aligned with the acceptance criteria
# (PREREG_AMENDMENT_4.json): a swing closes at the same load the evaluator calls
# 'placed', not at three times that load.
SWING_START, SWING_END = .05 * W, THRESH['place_load'] * W
# joint travel available to a saturated command inside one episode window
TRAVEL = rc.MAX_RATE * rc.PDT * T_EP


class Net:
    """Six fully independent sub-networks plus a shared value function.

    Declared engineering encoding (PREREG_AMENDMENT_11.json) for the single-leg step task:
    the leg prompt selects which sub-network acts, so the six programmes cannot interfere.
    Explicitly not a walking architecture: independent sub-networks cannot coordinate legs.
    """

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
                Xh = X[m]
                h0 = np.tanh(Xh @ self.p['w0'][h] + self.p['b0'][h])
                h1 = np.tanh(h0 @ self.p['w1'][h] + self.p['b1'][h])
                out[m] = h1 @ self.p['w2'][h] + self.p['b2'][h]
                self._cache[h] = (h0, h1)
        return None, None, out, value

    def mean_backward(self, X, _h1_unused, d):
        """Gradients of the acting sub-network only, and of nothing for the other legs."""
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
        """Gradients of each leg's own value function."""
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

    def save_to(self, path, extra):
        Z = {('actor_' + k): self.p[k].astype(np.float32)
             for k in ('w0', 'b0', 'w1', 'b1', 'w2', 'b2', 'wc0', 'bc0', 'wc', 'bc', 'logstd')}
        Z['critic_w'] = self.p['wc'].astype(np.float32)
        Z['critic_b'] = self.p['bc'].astype(np.float32)
        Z['logstd'] = self.p['logstd'].astype(np.float32)
        Z.update(extra)
        np.savez_compressed(path, **Z)


def onehot(i):
    v = np.zeros(NL)
    v[i] = 1.
    return v


def leg_joint_action_index(core, leg):
    L = rc.LEG_ORDER[leg]
    return [i for i, j in enumerate(core.actuator.joint_ids)
            if core.plant.joint(int(j)).name.startswith('joint_' + L)]


def excursion(core, leg, unit):
    """Fraction of the commanded lift travel already achieved, from joint angles only."""
    idx = leg_joint_action_index(core, leg)
    q = core.d.qpos[core.actuator.qpos_addresses[idx]]
    q0 = core.pose_q[idx]      # pose_q is already in actuator order
    u = unit[idx]
    den = float(np.sum(np.abs(u)))
    if den < 1e-9:
        return 0.
    got = float(np.sum(np.sign(u) * (q - q0)))
    # full travel of the direction is |u|<=1 over RATE_TRAVEL, so normalise by the travel
    return got / (den * TRAVEL)


def warm_start(core, net, rng, ticks=None, steps=None, lr=1e-3):
    ticks = int(os.environ.get('BC_TICKS', '1800')) if ticks is None else ticks
    steps = int(os.environ.get('BC_STEPS', '600')) if steps is None else steps
    """Supervised regression of the actor mean onto a declared teacher.

    WS_MODE=base imitates the frozen measured-motion base network's rate command.
    WS_MODE=demo imitates the screened lift direction of the prompted leg while that leg still
    carries load (LIFT_DEMO.json): a function of measured load only, no clock, no phase.
    """
    base = rc.BaseNet()
    demos = demo_directions() if WS_MODE == 'demo' else None
    X, Y = [], []
    core.reset(); base.reset(core.q0)
    for t in range(ticks):
        leg = (rc.LEG_ORDER.index(FIXED_LEG) if FIXED_LEG else int(rng.integers(0, NL)))
        obs = core.obs(onehot(leg))
        if demos is None:
            bt = base.target()
            base_rate = (bt - core.actuator.target_rad) / PDT
            a = np.clip(base_rate / rc.MAX_RATE, -1., 1.)
            core.tick_rate(base_rate)
        else:
            # timed demonstration: lift for DEMO_LIFT_TICKS, then press down along the vertical
            # Jacobian recomputed at the current pose. The teacher is allowed a schedule; the
            # network that imitates it is not (PREREG_AMENDMENT_6.json).
            k = t % T_EP
            a = teacher_action(core, leg, k)
            core.tick(a)
        X.append(obs); Y.append(np.arctanh(np.clip(a, -.95, .95)))
        if t % T_EP == T_EP - 1:
            core.reset(); base.reset(core.q0)
    X = np.asarray(X); Y = np.asarray(Y)
    losses = []
    for s in range(steps):
        ids = rng.integers(0, len(X), CFG['minibatch'])
        x, y = X[ids], Y[ids]
        h0, h1, mean, value = net.forward(x)
        d = (mean - y) / len(ids)
        grads = net.mean_backward(x, h1, d)
        for k in ('wc0', 'bc0', 'wc', 'bc', 'logstd'):
            grads[k] = np.zeros_like(net.p[k])
        net.step(grads, lr)
        if s % 100 == 0:
            h0, h1, mean, _ = net.forward(X)
            losses.append(dict(step=s, mse=float(np.mean((np.tanh(mean) - np.tanh(Y)) ** 2))))
    return dict(samples=int(len(X)), steps=steps, curve=losses,
                teacher='frozen measured-motion base network, initialisation only')


def episode_step(core, net, rng, leg, k, obs, st):
    h0, h1, mean, value = net.forward(obs[None])
    std = net.std()
    lat = mean[0] + std * rng.standard_normal(NJ)
    logp = float(np.sum(-.5 * ((lat - mean[0]) / std) ** 2 - np.log(std) - .5 * np.log(2 * np.pi)))
    a = np.tanh(lat)
    m = core.tick(a)
    loads = m['loads']
    sup = float(loads.sum() / W)
    padz = float(m['padz'][leg])
    fx = float(m['foot'][leg][0])
    rel = float(np.clip(1. - loads[leg] / (.30 * W), 0., 1.))
    # shaping is relative to the leg's own resting pad height, because the declared adhesion
    # presses the pads below the ground plane; the acceptance criterion stays absolute
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
    # potential-based task shaping (PREREG_AMENDMENT_2.json): the task quantities are a state
    # potential, so holding a lifted foot earns nothing further and only progress pays. The
    # earlier level form paid about 2 per tick for holding the foot up, which is why placing
    # never appeared even when lifting worked.
    # achievement flags, not instantaneous levels (PREREG_AMENDMENT_3.json): a level potential
    # still prefers ending the episode with the foot in the air, because that state has the
    # highest potential. Flags stay set once reached, so completing the step dominates.
    if rel >= 0.5:
        st['did_unload'] = True
    if st['max_lift'] >= THRESH['lift']:
        st['did_lift'] = True
    phi = (RW['unload'] * (1. if st.get('did_unload') else 0.)
           + RW['lift'] * (1. if st.get('did_lift') else 0.)
           + RW['advance_rate'] * float(np.clip(st['best_adv'] / THRESH['advance'], 0., 1.))
           + RW['place'] * (1. if st['placed'] else 0.))
    if st.get('phi_prev') is None:
        shaping = 0.
    else:
        shaping = CFG['gamma'] * phi - st['phi_prev']
    st['phi_prev'] = phi
    placed_hold = 0.
    if st['placed'] and loads[leg] >= SWING_END and padz <= THRESH['place_padz']:
        placed_hold = RW['place_hold']
    reward = (shaping
              + st.pop('place_bonus', 0.)
              + placed_hold
              - RW['nonfoot'] * float(m['nonfoot'] / W)
              - RW['support'] * max(0., SUP_TARGET - sup) ** 2
              - RW['com'] * max(0., COM_TARGET - m['com']) ** 2
              - RW['other_stance'] * excess
              - RW['act'] * float(np.mean(a ** 2))
              - RW['omega'] * float(np.sum(omega ** 2))
              - RW['grav'] * float(np.sum(grav[:2] ** 2)))
    term = bool(m['com'] < .95 or m['nonfoot'] > .05 * W)
    return a, lat, logp, float(value[0]), reward, term, dict(
        sup=sup, com=m['com'], nonfoot=float(m['nonfoot']), rel=rel, lift=lift,
        adv=float(st['best_adv']), placed=bool(st['placed']))


def evaluate(core, net, ticks=T_EP):
    """Deterministic per-leg check of the pre-registered criteria."""
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
            obs = core.obs(prompt)
            h0, h1, mean, _ = net.forward(obs[None])
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
        crit = dict(
            unload=bool(min_load / W <= THRESH['unload']),
            lift=bool(st['max_lift'] >= THRESH['lift']),
            advance=bool(st['best_adv'] >= THRESH['advance']),
            place=bool(m_load >= THRESH['place_load'] and padz_leg <= THRESH['place_padz']),
            safe=bool(min(sup) >= THRESH['support'] and max(nf) <= THRESH['nonfoot']
                      and min(com) >= THRESH['com']))
        out[rc.LEG_ORDER[leg]] = dict(
            min_load_over_body_weight=float(min_load / W),
            max_lift=float(st['max_lift']), best_advance=float(st['best_adv']),
            final_load_over_body_weight=m_load, final_padz=padz_leg,
            support_min=float(min(sup)), com_min=float(min(com)), nonfoot_max=float(max(nf)),
            other_stance_excess_max=float(stance_excess),
            criteria=crit, passed=bool(all(crit.values())))
    out['_summary'] = dict(passed_legs=[k for k in rc.LEG_ORDER if out[k]['passed']],
                           n_passed=int(sum(1 for k in rc.LEG_ORDER if out[k]['passed'])),
                           all_six=bool(all(out[k]['passed'] for k in rc.LEG_ORDER)))
    return out


def log_prob(lat, mean, std):
    z = (lat - mean) / std
    return np.sum(-.5 * z ** 2 - np.log(std) - .5 * np.log(2 * np.pi), axis=1), z


def optimize(net, O, A, LOGP, adv_n, ret, rng, lr):
    """PPO epochs with a cumulative KL budget per iteration.

    Evidence for this design (PPO_KL_DIAGNOSIS.json, from the inherited trainer): one Adam step
    at lr 3e-4 moved the policy by KL ~0.9 against the rollout policy, i.e. 30x the target, so
    the inherited early stop fired on the second minibatch of every epoch and each "update"
    performed exactly one minibatch step of an oversized move. Here the step size is fixed and
    modest, and the epoch loop stops when the drift measured on the current minibatch reaches
    target_kl, so an iteration consists of as many steps as fit inside the trust region.
    """
    total = 0; kls = []; stopped_at = None
    for ep in range(CFG['epochs']):
        perm = rng.permutation(len(O))
        mb = max(1, min(CFG['minibatch'], len(perm)))
        perm = perm[:len(perm) // mb * mb]
        for ids in perm.reshape(-1, mb):
            x, lat_t, oldlog, advb, retb = O[ids], A[ids], LOGP[ids], adv_n[ids], ret[ids]
            h0, h1, mean, value = net.forward(x)
            std = net.std()
            logp, z = log_prob(lat_t, mean, std)
            kl_now = float(np.mean(oldlog - logp))
            if kl_now > CFG['target_kl']:
                stopped_at = dict(epoch=ep, minibatch=total, kl=kl_now)
                break
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
            net.step(grads, lr)
            logpb, _zb = log_prob(lat_t, net.forward(x)[2], net.std())
            kls.append(float(np.mean(oldlog - logpb)))
            total += 1
        if stopped_at is not None:
            break
    drift = float(np.mean(LOGP - log_prob(A, net.forward(O)[2], net.std())[0]))
    return dict(lr=lr, minibatch_updates=total, stopped_at=stopped_at, drift_vs_rollout=drift,
                kl_step_mean=float(np.mean(kls)) if kls else None,
                kl_step_max=float(np.max(kls)) if kls else None,
                nominal_updates=CFG['epochs'] * max(1, len(O) // CFG['minibatch']))


def main():
    t0 = time.time()
    rng = np.random.default_rng(int(os.environ.get('SEED', '20261109')))
    core = rc.Core()
    net = Net(rng)
    log = dict(prereg=json.loads((HERE / 'PREREG.json').read_text()),
               prereg_amendment=json.loads((HERE / 'PREREG_AMENDMENT_1.json').read_text()),
               prereg_amendment_2=json.loads((HERE / 'PREREG_AMENDMENT_2.json').read_text())
               if (HERE / 'PREREG_AMENDMENT_2.json').exists() else None,
               prereg_amendment_3=json.loads((HERE / 'PREREG_AMENDMENT_3.json').read_text())
               if (HERE / 'PREREG_AMENDMENT_3.json').exists() else None,
               prereg_amendment_4=json.loads((HERE / 'PREREG_AMENDMENT_4.json').read_text())
               if (HERE / 'PREREG_AMENDMENT_4.json').exists() else None,
               prereg_amendment_5=json.loads((HERE / 'PREREG_AMENDMENT_5.json').read_text())
               if (HERE / 'PREREG_AMENDMENT_5.json').exists() else None,
               prereg_amendment_6=json.loads((HERE / 'PREREG_AMENDMENT_6.json').read_text())
               if (HERE / 'PREREG_AMENDMENT_6.json').exists() else None,
               prereg_obs_v2=json.loads((HERE / 'PREREG_OBS_V2.json').read_text())
               if (HERE / 'PREREG_OBS_V2.json').exists() else None,
               prereg_amendment_7=json.loads((HERE / 'PREREG_AMENDMENT_7.json').read_text())
               if (HERE / 'PREREG_AMENDMENT_7.json').exists() else None,
               prereg_amendment_8=json.loads((HERE / 'PREREG_AMENDMENT_8.json').read_text())
               if (HERE / 'PREREG_AMENDMENT_8.json').exists() else None,
               prereg_amendment_9=json.loads((HERE / 'PREREG_AMENDMENT_9.json').read_text())
               if (HERE / 'PREREG_AMENDMENT_9.json').exists() else None,
               prereg_amendment_10=json.loads((HERE / 'PREREG_AMENDMENT_10.json').read_text())
               if (HERE / 'PREREG_AMENDMENT_10.json').exists() else None,
               prereg_amendment_11=json.loads((HERE / 'PREREG_AMENDMENT_11.json').read_text())
               if (HERE / 'PREREG_AMENDMENT_11.json').exists() else None,
               prereg_amendment_12=json.loads((HERE / 'PREREG_AMENDMENT_12.json').read_text())
               if (HERE / 'PREREG_AMENDMENT_12.json').exists() else None,
               seed=int(os.environ.get('SEED', '20261109')),
               architecture='independent per-leg sub-networks and value functions (amendments 11, 12)',
               bc_ticks=int(os.environ.get('BC_TICKS', '1800')),
               bc_steps=int(os.environ.get('BC_STEPS', '600')),
               fixed_leg=FIXED_LEG or None,
               leg_sequence=LEG_SEQ or None,
               seq_block=SEQ_BLOCK if LEG_SEQ else None,
               config={k: (float(v) if isinstance(v, float) else v) for k, v in CFG.items()},
               gamma=GAMMA, gae=GAE_LAMBDA, gae_horizon_s=HORIZON_S,
               reward_weights=RW, thresholds={k: float(v) for k, v in THRESH.items()},
               action_interface='rate = tanh(mean+std*noise) * 2.0 rad/s, full actuator authority',
               observation=dict(dim=OBS, proprioception=183, own_command_history=294, prompt=6))
    ws = warm_start(core, net, rng, ticks=WS_TICKS, steps=WS_STEPS)
    log['warm_start'] = ws
    print(json.dumps({'warm_start_mse': ws['curve'][-1]['mse'] if ws['curve'] else None}), flush=True)
    updates = []
    lr_state = [CFG['lr']]
    for u in range(CFG['updates']):
        O, A, LOGP, V, R, TERM, TRUNC = [], [], [], [], [], [], []
        sup_l, com_l, nf_l, rel_l, lift_l, adv_l, place_l = [], [], [], [], [], [], []
        steps = 0
        while steps < CFG['rollout']:
            if LEG_SEQ:
                seq = [rc.LEG_ORDER.index(x) for x in LEG_SEQ.split(',')]
                leg = seq[min(len(seq) - 1, u // SEQ_BLOCK)]
            else:
                leg = (rc.LEG_ORDER.index(FIXED_LEG) if FIXED_LEG
                       else int(rng.integers(0, NL)))
            prompt = onehot(leg)
            core.reset()
            st = dict(swing=False, start_x=0., max_lift=0., best_adv=0., placed=False, prev_x=0.,
                      rest_z=float(core.measure()['padz'][leg]))
            last = T_EP - 1
            for k in range(T_EP):
                obs = core.obs(prompt)
                a, lat, logp, value, reward, term, info = episode_step(core, net, rng, leg, k, obs, st)
                O.append(obs); A.append(lat); LOGP.append(logp); V.append(value); R.append(reward)
                TERM.append(term); TRUNC.append((k == last) and not term)
                sup_l.append(info['sup']); com_l.append(info['com']); nf_l.append(info['nonfoot'])
                rel_l.append(info['rel']); lift_l.append(info['lift']); adv_l.append(info['adv'])
                place_l.append(1. if info['placed'] else 0.)
                steps += 1
                if term or k == last:
                    break
                if steps >= CFG['rollout']:
                    TRUNC[-1] = True
                    break
        O = np.asarray(O); A = np.asarray(A); V = np.asarray(V); R = np.asarray(R)
        TERM = np.asarray(TERM); TRUNC = np.asarray(TRUNC); LOGP = np.asarray(LOGP)
        adv = np.zeros_like(R); last = 0.
        for i in range(len(R) - 1, -1, -1):
            nxt = V[i + 1] if (i + 1 < len(V) and not TERM[i] and not TRUNC[i]) else 0.
            if TRUNC[i]:
                nxt = V[i]                      # bootstrap from the state at the time limit
            td = R[i] + CFG['gamma'] * nxt - V[i]
            cut = TERM[i] or TRUNC[i]
            last = td + CFG['gamma'] * CFG['gae'] * (0. if cut else 1.) * last
            adv[i] = last
        ret = adv + V
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        opt = optimize(net, O, A, LOGP, adv_n, ret, rng, lr_state[0])
        lr_state[0] = opt['lr']
        row = dict(update=u, decisions=int(len(R)), reward_mean=float(R.mean()), reward_sum=float(R.sum()),
                   terminated_fraction=float(TERM.mean()), truncated_fraction=float(TRUNC.mean()),
                   kl_step_mean=opt['kl_step_mean'], kl_step_max=opt['kl_step_max'],
                   drift_vs_rollout=opt['drift_vs_rollout'],
                   minibatch_updates=opt['minibatch_updates'],
                   nominal_minibatch_updates=opt['nominal_updates'], lr=opt['lr'],
                   stopped_at=opt['stopped_at'],
                   support_mean=float(np.mean(sup_l)), support_min=float(np.min(sup_l)),
                   com_min=float(np.min(com_l)), nonfoot_max=float(np.max(nf_l)),
                   unload_frac=float(np.mean(rel_l)), lift_mean=float(np.mean(lift_l)),
                   advance_max=float(np.max(adv_l)), placed_fraction=float(np.mean(place_l)),
                   logstd=float(np.mean(net.p['logstd'])))
        updates.append(row)
        print(json.dumps(row), flush=True)
        if (u + 1) % CFG['eval_every'] == 0 or u == CFG['updates'] - 1:
            ev = evaluate(core, net)
            net.save_to(HERE / ('MODEL_singleleg_%s%s_u%02d.npz' % (WS_MODE, TAG, u + 1)),
                        dict(scheme=np.array('repaired_direct_rate_singleleg'),
                             global_updates=u + 1, obs_dim=OBS))
            (HERE / ('EVAL_%s%s_u%02d.json' % (WS_MODE, TAG, u + 1))).write_text(json.dumps(ev, indent=2) + '\n')
            print(json.dumps({'EVAL': u + 1,
                              'summary': ev['_summary'],
                              'per_leg': {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                              for kk, vv in v.items()}
                                          for k, v in ev.items() if k != '_summary'}}), flush=True)
    net.save_to(HERE / ('MODEL_singleleg_%s%s.npz' % (WS_MODE, TAG)), dict(scheme=np.array('repaired_direct_rate_singleleg_' + WS_MODE),
                                                   global_updates=CFG['updates'], obs_dim=OBS))
    log['updates'] = updates
    log['final_eval'] = evaluate(core, net)
    log['wall_s'] = time.time() - t0
    log['physics_ticks'] = int(CFG['updates'] * CFG['rollout'])
    (HERE / ('TRAIN_LOG_singleleg_%s%s.json' % (WS_MODE, TAG))).write_text(json.dumps(log, indent=2) + '\n')
    print(json.dumps({'DONE': True, 'wall_s': round(log['wall_s'], 1),
                      'eval': log['final_eval']}, indent=2))


if __name__ == '__main__':
    main()
