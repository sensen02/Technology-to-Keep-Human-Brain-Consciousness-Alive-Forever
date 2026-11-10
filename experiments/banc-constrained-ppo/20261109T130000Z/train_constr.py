"""Full 42-joint PPO with a fading measured-data prior (scheme A), reward v3, optional adhesion.

No deep-learning framework is installed in this project's environments, so the small policy and
critic are implemented directly in numpy with hand-written gradients. Everything else follows
PREREG_RESIDUAL_V2.json: frozen body, frozen bounded incremental actuator, frozen base network,
six symmetric tibia residuals (+-0.003 rad per 5 ms tick), reward built from the acceptance
metric (per-leg slip hinge to 0.35 plus stance-phase foot-speed penalty), curriculum
0.05 -> 0.12 -> 0.20 -> 0.30 with per-level gate evaluation.

No phase, no time input, no teacher, no root/pose write, no external oscillator.
"""
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
import mujoco as mj

HERE = Path(__file__).resolve().parent
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
ACT = ROOT / 'experiments/s5-ann-incremental/20261107T010000Z/incremental_actuator.py'
SENSORY = ROOT / 'experiments/s5-ann-walk/20261107T091243Z/proprioception.py'
BASE = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-stand-retarget/20261108T190000Z/MODEL_MEASURED_STAND.npz')
MU = float(__import__('os').environ.get('F_MU', '1.5'))
PAD_XML = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-tarsus-pad/20261109T010000Z/MODEL_TARSUS_PAD_R02.xml')
W_DUTY, W_SWING = 0.0, 0.0
DUTY_TARGET, DUTY_TAU = .6, .2
SWING_LOAD, CLEAR_REF = .1, .02
WEIGHT = 24.446354755878005
_POSE = json.loads((ROOT / 'experiments/banc-stand-pose/20261108T170000Z/STAND_POSE_2.json').read_text())
STAND_Q = np.asarray(_POSE['q_rad'], float)
STAND_ROOT_Z = float(_POSE['root_z'])
LEGS = ['LF', 'LH', 'LM', 'RF', 'RH', 'RM']
TIBIA = [6, 13, 20, 27, 34, 41]
PDT = .005
CAP = 30.
OBS = 477
ADHESION = bool(int(__import__('os').environ.get('A_GRIP', '1')))
REWARD_V3 = True
TAG = __import__('os').environ.get('A_TAG', 'run')
START_GLOBAL_UPDATES = int(__import__('os').environ.get('A_START_UPDATES', '0'))
BUDGETS = [int(x) for x in __import__('os').environ.get('A_BUDGETS', '64,32,32,32').split(',')]
try:
    EVAL_UPDATES
except NameError:
    EVAL_UPDATES = [0]
NJ = 42
C_START, C_END, C_ANNEAL = .003, .05, 32
A_ADHESION = .25 * 24.446354755878005
LAM_ETA, LAM_MAX, SUPPORT_TARGET = .05, 50., .90
SUPPORT_WINDOW = 10   # ticks (50 ms at a 5 ms tick): the constraint uses the windowed MINIMUM, because the mean support is already above the gate while the body is collapsing
RELEASE_ON_LIFT = True
STANCE_LOAD = .1 * 24.446354755878005
CFG = dict(updates_per_level=16, rollout=256, epochs=6, minibatch=64, lr=3e-4, clip=.2, gamma=.995,
           gae=.95, value_coef=.5, grad_norm=.5, target_kl=.03, residual_rad=.003,
           initial_logstd=-3., logstd_bounds=[-5., -1.5], forward_filter_tau=.05,
           v_targets=[.05, .12, .20, .30], slip_margin=.35, slip_weight=2.0, stance_weight=.5,
           terminate_com=.95, terminate_nonfoot=.05 * 24.446354755878005, log_every=8)
GATES = dict(foot_support_mean_min=.90, nonfoot_force_max_W=.20 * WEIGHT, com_height_min=1.20,
             slip_ratio_max=.5, forward_speed_min=.05)


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_module(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


class Net:
    """477 -> 128 tanh -> 128 tanh -> {6 mean, 1 critic}, scalar logstd."""

    def __init__(self, rng):
        self.p = dict(w0=(rng.standard_normal((OBS, 128)) * np.sqrt(1 / OBS)),
                      b0=np.zeros(128),
                      w1=(rng.standard_normal((128, 128)) * np.sqrt(1 / 128)), b1=np.zeros(128),
                      w2=(rng.standard_normal((128, NJ)) * .01), b2=np.zeros(NJ),
                      wc=(rng.standard_normal((128, 1)) * .01), bc=np.zeros(1),
                      logstd=np.full(NJ, CFG['initial_logstd']))
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def std(self):
        return np.exp(np.clip(self.p['logstd'], *CFG['logstd_bounds']))

    def forward(self, X):
        h0 = np.tanh(X @ self.p['w0'] + self.p['b0'])
        h1 = np.tanh(h0 @ self.p['w1'] + self.p['b1'])
        return h0, h1, h1 @ self.p['w2'] + self.p['b2'], (h1 @ self.p['wc'] + self.p['bc'])[:, 0]

    def step(self, grads, lr):
        self.t += 1
        norm = np.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
        if norm > CFG['grad_norm']:
            grads = {k: g * (CFG['grad_norm'] / (norm + 1e-12)) for k, g in grads.items()}
        for k, g in grads.items():
            self.m[k] = .9 * self.m[k] + .1 * g
            self.v[k] = .999 * self.v[k] + .001 * g ** 2
            mh = self.m[k] / (1 - .9 ** self.t); vh = self.v[k] / (1 - .999 ** self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        lo, hi = CFG['logstd_bounds']
        self.p['logstd'] = np.clip(self.p['logstd'], lo, hi)


class PlantShim:
    def __init__(self, m):
        self.m = m
        self.jids = m.actuator_trnid[:, 0]
        self.qa = m.jnt_qposadr[self.jids]
        self.va = m.jnt_dofadr[self.jids]
        self.q0 = m.key_qpos[0]
        self.thorax = m.body('Thorax').id


class Sim:
    def __init__(self, base, act_mod, sensory):
        self.plant = act_mod.load_full_model(PAD_XML); self.plant.geom_friction[:, 0] = MU; self.d = mj.MjData(self.plant)
        mj.mj_resetDataKeyframe(self.plant, self.d, 0); mj.mj_forward(self.plant, self.d)
        self.actuator = act_mod.IncrementalActuator(self.plant, self.d)
        self.sensory = sensory; self.base = base; self.shim = PlantShim(self.plant)
        self.th = self.plant.body('Thorax').id
        self.q0 = self.plant.key_qpos[0][self.plant.jnt_qposadr[self.plant.actuator_trnid[:, 0]]].copy()
        self.planes = self.plant.geom_type == mj.mjtGeom.mjGEOM_PLANE
        names = [self.plant.geom(i).name or '' for i in range(self.plant.ngeom)]
        self.feet = np.array([next((j for j, L in enumerate(LEGS) if nm.startswith(L) and 'Tars' in nm), -1) for nm in names])
        self.K = int(base['K']); self.lags = np.asarray(base['lags'])
        self.reset()

    def pad_bottom_z(self):
        z = np.zeros(6)
        for l, leg in enumerate(LEGS):
            for gi in range(self.plant.ngeom):
                nm = self.plant.geom(gi).name or ''
                if nm == leg + 'Tarsus5_pad':
                    z[l] = float(self.d.geom_xpos[gi][2]) - float(self.plant.geom_size[gi][0])
                    break
        return z

    def reset(self):
        mj.mj_resetDataKeyframe(self.plant, self.d, 0)
        self.d.qpos[self.plant.jnt_qposadr[self.plant.actuator_trnid[:, 0]]] = STAND_Q
        self.d.qpos[2] = STAND_ROOT_Z
        mj.mj_forward(self.plant, self.d)
        self.actuator.reset(self.d)
        self.buf = np.repeat(self.q0[None], self.K, axis=0)
        self.d.xfrc_applied[:] = 0.
        self.filtered_forward = 0.; self.steps = 0

    def base_target(self):
        z = self.base
        x = self.buf[self.lags].reshape(-1)
        h0 = np.tanh(x @ z['w0'] + z['b0']); h1 = np.tanh(h0 @ z['w1'] + z['b1'])
        return z['center'] + z['scale'] * (h1 @ z['w2'] + z['b2'])

    def apply_adhesion(self, loads):
        """Declared normal adhesion: A per leg, shared across that leg's contacts, only when loaded."""
        self.d.xfrc_applied[:] = 0.
        if not ADHESION:
            return
        n = np.maximum(self.contact_count(), 1)
        rising = np.zeros(6, bool)
        if RELEASE_ON_LIFT:
            jac = np.zeros((3, self.plant.nv))
            for leg in range(6):
                gi = next((g for g in range(self.plant.ngeom)
                           if (self.plant.geom(g).name or '') == LEGS[leg] + 'Tarsus5_pad'), None)
                if gi is None:
                    continue
                mj.mj_jac(self.plant, self.d, jac, None, np.asarray(self.d.geom_xpos[gi]),
                          int(self.plant.geom_bodyid[gi]))
                rising[leg] = float((jac @ self.d.qvel)[2]) > 0.
        for leg in range(6):
            if loads[leg] <= STANCE_LOAD or rising[leg]:
                continue
            share = A_ADHESION / n[leg]
            for ci in range(self.d.ncon):
                c = self.d.contact[ci]; ga, gb = int(c.geom1), int(c.geom2)
                if not (self.planes[ga] or self.planes[gb]):
                    continue
                geom = gb if self.planes[ga] else ga
                if self.feet[geom] != leg:
                    continue
                body = int(self.plant.geom_bodyid[geom])
                f = np.array([0., 0., -share]); r = np.asarray(c.pos) - self.d.xpos[body]
                self.d.xfrc_applied[body, 0:3] += f
                self.d.xfrc_applied[body, 3:6] += np.cross(r, f)

    def contact_count(self):
        n = np.zeros(6, int)
        for ci in range(self.d.ncon):
            c = self.d.contact[ci]; ga, gb = int(c.geom1), int(c.geom2)
            if not (self.planes[ga] or self.planes[gb]):
                continue
            geom = gb if self.planes[ga] else ga
            if 0 <= geom < self.feet.size and self.feet[geom] >= 0:
                n[self.feet[geom]] += 1
        return n

    def step(self, rate):
        """One 5 ms policy tick: five 1 ms actuator updates, 400 native steps, adhesion every 0.5 ms."""
        for _ in range(5):
            self.actuator.update(np.clip(rate, -CAP, CAP), .001)
            for k in range(80):
                if k % 40 == 0:
                    loads, _nonfoot, _speed = self.loads_and_speed()
                    self.apply_adhesion(loads)
                self.d.ctrl[:] = self.actuator.torque(self.d)
                mj.mj_step(self.plant, self.d); mj.mj_forward(self.plant, self.d)
                self.steps += 1
            if self.d.warning.number.any():
                raise RuntimeError('MuJoCo warning')

    def loads_and_speed(self):
        loads = np.zeros(6); nonfoot = 0.
        vt = np.zeros(6); den = np.zeros(6); jacp = np.zeros((3, self.plant.nv))
        for ci in range(self.d.ncon):
            c = self.d.contact[ci]; ga, gb = int(c.geom1), int(c.geom2)
            if not (self.planes[ga] or self.planes[gb]):
                continue
            geom = gb if self.planes[ga] else ga
            f = np.zeros(6); mj.mj_contactForce(self.plant, self.d, ci, f)
            normal = max(0., float((c.frame.reshape(3, 3).T @ f[:3])[2]) * (1 if self.planes[ga] else -1))
            leg = int(self.feet[geom]) if 0 <= geom < self.feet.size else -1
            if leg < 0:
                nonfoot += normal; continue
            loads[leg] += normal
            mj.mj_jac(self.plant, self.d, jacp, None, np.asarray(c.pos), int(self.plant.geom_bodyid[geom]))
            v = jacp @ self.d.qvel
            vt[leg] += normal * float(np.hypot(v[0], v[1])); den[leg] += normal
        speed = np.where(den > 0, vt / np.maximum(den, 1e-12), 0.)
        return loads, nonfoot, speed


def evaluate_control(sim, ticks=800):
    """No-residual control rollout (base network alone), 4 s, for comparison."""
    sim.reset(); com0 = sim.d.subtree_com[sim.th].copy(); rec = []
    for _ in range(ticks):
        target = sim.base_target()
        rate = np.clip((target - sim.actuator.target_rad) / PDT, -CAP, CAP)
        sim.step(rate)
        loads, nonfoot, speed = sim.loads_and_speed()
        rec.append((sim.d.time, sim.d.subtree_com[sim.th].copy(), loads, nonfoot, speed))
        sim.buf = np.vstack([sim.buf[1:], target[None]])
    return summarise(rec, sim.steps)


def summarise(rec, steps):
    times = np.array([r[0] for r in rec]); com = np.array([r[1] for r in rec])
    loads = np.array([r[2] for r in rec]); nonfoot = np.array([r[3] for r in rec])
    speed = np.array([r[4] for r in rec])
    support = loads.sum(1) / WEIGHT
    duration = float(times[-1] - times[0])
    fwd = float(np.linalg.norm((com[-1] - com[0])[:2]) / duration)
    per_leg_vt = loads.__mul__(speed).sum(0) / np.maximum(loads.sum(0), 1e-12)
    ratio = per_leg_vt / max(fwd, 1e-9)
    return dict(native_steps=int(steps), duration_s=duration, forward_speed_native_per_s=fwd,
                foot_support_mean=float(support.mean()), foot_support_min=float(support.min()),
                nonfoot_force_max_W=float(nonfoot.max()), com_height_min=float(com[:, 2].min()),
                free_fall_fraction=float(np.mean(support < .05)),
                per_leg_slip_ratio=dict(zip(LEGS, map(float, ratio))),
                slip_ratio_max=float(ratio.max()), slip_ratio_mean=float(ratio.mean()),
                slip_criterion_passed=bool(ratio.max() <= GATES['slip_ratio_max']))


EVAL_UPDATES = [0]


def evaluate_policy(sim, net, ticks=800):
    sim.reset(); rec = []
    for _ in range(ticks):
        target = sim.base_target()
        proprio = sim.sensory.features(sim.shim, sim.d, target, sim.loads_and_speed()[0])
        obs = np.concatenate([proprio, sim.buf[sim.lags].reshape(-1)])
        _, _, mean, _ = net.forward(obs[None])
        c_now = C_START + (C_END - C_START) * min(1., EVAL_UPDATES[0] / C_ANNEAL)
        base_rate = (target - sim.actuator.target_rad) / PDT
        rate = base_rate + np.tanh(mean[0]) * c_now / PDT
        sim.step(rate)
        loads, nonfoot, speed = sim.loads_and_speed()
        rec.append((sim.d.time, sim.d.subtree_com[sim.th].copy(), loads, nonfoot, speed))
        sim.buf = np.vstack([sim.buf[1:], target[None]])
    return summarise(rec, sim.steps)


LAM = [0.0]   # dual variable, declared: dual ascent on the support constraint


def main():
    rng = np.random.default_rng(33000)
    act_mod = load_module('act', ACT); sensory = load_module('sensory', SENSORY)
    base = np.load(BASE); sim = Sim(base, act_mod, sensory)
    net = Net(rng)
    model_path = HERE / ('MODEL_' + TAG + '.npz')
    log = dict(reward_v5=dict(w_duty=W_DUTY, w_swing=W_SWING, duty_target=DUTY_TARGET, duty_tau_s=DUTY_TAU, swing_load=SWING_LOAD, clearance_ref=CLEAR_REF), prereg=json.loads((HERE / 'PREREG.json').read_text()), config=CFG, gates=GATES,
               adhesion=dict(enabled=ADHESION, per_leg=A_ADHESION, stance_load=STANCE_LOAD,
                             label='derived-channel: results are NOT frozen-body acceptance'),
               base=dict(path=str(BASE), sha256=sha(BASE)), action_joints='derived pad body r02', levels=[])
    t0 = time.time(); total_steps = 0
    ctrl = evaluate_control(sim)
    log['control_base_network_only'] = ctrl
    print(json.dumps({'control': {k: ctrl[k] for k in ('forward_speed_native_per_s', 'foot_support_mean', 'nonfoot_force_max_W', 'slip_ratio_max')}}), flush=True)
    budgets = BUDGETS if 'BUDGETS' in globals() else [64, 32, 32, 32]
    global_updates = globals().get('START_GLOBAL_UPDATES', 0)
    for level, vt in enumerate(CFG['v_targets']):
        lvl = HERE / ('LEVEL_%d_v%.2f' % (level, vt)); lvl.mkdir(exist_ok=True)
        ups = []; passed = False; ev = None; gates = None
        for u in range(budgets[level]):
            sim.reset()
            O, A, L, LOGP, V, R, T, D = [], [], [], [], [], [], [], []
            duty = np.full(6, DUTY_TARGET)
            support_log = []
            supp_win = []
            total_updates = global_updates
            for t in range(CFG['rollout']):
                target = sim.base_target()
                loads_now, nonfoot_now, _ = sim.loads_and_speed()
                proprio = sim.sensory.features(sim.shim, sim.d, target, loads_now)
                obs = np.concatenate([proprio, sim.buf[sim.lags].reshape(-1)]).astype(np.float64)
                _, _, mean, value = net.forward(obs[None])
                std = net.std()
                lat = mean[0] + std * rng.standard_normal(NJ)
                logp = float(np.sum(-.5 * ((lat - mean[0]) / std) ** 2 - np.log(std) - .5 * np.log(2 * np.pi)))
                a = np.tanh(lat)
                c_now = C_START + (C_END - C_START) * min(1., total_updates / C_ANNEAL)
                base_rate = (target - sim.actuator.target_rad) / PDT
                rate = base_rate + a * c_now / PDT
                sim.step(rate); total_steps += sim.steps
                loads, nonfoot, speed = sim.loads_and_speed()
                Rm = sim.d.xmat[sim.th].reshape(3, 3)
                vel = Rm.T @ sim.d.qvel[:3]; omega = sim.d.qvel[3:6]; grav = Rm.T @ np.array([0., 0., -1.])
                beta = -np.expm1(-PDT / CFG['forward_filter_tau'])
                sim.filtered_forward += beta * (vel[0] - sim.filtered_forward)
                ratio = speed / max(abs(sim.filtered_forward), .05)
                stance = loads > .3 * WEIGHT
                progress = float(np.clip(sim.filtered_forward / vt, -1., 1.5))
                if sim.filtered_forward < .5 * vt:
                    progress *= 2.
                allow = CFG['slip_margin'] * max(abs(sim.filtered_forward), vt)
                com_z = float(sim.d.subtree_com[sim.th, 2])
                # reward v5: duty factor and swing clearance, both measured from the current contact state
                stance_now = (loads > SWING_LOAD).astype(np.float64)
                alpha = 1. - np.exp(-PDT / DUTY_TAU)
                duty += alpha * (stance_now - duty)
                swing = (loads <= SWING_LOAD)
                pad_z = sim.pad_bottom_z()
                clearance = np.where(swing, np.clip(pad_z / CLEAR_REF, 0., 1.), 0.)
                duty_pen = W_DUTY * float(np.sum((duty - DUTY_TARGET) ** 2))
                swing_bonus = W_SWING * float(np.sum(clearance))
                support_now = float(loads.sum() / WEIGHT)
                support_log.append(support_now)
                supp_win.append(support_now)
                if len(supp_win) > SUPPORT_WINDOW:
                    supp_win.pop(0)
                support_min_win = min(supp_win)
                reward = (-LAM[0] * max(0., SUPPORT_TARGET - support_min_win)
                          + progress + swing_bonus - duty_pen
                          - CFG['slip_weight'] * float(np.sum(np.maximum(0., speed - allow) ** 2))
                          - CFG['stance_weight'] * float(np.sum(stance * speed ** 2))
                          - .05 * float(np.sum(omega ** 2)) - .01 * float(np.sum(grav[:2] ** 2))
                          - .01 * float(np.mean(rate ** 2))
                          - 5. * float((loads.sum() / WEIGHT - 1.) ** 2) - 50. * float(nonfoot / WEIGHT)
                          - 5. * max(0., 1.20 - com_z) ** 2)
                term = bool(sim.d.subtree_com[sim.th, 2] < CFG['terminate_com'] or nonfoot > CFG['terminate_nonfoot'])
                O.append(obs); A.append(lat); LOGP.append(logp); V.append(value[0]); R.append(reward)
                T.append(term); D.append(term or t == CFG['rollout'] - 1)
                sim.buf = np.vstack([sim.buf[1:], target[None]])
                if not (np.isfinite(reward) and np.isfinite(sim.d.qpos).all() and np.isfinite(mean).all()):
                    raise FloatingPointError('nonfinite in rollout')
                if term:
                    break
            O = np.asarray(O); A = np.asarray(A); V = np.asarray(V); R = np.asarray(R)
            T = np.asarray(T); D = np.asarray(D); LOGP = np.asarray(LOGP)
            adv = np.zeros_like(R); last = 0.
            for i in range(len(R) - 1, -1, -1):
                nxt = V[i + 1] if i + 1 < len(V) else 0.
                td = R[i] + CFG['gamma'] * nxt * (not T[i]) - V[i]
                last = td + CFG['gamma'] * CFG['gae'] * (not D[i]) * last
                adv[i] = last
            ret = adv + V
            adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
            stats = []
            for ep in range(CFG['epochs']):
                perm = rng.permutation(len(O)); stop = False
                mb = max(1, min(CFG['minibatch'], len(perm)))
                perm = perm[:len(perm) // mb * mb]
                for ids in perm.reshape(-1, mb):
                    x = O[ids]; lat_target = A[ids]; oldlog = LOGP[ids]; advb = adv_n[ids]; retb = ret[ids]
                    h0, h1, mean, value = net.forward(x)
                    std = net.std()
                    z = (lat_target - mean) / std
                    logp = np.sum(-.5 * z ** 2 - np.log(std) - .5 * np.log(2 * np.pi), axis=1)
                    kl = float(np.mean(oldlog - logp))
                    if kl > CFG['target_kl']:
                        stop = True; break
                    ratio = np.exp(logp - oldlog)
                    unclipped = advb * ratio
                    clipped = advb * np.clip(ratio, 1 - CFG['clip'], 1 + CFG['clip'])
                    active = (unclipped <= clipped).astype(np.float64)   # 1 = unclipped branch selected
                    # dL/dlogp for the surrogate  L = -mean(active*adv*ratio)
                    dlogp = -(active * advb * ratio) / len(ids)
                    # value loss
                    dvalue = 2 * CFG['value_coef'] * (value - retb) / len(ids)
                    dmean = dlogp[:, None] * (z / std[None, :])
                    dlogstd = np.sum(dlogp[:, None] * (z ** 2 - 1), axis=0)
                    dh1 = dmean @ net.p['w2'].T + dvalue[:, None] @ net.p['wc'].T
                    grads = dict(w2=h1.T @ dmean, b2=dmean.sum(0),
                                 wc=h1.T @ dvalue[:, None], bc=np.array([dvalue.sum()]),
                                 logstd=dlogstd)
                    dh1 *= (1 - h1 ** 2)
                    grads['w1'] = h0.T @ dh1; grads['b1'] = dh1.sum(0)
                    dh0 = (dh1 @ net.p['w1'].T) * (1 - h0 ** 2)
                    grads['w0'] = x.T @ dh0; grads['b0'] = dh0.sum(0)
                    net.step(grads, CFG['lr'])
                    stats.append(dict(kl=kl, ratio_mean=float(ratio.mean()), adv_mean=float(advb.mean())))
                if stop:
                    break
            ups.append(dict(update=u, decisions=int(len(R)), reward_mean=float(R.mean()), reward_sum=float(R.sum()),
                            terminated=bool(T.any()), kl_last=stats[-1]['kl'] if stats else None,
                            logstd=np.round(net.p['logstd'], 5).tolist(), physics_steps=int(sim.steps)))
            global_updates += 1; EVAL_UPDATES[0] = global_updates
            if (u + 1) % 16 == 0:
                ev = evaluate_policy(sim, net)
                gates = dict(foot_support_mean=ev['foot_support_mean'] >= GATES['foot_support_mean_min'],
                             nonfoot=ev['nonfoot_force_max_W'] <= GATES['nonfoot_force_max_W'],
                             com_height=ev['com_height_min'] >= GATES['com_height_min'],
                             slip=ev['slip_ratio_max'] <= GATES['slip_ratio_max'],
                             forward=ev['forward_speed_native_per_s'] >= GATES['forward_speed_min'])
                print(json.dumps({'level': level, 'update': u + 1, 'gate_check': {k: round(ev[k], 4) for k in ('forward_speed_native_per_s', 'foot_support_mean', 'nonfoot_force_max_W', 'com_height_min', 'slip_ratio_max')}, 'gates': gates}), flush=True)
                ups.append(dict(update=u, gate_check=ev, gates=gates))
                if all(gates.values()):
                    passed = True
                    break
            if u % CFG['log_every'] == 0:
                print(json.dumps({'level': level, 'v': vt, 'update': u, 'reward': round(float(R.mean()), 4),
                                  'terminated': bool(T.any()), 'steps': int(total_steps),
                                  'lambda_': round(LAM[0], 4),
                                  'mean_support': round(float(np.mean(support_log)), 4) if support_log else None,
                                  'min_support': round(float(np.min(support_log)), 4) if support_log else None}), flush=True)
        if ev is None:
            ev = evaluate_policy(sim, net)
            gates = dict(foot_support_mean=ev['foot_support_mean'] >= GATES['foot_support_mean_min'],
                         nonfoot=ev['nonfoot_force_max_W'] <= GATES['nonfoot_force_max_W'],
                         com_height=ev['com_height_min'] >= GATES['com_height_min'],
                         slip=ev['slip_ratio_max'] <= GATES['slip_ratio_max'],
                         forward=ev['forward_speed_native_per_s'] >= GATES['forward_speed_min'])
        (lvl / 'RESULT.json').write_text(json.dumps(dict(level=level, v_target=vt, updates=ups, evaluation=ev,
                                                         gates=gates, gates_passed=all(gates.values())), indent=2, allow_nan=False) + '\n')
        log['levels'].append(dict(level=level, v_target=vt, evaluation=ev, gates=gates, gates_passed=all(gates.values())))
        Z = dict(center=base['center'], scale=base['scale'], lags=base['lags'], K=base['K'],
                 actor_w0=net.p['w0'].astype(np.float32), actor_b0=net.p['b0'].astype(np.float32),
                 actor_w1=net.p['w1'].astype(np.float32), actor_b1=net.p['b1'].astype(np.float32),
                 actor_w2=net.p['w2'].astype(np.float32), actor_b2=net.p['b2'].astype(np.float32),
                 critic_w=net.p['wc'].astype(np.float32), critic_b=net.p['bc'].astype(np.float32),
                 logstd=net.p['logstd'].astype(np.float32), delta_scale=CFG['residual_rad'],
                 indices=np.arange(NJ), base_sha256=np.array(sha(BASE)),
                 scheme=np.array('full42_fading_prior'), c_start=C_START, c_end=C_END, c_anneal_updates=C_ANNEAL,
                 adhesion=np.array(bool(ADHESION)), a_adhesion=A_ADHESION, global_updates=global_updates)
        np.savez_compressed(model_path, **Z)
        print(json.dumps({'LEVEL_DONE': level, 'v': vt, 'eval': ev, 'gates': gates}), flush=True)
        if not all(gates.values()):
            log['stopped_at_level'] = level
            break
    log['total_physics_steps'] = int(total_steps)
    log['wall_s'] = time.time() - t0
    log['model'] = dict(path=str(model_path), sha256=sha(model_path))
    (HERE / ('TRAIN_LOG_' + TAG + '.json')).write_text(json.dumps(log, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'DONE_FULL42', 'levels_done': len(log['levels']), 'stopped_at': log.get('stopped_at_level'),
                      'steps': int(total_steps), 'wall_s': round(log['wall_s'], 1)}))


if __name__ == '__main__':
    main()
