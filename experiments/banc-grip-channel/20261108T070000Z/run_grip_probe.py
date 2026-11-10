"""B3: declared grip channel + evaluation of the existing controllers with grip OFF and ON.

The body XML is untouched. Grip is an explicit external-force channel with a cap, a stance
condition and full logging, applied every 0.5 ms. Every number produced with grip ON is labelled
'derived-channel' and is never reported as frozen-body acceptance.
"""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import mujoco as mj

HERE = Path(__file__).resolve().parent
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
A_DIR = ROOT / 'experiments/banc-locomotion-dataset/20261108T011500Z'
V2_DIR = ROOT / 'experiments/banc-residual-v2/20261108T033000Z'
WEIGHT = 24.446354755878005
LEGS = ['LF', 'LH', 'LM', 'RF', 'RH', 'RM']
TIBIA = [6, 13, 20, 27, 34, 41]
PDT = .005; CAP = 30.
K_GRIP = 5000.
LAW = 'tangential'   # 'tangential' (B3) or 'normal' (B4); set by --law
F_MAX_PER_LEG = .5 * WEIGHT
STANCE_LOAD = .1 * WEIGHT


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Scenario:
    """Holds the plant, the actuator, the controllers and the optional grip channel."""

    def __init__(self, grip: bool, model_path=None):
        sys.path.insert(0, str(ROOT / 'experiments/s5-ann-incremental/20261107T010000Z'))
        import importlib.util
        spec = importlib.util.spec_from_file_location('act', ROOT / 'experiments/s5-ann-incremental/20261107T010000Z/incremental_actuator.py')
        act = importlib.util.module_from_spec(spec); sys.modules['act'] = act; spec.loader.exec_module(act)
        self.m = act.load_full_model() if model_path is None else mj.MjModel.from_xml_path(str(model_path))
        self.d = mj.MjData(self.m); mj.mj_resetDataKeyframe(self.m, self.d, 0); mj.mj_forward(self.m, self.d)
        self.actuator = act.IncrementalActuator(self.m, self.d)
        self.th = self.m.body('Thorax').id
        self.q0 = self.m.key_qpos[0][self.m.jnt_qposadr[self.m.actuator_trnid[:, 0]]].copy()
        self.planes = self.m.geom_type == mj.mjtGeom.mjGEOM_PLANE
        names = [self.m.geom(i).name or '' for i in range(self.m.ngeom)]
        self.feet = np.array([next((j for j, L in enumerate(LEGS) if nm.startswith(L) and 'Tars' in nm), -1) for nm in names])
        self.tarsus_body = np.array([next((self.m.body(LEG + 'Tarsus5').id for LEG in LEGS if nm.startswith(LEG) and 'Tars' in nm), -1) for nm in names])
        self.grip = grip
        self.base = np.load(A_DIR / 'MODEL_MEASURED.npz')
        self.K = int(self.base['K']); self.lags = np.asarray(self.base['lags'])
        self.grip_log = dict(peak_force=0., impulse=0., cap_hits=0, active_updates=0, updates=0)
        self.reset()

    def reset(self):
        mj.mj_resetDataKeyframe(self.m, self.d, 0); mj.mj_forward(self.m, self.d)
        self.d.xfrc_applied[:] = 0.
        self.actuator.reset(self.d)
        self.buf = np.repeat(self.q0[None], self.K, axis=0)
        self.steps = 0
        self.grip_log = dict(peak_force=0., impulse=0., cap_hits=0, active_updates=0, updates=0)

    def base_target(self):
        z = self.base
        x = self.buf[self.lags].reshape(-1)
        h0 = np.tanh(x @ z['w0'] + z['b0']); h1 = np.tanh(h0 @ z['w1'] + z['b1'])
        return z['center'] + z['scale'] * (h1 @ z['w2'] + z['b2'])

    def measure(self):
        loads = np.zeros(6); nonfoot = 0.
        vt = np.zeros(6); den = np.zeros(6)
        contacts = []
        jacp = np.zeros((3, self.m.nv))
        for ci in range(self.d.ncon):
            c = self.d.contact[ci]; ga, gb = int(c.geom1), int(c.geom2)
            if not (self.planes[ga] or self.planes[gb]):
                continue
            geom = gb if self.planes[ga] else ga
            f = np.zeros(6); mj.mj_contactForce(self.m, self.d, ci, f)
            normal = max(0., float((c.frame.reshape(3, 3).T @ f[:3])[2]) * (1 if self.planes[ga] else -1))
            leg = int(self.feet[geom]) if 0 <= geom < self.feet.size else -1
            if leg < 0:
                nonfoot += normal; continue
            loads[leg] += normal
            body = int(self.m.geom_bodyid[geom])
            mj.mj_jac(self.m, self.d, jacp, None, np.asarray(c.pos), body)
            v = jacp @ self.d.qvel
            vt[leg] += normal * float(np.hypot(v[0], v[1])); den[leg] += normal
            contacts.append((leg, body, np.asarray(c.pos).copy(), v.copy(), normal))
        speed = np.where(den > 0, vt / np.maximum(den, 1e-12), 0.)
        return loads, nonfoot, speed, contacts

    def apply_grip(self, contacts):
        self.d.xfrc_applied[:] = 0.
        if not self.grip:
            return
        self.grip_log['updates'] += 1
        per_leg = np.zeros(6); applied_any = False
        # declare-correct allocation: the cap is PER LEG, shared across that leg's contacts
        counts = np.zeros(6, int)
        for leg, body, pos, v, normal in contacts:
            if normal > STANCE_LOAD:
                counts[leg] += 1
        for leg, body, pos, v, normal in contacts:
            if normal <= STANCE_LOAD:
                continue
            share = F_MAX_PER_LEG / max(counts[leg], 1)
            if LAW == 'normal':
                f = np.array([0., 0., -share])
            else:
                f = -K_GRIP * np.array([v[0], v[1], 0.])
                mag = float(np.linalg.norm(f))
                if mag > share:
                    f *= share / mag; self.grip_log['cap_hits'] += 1
            per_leg[leg] += float(np.linalg.norm(f))
            r = pos - self.d.xpos[body]
            self.d.xfrc_applied[body, 0:3] += f
            self.d.xfrc_applied[body, 3:6] += np.cross(r, f)
            applied_any = True
        if applied_any:
            self.grip_log['active_updates'] += 1
            self.grip_log['peak_force'] = max(self.grip_log['peak_force'], float(per_leg.max()))
            if float(per_leg.max()) > F_MAX_PER_LEG + 1e-9:
                self.grip_log['cap_violations'] = self.grip_log.get('cap_violations', 0) + 1
            self.grip_log['impulse'] += float(per_leg.sum()) * .0005

    def step(self, rate, residual=None):
        """One 5 ms policy tick = 5 x (1 ms actuator update + 80 native steps)."""
        for _ in range(5):
            self.actuator.update(np.clip(rate, -CAP, CAP), .001)
            for k in range(80):
                if k % 40 == 0:                     # grip refreshes every 0.5 ms, as declared
                    loads, nonfoot, speed, contacts = self.measure()
                    self.apply_grip(contacts)
                self.d.ctrl[:] = self.actuator.torque(self.d)
                mj.mj_step(self.m, self.d); mj.mj_forward(self.m, self.d)
                self.steps += 1
            if self.d.warning.number.any():
                raise RuntimeError('MuJoCo warning')


def run(grip, mode, ticks=800, assist_ticks=0):
    sc = Scenario(grip=grip)
    seg = None
    if assist_ticks:
        ts = np.load(A_DIR / 'TRAINING_SET.npz')
        seg = ts['targets_rad'][int(ts['offsets'][0]):int(ts['offsets'][1])][:assist_ticks]
        if seg.shape[0] >= sc.K:
            sc.buf = seg[:sc.K].copy()
    z = np.load(V2_DIR / 'MODEL_RESIDUAL_V2.npz')
    P = dict(w0=z['actor_w0'].astype(np.float64), b0=z['actor_b0'].astype(np.float64),
             w1=z['actor_w1'].astype(np.float64), b1=z['actor_b1'].astype(np.float64),
             w2=z['actor_w2'].astype(np.float64), b2=z['actor_b2'].astype(np.float64))
    sys.path.insert(0, str(ROOT / 'experiments/s5-ann-walk/20261107T091243Z'))
    import importlib.util
    spec = importlib.util.spec_from_file_location('sensory', ROOT / 'experiments/s5-ann-walk/20261107T091243Z/proprioception.py')
    sensory = importlib.util.module_from_spec(spec); sys.modules['sensory'] = sensory; spec.loader.exec_module(sensory)

    class Shim:
        def __init__(self, m):
            self.m = m; self.jids = m.actuator_trnid[:, 0]; self.qa = m.jnt_qposadr[self.jids]
            self.va = m.jnt_dofadr[self.jids]; self.q0 = m.key_qpos[0]; self.thorax = m.body('Thorax').id
    shim = Shim(sc.m)
    rec = []; phase = []
    for t in range(ticks):
        target = sc.base_target()
        loads, nonfoot, speed, contacts = sc.measure()
        proprio = sensory.features(shim, sc.d, target, loads)
        obs = np.concatenate([proprio, sc.buf[sc.lags].reshape(-1)])
        h0 = np.tanh(obs @ P['w0'] + P['b0']); h1 = np.tanh(h0 @ P['w1'] + P['b1'])
        res = np.zeros(42); res[TIBIA] = np.tanh(h1 @ P['w2'] + P['b2']) * float(z['delta_scale']) / PDT
        if seg is not None and t < seg.shape[0]:
            rate = np.clip((seg[t] - sc.actuator.target_rad) / PDT, -CAP, CAP); phase.append('assist')
        else:
            rate = np.clip((target - sc.actuator.target_rad) / PDT, -CAP, CAP) + res; phase.append('release')
        sc.step(np.clip(rate, -CAP, CAP))
        loads, nonfoot, speed, contacts = sc.measure()
        rec.append((sc.d.time, sc.d.subtree_com[sc.th].copy(), loads, nonfoot, speed))
        sc.buf = np.vstack([sc.buf[1:], target[None]])
    phase = np.array(phase)
    out = dict(mode=mode, grip=grip, native_steps=int(sc.steps), labels=dict(grip='derived-channel' if grip else 'frozen-body'),
               grip_log=sc.grip_log, prereg=str(HERE / 'PREREG.json'))
    for tag, mask in (('assist', phase == 'assist'), ('release', phase == 'release'), ('all', np.ones(len(phase), bool))):
        if not mask.any():
            continue
        idx = np.flatnonzero(mask)
        com = np.array([rec[i][1] for i in idx]); loads = np.array([rec[i][2] for i in idx])
        nonfoot = np.array([rec[i][3] for i in idx]); speed = np.array([rec[i][4] for i in idx])
        times = np.array([rec[i][0] for i in idx])
        support = loads.sum(1) / WEIGHT
        dur = float(times[-1] - times[0]); fwd = float(np.linalg.norm((com[-1] - com[0])[:2]) / dur)
        per_leg = loads.__mul__(speed).sum(0) / np.maximum(loads.sum(0), 1e-12)
        ratio = per_leg / max(fwd, 1e-9)
        out[tag] = dict(duration_s=dur, forward_speed_native_per_s=fwd, foot_support_mean=float(support.mean()),
                        foot_support_min=float(support.min()), nonfoot_force_max_W=float(nonfoot.max()),
                        com_height_min=float(com[:, 2].min()), free_fall_fraction=float(np.mean(support < .05)),
                        per_leg_slip_ratio=dict(zip(LEGS, map(float, ratio))), slip_ratio_max=float(ratio.max()),
                        slip_ratio_mean=float(ratio.mean()), slip_criterion_passed=bool(ratio.max() <= .5))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['cold', 'assist'], default='cold')
    ap.add_argument('--law', choices=['tangential', 'normal'], default='tangential')
    args = ap.parse_args()
    global LAW
    LAW = args.law
    ticks = 800; assist = 200 if args.mode == 'assist' else 0
    results = {}
    for grip in (False, True):
        key = ('grip_ON' if grip else 'grip_OFF') + '_' + args.mode
        results[key] = run(grip, args.mode, ticks=ticks, assist_ticks=assist)
        r = results[key]
        tag = 'release' if args.mode == 'assist' else 'all'
        print(json.dumps({key: {k: (round(r[tag][k], 4) if isinstance(r[tag][k], float) else r[tag][k])
                                for k in ('forward_speed_native_per_s', 'foot_support_mean', 'nonfoot_force_max_W',
                                          'com_height_min', 'slip_ratio_max', 'slip_ratio_mean')}}), flush=True)
    summary = dict(schema='grip-probe/2', mode=args.mode, law=LAW, k_grip=K_GRIP, f_max_per_leg=F_MAX_PER_LEG,
                   stance_load=STANCE_LOAD, results=results, prereg_sha256=sha(HERE / 'PREREG.json'))
    out = HERE / ('RESULT_%s_%s.json' % (LAW, args.mode))
    out.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'saved': str(out)}))


if __name__ == '__main__':
    main()
