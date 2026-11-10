"""Frozen-body and derived-channel evaluation of the full 42-joint PPO controller.

grip OFF = frozen body: this is the honest acceptance number.
grip ON  = declared adhesion channel: derived-channel only.
"""
from pathlib import Path
import argparse, json, sys
import numpy as np
import mujoco as mj
HERE = Path(__file__).resolve().parent
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
A_DIR = ROOT / 'experiments/banc-locomotion-dataset/20261108T011500Z'
WEIGHT = 24.446354755878005
LEGS = ['LF', 'LH', 'LM', 'RF', 'RH', 'RM']
PDT = .005; CAP = 30.; C = .05
A_ADH = .5 * WEIGHT; STANCE = .1 * WEIGHT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=str(HERE / 'MODEL_full42_after128.npz'))
    ap.add_argument('--grip', type=int, default=0)
    ap.add_argument('--assist', type=int, default=200)
    ap.add_argument('--release', type=int, default=600)
    args = ap.parse_args()
    import importlib.util
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, str(path))
        m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
    act = load('act', ROOT / 'experiments/s5-ann-incremental/20261107T010000Z/incremental_actuator.py')
    sensory = load('sensory', ROOT / 'experiments/s5-ann-walk/20261107T091243Z/proprioception.py')
    src = load('tf', HERE / 'train_full42.py')
    plant = act.load_full_model(); d = mj.MjData(plant)
    mj.mj_resetDataKeyframe(plant, d, 0); mj.mj_forward(plant, d)
    actuator = act.IncrementalActuator(plant, d)
    th = plant.body('Thorax').id
    q0 = plant.key_qpos[0][plant.jnt_qposadr[plant.actuator_trnid[:, 0]]].copy()
    planes = plant.geom_type == mj.mjtGeom.mjGEOM_PLANE
    names = [plant.geom(i).name or '' for i in range(plant.ngeom)]
    feet = np.array([next((j for j, L in enumerate(LEGS) if nm.startswith(L) and 'Tars' in nm), -1) for nm in names])
    class Shim:
        def __init__(self, m):
            self.m = m; self.jids = m.actuator_trnid[:, 0]; self.qa = m.jnt_qposadr[self.jids]
            self.va = m.jnt_dofadr[self.jids]; self.q0 = m.key_qpos[0]; self.thorax = m.body('Thorax').id
    shim = Shim(plant)
    base = np.load(A_DIR / 'MODEL_MEASURED.npz')
    K = int(base['K']); lags = np.asarray(base['lags'])
    z = np.load(args.model)
    P = dict(w0=z['actor_w0'].astype(np.float64), b0=z['actor_b0'].astype(np.float64),
             w1=z['actor_w1'].astype(np.float64), b1=z['actor_b1'].astype(np.float64),
             w2=z['actor_w2'].astype(np.float64), b2=z['actor_b2'].astype(np.float64))
    ts = np.load(A_DIR / 'TRAINING_SET.npz')
    seg = ts['targets_rad'][int(ts['offsets'][0]):int(ts['offsets'][1])][:args.assist]
    mj.mj_resetDataKeyframe(plant, d, 0); mj.mj_forward(plant, d); actuator.reset(d)
    buf = seg[:K].copy() if seg.shape[0] >= K else np.repeat(q0[None], K, axis=0)
    jacp = np.zeros((3, plant.nv))

    def measure():
        loads = np.zeros(6); nonfoot = 0.; vt = np.zeros(6); den = np.zeros(6); conts = []
        for ci in range(d.ncon):
            c = d.contact[ci]; ga, gb = int(c.geom1), int(c.geom2)
            if not (planes[ga] or planes[gb]):
                continue
            geom = gb if planes[ga] else ga
            f = np.zeros(6); mj.mj_contactForce(plant, d, ci, f)
            normal = max(0., float((c.frame.reshape(3, 3).T @ f[:3])[2]) * (1 if planes[ga] else -1))
            leg = int(feet[geom]) if 0 <= geom < feet.size else -1
            if leg < 0:
                nonfoot += normal; continue
            loads[leg] += normal
            body = int(plant.geom_bodyid[geom])
            mj.mj_jac(plant, d, jacp, None, np.asarray(c.pos), body)
            v = jacp @ d.qvel
            vt[leg] += normal * float(np.hypot(v[0], v[1])); den[leg] += normal
            conts.append((leg, body, np.asarray(c.pos).copy()))
        speed = np.where(den > 0, vt / np.maximum(den, 1e-12), 0.)
        return loads, nonfoot, speed, conts

    def target_net():
        x = buf[lags].reshape(-1)
        h0 = np.tanh(x @ base['w0'] + base['b0']); h1 = np.tanh(h0 @ base['w1'] + base['b1'])
        return base['center'] + base['scale'] * (h1 @ base['w2'] + base['b2'])

    rec = []; phase = []
    for t in range(args.assist + args.release):
        tgt = target_net()
        loads, nonfoot, speed, conts = measure()
        proprio = sensory.features(shim, d, tgt, loads)
        obs = np.concatenate([proprio, buf[lags].reshape(-1)])
        h0 = np.tanh(obs @ P['w0'] + P['b0']); h1 = np.tanh(h0 @ P['w1'] + P['b1'])
        pol = np.tanh(h1 @ P['w2'] + P['b2'])
        base_rate = (tgt - actuator.target_rad) / PDT
        if t < seg.shape[0]:
            rate = np.clip((seg[t] - actuator.target_rad) / PDT, -CAP, CAP); phase.append('assist')
        else:
            rate = base_rate + pol * C / PDT; phase.append('release')
        for _ in range(5):
            actuator.update(np.clip(rate, -CAP, CAP), .001)
            for k in range(80):
                if args.grip and k % 40 == 0:
                    d.xfrc_applied[:] = 0.
                    l, _, _, cs = measure()
                    if cs:
                        cnt = np.zeros(6, int)
                        for leg, body, pos in cs:
                            cnt[leg] += 1
                        for leg, body, pos in cs:
                            if l[leg] <= STANCE:
                                continue
                            f = np.array([0., 0., -A_ADH / max(cnt[leg], 1)])
                            d.xfrc_applied[body, 0:3] += f
                            d.xfrc_applied[body, 3:6] += np.cross(pos - d.xpos[body], f)
                d.ctrl[:] = actuator.torque(d)
                mj.mj_step(plant, d); mj.mj_forward(plant, d)
        loads, nonfoot, speed, conts = measure()
        rec.append((d.time, d.subtree_com[th].copy(), loads, nonfoot, speed))
        buf = np.vstack([buf[1:], tgt[None]])
    phase = np.array(phase)
    out = dict(model=str(args.model), model_sha256=src.sha(args.model), grip=bool(args.grip),
               labels=dict(grip='derived-channel (NOT frozen-body acceptance)' if args.grip else 'frozen-body'),
               amplitude_C=C, native_steps=int(plant.opt.timestep and (args.assist + args.release) * 400))
    for tag, mask in (('assist', phase == 'assist'), ('release', phase == 'release')):
        idx = np.flatnonzero(mask)
        times = np.array([rec[i][0] for i in idx]); com = np.array([rec[i][1] for i in idx])
        loads = np.array([rec[i][2] for i in idx]); nonfoot = np.array([rec[i][3] for i in idx])
        speed = np.array([rec[i][4] for i in idx])
        support = loads.sum(1) / WEIGHT
        dur = float(times[-1] - times[0]); fwd = float(np.linalg.norm((com[-1] - com[0])[:2]) / dur)
        per_leg = loads.__mul__(speed).sum(0) / np.maximum(loads.sum(0), 1e-12)
        ratio = per_leg / max(fwd, 1e-9)
        out[tag] = dict(duration_s=dur, forward_speed_native_per_s=fwd, foot_support_mean=float(support.mean()),
                        foot_support_min=float(support.min()), nonfoot_force_max_W=float(nonfoot.max()),
                        com_height_min=float(com[:, 2].min()), free_fall_fraction=float(np.mean(support < .05)),
                        per_leg_slip_ratio=dict(zip(LEGS, map(float, ratio))),
                        slip_ratio_max=float(ratio.max()), slip_ratio_mean=float(ratio.mean()))
    r = out['release']
    out['release_gates'] = dict(foot_support_mean=r['foot_support_mean'] >= .9,
                                nonfoot=r['nonfoot_force_max_W'] <= .2 * WEIGHT,
                                com_height=r['com_height_min'] >= 1.20,
                                slip=r['slip_ratio_max'] <= .5,
                                forward=r['forward_speed_native_per_s'] >= .05)
    out['release_gates_passed'] = all(out['release_gates'].values())
    name = ('EVAL_gripON' if args.grip else 'EVAL_gripOFF') + '.json'
    (HERE / name).write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'grip': bool(args.grip), 'release': {k: round(out['release'][k], 4) for k in
          ('forward_speed_native_per_s', 'foot_support_mean', 'nonfoot_force_max_W', 'com_height_min', 'slip_ratio_max', 'slip_ratio_mean')},
          'gates': out['release_gates']}))


if __name__ == '__main__':
    main()
