"""Full-frame close-up of one foot on the floor, with its tangential speed against the allowed speed."""
from pathlib import Path
import argparse, json, os, subprocess, sys, importlib.util
import numpy as np
os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco as mj
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
SRC = ROOT / 'experiments/banc-stand-retarget/20261108T190000Z'
WEIGHT = 24.446354755878005
LEGS = ['LF', 'LH', 'LM', 'RF', 'RH', 'RM']
PDT = .005; CAP = 30.; C = .05
A_ADH = .5 * WEIGHT; STANCE = .1 * WEIGHT


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grip', type=int, default=1)
    ap.add_argument('--leg', type=str, default='LM')
    ap.add_argument('--assist', type=int, default=200)
    ap.add_argument('--release', type=int, default=600)
    ap.add_argument('--every', type=int, default=4)
    args = ap.parse_args()
    act = load('act', ROOT / 'experiments/s5-ann-incremental/20261107T010000Z/incremental_actuator.py')
    sensory = load('sensory', ROOT / 'experiments/s5-ann-walk/20261107T091243Z/proprioception.py')
    tf = load('tf', SRC / 'train_full42_stand.py')
    plant = act.load_full_model(); d = mj.MjData(plant)
    mj.mj_resetDataKeyframe(plant, d, 0)
    d.qpos[plant.jnt_qposadr[plant.actuator_trnid[:, 0]]] = tf.STAND_Q
    d.qpos[2] = tf.STAND_ROOT_Z
    mj.mj_forward(plant, d)
    actuator = act.IncrementalActuator(plant, d); actuator.reset(d)
    th = plant.body('Thorax').id
    planes = plant.geom_type == mj.mjtGeom.mjGEOM_PLANE
    gname = [plant.geom(i).name or '' for i in range(plant.ngeom)]
    feet = np.array([next((j for j, L in enumerate(LEGS) if nm.startswith(L) and 'Tars' in nm), -1) for nm in gname])
    track = LEGS.index(args.leg)
    for g in range(plant.ngeom):
        if feet[g] == track:
            plant.geom_rgba[g] = [1.0, 0.85, 0.1, 1.0]      # tracked leg in yellow
        elif feet[g] >= 0:
            plant.geom_rgba[g] = [0.1, 0.8, 0.95, 1.0]
    class Shim:
        def __init__(self, m):
            self.m = m; self.jids = m.actuator_trnid[:, 0]; self.qa = m.jnt_qposadr[self.jids]
            self.va = m.jnt_dofadr[self.jids]; self.q0 = m.key_qpos[0]; self.thorax = m.body('Thorax').id
    shim = Shim(plant)
    base = np.load(SRC / 'MODEL_MEASURED_STAND.npz'); K = int(base['K']); lags = np.asarray(base['lags'])
    z = np.load(SRC / 'MODEL_stand.npz')
    P = dict(w0=z['actor_w0'].astype(np.float64), b0=z['actor_b0'].astype(np.float64),
             w1=z['actor_w1'].astype(np.float64), b1=z['actor_b1'].astype(np.float64),
             w2=z['actor_w2'].astype(np.float64), b2=z['actor_b2'].astype(np.float64))
    ts = np.load(SRC / 'TRAINING_SET_STAND.npz')
    seg = ts['targets_rad'][int(ts['offsets'][0]):int(ts['offsets'][1])][:args.assist]
    buf = seg[:K].copy() if seg.shape[0] >= K else np.repeat(tf.STAND_Q[None], K, axis=0)
    jacp = np.zeros((3, plant.nv)); jacr = np.zeros((3, plant.nv))
    renderer = mj.Renderer(plant, height=432, width=560)
    vopt = mj.MjvOption()
    for flag in ('mjVIS_TENDON', 'mjVIS_ACTUATOR', 'mjVIS_JOINT', 'mjVIS_COM', 'mjVIS_CONTACTPOINT'):
        vopt.flags[getattr(mj.mjtVisFlag, flag)] = False

    def measure():
        loads = np.zeros(6); nonfoot = 0.; vt = np.zeros(6); den = np.zeros(6); conts = []; tv = np.zeros(3)
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
            tv_here = np.array([v[0], v[1], 0.])
            vt[leg] += normal * float(np.hypot(v[0], v[1])); den[leg] += normal
            conts.append((leg, body, np.asarray(c.pos).copy()))
            if leg == track:
                tv = tv_here
        speed = np.where(den > 0, vt / np.maximum(den, 1e-12), 0.)
        return loads, nonfoot, speed, conts, tv

    def target_net():
        x = buf[lags].reshape(-1)
        h0 = np.tanh(x @ base['w0'] + base['b0']); h1 = np.tanh(h0 @ base['w1'] + base['b1'])
        return base['center'] + base['scale'] * (h1 @ base['w2'] + base['b2'])

    times, com, phase, track_speed, track_pos, allowed = [], [], [], [], [], []
    for t in range(args.assist + args.release):
        tgt = target_net()
        loads, nonfoot, speed_pre, conts, _ = measure()
        proprio = sensory.features(shim, d, tgt, loads)
        obs = np.concatenate([proprio, buf[lags].reshape(-1)])
        h0 = np.tanh(obs @ P['w0'] + P['b0']); h1 = np.tanh(h0 @ P['w1'] + P['b1'])
        pol = np.tanh(h1 @ P['w2'] + P['b2'])
        if t < seg.shape[0]:
            rate = np.clip((seg[t] - actuator.target_rad) / PDT, -CAP, CAP)
        else:
            rate = ((tgt - actuator.target_rad) / PDT) + pol * C / PDT
        for _ in range(5):
            actuator.update(np.clip(rate, -CAP, CAP), .001)
            for k in range(80):
                if args.grip and k % 40 == 0:
                    d.xfrc_applied[:] = 0.
                    l, _, _, cs, _ = measure()
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
        loads, nonfoot, speed, conts, tv = measure()
        times.append(float(d.time)); com.append(d.subtree_com[th].copy())
        phase.append('assist' if t < seg.shape[0] else 'release')
        track_speed.append(float(np.hypot(tv[0], tv[1])))
        tp = None
        for leg, body, pos in conts:
            if leg == track:
                tp = pos; break
        track_pos.append(tp if tp is not None else d.xpos[plant.body(args.leg + 'Tarsus5').id].copy())
        buf = np.vstack([buf[1:], tgt[None]])
    times = np.array(times); com = np.array(com); ts_ = np.array(track_speed)
    rel = np.array([i for i, p in enumerate(phase) if p == 'release'])
    dur = float(times[-1] - times[0]); fwd = float(np.linalg.norm((com[-1] - com[0])[:2]) / dur)
    frames = []
    for i in range(0, len(times), args.every):
        cam = mj.MjvCamera(); p = track_pos[i]
        cam.lookat[:] = [p[0], p[1], 0.02]; cam.distance = 0.34; cam.elevation = -7; cam.azimuth = 150
        renderer.update_scene(d, camera=cam, scene_option=vopt)
        frames.append((i, renderer.render().copy()))
    # NOTE: rendering uses the final state; re-render with stored states instead
    out = HERE / ('closeup_grip%d_%s' % (args.grip, args.leg)); out.mkdir(exist_ok=True)
    print('need state replay for per-frame cameras', file=sys.stderr)
    print(json.dumps(dict(frames_collected=len(frames), forward=fwd)))
    (HERE / ('CLOSEUP_TRACE_grip%d_%s.json' % (args.grip, args.leg))).write_text(json.dumps(dict(
        grip=bool(args.grip), leg=args.leg, times=np.round(times, 4).tolist(), forward_speed=fwd,
        foot_speed=np.round(ts_, 5).tolist(), phase=phase,
        allowed_speed=float(.5 * fwd),
        release_foot_speed_mean=float(ts_[rel].mean()), release_foot_speed_max=float(ts_[rel].max())) ) + '\n')


if __name__ == '__main__':
    main()
