"""Full-frame close-up of one foot on the floor, rendered inside the run, with its speed against the allowed speed."""
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
            plant.geom_rgba[g] = [1.0, 0.8, 0.1, 1.0]
        elif feet[g] >= 0:
            plant.geom_rgba[g] = [0.15, 0.75, 0.95, 1.0]
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
    jacp = np.zeros((3, plant.nv))
    renderer = mj.Renderer(plant, height=432, width=560)
    vopt = mj.MjvOption()
    for flag in ('mjVIS_TENDON', 'mjVIS_ACTUATOR', 'mjVIS_JOINT', 'mjVIS_COM', 'mjVIS_CONTACTPOINT'):
        vopt.flags[getattr(mj.mjtVisFlag, flag)] = False

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

    times, com, phase, speeds_all, track_pos = [], [], [], [], []
    frames = []
    for t in range(args.assist + args.release):
        tgt = target_net()
        loads, nonfoot, speed_pre, conts = measure()
        proprio = sensory.features(shim, d, tgt, loads)
        obs = np.concatenate([proprio, buf[lags].reshape(-1)])
        h0 = np.tanh(obs @ P['w0'] + P['b0']); h1 = np.tanh(h0 @ P['w1'] + P['b1'])
        pol = np.tanh(h1 @ P['w2'] + P['b2'])
        rel_now = t >= seg.shape[0]
        rate = (((tgt - actuator.target_rad) / PDT) + pol * C / PDT) if rel_now else np.clip((seg[t] - actuator.target_rad) / PDT, -CAP, CAP)
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
        times.append(float(d.time)); com.append(d.subtree_com[th].copy())
        phase.append('release' if rel_now else 'assist'); speeds_all.append(speed.copy())
        tp = None
        for leg, body, pos in conts:
            if leg == track:
                tp = pos; break
        track_pos.append(tp)
        buf = np.vstack([buf[1:], tgt[None]])
        if t % args.every == 0:
            p = tp if tp is not None else d.xpos[plant.body(args.leg + 'Tarsus5').id]
            cam = mj.MjvCamera(); cam.lookat[:] = [p[0], p[1], 0.015]
            cam.distance = 0.30; cam.elevation = -7; cam.azimuth = 150
            renderer.update_scene(d, camera=cam, scene_option=vopt)
            frames.append((t, renderer.render().copy()))
    times = np.array(times); com = np.array(com); speeds_all = np.array(speeds_all)
    rel = np.array([i for i, p in enumerate(phase) if p == 'release'])
    dur = float(times[-1] - times[0]); fwd = float(np.linalg.norm((com[-1] - com[0])[:2]) / dur)
    dur_rel = float(times[rel[-1]] - times[rel[0]])
    fwd_rel = float(np.linalg.norm((com[rel[-1]] - com[rel[0]])[:2]) / dur_rel)
    foot = speeds_all[:, track]
    allowed = .5 * max(abs(fwd_rel), .05)
    tr = dict(grip=bool(args.grip), leg=args.leg, native_steps=int((args.assist + args.release) * 400),
              duration_s=dur, forward_full=float(fwd), forward_release=float(fwd_rel),
              allowed_foot_speed=float(allowed), frames=len(frames),
              release_foot_speed_mean=float(foot[rel].mean()), release_foot_speed_max=float(foot[rel].max()),
              release_slip_ratio=float(foot[rel].mean() / max(abs(fwd_rel), 1e-9)),
              times=np.round(times, 4).tolist(), foot_speed=np.round(foot, 5).tolist(), phase=phase)
    (HERE / ('CLOSEUP_TRACE_grip%d_%s.json' % (args.grip, args.leg))).write_text(json.dumps(tr) + '\n')
    out = HERE / ('closeup_grip%d_%s' % (args.grip, args.leg)); out.mkdir(exist_ok=True)
    for t, img in frames:
        fig = plt.figure(figsize=(7.2, 8.4), dpi=100)
        gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=.16)
        ax0 = fig.add_subplot(gs[0]); ax0.imshow(img); ax0.axis('off')
        ax0.set_title('%s foot close-up (yellow = tracked leg, cyan = other feet), checkered floor is world-fixed' % args.leg, fontsize=8)
        ax1 = fig.add_subplot(gs[1])
        sel = np.arange(0, t + 1)
        ax1.plot(times[sel], foot[sel], color='tab:blue', lw=1.2, label='%s foot tangential speed' % args.leg)
        ax1.axhline(allowed, color='k', ls='--', lw=1.0, label='speed allowed by the 0.5 criterion (%.3f)' % allowed)
        if len(rel):
            ax1.axvspan(times[rel[0]], times[-1], color='tab:orange', alpha=.10)
        ax1.set(xlabel='time (s)', ylabel='foot speed (native/s)', ylim=(0, max(.05, float(np.max(foot)) * 1.15)))
        ax1.legend(fontsize=7, loc='upper left')
        now = foot[t]
        ax1.set_title('this instant: %.3f native/s  =  %.2f x the allowed speed' % (now, now / allowed), fontsize=8)
        ph = 'release: AUTONOMOUS' if phase[t] == 'release' else 'assist: measured targets replayed (not autonomous)'
        fig.suptitle('grip %s | %s | t = %.2f s' % ('ON (derived channel)' if args.grip else 'OFF (frozen body)', ph, times[t]), fontsize=9)
        fig.savefig(out / ('f%04d.png' % t)); plt.close(fig)
    try:
        renderer.close()
    except Exception:
        pass
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-framerate', '25', '-pattern_type', 'glob', '-i', str(out / '*.png'),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                    str(HERE / ('closeup_grip%d_%s.mp4' % (args.grip, args.leg)))], check=True, capture_output=True)
    print(json.dumps({k: tr[k] for k in ('grip', 'leg', 'frames', 'forward_release', 'allowed_foot_speed',
                                         'release_foot_speed_mean', 'release_foot_speed_max', 'release_slip_ratio')}))


if __name__ == '__main__':
    main()
