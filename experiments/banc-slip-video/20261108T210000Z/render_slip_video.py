"""Render the stand-pose controller so the foot slip is visible, with the measured numbers overlaid."""
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
    # tint the foot geoms so the contact points are visible (visual only)
    for g in range(plant.ngeom):
        if feet[g] >= 0:
            plant.geom_rgba[g] = [0.05, 0.85, 0.95, 1.0]
    class Shim:
        def __init__(self, m):
            self.m = m; self.jids = m.actuator_trnid[:, 0]; self.qa = m.jnt_qposadr[self.jids]
            self.va = m.jnt_dofadr[self.jids]; self.q0 = m.key_qpos[0]; self.thorax = m.body('Thorax').id
    shim = Shim(plant)
    base = np.load(SRC / 'MODEL_MEASURED_STAND.npz')
    K = int(base['K']); lags = np.asarray(base['lags'])
    z = np.load(SRC / 'MODEL_stand.npz')
    P = dict(w0=z['actor_w0'].astype(np.float64), b0=z['actor_b0'].astype(np.float64),
             w1=z['actor_w1'].astype(np.float64), b1=z['actor_b1'].astype(np.float64),
             w2=z['actor_w2'].astype(np.float64), b2=z['actor_b2'].astype(np.float64))
    ts = np.load(SRC / 'TRAINING_SET_STAND.npz')
    seg = ts['targets_rad'][int(ts['offsets'][0]):int(ts['offsets'][1])][:args.assist]
    buf = seg[:K].copy() if seg.shape[0] >= K else np.repeat(tf.STAND_Q[None], K, axis=0)
    jacp = np.zeros((3, plant.nv))
    renderer = mj.Renderer(plant, height=360, width=600)   # model framebuffer is 640x480
    vopt = mj.MjvOption()
    vopt.flags[mj.mjtVisFlag.mjVIS_TENDON] = False
    vopt.flags[mj.mjtVisFlag.mjVIS_ACTUATOR] = False
    vopt.flags[mj.mjtVisFlag.mjVIS_JOINT] = False
    vopt.flags[mj.mjtVisFlag.mjVIS_COM] = False
    vopt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    opt = plant.vis.global_
    plant.vis.global_.fovy = 45

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

    frames = []; trace = []
    time_s = []; com = []; speeds = []; loads_hist = []; phase_hist = []
    wall = 0
    for t in range(args.assist + args.release):
        tgt = target_net()
        loads, nonfoot, speed_pre, conts = measure()
        proprio = sensory.features(shim, d, tgt, loads)
        obs = np.concatenate([proprio, buf[lags].reshape(-1)])
        h0 = np.tanh(obs @ P['w0'] + P['b0']); h1 = np.tanh(h0 @ P['w1'] + P['b1'])
        pol = np.tanh(h1 @ P['w2'] + P['b2'])
        if t < seg.shape[0]:
            rate = np.clip((seg[t] - actuator.target_rad) / PDT, -CAP, CAP); phase = 'assist (measured targets replayed, NOT autonomous)'
        else:
            rate = ((tgt - actuator.target_rad) / PDT) + pol * C / PDT; phase = 'release (autonomous: base network + PPO policy)'
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
                mj.mj_step(plant, d); mj.mj_forward(plant, d); wall += 1
        loads, nonfoot, speed, conts = measure()
        time_s.append(float(d.time)); com.append(d.subtree_com[th].copy())
        speeds.append(speed); loads_hist.append(loads); phase_hist.append('assist' if t < seg.shape[0] else 'release')
        buf = np.vstack([buf[1:], tgt[None]])
        if t % args.every == 0:
            cam_w = mj.MjvCamera(); cam_w.lookat[:] = [com[-1][0], com[-1][1], 0.30]
            cam_w.distance = 3.1; cam_w.elevation = -20; cam_w.azimuth = 130
            renderer.update_scene(d, camera=cam_w, scene_option=vopt)
            wide = renderer.render().copy()
            # close-up centred on the front-left foot's material point, so the contact patch fills the frame
            footpos = None
            for ci in range(d.ncon):
                c = d.contact[ci]; ga, gb = int(c.geom1), int(c.geom2)
                if planes[ga] or planes[gb]:
                    geom = gb if planes[ga] else ga
                    if 0 <= geom < feet.size and feet[geom] == 2:
                        footpos = np.asarray(c.pos).copy(); break
            if footpos is None:
                footpos = np.array([com[-1][0], com[-1][1], 0.02])
            cam_c = mj.MjvCamera(); cam_c.lookat[:] = [footpos[0], footpos[1], 0.02]
            cam_c.distance = 0.42; cam_c.elevation = -9; cam_c.azimuth = 150
            renderer.update_scene(d, camera=cam_c, scene_option=vopt)
            close = renderer.render().copy()
            frames.append((wide, close, t))
    # per-leg slip ratio over the release phase, as reported
    rel = np.array([i for i, p in enumerate(phase_hist) if p == 'release'])
    com_a = np.array(com); times = np.array(time_s)
    dur = float(times[-1] - times[0])
    # the gate quantity is measured over the RELEASE window only, so use the same window here
    dur_rel = float(times[rel[-1]] - times[rel[0]])
    fwd = float(np.linalg.norm((com_a[rel[-1]] - com_a[rel[0]])[:2]) / dur_rel)
    sp = np.array(speeds); ld = np.array(loads_hist)
    per_leg_vt = ld[rel].__mul__(sp[rel]).sum(0) / np.maximum(ld[rel].sum(0), 1e-12)
    ratio = per_leg_vt / max(fwd, 1e-9)
    TRACE = dict(grip=bool(args.grip), assist_ticks=int(args.assist), release_ticks=int(args.release),
                 native_steps=int(wall), duration_s=dur, release_duration_s=dur_rel,
                 forward_speed_native_per_s=fwd, forward_note='release window only, same definition as the verifier',
                 per_leg_slip_ratio=dict(zip(LEGS, map(float, ratio))), slip_ratio_max=float(ratio.max()),
                 slip_ratio_mean=float(ratio.mean()),
                 com_z_min=float(com_a[rel][:, 2].min()),
                 times=np.round(times, 4).tolist(),
                 per_tick_leg_speed=np.round(sp, 5).tolist(),
                 per_tick_load_share=np.round(ld / WEIGHT, 4).tolist(),
                 phase=phase_hist, frames=len(frames), frame_stride=int(args.every),
                 frame_rate=float(1.0 / (args.every * PDT)))
    (HERE / ('SLIP_TRACE_grip%d.json' % args.grip)).write_text(json.dumps(TRACE) + '\n')
    # composite frames
    out = HERE / ('frames_grip%d' % args.grip); out.mkdir(exist_ok=True)
    for wide, close, t in frames:
        fig = plt.figure(figsize=(11.0, 7.6), dpi=100)
        gs = fig.add_gridspec(2, 3, height_ratios=[1.45, 1.0], hspace=.12, wspace=.06)
        ax0 = fig.add_subplot(gs[0, :2]); ax0.imshow(wide); ax0.axis('off')
        ax0.set_title('wide view (camera follows the fly; the checkered floor is world-fixed, so any foot that slides shows up as motion against it)', fontsize=7)
        ax1 = fig.add_subplot(gs[0, 2]); ax1.imshow(close); ax1.axis('off')
        ax1.set_title('close-up of the feet on the floor', fontsize=7)
        ax2 = fig.add_subplot(gs[1, :2])
        cur = np.round(speeds[t] / max(1e-9, np.abs(sp[t]).max() if False else 1.0), 4)
        ax2.bar(LEGS, speeds[t], color=['tab:blue' if i < 3 else 'tab:red' for i in range(6)])
        ax2.axhline(.5 * max(fwd, .05), color='k', ls='--', lw=.9)
        ax2.set(ylabel='foot tangential speed\n(native/s, force-weighted)', ylim=(0, max(.6, float(np.max(speeds)) * 1.15)))
        ax2.set_title('foot speed at the contact this instant (dashed = the foot speed the 0.5 criterion allows at this forward speed %.3f)' % fwd, fontsize=7)
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.bar(LEGS, ld[t] / WEIGHT, color='tab:purple')
        ax3.axhline(.3, color='k', ls=':', lw=.8)
        ax3.set(ylabel='foot load / body weight', ylim=(0, max(1.2, float((ld / WEIGHT).max()) * 1.1)))
        ax3.set_title('who is actually loaded', fontsize=7)
        ph = 'assist: measured targets replayed (not autonomous)' if phase_hist[t] == 'assist' else 'release: AUTONOMOUS (base network + PPO)'
        upto = [i for i in range(t + 1) if phase_hist[i] == 'release']
        if upto:
            sel = np.array(upto)
            vt_run = (ld[sel] * sp[sel]).sum(0) / np.maximum(ld[sel].sum(0), 1e-12)
        else:
            vt_run = np.zeros(6)
        slip_now = {L: float(vt_run[i] / max(fwd, .05)) for i, L in enumerate(LEGS)}
        fig.suptitle('stand-pose controller, grip %s  |  t = %.2f s  |  %s\nrunning per-leg slip ratio so far in release: %s   (final over 3 s: max %.2f, mean %.2f)'
                     % ('ON (derived channel)' if args.grip else 'OFF (frozen body)', times[t], ph,
                        '  '.join('%s %.2f' % (L, v) for L, v in slip_now.items()), ratio.max(), ratio.mean()), fontsize=8)
        fig.savefig(out / ('f%04d.png' % t))
        plt.close(fig)
    try:
        renderer.close()
    except Exception:
        pass
    cmd = ['ffmpeg', '-y', '-v', 'error', '-framerate', '25', '-pattern_type', 'glob', '-i', str(out / '*.png'),
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
           str(HERE / ('slip_grip%d.mp4' % args.grip))]
    subprocess.run(cmd, check=True, capture_output=True)
    print(json.dumps({k: TRACE[k] for k in ('grip', 'native_steps', 'forward_speed_native_per_s', 'slip_ratio_max', 'slip_ratio_mean', 'com_z_min', 'frames', 'frame_rate')}))
    print(json.dumps(dict(per_leg_slip_ratio=TRACE['per_leg_slip_ratio'])))
    print('video', str(HERE / ('slip_grip%d.mp4' % args.grip)))


if __name__ == '__main__':
    main()
