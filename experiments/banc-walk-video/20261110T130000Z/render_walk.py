"""Render the current best configuration. No physics change, no scripted motion."""
from pathlib import Path
import json, os, subprocess, sys, importlib.util, argparse
import numpy as np
os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco as mj
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
PAD_XML = ROOT / 'experiments/banc-triad-pad/20261110T010000Z/MODEL_TRIAD_x200.xml'
POSE = ROOT / 'experiments/banc-triad-stand-pose/20261110T030000Z/STAND_POSE_TRIAD_1.45.json'
MODEL = ROOT / 'experiments/banc-adh-namebug/20261110T150000Z/MODEL_adhfix.npz'
WEIGHT = 24.446354755878005
LEGS = ['LF', 'LH', 'LM', 'RF', 'RH', 'RM']
PDT = .001; BASE_EVERY = 5; CAP = 30.; C = .02
A_ADH = .25 * WEIGHT; STANCE = .1 * WEIGHT; MU = 1.5


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['cold', 'assist'], default='cold')
    ap.add_argument('--ticks', type=int, default=3200)
    ap.add_argument('--assist', type=int, default=1000)
    ap.add_argument('--every', type=int, default=8)
    args = ap.parse_args()
    act = load('act', ROOT / 'experiments/s5-ann-incremental/20261107T010000Z/incremental_actuator.py')
    sensory = load('sensory', ROOT / 'experiments/s5-ann-walk/20261107T091243Z/proprioception.py')
    tf = load('tf', ROOT / 'experiments/banc-stand-retarget/20261108T190000Z/train_full42_stand.py')
    plant = act.load_full_model(PAD_XML); plant.geom_friction[:, 0] = MU
    d = mj.MjData(plant)
    pose = json.loads(POSE.read_text())
    jids = plant.actuator_trnid[:, 0]
    mj.mj_resetDataKeyframe(plant, d, 0)
    d.qpos[plant.jnt_qposadr[jids]] = np.asarray(pose['q_rad'], float); d.qpos[2] = float(pose['root_z'])
    mj.mj_forward(plant, d)
    actuator = act.IncrementalActuator(plant, d); actuator.reset(d)
    th = plant.body('Thorax').id
    planes = plant.geom_type == mj.mjtGeom.mjGEOM_PLANE
    names = [plant.geom(i).name or '' for i in range(plant.ngeom)]
    pad_geoms = {L: [i for i, nm in enumerate(names) if nm.startswith(L + 'Tarsus5_pad')] for L in LEGS}
    geom_leg = np.full(plant.ngeom, -1)
    for li, L in enumerate(LEGS):
        for g in pad_geoms[L]:
            geom_leg[g] = li
    for g in range(plant.ngeom):
        nm = names[g]
        if geom_leg[g] >= 0 or any(nm.startswith(L + 'Tarsus') for L in LEGS):
            plant.geom_rgba[g] = [0.05, 0.85, 0.95, 1.0]

    class Shim:
        def __init__(s, m):
            s.m = m; s.jids = m.actuator_trnid[:, 0]; s.qa = m.jnt_qposadr[s.jids]
            s.va = m.jnt_dofadr[s.jids]; s.q0 = m.key_qpos[0]; s.thorax = m.body('Thorax').id
    shim = Shim(plant)
    base = np.load(ROOT / 'experiments/banc-stand-retarget/20261108T190000Z/MODEL_MEASURED_STAND.npz')
    K = int(base['K']); lags = np.asarray(base['lags'])
    z = np.load(MODEL)
    P = dict((k[6:], z[k].astype(np.float64)) for k in z.files if k.startswith('actor_'))
    ts = np.load(ROOT / 'experiments/banc-stand-retarget/20261108T190000Z/TRAINING_SET_STAND.npz')
    off = np.asarray(ts['offsets'], dtype=int); tar = ts['targets_rad'].astype(np.float64)
    need = int(np.ceil(args.assist / 5)); parts = []; have = 0
    for k in range(len(off) - 1):
        a, b = int(off[k]), int(off[k + 1])
        if b - a >= 20 and have < need:
            parts.append(tar[a:b]); have += b - a
    seg = np.vstack(parts) if parts else np.repeat(np.asarray(pose['q_rad'], float)[None], 200, axis=0)
    buf = np.repeat(np.asarray(pose['q_rad'], float)[None], K, axis=0)
    jacp = np.zeros((3, plant.nv))
    renderer = mj.Renderer(plant, height=400, width=640)
    vopt = mj.MjvOption()
    for fl in ('mjVIS_TENDON', 'mjVIS_ACTUATOR', 'mjVIS_JOINT', 'mjVIS_COM', 'mjVIS_CONTACTPOINT'):
        vopt.flags[getattr(mj.mjtVisFlag, fl)] = False
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
            li = int(geom_leg[geom])
            if li < 0:
                nonfoot += normal; continue
            loads[li] += normal
            body = int(plant.geom_bodyid[geom])
            mj.mj_jac(plant, d, jacp, None, np.asarray(c.pos), body)
            v = jacp @ d.qvel
            vt[li] += normal * float(np.hypot(v[0], v[1])); den[li] += normal
            conts.append((li, body, np.asarray(c.pos).copy()))
        speed = np.where(den > 0, vt / np.maximum(den, 1e-12), 0.)
        return loads, nonfoot, speed, conts

    def adhesion():
        d.xfrc_applied[:] = 0.
        l, _, _, cs = measure()
        if not cs:
            return
        cnt = np.zeros(6, int)
        for li, body, pos in cs:
            cnt[li] += 1
        rising = np.zeros(6, bool)
        for li, L in enumerate(LEGS):
            for g in pad_geoms[L]:
                mj.mj_jac(plant, d, jacp, None, np.asarray(d.geom_xpos[g]), int(plant.geom_bodyid[g]))
                if float((jacp @ d.qvel)[2]) > 0.:
                    rising[li] = True; break
        for li, body, pos in cs:
            if l[li] <= STANCE or rising[li]:
                continue
            f = np.array([0., 0., -A_ADH / max(cnt[li], 1)])
            d.xfrc_applied[body, 0:3] += f
            d.xfrc_applied[body, 3:6] += np.cross(pos - d.xpos[body], f)

    def net_target():
        x = buf[lags].reshape(-1)
        h0 = np.tanh(x @ base['w0'] + base['b0']); h1 = np.tanh(h0 @ base['w1'] + base['b1'])
        return base['center'] + base['scale'] * (h1 @ base['w2'] + base['b2'])

    def policy_mean(obs):
        h0 = np.tanh(obs @ P['w0'] + P['b0']); h1 = np.tanh(h0 @ P['w1'] + P['b1'])
        return np.tanh(h1 @ P['w2'] + P['b2'])

    tgt = net_target(); frames = []
    com0 = d.subtree_com[th].copy(); com = []; times = []; speeds = []; loadh = []; phaseh = []; nonfoots = []
    for t in range(args.ticks):
        if t % BASE_EVERY == 0:
            tgt = net_target(); buf = np.vstack([buf[1:], tgt[None]])
        loads, nonfoot, _, _ = measure()
        proprio = sensory.features(shim, d, tgt, loads)
        obs = np.concatenate([proprio, buf[lags].reshape(-1)])
        if args.mode == 'assist' and t < args.assist:
            rate = np.clip((seg[min(t // 5, len(seg) - 1)] - actuator.target_rad) / PDT, -CAP, CAP)
            phaseh.append('assist')
        else:
            rate = (tgt - actuator.target_rad) / PDT + policy_mean(obs) * C / PDT
            phaseh.append('release' if args.mode == 'assist' else 'autonomous')
        actuator.update(np.clip(rate, -CAP, CAP), .001)
        for k in range(80):
            if k % 40 == 0:
                adhesion()
            d.ctrl[:] = actuator.torque(d)
            mj.mj_step(plant, d); mj.mj_forward(plant, d)
        loads, nonfoot, speed, _ = measure()
        com.append(d.subtree_com[th].copy()); times.append(float(d.time))
        speeds.append(speed); loadh.append(loads); nonfoots.append(nonfoot)
        if t % args.every == 0:
            cw = mj.MjvCamera(); cw.lookat[:] = [com[-1][0], com[-1][1], .32]
            cw.distance = 4.6; cw.elevation = -26; cw.azimuth = 130
            renderer.update_scene(d, camera=cw, scene_option=vopt)
            wide = renderer.render().copy()
            cc = mj.MjvCamera(); cc.lookat[:] = [com[-1][0], com[-1][1], .06]
            cc.distance = 2.0; cc.elevation = -16; cc.azimuth = 150
            renderer.update_scene(d, camera=cc, scene_option=vopt)
            feet = renderer.render().copy()
            frames.append((wide, feet, t))
    try:
        renderer.close()
    except Exception:
        pass
    com_a = np.array(com); ld = np.array(loadh); sp = np.array(speeds)
    dur = float(times[-1] - times[0])
    fwd = float(np.hypot(*(com_a[-1] - com_a[0])[:2]) / dur)
    tag = 'cold_start' if args.mode == 'cold' else 'assist_then_release'
    out = HERE / ('frames_' + tag); out.mkdir(exist_ok=True)
    for f in out.glob('*.png'):
        f.unlink()
    for wide, feet, t in frames:
        vt_run = (ld[:t + 1] * sp[:t + 1]).sum(0) / np.maximum(ld[:t + 1].sum(0), 1e-12)
        fig = plt.figure(figsize=(12.4, 7.2), dpi=100)
        gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1.0], hspace=.14, wspace=.06)
        a0 = fig.add_subplot(gs[0, :2]); a0.imshow(wide); a0.axis('off')
        a0.set_title('whole body; camera follows the fly, floor is world-fixed', fontsize=7)
        a1 = fig.add_subplot(gs[0, 2]); a1.imshow(feet); a1.axis('off')
        a1.set_title('legs and feet close up (tinted cyan for visibility only)', fontsize=7)
        a2 = fig.add_subplot(gs[1, :2])
        a2.bar(LEGS, sp[t], color=['tab:blue'] * 3 + ['tab:red'] * 3)
        a2.axhline(.5 * max(fwd, .05), color='k', ls='--', lw=.9)
        a2.set(ylabel='foot speed (native/s)')
        a2.set_title('instant foot tangential speed; dashed = the 0.5 criterion allowance', fontsize=7)
        a3 = fig.add_subplot(gs[1, 2])
        a3.bar(LEGS, ld[t] / WEIGHT, color='tab:purple'); a3.axhline(.3, color='k', ls=':', lw=.8)
        a3.set(ylabel='load / body weight', ylim=(0, max(1.2, float((ld / WEIGHT).max()) * 1.1)))
        a3.set_title('who is loaded', fontsize=7)
        ph = 'AUTONOMOUS' if phaseh[t] in ('autonomous', 'release') else 'ASSIST: measured targets replayed (not autonomous)'
        slip = '  '.join('%s %.2f' % (L, float(vt_run[i] / max(fwd, .05))) for i, L in enumerate(LEGS))
        fig.suptitle('NOT accepted walking  |  t = %.2f s  |  %s  |  slow motion 5x\nrunning per-leg slip ratio (foot speed / forward speed): %s   |   forward %.3f native/s   |   non-foot %.1f W'
                      % (times[t], ph, slip, fwd, nonfoots[t]), fontsize=8)
        fig.savefig(out / ('f%04d.png' % t)); plt.close(fig)
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-framerate', '25', '-pattern_type', 'glob',
                    '-i', str(out / '*.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', str(HERE / (tag + '.mp4'))], check=True, capture_output=True)
    vt_all = (ld * sp).sum(0) / np.maximum(ld.sum(0), 1e-12)
    res = dict(mode=args.mode, ticks=int(args.ticks), native_time_s=dur, forward_native_per_s=fwd,
               forward_mm_s_equivalent=float(fwd), frames=len(frames), video=str(HERE / (tag + '.mp4')),
               per_leg_slip_ratio=dict(zip(LEGS, map(float, vt_all / max(fwd, .05)))),
               com_z_min=float(com_a[:, 2].min()), com_z_start=float(com_a[0, 2]), com_z_end=float(com_a[-1, 2]),
               nonfoot_max_W=float(np.max(nonfoots)), displacement_native=np.round((com_a[-1] - com_a[0])[:2], 5).tolist(),
               note='no physics change; camera follows the fly but the floor is world-fixed; slow motion 5x',
               label='derived body/actuator/adhesion + replacement ANN; NOT accepted walking')
    (HERE / ('RENDER_%s.json' % tag)).write_text(json.dumps(res, indent=2, default=float) + '\n')
    print(json.dumps({k: res[k] for k in ('mode', 'native_time_s', 'forward_native_per_s', 'frames', 'com_z_min', 'nonfoot_max_W')}))
    print('video', res['video'])


if __name__ == '__main__':
    main()
