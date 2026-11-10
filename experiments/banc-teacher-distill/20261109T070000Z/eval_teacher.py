"""Evaluate the trained stand-pose controller at a range of ground friction coefficients (no retraining)."""
from pathlib import Path
import argparse, hashlib, json, sys, time, importlib.util
import numpy as np
import mujoco as mj
HERE = Path(__file__).resolve().parent
PAD_XML = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-tarsus-pad/20261109T010000Z/MODEL_TARSUS_PAD_R02.xml')
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
SRC = ROOT / 'experiments/banc-stand-retarget/20261108T190000Z'
WEIGHT = 24.446354755878005
LEGS = ['LF', 'LH', 'LM', 'RF', 'RH', 'RM']
PDT = .005; CAP = 30.; C = .05
A_FRAC = float(__import__('os').environ.get('ADH_FRAC', '0.5'))
RELEASE_ON_LIFT = bool(int(__import__('os').environ.get('ADH_RELEASE', '0')))
PAD_BODY = bool(int(__import__('os').environ.get('PAD_BODY', '1')))
FREEZE_XML = None
A_ADH = A_FRAC * WEIGHT; STANCE = .1 * WEIGHT


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


def checksum(plant):
    parts = [plant.geom_pos, plant.geom_size, plant.geom_quat, plant.body_mass, plant.body_inertia,
             plant.jnt_range, plant.actuator_gainprm, plant.actuator_biasprm, plant.actuator_forcerange, plant.opt.gravity]
    h = hashlib.sha256()
    for a in parts:
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mu', type=float, required=True)
    ap.add_argument('--base', default=str(HERE / 'MODEL_MEASURED_TEACHER.npz'))
    ap.add_argument('--open', type=int, default=0)
    ap.add_argument('--model', default=str(HERE / 'MODEL_MEASURED_TEACHER.npz'))
    ap.add_argument('--grip', type=int, default=0)
    ap.add_argument('--assist', type=int, default=200)
    ap.add_argument('--release', type=int, default=600)
    args = ap.parse_args()
    t0 = time.time()
    act = load('act', ROOT / 'experiments/s5-ann-incremental/20261107T010000Z/incremental_actuator.py')
    sensory = load('sensory', ROOT / 'experiments/s5-ann-walk/20261107T091243Z/proprioception.py')
    tf = load('tf', SRC / 'train_full42_stand.py')
    plant = act.load_full_model(PAD_XML) if PAD_BODY else act.load_full_model(); d = mj.MjData(plant)
    pre = checksum(plant)
    fric_before = plant.geom_friction.copy()
    plant.geom_friction[:, 0] = args.mu
    post = checksum(plant)
    mj.mj_resetDataKeyframe(plant, d, 0)
    d.qpos[plant.jnt_qposadr[plant.actuator_trnid[:, 0]]] = tf.STAND_Q
    d.qpos[2] = tf.STAND_ROOT_Z
    mj.mj_forward(plant, d)
    actuator = act.IncrementalActuator(plant, d); actuator.reset(d)
    th = plant.body('Thorax').id
    planes = plant.geom_type == mj.mjtGeom.mjGEOM_PLANE
    gname = [plant.geom(i).name or '' for i in range(plant.ngeom)]
    # in the pad body only the pad spheres make foot contact; map them explicitly and assert it
    if PAD_BODY:
        feet = np.array([next((j for j, L in enumerate(LEGS) if nm == L + 'Tarsus5_pad'), -1) for nm in gname])
        foot_geoms = [i for i in range(plant.ngeom) if (gname[i] or '').endswith('_pad')]
    else:
        feet = np.array([next((j for j, L in enumerate(LEGS) if nm.startswith(L) and 'Tarsus' in nm), -1) for nm in gname])
        foot_geoms = [i for i in range(plant.ngeom) if feet[i] >= 0]
    assert all(feet[g] >= 0 for g in foot_geoms), 'foot geoms must map onto the six legs'
    class Shim:
        def __init__(self, m):
            self.m = m; self.jids = m.actuator_trnid[:, 0]; self.qa = m.jnt_qposadr[self.jids]
            self.va = m.jnt_dofadr[self.jids]; self.q0 = m.key_qpos[0]; self.thorax = m.body('Thorax').id
    shim = Shim(plant)
    base = np.load(args.base); K = int(base['K']); lags = np.asarray(base['lags'])
    z = np.load(args.model) if not args.open else None
    P = None if args.open else dict(w0=z['actor_w0'].astype(np.float64), b0=z['actor_b0'].astype(np.float64),
             w1=z['actor_w1'].astype(np.float64), b1=z['actor_b1'].astype(np.float64),
             w2=z['actor_w2'].astype(np.float64), b2=z['actor_b2'].astype(np.float64))
    ts = np.load(HERE / 'TRAINING_SET_TEACHER.npz')
    seg = ts['targets_rad'][int(ts['offsets'][0]):int(ts['offsets'][1])][:args.assist]
    buf = seg[:K].copy() if seg.shape[0] >= K else np.repeat(tf.STAND_Q[None], K, axis=0)
    jacp = np.zeros((3, plant.nv)); jacp_v = np.zeros((3, plant.nv))

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

    rec = []; phase = []; steps = 0
    for t in range(args.assist + args.release):
        tgt = target_net()
        loads, nonfoot, speed, conts = measure()
        proprio = sensory.features(shim, d, tgt, loads)
        if args.open:
            rate = np.clip((tgt - actuator.target_rad) / PDT, -CAP, CAP); phase.append('release' if t >= seg.shape[0] else 'assist')
        else:
            obs = np.concatenate([proprio, buf[lags].reshape(-1)])
            h0 = np.tanh(obs @ P['w0'] + P['b0']); h1 = np.tanh(h0 @ P['w1'] + P['b1'])
            pol = np.tanh(h1 @ P['w2'] + P['b2'])
        if args.open:
            pass
        elif t < seg.shape[0]:
            rate = np.clip((seg[t] - actuator.target_rad) / PDT, -CAP, CAP); phase.append('assist')
        else:
            rate = ((tgt - actuator.target_rad) / PDT) + pol * C / PDT; phase.append('release')
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
                        rising = np.zeros(6, bool)
                        if RELEASE_ON_LIFT:
                            for leg in range(6):
                                gi = next((g for g in foot_geoms if feet[g] == leg), None)
                                if gi is None:
                                    continue
                                mj.mj_jac(plant, d, jacp_v, None, np.asarray(d.geom_xpos[gi]), int(plant.geom_bodyid[gi]))
                                rising[leg] = float((jacp_v @ d.qvel)[2]) > 0.
                        for leg, body, pos in cs:
                            if l[leg] <= STANCE or rising[leg]:
                                continue
                            f = np.array([0., 0., -A_ADH / max(cnt[leg], 1)])
                            d.xfrc_applied[body, 0:3] += f
                            d.xfrc_applied[body, 3:6] += np.cross(pos - d.xpos[body], f)
                d.ctrl[:] = actuator.torque(d)
                mj.mj_step(plant, d); mj.mj_forward(plant, d); steps += 1
        loads, nonfoot, speed, conts = measure()
        padz = np.zeros(6)
        for l, leg in enumerate(LEGS):
            for gi in range(plant.ngeom):
                if (plant.geom(gi).name or '') == leg + 'Tarsus5_pad':
                    padz[l] = float(d.geom_xpos[gi][2]) - float(plant.geom_size[gi][0]); break
        rec.append((d.time, d.subtree_com[th].copy(), loads, nonfoot, speed, padz / .02))
        buf = np.vstack([buf[1:], tgt[None]])
    phase = np.array(phase)
    idx_rel = np.flatnonzero(phase == 'release')
    dst = idx_rel[100:]
    out = dict(mu=args.mu, grip=bool(args.grip), model=str(args.model), native_steps=int(steps), wall_s=time.time() - t0,
               label='teacher imitation (replacement ANN imitating the official reference teacher); own body, own actuator; NOT frozen-body acceptance',
               adhesion=dict(a_frac=A_FRAC, release_on_lift=RELEASE_ON_LIFT, pad_body=PAD_BODY, stance_load=STANCE),
               pad_xml=str(PAD_XML), pad_sha256=hashlib.sha256(PAD_XML.read_bytes()).hexdigest(),
               friction_before=dict(zip([gname[i] or ('geom%d' % i) for i in range(plant.ngeom)],
                                        np.round(fric_before, 4).tolist())),
               friction_after_first_component=args.mu,
               checksum_geometry_before=pre, checksum_geometry_after=post,
               checksum_note='equal checksums would mean the write did not land; geometry, mass, ranges, gains and gravity are hashed, friction is not',
               per_leg_stance_share=None)
    for tag, idx in (('release', idx_rel), ('steady', dst)):
        times = np.array([rec[i][0] for i in idx]); com = np.array([rec[i][1] for i in idx])
        loads = np.array([rec[i][2] for i in idx]); nonfoot = np.array([rec[i][3] for i in idx])
        speed = np.array([rec[i][4] for i in idx])
        support = loads.sum(1) / WEIGHT
        dur = float(times[-1] - times[0]); fwd = float(np.linalg.norm((com[-1] - com[0])[:2]) / dur)
        per_leg = loads.__mul__(speed).sum(0) / np.maximum(loads.sum(0), 1e-12)
        ratio = per_leg / max(fwd, 1e-9)
        allowed = .5 * max(abs(fwd), .05)
        stance = loads > .3 * WEIGHT
        frac = [(float((speed[stance[:, i], i] > allowed).mean()) if stance[:, i].any() else None) for i in range(6)]
        out[tag] = dict(duration_s=dur, forward_speed_native_per_s=fwd, foot_support_mean=float(support.mean()),
                        foot_support_min=float(support.min()), nonfoot_force_max_W=float(nonfoot.max()),
                        com_height_min=float(com[:, 2].min()), free_fall_fraction=float(np.mean(support < .05)),
                        per_leg_slip_ratio=dict(zip(LEGS, map(float, ratio))),
                        slip_ratio_max=float(ratio.max()), slip_ratio_mean=float(ratio.mean()),
                        allowed_foot_speed=allowed,
                        stance_speed_over_allowed=dict(zip(LEGS, frac)),
                        per_leg_stance_share=dict(zip(LEGS, map(float, (loads.sum(0) / np.maximum(loads.sum(), 1e-9))))),
                        per_leg_stance_fraction=dict(zip(LEGS, map(float, (loads > .1 * WEIGHT).mean(0)))),
                        per_leg_swing_clearance_mean=dict(zip(LEGS, [float(np.clip(rec[i][5], 0, 1).mean()) if (loads[:, i] <= .1 * WEIGHT).any() else 0.0 for i in range(6)])))
    r = out['release']
    out['release_gates'] = dict(foot_support_mean=r['foot_support_mean'] >= .9, nonfoot=r['nonfoot_force_max_W'] <= .2 * WEIGHT,
                               com_height=r['com_height_min'] >= 1.20, slip=r['slip_ratio_max'] <= .5,
                               forward=r['forward_speed_native_per_s'] >= .05)
    out['release_gates_passed'] = all(out['release_gates'].values())
    tag = Path(args.model).stem.replace('MODEL_', '')
    (HERE / ('TEACHER_%s_open%d_grip%d_pad%d_mu%.2f.json' % (tag, args.open, args.grip, int(PAD_BODY), args.mu))).write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(model=tag, mu=args.mu, grip=bool(args.grip),
                          release={k: round(r[k], 4) for k in ('forward_speed_native_per_s', 'foot_support_mean', 'nonfoot_force_max_W', 'com_height_min', 'slip_ratio_max', 'slip_ratio_mean')},
                          steady_mean_slip=round(out['steady']['slip_ratio_mean'], 4), gates=out['release_gates'])))


if __name__ == '__main__':
    main()
