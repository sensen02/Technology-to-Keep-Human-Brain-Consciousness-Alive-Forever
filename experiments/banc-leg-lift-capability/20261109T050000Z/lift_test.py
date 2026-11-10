"""Open-loop capability probe: can the actuator lift one foot? No learning involved."""
from pathlib import Path
import argparse, json, sys, importlib.util
import numpy as np
import mujoco as mj
HERE = Path(__file__).resolve().parent
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
SRC = ROOT / 'experiments/banc-stand-retarget/20261108T190000Z'
PAD = ROOT / 'experiments/banc-tarsus-pad/20261109T010000Z/MODEL_TARSUS_PAD_R02.xml'
LEGS = ['LF', 'LH', 'LM', 'RF', 'RH', 'RM']
WEIGHT = 24.446354755878005
PDT = .005; CAP = 30.
A_ADH = .5 * WEIGHT; STANCE = .1 * WEIGHT
DELTA = .4


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--leg', required=True, help='comma separated legs, e.g. LF or LF,RM,LH')
    ap.add_argument('--grip', type=int, default=0)
    ap.add_argument('--pad', type=int, default=1, help='1 = derived pad body, 0 = frozen mesh-foot body')
    ap.add_argument('--mu', type=float, default=1.5)
    ap.add_argument('--hold_ms', type=int, default=400)
    args = ap.parse_args()
    act = load('act', ROOT / 'experiments/s5-ann-incremental/20261107T010000Z/incremental_actuator.py')
    tf = load('tf', SRC / 'train_full42_stand.py')
    plant = act.load_full_model(PAD) if args.pad else act.load_full_model()
    plant.geom_friction[:, 0] = args.mu
    d = mj.MjData(plant)
    names = [plant.actuator(i).name or '' for i in range(plant.nu)]
    jids = plant.actuator_trnid[:, 0]
    jname = [plant.joint(i).name for i in jids]
    assert len(jname) == 42
    plant.geom_friction[:, 0] = args.mu
    mj.mj_resetDataKeyframe(plant, d, 0)
    d.qpos[plant.jnt_qposadr[jids]] = tf.STAND_Q
    d.qpos[2] = tf.STAND_ROOT_Z
    mj.mj_forward(plant, d)
    actuator = act.IncrementalActuator(plant, d); actuator.reset(d)
    actuator.target_rad[:] = np.asarray(tf.STAND_Q, float) if hasattr(actuator, 'target_rad') else None
    actuator.reset(d)
    th = plant.body('Thorax').id
    planes = plant.geom_type == mj.mjtGeom.mjGEOM_PLANE
    gname = [plant.geom(i).name or '' for i in range(plant.ngeom)]
    pad_geom = {}
    for l, leg in enumerate(LEGS):
        for gi in range(plant.ngeom):
            if gname[gi] == leg + 'Tarsus5_pad':
                pad_geom[leg] = gi
            elif gname[gi] == leg + 'Tarsus5_geom':
                pad_geom.setdefault(leg, gi)
    # the joints this leg will be displaced
    legs = [s.strip() for s in args.leg.split(',') if s.strip()]
    assert all(l in LEGS for l in legs), legs
    idx = [i for i, j in enumerate(jname) if any(j.startswith('joint_' + l) for l in legs)]
    lift = np.zeros(42)
    for i in idx:
        j = jname[i]
        if 'Coxa_pitch' in j or 'Trochanter_pitch' in j or 'Tibia_pitch' in j or 'Coxa_roll' in j:
            lift[i] = DELTA
        elif 'Trochanter_roll' in j:
            lift[i] = -.5 * DELTA
    lo = plant.jnt_range[jids][:, 0]; hi = plant.jnt_range[jids][:, 1]

    def measure():
        loads = np.zeros(6); nonfoot = 0.
        for ci in range(d.ncon):
            c = d.contact[ci]; ga, gb = int(c.geom1), int(c.geom2)
            if not (planes[ga] or planes[gb]):
                continue
            geom = gb if planes[ga] else ga
            f = np.zeros(6); mj.mj_contactForce(plant, d, ci, f)
            n = max(0., float((c.frame.reshape(3, 3).T @ f[:3])[2]) * (1 if planes[ga] else -1))
            leg = next((l for l, g in pad_geom.items() if g == geom), None)
            if leg is None:
                nonfoot += n
            else:
                loads[LEGS.index(leg)] += n
        return loads, nonfoot

    zero = np.asarray(tf.STAND_Q, float)
    hold = np.clip(zero, lo + .01, hi - .01)
    seq = []
    for _ in range(40): seq.append(hold)                     # 200 ms
    ramp = np.linspace(0, 1, 20)                              # 100 ms
    for r in ramp: seq.append(np.clip(zero + r * lift, lo + .01, hi - .01))
    for _ in range(args.hold_ms // 5): seq.append(np.clip(zero + lift, lo + .01, hi - .01))
    for r in ramp[::-1]: seq.append(np.clip(zero + r * lift, lo + .01, hi - .01))
    for _ in range(40): seq.append(hold)
    rec = []; torque_demand = 0.
    for tgt in seq:
        rate = (tgt - actuator.target_rad) / PDT
        for _ in range(5):
            actuator.update(np.clip(rate, -CAP, CAP), .001)
            for k in range(80):
                if args.grip and k % 40 == 0:
                    d.xfrc_applied[:] = 0.
                    l, _ = measure()
                    if l.any():
                        for leg, gi in pad_geom.items():
                            if l[LEGS.index(leg)] <= STANCE:
                                continue
                            body = int(plant.geom_bodyid[gi])
                            f = np.array([0., 0., -A_ADH])
                            d.xfrc_applied[body, 0:3] += f
                            d.xfrc_applied[body, 3:6] += np.cross(d.geom_xpos[gi] - d.xpos[body], f)
                d.ctrl[:] = actuator.torque(d)
                torque_demand = max(torque_demand, float(np.max(np.abs(d.ctrl))))
                mj.mj_step(plant, d); mj.mj_forward(plant, d)
        loads, nonfoot = measure()
        heights = []
        for l in legs:
            gi = pad_geom[l]
            heights.append(float(d.geom_xpos[gi][2]) - (float(plant.geom_size[gi][0]) if gname[gi].endswith('_pad') else 0.))
        pz = max(heights)
        gi = pad_geom[legs[0]]
        rec.append((float(d.time), pz, float(d.subtree_com[th][2]), float(loads.sum() / WEIGHT),
                    float(nonfoot), float(max(loads[LEGS.index(l)] for l in legs) / WEIGHT)))
    rec = np.array(rec)
    lim = float(np.max(plant.actuator_ctrlrange[:, 1]))
    per_joint_limit = plant.actuator_ctrlrange[:, 1].copy()
    out = dict(leg=args.leg, grip=bool(args.grip), pad=bool(args.pad), mu=args.mu,
               delta_rad=DELTA, lifted_joint_names=[jname[i] for i in idx],
               lifted_joint_targets=np.round(np.clip(zero + lift, lo + .01, hi - .01)[idx], 4).tolist(),
               legs=legs, hold_ms=args.hold_ms,
               pad_start_height=float(rec[0, 1]), pad_max_height=float(rec[:, 1].max()),
               pad_final_height=float(rec[-1, 1]), foot_left_floor=bool(rec[:, 1].max() > 0.002),
               com_z_min=float(rec[:, 2].min()), nonfoot_peak_W=float(rec[:, 4].max()),
               support_min=float(rec[:, 3].min()),
               lifted_leg_load_at_peak=float(rec[np.argmax(rec[:, 1]), 5]),
               peak_torque_demand=float(torque_demand), actuator_force_limit=lim,
               torque_within_limit=bool(torque_demand <= lim),
               torque_demand_per_joint_limit_ratio=float(torque_demand / min(float(per_joint_limit[i]) for i in idx)),
               trace=np.round(rec, 4).tolist(),
               note='open-loop capability probe; the target is set directly, no policy, no learning')
    tag = 'pad' if args.pad else 'frozen'
    (HERE / ('LIFT_%s_%dms_grip%d_%s.json' % (args.leg.replace(',', '+'), args.hold_ms, args.grip, tag))).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('legs', 'hold_ms', 'grip', 'pad_max_height', 'foot_left_floor', 'com_z_min',
                                          'nonfoot_peak_W', 'support_min', 'peak_torque_demand', 'actuator_force_limit', 'torque_within_limit')}))


if __name__ == '__main__':
    main()
