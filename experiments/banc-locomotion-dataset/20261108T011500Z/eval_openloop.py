"""Open-loop physical test of the measured-data base network on the frozen 42-joint body.

The network is the only controller: it sees its own 42-dim target history (seven lags),
predicts the next target, and the frozen bounded incremental actuator turns the change into
rates. No teacher, no phase, no root/pose write, no ANN residual, no connectome.

Reported metrics are MY OWN recomputation of the quantities that matter for the acceptance
debate (support, non-foot contact, height, free fall, per-leg material slip). They are not the
frozen D-gate suite, whose assumptions need an assist phase this run does not have.
"""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import mujoco as mj

HERE = Path(__file__).resolve().parent
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
ACT = ROOT / 'experiments/s5-ann-incremental/20261107T010000Z/incremental_actuator.py'
WEIGHT = 24.446354755878005
LEGS = ['LF', 'LH', 'LM', 'RF', 'RH', 'RM']


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=str(HERE / 'MODEL_MEASURED.npz'))
    ap.add_argument('--seconds', type=float, default=4.)
    ap.add_argument('--out', default='OPENLOOP_MEASURED')
    ap.add_argument('--warm-ticks', type=int, default=0,
                    help='DIAGNOSTIC ONLY: replay measured targets for this many ticks before self-driving')
    args = ap.parse_args()
    sys.path.insert(0, str(ACT.parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location('act', ACT)
    act = importlib.util.module_from_spec(spec); sys.modules['act'] = act; spec.loader.exec_module(act)
    plant = act.load_full_model()
    d = mj.MjData(plant); mj.mj_resetDataKeyframe(plant, d, 0); mj.mj_forward(plant, d)
    actuator = act.IncrementalActuator(plant, d)
    actuator.reset(d)
    z = np.load(args.model)
    center = z['center']; scale = z['scale']; lags = np.asarray(z['lags']); K = int(z['K'])
    w0, b0, w1, b1, w2, b2 = (z[k].astype(np.float64) for k in ('w0', 'b0', 'w1', 'b1', 'w2', 'b2'))
    jids = plant.actuator_trnid[:, 0]; qa = plant.jnt_qposadr[jids]
    th = plant.body('Thorax').id
    q0 = plant.key_qpos[0][qa].copy()
    buf = np.repeat(q0[None], K, axis=0)
    warm = None
    if args.warm_ticks:
        ts = np.load(HERE / 'TRAINING_SET.npz')
        offs = ts['offsets']
        seg = ts['targets_rad'][int(offs[0]):int(offs[1])]
        warm = seg[:args.warm_ticks]
        if warm.shape[0] >= K:
            buf = warm[:K].copy()
    planes = plant.geom_type == mj.mjtGeom.mjGEOM_PLANE
    names = [plant.geom(i).name or '' for i in range(plant.ngeom)]
    feet = np.array([next((j for j, L in enumerate(LEGS) if nm.startswith(L) and 'Tars' in nm), -1) for nm in names])
    com0 = d.subtree_com[th].copy()
    ticks = int(round(args.seconds / .005))
    samples = []
    slipsum = np.zeros(6); loadsum = np.zeros(6)
    steps = 0
    for k in range(ticks):
        if warm is not None and k < warm.shape[0]:
            nxt = warm[k].astype(np.float64)
        else:
            x = buf[lags].reshape(-1)
            h0 = np.tanh(x @ w0 + b0); h1 = np.tanh(h0 @ w1 + b1)
            nxt = center + scale * (h1 @ w2 + b2)
        rate = np.clip((nxt - buf[-1]) / .005, -30., 30.)
        actuator.update(rate, .001)
        for sub in range(5):
            for _ in range(80):
                d.ctrl[:] = actuator.torque(d)
                mj.mj_step(plant, d); mj.mj_forward(plant, d)
                steps += 1
                if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
                    raise FloatingPointError('nonfinite body state')
            # measure every 0.5 ms
            loads = np.zeros(6); nonfoot = 0.; vt_num = np.zeros(6); vt_den = np.zeros(6)
            jacp = np.zeros((3, plant.nv))
            for ci in range(d.ncon):
                c = d.contact[ci]; ga, gb = int(c.geom1), int(c.geom2)
                if not (planes[ga] or planes[gb]):
                    continue
                geom = gb if planes[ga] else ga
                leg = int(feet[geom]) if 0 <= geom < feet.size else -1
                f = np.zeros(6); mj.mj_contactForce(plant, d, ci, f)
                normal = float((c.frame.reshape(3, 3).T @ f[:3])[2]) * (1 if planes[ga] else -1)
                normal = max(0., normal)
                body = int(plant.geom_bodyid[geom])
                mj.mj_jac(plant, d, jacp, None, np.asarray(c.pos), body)
                v = jacp @ d.qvel
                vt = float(np.hypot(v[0], v[1]))
                if leg >= 0:
                    loads[leg] += normal; vt_num[leg] += normal * vt; vt_den[leg] += normal
                else:
                    nonfoot += normal
            samples.append(dict(t=d.time, loads=loads.copy(), nonfoot=nonfoot,
                                com=d.subtree_com[th].copy(), vt_num=vt_num.copy(), vt_den=vt_den.copy()))
        buf = np.vstack([buf[1:], nxt[None]])
    arr_t = np.array([s['t'] for s in samples])
    loads = np.array([s['loads'] for s in samples]); nonfoot = np.array([s['nonfoot'] for s in samples])
    com = np.array([s['com'] for s in samples])
    vt_num = np.array([s['vt_num'] for s in samples]); vt_den = np.array([s['vt_den'] for s in samples])
    support = loads.sum(1) / WEIGHT
    forward = com[-1] - com[0]
    duration = float(arr_t[-1] - arr_t[0])
    fwd_speed = float(np.linalg.norm(forward[:2]) / duration)
    per_leg_vt = np.where(vt_den.sum(0) > 0, vt_num.sum(0) / np.maximum(vt_den.sum(0), 1e-12), 0.)
    ratio = per_leg_vt / max(fwd_speed, 1e-9)
    metrics = dict(
        status='OPENLOOP_MEASURED_DATA_NETWORK',
        ticks=ticks, native_steps=steps, duration_s=duration,
        forward_displacement=list(map(float, forward)), forward_speed_native_per_s=fwd_speed,
        foot_support_mean=float(support.mean()), foot_support_min=float(support.min()),
        nonfoot_force_max_W=float(nonfoot.max()), nonfoot_force_mean_W=float(nonfoot.mean()),
        com_height_min=float(com[:, 2].min()), com_height_mean=float(com[:, 2].mean()),
        free_fall_fraction=float(np.mean(support < .05)),
        per_leg_material_speed=dict(zip(LEGS, map(float, per_leg_vt))),
        per_leg_slip_ratio=dict(zip(LEGS, map(float, ratio))),
        slip_ratio_max=float(ratio.max()), slip_ratio_mean=float(ratio.mean()),
        slip_criterion_threshold=.5, slip_criterion_passed=bool(ratio.max() <= .5),
        warnings=int(d.warning.number.sum()),
        warm_ticks=int(args.warm_ticks),
        slip_metric_note='the per-leg slip ratio is only meaningful when the feet carry the body; if foot_support_mean is low and nonfoot force is high the ratio is degenerate (few loaded contacts) and must not be read as a pass',
        provenance=dict(controller='base network trained on filtered measured motion; no teacher, no residual, no connectome'
                                     + ('; DIAGNOSTIC warm start replaying measured targets for %d ticks' % args.warm_ticks if args.warm_ticks else ''),
                        actuator='frozen bounded incremental actuator, 30 rad/s, 1 ms integration, finite PD128',
                        metric_note='own recomputation: mj_jac at contact point, force-weighted tangential speed per leg, forward speed = COM displacement / duration; not the frozen D-gate suite'),
        source_hashes={str(Path(args.model)): sha(args.model), str(ACT): sha(ACT)})
    out = HERE / args.out; out.mkdir(exist_ok=True)
    np.savez_compressed(out / 'TRACE.npz', time=arr_t, loads=loads, nonfoot=nonfoot, com=com,
                        vt_num=vt_num, vt_den=vt_den)
    (out / 'RESULT.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: metrics[k] for k in ('forward_speed_native_per_s', 'foot_support_mean', 'nonfoot_force_max_W',
                                              'com_height_min', 'free_fall_fraction', 'slip_ratio_max',
                                              'slip_criterion_passed')}))


if __name__ == '__main__':
    main()
