"""Declared algorithm-assisted whole-body walking warm-up. NOT neural autonomous walking.

Every earlier attempt stepped a single leg while the other five held fixed targets. This drives all
42 joints with an alternating tripod gait expressed as body-frame foot trajectories:

  stance : the foot target ramps backwards in the thorax frame, which is what a planted foot looks
           like while the body advances, so the legs push the ground and the body moves as a result
  swing  : the foot target ramps forwards with an added lift

Tracking is resolved-rate control on the thorax-relative Jacobian (verified by central differences in
frame_audit.py), plus a body-height feedback term. Nothing writes qpos/qvel after reset, nothing
translates the root, nothing holds the pose, and no external support force is applied: the body moves
only because its own legs push the ground through the unchanged incremental actuators.

The gate evaluation is copied verbatim in structure from the frozen verifier used by the earlier
42-joint whole-body experiments so the thresholds and the slip definition cannot drift.
"""
from pathlib import Path
import json, os, sys, itertools
import numpy as np
import mujoco as mj

HERE = Path(__file__).resolve().parent
RECOVERY = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261109T202355Z')
sys.path.insert(0, str(RECOVERY))
from frame_audit import rc, T, skew  # rc = repair_core, T = train_single_leg
import core_v2

Core = core_v2.Core
LEGS = rc.LEG_ORDER
NL = rc.NL
NJ = rc.NJ
PDT = rc.PDT
MAX_RATE = rc.MAX_RATE
W = rc.WEIGHT

# Frozen gates, copied from banc-strong-actuator/20261109T230000Z/train_strong.py
GATES = dict(foot_support_mean_min=.90, nonfoot_force_max_W=.20 * W, com_height_min=1.20,
             slip_ratio_max=.5, forward_speed_min=.05)

# Alternating tripod: {front-left, middle-right, rear-left} against {front-right, middle-left, rear-right}
TRIPOD_A = ('LF', 'RM', 'LH')
TRIPOD_B = ('RF', 'LM', 'RH')
assert sorted(TRIPOD_A + TRIPOD_B) == sorted(LEGS), (TRIPOD_A, TRIPOD_B)


def rel_and_world(core, leg):
    """Thorax-relative position Jacobian and the world Jacobian for one leg's pad centre."""
    m, d = core.plant, core.d
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mj.mj_jac(m, d, jp, jr, d.xpos[core.th], core.th)
    pads = core.pads[LEGS[leg]]
    p = np.mean([d.geom_xpos[g] for g in pads], axis=0)
    J = np.zeros((3, m.nv))
    for g in pads:
        j = np.zeros_like(J)
        mj.mj_jac(m, d, j, None, d.geom_xpos[g], int(m.geom_bodyid[g]))
        J += j / len(pads)
    R = d.xmat[core.th].reshape(3, 3)
    relative = R.T @ (J - jp + skew(p - d.xpos[core.th]) @ jr)
    return relative, J


def tripod_targets(leg, k, period, stride, lift):
    """Body-frame (delta_x, delta_z) target for this leg at policy tick k."""
    offs = 0. if LEGS[leg] in TRIPOD_A else period // 2
    ph = ((k + offs) % period) / period            # 0..1
    half = 0.5
    if ph < half:                                   # stance: +S/2 -> -S/2
        u = ph / half
        return stride * (0.5 - u), 0.
    u = (ph - half) / (1. - half)                   # swing: -S/2 -> +S/2
    return stride * (-0.5 + u), lift * np.sin(np.pi * u)


def run(period=60, stride=0.06, lift=0.08, height_gain=3.0, tau=0.05,
        ticks=3200, com_target=None, record=True, debug=False, padz_gain=4.0, z_mode='mixed'):
    core = Core(); core.reset()
    rest = [np.asarray(core.measure()['foot'][l], float) for l in range(NL)]
    com0 = float(core.d.subtree_com[core.th][2])
    # Sag only: ask for more leg extension when the body is lower than the reset height, and never
    # push the body upward past it. The reset height is the only reference that is not a scripted pose.
    target_z = com_target if com_target else com0 - 0.02
    rec = []
    for k in range(ticks):
        a = np.zeros(NJ)
        com = float(core.d.subtree_com[core.th][2])
        # Negative world z velocity pushes a stance foot down, which is what lifts the body.
        push = float(np.clip(-height_gain * (target_z - com), -0.6, 0.05))
        for leg in range(NL):
            rel, world = rel_and_world(core, leg)
            ids = T.leg_joint_action_index(core, leg)
            ds = core.a_dof[ids]
            dx, dz = tripod_targets(leg, k, period, stride, lift)
            cur = np.asarray(core.measure()['foot'][leg], float)
            padz = float(core.measure()['padz'][leg])
            if z_mode == 'body':
                # Rigid body-frame tracking: over-constrains the trunk against the planted feet.
                target = np.array([rest[leg][0] + dx, rest[leg][1], rest[leg][2] + dz])
                desired = (target - cur) / tau
                A = rel[:, ds]; full = rel
            else:
                # Mixed: fore-aft and lateral tracked in the thorax frame (this is what propels the
                # body), vertical tracked in the world frame against the ground plane so the trunk
                # stays free to pitch. The push term only adds downward stance force when the body sags.
                vz = padz_gain * ((dz if dz > 0 else 0.) - padz) + (push if dz <= 0 else 0.)
                desired = np.array([(rest[leg][0] + dx - cur[0]) / tau,
                                    (rest[leg][1] - cur[1]) / tau,
                                    float(np.clip(vz, -2.5, 2.5))])
                A = np.vstack([rel[0, ds], rel[1, ds], world[2, ds]])
                full = np.vstack([rel[0], rel[1], world[2]])
            root_v = core.d.qvel.copy(); root_v[core.a_dof] = 0.
            rhs = desired - full @ root_v
            dq = A.T @ np.linalg.solve(A @ A.T + 1e-2 * np.eye(3), rhs)
            # Soft-limited so that a large request keeps its direction instead of distorting it the
            # way a hard clip does when several joints saturate together.
            a[ids] = dq / MAX_RATE / (1. + np.abs(dq) / MAX_RATE)
        m = core.tick(np.clip(a, -1., 1.))
        if debug and k % 25 == 0:
            print('k %4d com %.4f sup %.3f nf %6.2f lift %s' % (
                k, float(core.d.subtree_com[core.th][2]), float(m['loads'].sum() / W),
                float(m['nonfoot']), np.round(m['padz'], 3)), flush=True)
        if record:
            loads = m['loads']
            # Copies, not views: these arrays point into MuJoCo's live buffers, so recording the view
            # would make every sample read the final state and report zero body travel.
            rec.append((float(core.d.time), np.array(core.d.subtree_com[core.th], float),
                        loads.copy(), float(m['nonfoot']), np.array(m['speed'], float)))
    return summarise(rec, ticks * 80), core


def summarise(rec, steps):
    """Frozen verifier summary, same definitions as the earlier 42-joint experiments."""
    times = np.array([r[0] for r in rec]); com = np.array([r[1] for r in rec])
    loads = np.array([r[2] for r in rec]); nonfoot = np.array([r[3] for r in rec])
    speed = np.array([r[4] for r in rec])
    support = loads.sum(1) / W
    duration = float(times[-1] - times[0])
    # Displacement norm counts drift in any direction, so a controller that merely slides sideways
    # reads as "forward". Record the signed displacement along the initial heading and the lateral
    # component separately, and use the signed forward value for the gate.
    R0 = np.asarray(com[0]) * 0.
    d_xy = (com[-1] - com[0])[:2]
    fwd_signed = float(d_xy[0] / duration) if duration > 0 else 0.
    lateral = float(abs(d_xy[1]) / duration) if duration > 0 else 0.
    fwd = float(np.linalg.norm(d_xy) / duration) if duration > 0 else 0.
    per_leg_vt = loads.__mul__(speed).sum(0) / np.maximum(loads.sum(0), 1e-12)
    ratio = per_leg_vt / max(fwd, 1e-9)
    out = dict(native_steps=int(steps), duration_s=duration,
               forward_speed_native_per_s=fwd,
               forward_signed_native_per_s=fwd_signed,
               lateral_speed_native_per_s=lateral,
               foot_support_mean=float(support.mean()), foot_support_min=float(support.min()),
               nonfoot_force_max_W=float(nonfoot.max()), com_height_min=float(com[:, 2].min()),
               free_fall_fraction=float(np.mean(support < .05)),
               per_leg_slip_ratio=dict(zip(LEGS, map(float, ratio))),
               slip_ratio_max=float(ratio.max()), slip_ratio_mean=float(ratio.mean()),
               com_start=float(com[0][2]), com_end=float(com[-1][2]),
               com_xy_travel=float(np.linalg.norm((com[-1] - com[0])[:2])))
    out['gates'] = dict(
        support=bool(out['foot_support_mean'] >= GATES['foot_support_mean_min']),
        nonfoot=bool(out['nonfoot_force_max_W'] <= GATES['nonfoot_force_max_W']),
        com=bool(out['com_height_min'] >= GATES['com_height_min']),
        forward=bool(out['forward_signed_native_per_s'] >= GATES['forward_speed_min']),
        slip=bool(out['slip_ratio_max'] <= GATES['slip_ratio_max']))
    out['all_gates'] = bool(all(out['gates'].values()))
    return out


def main():
    ticks = int(os.environ.get('GAIT_TICKS', '3200'))
    grid = [dict(period=p, stride=s, lift=l, height_gain=h)
            for p, s, l, h in itertools.product([40, 60, 80], [0.04, 0.08], [0.06, 0.10], [0., 6.])]
    out = dict(label='declared algorithm-assisted tripod gait, all 42 joints; NOT neural walking',
               prereg='PREREG.json', gates=GATES, ticks=ticks, results=[])
    best = None
    for cfg in grid:
        s, _ = run(ticks=ticks, **cfg)
        row = dict(**cfg, **s)
        out['results'].append(row)
        key = 'p%d s%.2f l%.2f h%.0f' % (cfg['period'], cfg['stride'], cfg['lift'], cfg['height_gain'])
        print('%-22s fwd %.4f supmean %.3f supmin %.3f nonfoot %6.2f com %.3f slip %.2f  %s' % (
            key, s['forward_speed_native_per_s'], s['foot_support_mean'], s['foot_support_min'],
            s['nonfoot_force_max_W'], s['com_height_min'], s['slip_ratio_max'],
            ''.join(k[0].upper() for k, v in s['gates'].items() if v)), flush=True)
        (HERE / 'GAIT_SWEEP.json').write_text(json.dumps(out, indent=2) + '\n')
        score = s['foot_support_mean'] + s['forward_speed_native_per_s']
        if best is None or score > best[0]:
            best = (score, row)
    out['best'] = best[1] if best else None
    out['any_all_gates'] = any(r['all_gates'] for r in out['results'])
    (HERE / 'GAIT_SWEEP.json').write_text(json.dumps(out, indent=2) + '\n')
    print('any configuration passing all five frozen gates:', out['any_all_gates'])


if __name__ == '__main__':
    main()
