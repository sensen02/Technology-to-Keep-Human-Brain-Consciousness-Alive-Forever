"""Gait-speed probe: does the retargeted reference actually WALK this body?

Motivation (measured, not assumed). The S5 walking run played the retargeted
reference (GAIT.npz, derived from real untethered fly walking whose natural
cycle is 0.135 s = 7.41 Hz) at 1 Hz, i.e. 7.4x too slow. Re-measuring that run
from its own arrays shows the failure mode: feet stay in ground contact 85-100%
of the time and slide backwards on the floor while the body advances only
0.10 native per cycle instead of the ~0.8 the reference sweeps. So the earlier
"walking" was traction-limited dragging, and the ANN collapse at 2.8 s was the
tail of that, not the cause.

This probe is plant + incremental actuator only. It runs NO neural network: the
teacher reference is an algorithm stage, and nothing here is claimed as neural
control. It measures, per frequency and stride scale:

  body advance per gait cycle, real forward speed,
  stance slip (world drift of a foot while it is loaded),
  per-leg lift-off count and peak clearance,
  foot support / non-foot contact (D gates),
  COM height, root attitude, and whether the body is still standing at the end.

Success criterion for "the reference can walk": stance slip small compared with
body advance, all six legs lifting once per cycle, feet carrying the body.
Units are native; SI force values remain NOT_ESTABLISHED.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

PROJ = '/run/media/sensen/Data2/cell_wound_prototype'
ANN = PROJ + '/experiments/s5-ann-incremental/20261107T010000Z'
sys.path[:0] = [PROJ, PROJ + '/tools', ANN]
import mujoco as mj
import dynamic_support_metrics as dsm
from incremental_actuator import IncrementalActuator
from plant_probe import Plant, sha

GAIT_NPZ = Path('/home/sensen/Desktop/engine修复验收/s5-drive-chain-diagnosis/'
                '20261105T054144Z/baseline/EVAL/RUN/GAIT.npz')
OUT = Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261106T224246Z')
DT = 0.0000125          # physics
CDT = 0.001             # actuator / command tick
WEIGHT = 24.446354755878005
SCALE = 2.0             # actuator gain scale used by every S5 run
NATIVE_HZ = 7.4074074074074066   # 1 / 0.135 s, natural cycle of the source gait

# A foot counts as loaded on a 5 ms body frame when it carries this fraction of
# body weight for at least 90% of that frame's 80 native substeps.
LOADED = 1e-9


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)) + '\n')


class GaitRuntime:
    """Reference playback through the incremental target actuator (no network)."""

    def __init__(self, freq, stride=1.0, ramp=0.3, rate_cap=30.0, height=0.0,
                 smooth_ramp=True, yaw_gain=0.0, yaw_rate_gain=0.0, yaw_cap=0.35,
                 yaw_bias=0.0, target_yaw=0.0, stride_left=None, stride_right=None,
                 turn_gain=0.0, turn_cap=0.45):
        self.p = Plant()
        self.m = self.p.m
        self.freq = float(freq)
        self.stride = float(stride)
        self.ramp = float(ramp)
        self.rate_cap = float(rate_cap)
        self.height = float(height)      # additive coxa pitch offset, rad
        self.smooth_ramp = bool(smooth_ramp)
        self.yaw_gain = float(yaw_gain)
        self.yaw_rate_gain = float(yaw_rate_gain)
        self.yaw_cap = float(yaw_cap)
        self.yaw_bias = float(yaw_bias)
        self.target_yaw = float(target_yaw)
        g = np.load(GAIT_NPZ)
        if not np.array_equal(g['qpos_addresses'], self.p.qa):
            raise ValueError('GAIT qpos addresses do not match the plant actuator order')
        self.spline = CubicSpline(g['phase'], g['qpos_hinges'], axis=0, bc_type='periodic')
        self.q0 = self.p.q0[self.p.qa].copy()
        self.lo = self.m.jnt_range[self.p.jids, 0]
        self.hi = self.m.jnt_range[self.p.jids, 1]
        # scaled reference: keep the standing pose as the neutral point
        self.ref0 = np.clip(self.spline(0.0), self.lo, self.hi)
        self.cycle_s = 1.0 / self.freq
        # Actuator order is LF, LH, LM, RF, RH, RM (NOT the LEGS order used for
        # contact rows), and leg RF lists coxa ROLL before coxa YAW. So the
        # steering joint indices are explicit, not 7*leg.
        self.yaw_joints = np.array([0, 7, 14, 22, 28, 35])
        self.side = np.array([1., 1., 1., -1., -1., -1.])   # left +1, right -1
        self.gait_speed = 2 * np.pi * self.freq
        # differential stride: legs 0..2 are left, 3..5 right
        s_l = self.stride if stride_left is None else float(stride_left)
        s_r = self.stride if stride_right is None else float(stride_right)
        self.stride_vec = np.r_[np.full(21, s_l), np.full(21, s_r)]
        self.turn_gain = float(turn_gain)
        self.turn_cap = float(turn_cap)

    def phase_of(self, t):
        """Phase with a smooth speed onset so the first cycle is not a lunge."""
        if t >= self.ramp:
            return 0.5 * 2 * np.pi * self.freq * self.ramp + 2 * np.pi * self.freq * (t - self.ramp)
        u = t / self.ramp
        return 2 * np.pi * self.freq * self.ramp * (u ** 3 - 0.5 * u ** 4)

    @staticmethod
    def yaw_of(data):
        w, x, y, z = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def ref(self, phase, scale=None):
        sv = self.stride_vec if scale is None else self.stride_vec * scale
        pose = self.ref0 + sv * (np.clip(self.spline(phase % (2 * np.pi)),
                                         self.lo, self.hi) - self.ref0)
        if self.height:
            pose = pose + self.height
        return np.clip(pose, self.lo, self.hi)

    def desired(self, t, scale=None):
        if t < self.ramp and not self.smooth_ramp:
            u = t / self.ramp
            blend = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
            return (1 - blend) * self.q0 + blend * self.ref(0.0, scale)
        if t < self.ramp:
            u = t / self.ramp
            blend = 3 * u ** 2 - 2 * u ** 3
            return self.q0 + blend * (self.ref(self.phase_of(t), scale) - self.q0)
        return self.ref(self.phase_of(t), scale)

    def heading_term(self, data, prev_yaw, dt):
        """Steer term: differential coxa-yaw from heading error and yaw rate.

        Yaw rate is read from the free joint's own angular velocity (qvel[5] is
        the body-z component), so no finite difference is needed.
        """
        if not (self.yaw_gain or self.yaw_rate_gain or self.yaw_bias):
            return np.zeros(42)
        err = self.target_yaw - self.yaw_of(data)
        cmd = self.yaw_gain * err - self.yaw_rate_gain * float(data.qvel[5])
        cmd = float(np.clip(cmd, -self.yaw_cap, self.yaw_cap))
        out = np.zeros(42)
        out[self.yaw_joints] = self.side * (cmd + self.yaw_bias)
        return out

    def turn_scale(self, data):
        """Steer by asymmetric stride: outer legs step further than inner legs."""
        if not self.turn_gain:
            return np.ones(42)
        cmd = float(np.clip(self.turn_gain * (self.target_yaw - self.yaw_of(data)),
                            -self.turn_cap, self.turn_cap))
        out = np.ones(42)
        out[:21] += cmd
        out[21:] -= cmd
        return out

    def run(self, folder, seconds):
        folder.mkdir(parents=True, exist_ok=True)
        d = self.p.fresh()
        lim = self.p.limits
        a = IncrementalActuator(self.m, d, kp=lim / .2 * SCALE,
                                kd=lim * .005 * np.sqrt(SCALE),
                                max_rate_rad_s=self.rate_cap)
        rows, body, track = [], [], []
        nticks = int(round(seconds / CDT))
        start = time.monotonic()
        prev_yaw = self.yaw_of(d)
        for k in range(nticks):
            t = d.time
            want = self.desired(t, self.turn_scale(d)) + self.heading_term(d, prev_yaw, 5 * CDT)
            if k % 5 == 0:
                prev_yaw = self.yaw_of(d)
            rate = np.clip((want - a.target_rad) / CDT, -self.rate_cap, self.rate_cap)
            a.update(rate, CDT)
            for _ in range(80):
                d.ctrl[:] = a.torque(d)
                assert not d.qfrc_applied.any() and not d.xfrc_applied.any()
                mj.mj_step(self.m, d)
                mj.mj_forward(self.m, d)
                rows.append(self.p.contact(d))
            if k % 5 == 4:
                body.append(self.p.bodyrow(d))
                track.append(np.r_[a.target_rad, want])
            if (k + 1) % 1000 == 0:
                print(json.dumps(dict(tag=folder.name, sim_s=round((k + 1) / 1000, 2),
                                      wall_s=round(time.monotonic() - start, 1))), flush=True)
        rows = np.asarray(rows)
        np.savez_compressed(folder / 'NATIVE.npz', rows=rows)
        bd = {key: np.array([x[key] for x in body]) for key in body[0]}
        np.savez_compressed(folder / 'BODY.npz', **bd)
        tr = np.asarray(track)
        np.savez_compressed(folder / 'TRACK.npz', target=tr[:, :42], reference=tr[:, 42:])
        return rows, bd, tr


def legs_loaded(rows, nframes):
    """(nframes,6) boolean: foot carries load for >=90% of that frame's substeps."""
    per = rows.shape[0] // nframes
    f = rows[:, 1:7].reshape(nframes, per, 6)
    return (f > LOADED).mean(1) > 0.9


def runs_of(mask):
    """Contiguous True runs as (start, stop) half-open index pairs."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[cuts + 1]]
    stops = np.r_[idx[cuts] + 1, idx[-1] + 1]
    return list(zip(starts.tolist(), stops.tolist()))


def gait_metrics(folder, ramp):
    rows = np.load(folder / 'NATIVE.npz')['rows']
    z = np.load(folder / 'BODY.npz')
    body = {k: z[k].copy() for k in z.files}
    bt, com, feet, qpos = body['time'], body['com'], body['feet'], body['qpos']
    nf = len(bt)
    loaded = legs_loaded(rows, nf)

    sel = bt >= ramp
    rel = rows[rows[:, 0] >= ramp]
    # --- attitude: root quaternion -> roll / pitch of body z-axis
    w, x, y, zq = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
    up_z = 1 - 2 * (x ** 2 + y ** 2)          # third component of rotated +z
    tilt = np.degrees(np.arccos(np.clip(up_z, -1, 1)))

    # --- per-leg stance slip and swing clearance
    per_leg = []
    for L in range(6):
        st = loaded[:, L]
        slip, advance = [], []
        for a0, b0 in runs_of(st):
            if b0 - a0 >= 4:                    # >=20 ms loaded
                slip.append(float(feet[b0 - 1, L, 0] - feet[a0, L, 0]))
                advance.append(float(com[b0 - 1, 0] - com[a0, 0]))
        lifts = [(a0, b0) for a0, b0 in runs_of(~st) if b0 - a0 >= 4 and bt[a0] >= ramp]
        clear = [float(feet[max(a0, b0 - 1), L, 2]) for a0, b0 in lifts]
        per_leg.append(dict(
            duty=float(st[sel].mean()),
            lift_offs=len(lifts),
            clearances=clear,
            stance_runs=[dict(slip=round(s, 4), body_advance=round(adv, 4))
                         for s, adv in zip(slip, advance)],
            stance_slip_median=float(np.median(np.abs(slip))) if slip else None,
            stance_advance_median=float(np.median(advance)) if advance else None))

    foot_support = rel[:, 1:7].sum(1) / WEIGHT
    i0 = max(int(np.argmax(sel)) - 1, 0)
    dt_run = float(bt[-1] - bt[i0])
    adv = float(com[-1, 0] - com[i0, 0])
    xy = com[sel][:, :2]
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    path = float(step.sum())
    direct = float(np.linalg.norm(xy[-1] - xy[0]))
    yaw = np.arctan2(2 * (qpos[:, 3] * qpos[:, 6] + qpos[:, 4] * qpos[:, 5]),
                     1 - 2 * (qpos[:, 5] ** 2 + qpos[:, 6] ** 2))
    yaw_u = np.unwrap(yaw)
    return dict(
        seconds=dt_run,
        forward_advance=adv,
        forward_speed=float(adv / dt_run),
        path_length=path,
        path_speed=float(path / dt_run),
        straightness=float(direct / path) if path > 0 else 0.0,
        yaw_start=float(yaw_u[i0]), yaw_end=float(yaw_u[-1]),
        yaw_change=float(yaw_u[-1] - yaw_u[i0]),
        lateral_drift=float(com[-1, 1] - com[np.argmax(sel) - 1, 1]),
        com_z_min=float(com[sel, 2].min()), com_z_mean=float(com[sel, 2].mean()),
        com_z_end=float(com[-1, 2]),
        tilt_max_deg=float(tilt[sel].max()), tilt_end_deg=float(tilt[-1]),
        foot_support_mean=float(foot_support.mean()),
        foot_support_min=float(foot_support.min()),
        nonfoot_max=float(rel[:, 7].max()),
        nonfoot_fraction=float((rel[:, 7] > LOADED).mean()),
        bearing_legs_median=float(np.median((rel[:, 1:7] > LOADED).sum(1))),
        per_leg=per_leg,
        standing=bool(com[-1, 2] > 0.85 * com[np.argmax(sel) - 1, 2] and tilt[-1] < 45),
        gate=dsm.evaluate_native(
            dict(foot_support_mean=float(rows[(rows[:, 0] >= max(ramp - .2, 0))
                                              & (rows[:, 0] < ramp)][:, 1:7].sum(1).mean() / WEIGHT),
                 com_z_mean=float(com[max(np.argmax(sel) - 1, 0), 2])),
            rel)['checks'],
    )


def track_error(folder):
    z = np.load(folder / 'TRACK.npz')
    tgt, ref = z['target'], z['reference']
    return dict(target_vs_reference_rms=float(np.sqrt(((tgt - ref) ** 2).mean())),
                target_step_rms=float(np.sqrt((np.diff(tgt, axis=0) / 0.005) ** 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--freqs', default='1.0,2.0,3.7,5.5,7.4,9.0')
    ap.add_argument('--strides', default='1.0')
    ap.add_argument('--seconds', type=float, default=1.5)
    ap.add_argument('--ramp', type=float, default=0.3)
    ap.add_argument('--rate-cap', type=float, default=30.0)
    ap.add_argument('--yaw-gain', type=float, default=0.0)
    ap.add_argument('--yaw-rate-gain', type=float, default=0.0)
    ap.add_argument('--yaw-cap', type=float, default=0.35)
    ap.add_argument('--yaw-bias', type=float, default=0.0)
    ap.add_argument('--stride-left', type=float, default=None)
    ap.add_argument('--stride-right', type=float, default=None)
    ap.add_argument('--turn-gain', type=float, default=0.0)
    ap.add_argument('--turn-cap', type=float, default=0.45)
    ap.add_argument('--legacy-ramp', action='store_true',
                    help='old ramp: hold phase 0, then jump to gait speed')
    ap.add_argument('--tag', default='sweep')
    args = ap.parse_args()

    root = OUT / args.tag
    root.mkdir(parents=True, exist_ok=True)
    dump(root / 'PREREGISTRATION.json', dict(
        importance='核心必需',
        goal='decide whether the retargeted real-fly gait reference can produce real, '
             'planted-foot walking of the full body on the incremental actuator route',
        why='1 Hz playback of a 7.4 Hz reference slid the feet on the floor; measure the '
            'frequency/stride dependence instead of assuming',
        frequency_hz=[float(x) for x in args.freqs.split(',')],
        stride_scale=[float(x) for x in args.strides.split(',')],
        seconds=args.seconds, ramp_s=args.ramp, rate_cap_rad_s=args.rate_cap,
        natural_cycle_hz=NATIVE_HZ, smooth_ramp=not args.legacy_ramp,
        differential_stride=dict(left=args.stride_left, right=args.stride_right,
                                 turn_gain=args.turn_gain, turn_cap=args.turn_cap),
        heading_feedback=dict(yaw_gain=args.yaw_gain, yaw_rate_gain=args.yaw_rate_gain,
                              cap_rad=args.yaw_cap, bias_rad=args.yaw_bias,
                              actuator='differential thorax-coxa yaw target, left +, right -',
                              note='algorithm stabiliser; declared, not neural'),
        controller='algorithm gait reference only; NO neural network runs in this probe',
        metrics=dict(real_walking='stance slip small vs body advance, all six legs lift '
                                  'once per cycle, feet carry the body, body still upright, '
                                  'heading held',
                     fail='non-foot contact, COM collapse, tilt > 45 deg, free fall'),
        hashes={str(GAIT_NPZ): sha(GAIT_NPZ), str(Path(__file__)): sha(Path(__file__))},
        units='native; absolute SI forces NOT_ESTABLISHED'))
    results = []
    for f in [float(x) for x in args.freqs.split(',')]:
        for s in [float(x) for x in args.strides.split(',')]:
            tag = f'f{f:g}_s{s:g}'
            folder = root / tag
            rt = GaitRuntime(f, stride=s, ramp=args.ramp, rate_cap=args.rate_cap,
                             smooth_ramp=not args.legacy_ramp,
                             yaw_gain=args.yaw_gain, yaw_rate_gain=args.yaw_rate_gain,
                             yaw_cap=args.yaw_cap, yaw_bias=args.yaw_bias,
                             stride_left=args.stride_left, stride_right=args.stride_right,
                             turn_gain=args.turn_gain, turn_cap=args.turn_cap)
            t0 = time.monotonic()
            rows, body, tr = rt.run(folder, args.seconds)
            met = gait_metrics(folder, args.ramp)
            met.update(track_error(folder))
            met.update(freq_hz=f, stride=s, tag=tag, wall_s=round(time.monotonic() - t0, 1),
                       advance_per_cycle=met['forward_advance'] * rt.cycle_s / met['seconds'])
            dump(folder / 'RESULT.json', dict(importance='核心必需', **met))
            results.append(met)
            print(json.dumps(dict(tag=tag, adv=round(met['forward_advance'], 3),
                                  adv_per_cycle=round(met['advance_per_cycle'], 3),
                                  path=round(met['path_length'], 2),
                                  path_speed=round(met['path_speed'], 3),
                                  straight=round(met['straightness'], 3),
                                  yaw=round(met['yaw_change'], 3),
                                  support=round(met['foot_support_mean'], 3),
                                  nonfoot=round(met['nonfoot_max'], 1),
                                  comz_min=round(met['com_z_min'], 3),
                                  tilt_max=round(met['tilt_max_deg'], 1),
                                  standing=met['standing'],
                                  lift_offs=[x['lift_offs'] for x in met['per_leg']],
                                  slip=[None if x['stance_slip_median'] is None
                                        else round(x['stance_slip_median'], 3)
                                        for x in met['per_leg']]), ensure_ascii=False), flush=True)
    dump(root / 'SWEEP.json', dict(importance='核心必需', tag=args.tag, results=results))
    ok = [r for r in results if r['standing'] and r['nonfoot_max'] == 0]
    ok.sort(key=lambda r: -r['path_speed'])
    print(json.dumps(dict(best=[r['tag'] for r in ok[:3]],
                          worst=[r['tag'] for r in results if not r['standing']]),
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
