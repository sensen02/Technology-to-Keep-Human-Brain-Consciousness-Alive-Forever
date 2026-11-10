"""Does the incremental-target route physically walk under a real gait reference?

Stage A of the walking question. The teacher is the real untethered-fly gait
reference retargeted to this body (STANDARD_GAIT_FK source, GAIT.npz retarget),
not a hand-made sine. This is the user-authorized algorithm-assisted stage; it is
NOT autonomous neural walking and no claim of that is made here.

The actuator is the SAME incremental target integrator + finite PD used by the
standing/ANN route, with sweepable target-rate cap, driven by a rate teacher that
pulls the target toward the gait spline. The sweep answers one question only:
can this interface carry a walking reference at all, and at what cost?
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import mujoco as mj
from scipy.interpolate import CubicSpline

sys.path.insert(0, '/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-incremental/20261107T010000Z')
from incremental_actuator import IncrementalActuator
from plant_probe import Plant

GAIT_NPZ = Path('/home/sensen/Desktop/engine修复验收/s5-drive-chain-diagnosis/20261105T054144Z/baseline/EVAL/RUN/GAIT.npz')
OUT = Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T024636Z')
WEIGHT = 24.446354755878005
SCALE = 2.0


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)) + '\n')


def load_gait(plant):
    g = np.load(GAIT_NPZ)
    qa = g['qpos_addresses']
    assert np.array_equal(qa, plant.qa), 'gait columns must match actuator qpos order'
    spline = CubicSpline(g['phase'], g['qpos_hinges'], axis=0, bc_type='periodic')
    lo = plant.m.jnt_range[plant.jids, 0]
    hi = plant.m.jnt_range[plant.jids, 1]
    return spline, lo, hi


def run_teacher(plant, spline, lo, hi, freq, ramp_s, duration_s, rate_cap, tag):
    d = plant.fresh()
    limits = plant.limits
    a = IncrementalActuator(plant.m, d, kp=limits / .2 * SCALE,
                            kd=limits * .005 * np.sqrt(SCALE),
                            max_rate_rad_s=rate_cap)
    gait0 = np.clip(spline(0.0), lo, hi)
    rows = []
    body = []
    start = time.monotonic()
    total = int(round(duration_s / 0.001))
    peak_cmd_rate = 0.0
    clipped_ticks = 0
    for k in range(total):
        t = k * 0.001
        if t < ramp_s:
            u = t / ramp_s
            blend = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
            des = (1 - blend) * plant.q0[plant.qa] + blend * gait0
        else:
            phase = 2 * np.pi * freq * (t - ramp_s)
            des = np.clip(spline(phase % (2 * np.pi)), lo, hi)
        rate = (des - a.target_rad) / 0.001
        peak_cmd_rate = max(peak_cmd_rate, float(np.max(np.abs(rate))))
        if np.max(np.abs(rate)) > rate_cap:
            clipped_ticks += 1
        a.update(rate, 0.001)
        for _ in range(80):
            d.ctrl[:] = a.torque(d)
            mj.mj_step(plant.m, d)
            mj.mj_forward(plant.m, d)
            rows.append(plant.contact(d))
        body.append(plant.bodyrow(d))
        if not np.isfinite(d.qpos).all():
            raise RuntimeError('nonfinite state')
    rows = np.asarray(rows)
    body = {key: np.array([r[key] for r in body]) for key in body[0]}
    t = rows[:, 0]
    release = rows[t >= ramp_s]
    foot = release[:, 1:7].sum(1) / WEIGHT
    result = dict(tag=tag, importance='核心必需', freq_hz=freq, ramp_s=ramp_s, seconds=duration_s,
                  rate_cap_rad_s=rate_cap, gain_scale=SCALE,
                  note='algorithm-assisted gait reference on the incremental-target route; NOT neural autonomy',
                  root_start=body['qpos'][0, :3].tolist(), root_end=body['qpos'][-1, :3].tolist(),
                  forward_displacement=float(body['qpos'][-1, 0] - body['qpos'][0, 0]),
                  lateral_displacement=float(body['qpos'][-1, 1] - body['qpos'][0, 1]),
                  vertical_displacement=float(body['qpos'][-1, 2] - body['qpos'][0, 2]),
                  foot_support_mean=float(foot.mean()), foot_support_p10=float(np.percentile(foot, 10)),
                  nonfoot_positive_max=float(release[:, 7].max()),
                  nonfoot_contact_fraction=float((release[:, 7] > 1e-9).mean()),
                  com_z_min=float(release[:, 10].min()), com_z_end=float(release[-1, 10]),
                  bearing_legs_median=float(np.median((release[:, 1:7] > 1e-9).sum(1))),
                  per_leg_duty=[float((release[:, 1 + i] > 1e-9).mean()) for i in range(6)],
                  contact_liftoff_counts=[int(np.sum(np.diff((release[:, 1 + i] > 1e-9).astype(int)) == -1)) for i in range(6)],
                  peak_command_rate_rad_s=peak_cmd_rate,
                  rate_clipped_fraction=clipped_ticks / total,
                  warnings=int(d.warning.number.sum()),
                  wall_s=time.monotonic() - start)
    folder = OUT / tag
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / 'NATIVE.npz', rows=rows)
    np.savez_compressed(folder / 'BODY.npz', **body)
    dump(folder / 'RESULT.json', result)
    print(json.dumps(result), flush=True)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plant = Plant()
    spline, lo, hi = load_gait(plant)
    p = np.linspace(0, 2 * np.pi, 2001)
    v = spline(p, 1)
    peak_v = np.abs(v).max()  # rad per rad-of-phase
    plan = []
    for freq, rate_cap, tag in [(1.0, 40.0, 'teacher_f1.0'), (2.0, 80.0, 'teacher_f2.0')]:
        need = peak_v * 2 * np.pi * freq
        plan.append(dict(freq=freq, rate_cap=rate_cap, need=need, tag=tag))
        print(json.dumps(dict(tag=tag, freq=freq, rate_cap_needed_rad_s=float(need))), flush=True)
    summary = []
    for item in plan:
        summary.append(run_teacher(plant, spline, lo, hi, item['freq'], 0.5, 3.0,
                                   item['rate_cap'], item['tag']))
    dump(OUT / 'TEACHER_SUMMARY.json', summary)


if __name__ == '__main__':
    main()
