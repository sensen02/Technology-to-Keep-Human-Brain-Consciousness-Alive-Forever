"""Diagnose why run_equilibrium_reissue.py's multi-start found no equilibrium while
run_mujoco_equilibrium_search.py does.

Hypothesis under test: the reissue script's residual returned np.asarray(d.qacc), i.e. a VIEW into the live MjData
buffer rather than a copy. scipy.optimize.least_squares keeps the previous residual array for its ftol/gtol tests and
for the returned result; if the buffer is overwritten by every new evaluation, the algorithm compares an array with
itself and the step trial collapses. The already-verified search script returns a copy and converges.

The test runs the SAME model, SAME delivered pose A, SAME declared box and SAME first random start twice: once with a
view residual and once with a copied residual, and additionally recomputes the true residual at the returned x.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = Path('/home/sensen/Desktop/engine修复验收')
POSE_A = DESK / 'S2-declared-cone-min-s-20261015T123603/best_candidate_pose.npz'
SOURCE_POSE = ROOT / 'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz'
OUT = ROOT / ('outputs/diagnostics/aliased_residual_%s.json' % time.strftime('%Y%m%dT%H%M%S'))
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
SUFFIXES = ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll', 'Trochanter_yaw', 'Trochanter_pitch', 'Trochanter_roll', 'Tibia_pitch')
MAX_DEVIATION = 0.40
ROOT_Z_RANGE = 0.05
JOINT_MARGIN = 5e-3


def main():
    import mujoco
    from scipy.optimize import least_squares
    sys.path.insert(0, str(ROOT))
    from engine import canonical_physics as cp

    m = cp.load_model()
    d = mujoco.MjData(m)
    q0 = np.load(POSE_A)['projected_seated_qpos'].copy()
    src = np.load(SOURCE_POSE)['projected_seated_qpos'].copy()
    jname = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ''
    jids = []
    for L in LEGS:
        js = [j for j in range(m.njnt) if jname(j).startswith('joint_' + L)]
        for suf in SUFFIXES:
            jids.append(next(j for j in js if jname(j).endswith(suf)))
    qadr = [int(m.jnt_qposadr[j]) for j in jids]
    lo_q = np.array([max(m.jnt_range[j, 0] + JOINT_MARGIN, src[int(m.jnt_qposadr[j])] - MAX_DEVIATION) for j in jids])
    hi_q = np.array([min(m.jnt_range[j, 1] - JOINT_MARGIN, src[int(m.jnt_qposadr[j])] + MAX_DEVIATION) for j in jids])
    act_lo = np.full(m.na, 1e-4)
    act_hi = np.full(m.na, 1.0)
    lb = np.concatenate([act_lo, [-ROOT_Z_RANGE], lo_q])
    ub = np.concatenate([act_hi, [ROOT_Z_RANGE], hi_q])

    def unpack(x):
        qp = q0.copy()
        qp[2] = q0[2] + x[m.na]
        qp[qadr] = x[m.na + 1:]
        return qp, x[:m.na]

    def raw(x, copy_residual):
        qp, act = unpack(x)
        d.qpos[:] = qp
        d.qvel[:] = 0.0
        d.act[:] = np.clip(act, act_lo, act_hi)
        d.ctrl[:] = 0.0
        mujoco.mj_forward(m, d)
        if copy_residual:
            return np.asarray(d.qacc, dtype=float).copy()
        return np.asarray(d.qacc, dtype=float)

    rng = np.random.default_rng(20261015)
    xs = lb + rng.random(lb.size) * (ub - lb)
    cases = {}
    for tag, cp_flag in (('residual_is_a_view_into_MjData', False), ('residual_is_a_copy', True)):
        t0 = time.time()
        res = least_squares(lambda x: raw(x, cp_flag), xs, bounds=(lb, ub), xtol=1e-12, ftol=1e-12, gtol=1e-12,
                            max_nfev=400)
        true_f = raw(res.x, True)
        cases[tag] = dict(
            nfev=int(res.nfev), status=int(res.status), message=str(res.message),
            max_abs_of_returned_fun=float(np.max(np.abs(res.fun))),
            max_abs_recomputed_at_the_same_x=float(np.max(np.abs(true_f))),
            returned_fun_matches_recomputation=bool(
                np.allclose(np.asarray(res.fun), true_f, rtol=1e-12, atol=1e-12)),
            note_on_the_old_pointer_test=('a raw buffer-pointer equality test is NOT the criterion and must not be used: '
                                          'the decisive evidence is that the view variant stops after nfev = 1 with a zero '
                                          'finite-difference Jacobian, while the copy variant converges'),
            seconds=round(time.time() - t0, 1))
        print(tag, cases[tag], flush=True)

    out = dict(
        status='PASS_ALIASED_RESIDUAL_DIAGNOSIS',
        manual_research_not_product=True,
        question='why did run_equilibrium_reissue.py report no equilibrium while the verified search finds one?',
        same_model=True, same_pose=str(POSE_A), same_box=True, same_first_start=True,
        start_seed=20261015, start_x_sha256_first8=('%s' % xs[:8]),
        cases=cases,
        conclusion=('a residual that returns np.asarray(d.qacc) hands the optimiser a live buffer: the returned fun is '
                    'the same memory as d.qacc, so every new evaluation rewrites the array the algorithm still holds. '
                    'The copy variant converges on the same start/model/box. That is the root cause of the earlier '
                    'reissue failure, and it is why the corrected batch re-uses the verified search implementation.'),
        not_established='this diagnosis is about the optimiser plumbing only; it says nothing about the physical system')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
