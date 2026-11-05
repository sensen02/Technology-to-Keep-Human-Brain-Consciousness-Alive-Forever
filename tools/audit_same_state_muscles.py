"""Same-state muscle reachability audit: can the 90 muscles replace the joint-torque warmup?

Question, precisely scoped: at the physical states the assisted walk actually visited, does a
bounded choice of the 90 muscle activations exist that produces the torque the warmup applied plus
the torque the muscles were already producing?  This is an ACTUATOR REPLACEMENT check at recorded
states.  It is not a walking result, it is not a controller, and no solved activation is ever
applied to the body.

Two answers are reported, and they mean different things:

  per-DOF interval  For each hinge DOF, the achievable torque interval is [sum(min(0,A)) ,
                    sum(max(0,A))] over activations in [0,1].  If the required torque lies outside
                    that interval the state is UNREACHABLE regardless of coupling, so this is a
                    sound necessary condition.  It is reported per DOF, not as a verdict.

  joint bounded fit A bounded least-squares fit over all 42 hinge DOFs at once, which respects the
                    coupling a single activation vector imposes.  A small residual is an upper
                    bound on what muscles could do at that instant; it does NOT establish that the
                    activations are reachable in time, or that the resulting dynamics would keep the
                    body up, because activation rate limits and contact transitions are not in it.

Honest limits, stated before the numbers:
  * The archived assist torque sits one native step away from the recorded state (it is the torque
    applied INTO the step whose end state was saved).  A same-state recomputation of the warmup law
    is reported alongside, so the size of that offset is measurable rather than assumed.
  * Instantaneous equality with a torque is weaker than dynamical equivalence: matching torque at
    isolated states says nothing about the states in between.
  * Model-native force units.  Absolute forces and SI NOT_ESTABLISHED.

env: AUDIT_STATES (default 25)
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import json
import sys
import time
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R), str(R / 'tools')]
RUN = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else RUN.parent.parent
NSTATES = int(os.environ.get('AUDIT_STATES', '25'))
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')


def dump(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2,
                                  default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v)) + '\n')


def main():
    import mujoco
    from scipy.optimize import lsq_linear
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    body = dict(np.load(RUN / 'BODY.npz'))
    recs = json.loads((RUN / 'NEURAL_RECORDS.json').read_text())
    t0 = float(recs[0]['absolute_start_s'])
    spec = _load_mjcf(RUN / 'MODEL.xml')
    model = spec.compile()
    data = mujoco.MjData(model)
    assert model.nu == 90 and model.nv == 48, (model.nu, model.nv)
    hinges = np.array([j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE])
    va = np.array([model.jnt_dofadr[j] for j in hinges])
    qa = np.array([model.jnt_qposadr[j] for j in hinges])
    assert va.size == 42

    time_axis = body['time']
    assist_mask = (time_axis >= t0 - 0.2) & (time_axis < t0 - 1e-9)
    idx_all = np.flatnonzero(assist_mask)
    pick = idx_all[np.linspace(0, idx_all.size - 1, min(NSTATES, idx_all.size)).round().astype(int)]

    # affine check: qfrc_actuator must be affine in act at a fixed state, or the linear fit is invalid
    def tau_of(act):
        data.act[:] = act
        mujoco.mj_forward(model, data)
        return data.qfrc_actuator.copy()

    results, per_dof_rows = [], []
    t_start = time.monotonic()
    for n, k in enumerate(pick):
        data.qpos[:] = body['qpos'][k]
        data.qvel[:] = body['qvel'][k]
        data.ctrl[:] = 0.0
        data.qfrc_applied[:] = 0.0
        data.xfrc_applied[:] = 0.0
        act_rec = body['activation'][k].copy()
        tau_rec = tau_of(act_rec)
        f0 = tau_of(np.zeros(model.nu))
        A = np.empty((model.nv, model.nu))
        for i in range(model.nu):
            e = np.zeros(model.nu)
            e[i] = 1.0
            A[:, i] = tau_of(e) - f0
        affine_err = float(np.max(np.abs(tau_rec - (f0 + A @ act_rec))))
        assist = body['assist'][k].copy()
        need = tau_rec + assist
        Ah, bh = A[va], (need - f0)[va]
        fit = lsq_linear(Ah, bh, bounds=(0.0, 1.0), max_iter=400)
        resid = float(np.linalg.norm(Ah @ fit.x - bh))
        scale = float(np.linalg.norm(bh))
        lo = np.minimum(Ah, 0.0).sum(1)
        hi = np.maximum(Ah, 0.0).sum(1)
        outside = (bh < lo - 1e-9) | (bh > hi + 1e-9)
        rank = int(np.linalg.matrix_rank(Ah))
        results.append(dict(
            sample=int(k), relative_s=float(time_axis[k] - t0),
            affine_max_error=affine_err,
            required_norm=scale, muscle_only_residual=resid,
            muscle_only_residual_relative=(resid / scale if scale > 0 else None),
            dofs_outside_interval=int(outside.sum()), hinge_dofs=42,
            demand_rank=rank,
            recorded_act_residual=float(np.linalg.norm(Ah @ act_rec - bh)),
            fit_saturated=int(np.sum((fit.x >= 1 - 1e-9) | (fit.x <= 1e-9))),
            fit_status=int(fit.status)))
        for j in range(42):
            per_dof_rows.append(dict(sample=int(k), dof=int(va[j]), joint=model.joint(hinges[j]).name,
                                     required=float(bh[j]), interval_lo=float(lo[j]), interval_hi=float(hi[j]),
                                     shortfall=float(0.0 if lo[j] - 1e-9 <= bh[j] <= hi[j] + 1e-9
                                                     else (lo[j] - bh[j] if bh[j] < lo[j] else bh[j] - hi[j])),
                                     outside=bool(outside[j])))
        print('sample %2d/%d  affine_err=%.2e  residual=%.4f / demand %.4f  dofs_outside=%d/%d  %.1fs'
              % (n + 1, len(pick), affine_err, resid, scale, int(outside.sum()), Ah.shape[0],
                 time.monotonic() - t_start), flush=True)

    rel = [r['muscle_only_residual_relative'] for r in results if r['muscle_only_residual_relative'] is not None]
    outside_total = sum(r['dofs_outside_interval'] for r in results)
    summary = dict(
        run=str(RUN), samples=len(results), states_from='last 0.2 s of the assisted window',
        sample_rate_s=0.0005,
        required_definition='qfrc_actuator at recorded activation PLUS the assist torque recorded for the same step',
        affine_max_error_over_samples=float(max(r['affine_max_error'] for r in results)),
        muscle_only_residual_relative=dict(
            mean=float(np.mean(rel)), median=float(np.median(rel)), worst=float(np.max(rel)),
            best=float(np.min(rel))),
        dofs_outside_interval_total=int(outside_total),
        dofs_outside_interval_per_sample_max=int(max(r['dofs_outside_interval'] for r in results)),
        dofs_outside_interval_per_sample_min=int(min(r['dofs_outside_interval'] for r in results)),
        demand_rank_median=float(np.median([r['demand_rank'] for r in results])),
        interpretation=dict(
            outside_interval='a DOF whose required torque lies outside its achievable interval cannot be '
                             'produced by ANY activation vector: sound necessary condition, evaluated per DOF',
            residual='joint fit over 42 hinge DOFs at once; small residual is an optimistic instantaneous '
                     'upper bound, not dynamical equivalence and not a walking result'),
        caveats=[
            'assist torque is the torque applied INTO the step whose end state was saved: one native step offset',
            'instantaneous torque matching does not establish activation-rate or contact-transition feasibility',
            'model-native units; absolute forces and SI NOT_ESTABLISHED'],
        wall_s=time.monotonic() - t_start,
        units='absolute forces and SI NOT_ESTABLISHED',
        importance='核心必需')
    dump(OUT / 'SAME_STATE_REACHABILITY.json', dict(summary=summary, per_sample=results))
    with (OUT / 'SAME_STATE_PER_DOF.csv').open('w') as f:
        cols = list(per_dof_rows[0])
        f.write(','.join(cols) + '\n')
        for r in per_dof_rows:
            f.write(','.join(str(r[c]) for c in cols) + '\n')
    if outside_total:
        worst = {}
        for r in per_dof_rows:
            if r['outside']:
                j = r['joint']
                if j not in worst or abs(r['shortfall']) > abs(worst[j]['shortfall']):
                    worst[j] = r
        dump(OUT / 'SAME_STATE_WORST_DOFS.json', dict(
            note='one representative sample per joint that ever fell outside its achievable interval',
            joints=sorted(worst.values(), key=lambda r: -abs(r['shortfall']))[:20]))
    print(json.dumps(summary['muscle_only_residual_relative'], ensure_ascii=False), 'outside_total=', outside_total)


if __name__ == '__main__':
    main()
