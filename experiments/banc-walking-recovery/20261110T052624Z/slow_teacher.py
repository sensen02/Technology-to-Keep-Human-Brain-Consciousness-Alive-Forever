"""Bounded algorithm-teacher diagnostic; no ANN or autonomous walking claim."""
from pathlib import Path
import json
import time
import numpy as np
import mujoco as mj
import physics as P
import identity as I

HERE = Path(__file__).resolve().parent
OUT = HERE / 'slow_teacher_diag'
CFG = dict(periods_ms=[60, 200, 400, 800], cycles=4, settle_ms=300,
           stride_native=.03, lift_native=.08, tau_s=.02, adhesion=False,
           sampling_s=.001, max_rate_rad_s=2., label='algorithm teacher only',
           historical_difference='adhesion disabled; zero-action settle300ms then rebase rest feet',
           acceptance='legacy branch engineering gates only, not native original certification',
           event_definition='load<=.05W initiates; max clearance>.015 and relative x advance>.03 during unloading; reload>=.1W and padz<=.004 completes',
           forward_caveat='stride/period at800ms=.0375<.05 gate; rate limitation is not body impossibility')


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def leg_ids(c):
    ids = [[i for i, j in enumerate(c.actuator.joint_ids)
            if c.plant.joint(int(j)).name.startswith('joint_' + leg)] for leg in P.LEGS]
    if any(len(v) != 7 for v in ids) or sorted(sum(ids, [])) != list(range(42)):
        raise ValueError('Expected seven distinct actuator joints per leg')
    return ids


def rel_jac(c, leg):
    model, data = c.plant, c.d
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mj.mj_jac(model, data, jp, jr, data.xpos[c.th], c.th)
    pads = c.pads[P.LEGS[leg]]
    point = np.mean([data.geom_xpos[g] for g in pads], axis=0)
    jac = np.zeros_like(jp)
    for g in pads:
        j = np.zeros_like(jp)
        mj.mj_jac(model, data, j, None, data.geom_xpos[g], int(model.geom_bodyid[g]))
        jac += j / len(pads)
    rot = data.xmat[c.th].reshape(3, 3)
    return rot.T @ (jac - jp + skew(point - data.xpos[c.th]) @ jr)


def teacher(c, rest, k, period, ids):
    """Exact old BC control algebra; snapshot physics, local Jacobian and phase."""
    a, raw, err = np.zeros(42), np.zeros(42), np.zeros((6, 3))
    measured = c.measure()
    rv = c.d.qvel.copy()
    rv[c.a_dof] = 0.
    for leg in range(6):
        rel = rel_jac(c, leg)
        ds = c.a_dof[ids[leg]]
        offset = 0 if P.LEGS[leg] in ('LF', 'RM', 'LH') else period // 2
        ph = ((k + offset) % period) / period
        cur = measured['foot'][leg]
        if ph < .5:
            u = ph / .5
            target = rest[leg] + np.array([CFG['stride_native'] * (.5-u), 0., 0.])
            error = target[:2] - cur[:2]
            mat = rel[:2, ds]
            dq = mat.T @ np.linalg.solve(mat @ mat.T + 1e-2*np.eye(2), error/CFG['tau_s'] - rel[:2] @ rv)
            err[leg, :2] = error
        else:
            u = (ph-.5)/.5
            target = rest[leg] + np.array([CFG['stride_native']*(-.5+u), 0., CFG['lift_native']*np.sin(np.pi*u)])
            error = target-cur
            mat = rel[:, ds]
            dq = mat.T @ np.linalg.solve(mat @ mat.T + 1e-2*np.eye(3), error/CFG['tau_s'] - rel @ rv)
            err[leg] = error
        raw[ids[leg]] = dq/P.MAX_RATE
        a[ids[leg]] = raw[ids[leg]]/(1.+np.abs(raw[ids[leg]]))
    return np.clip(a, -1., 1.), raw, err


def sample(c, m):
    return dict(time_s=float(c.d.time), com_native=c.d.subtree_com[c.th].copy(),
                load_native=m['loads'].copy(), padz_native=m['padz'].copy(),
                relative_foot_native=m['foot'].copy(), slip_speed_native_s=m['speed'].copy(),
                nonfoot_native=float(m['nonfoot']), qpos=c.d.qpos.copy(), qvel=c.d.qvel.copy(),
                world_foot_native=np.array([np.mean([c.d.geom_xpos[g] for g in c.pads[L]], axis=0) for L in P.LEGS]),
                requested_rate_rad_s=c.requested_rate.copy(), actual_joint_velocity_rad_s=c.d.qvel[c.a_dof].copy(),
                actuator_target_rad=c.actuator.target_rad.copy())


def event_counts(a, period):
    perleg = {}
    for j, leg in enumerate(P.LEGS):
        pending, events = None, []
        for i in range(len(a['time_s'])):
            load = a['load_native'][i, j]/P.W
            if pending is None and load <= .05:
                pending = dict(start=i, start_x=float(a['relative_foot_native'][i,j,0]),
                               max_advance=0., max_clearance=float(a['padz_native'][i,j]))
            if pending is not None:
                if load <= .05:
                    pending['max_advance'] = max(pending['max_advance'], float(a['relative_foot_native'][i,j,0])-pending['start_x'])
                    pending['max_clearance'] = max(pending['max_clearance'], float(a['padz_native'][i,j]))
                if load >= .1 and a['padz_native'][i,j] <= .004:
                    pending.update(end=i, cycle=int((i-1)//period),
                                   complete=bool(pending['max_advance']>.03 and pending['max_clearance']>.015))
                    events.append(pending)
                    pending = None
        complete = [e for e in events if e['complete']]
        perleg[leg] = dict(reloads=len(events), completed_events=len(complete),
                          cycles_with_completion=sorted(set(e['cycle'] for e in complete)),
                          events=events, unfinished=pending,
                          max_relative_x_range=float(np.ptp(a['relative_foot_native'][:,j,0])),
                          max_clearance=float(a['padz_native'][:,j].max()))
    return perleg


def main():
    if OUT.exists():
        raise FileExistsError('Refuse to overwrite prior diagnostic')
    OUT.mkdir()
    I.atomic_json(OUT/'CONFIG_PREREG.json', CFG)
    start = time.time()
    c0 = P.Core(adhesion=False)
    ids0 = leg_ids(c0)
    manifest = I.capture(c0, CFG, [Path(__file__), HERE/'physics.py', HERE/'identity.py'])
    manifest['config']['joint_names_by_leg'] = [[c0.plant.joint(int(c0.actuator.joint_ids[i])).name for i in ids] for ids in ids0]
    # capture again so config is included in its canonical identity digest.
    manifest = I.capture(c0, manifest['config'], [Path(__file__), HERE/'physics.py', HERE/'identity.py'])
    I.atomic_json(OUT/'INPUT_IDENTITY.json', manifest)
    results = []
    for period in CFG['periods_ms']:
        c = P.Core(adhesion=False)
        I.verify_runtime(manifest, c)
        ids = leg_ids(c)
        settling = []
        for _ in range(CFG['settle_ms']):
            settling.append(sample(c, c.tick(np.zeros(42))))
        rest = c.measure()['foot'].copy()
        initial = sample(c, c.measure())
        rows = [dict(**initial, raw_dq_over_max=np.zeros(42), tracking_error_native=np.zeros((6,3)))]
        for k in range(period*CFG['cycles']):
            action, raw, error = teacher(c, rest, k, period, ids)
            row = sample(c, c.tick(action))
            row.update(raw_dq_over_max=raw.copy(), tracking_error_native=error.copy())
            rows.append(row)
        arrays = {key: np.asarray([r[key] for r in rows]) for key in rows[0]}
        arrays.update(rest_foot_native=rest, leg_order=np.asarray(P.LEGS))
        np.savez_compressed(OUT/f'period_{period:04d}ms.npz', **arrays)
        np.savez_compressed(OUT/f'settle_{period:04d}ms.npz', **{key: np.asarray([r[key] for r in settling]) for key in settling[0]})
        rec = [(r['time_s'], r['com_native'], r['load_native'], r['nonfoot_native'], r['slip_speed_native_s']) for r in rows]
        summary = P.summarise(rec, period*4*80, initial_heading=c.initial_heading)
        raw = arrays['raw_dq_over_max'][1:]
        joint_diag = {leg: dict(raw_dq_over_max_abs_p50_p95_max=np.quantile(np.abs(raw[:,ix]), [.5,.95,1]).tolist(),
                                     raw_joint_overlimit_fraction=float(np.mean(np.abs(raw[:,ix])>1)),
                                     actual_rate_abs_max=float(np.abs(arrays['requested_rate_rad_s'][1:,ix]).max()),
                                     joint_velocity_abs_max=float(np.abs(arrays['actual_joint_velocity_rad_s'][1:,ix]).max()),
                                     tracking_error_norm_p95=float(np.quantile(np.linalg.norm(arrays['tracking_error_native'][1:,j],axis=1),.95)))
                      for j,(leg,ix) in enumerate(zip(P.LEGS,ids))}
        row = dict(period_ms=period, simulated_s=period*.004, summary=summary,
                   per_leg_joint_diagnostic=joint_diag, completion=event_counts(arrays,period),
                   com_displacement_native=(arrays['com_native'][-1]-arrays['com_native'][0]).tolist(),
                   settle_end_load_over_W=(arrays['load_native'][0]/P.W).tolist())
        results.append(row)
        I.atomic_json(OUT/'RESULTS.json', dict(config=CFG, runs=results, wall_s=time.time()-start, complete=False))
        print(json.dumps(dict(period_ms=period, forward=summary['forward_signed_native_per_s'],
                              support=summary['foot_support_mean'], nonfoot=summary['nonfoot_force_max_native'],
                              com=summary['com_height_min'], slip=summary['slip_ratio_max'],
                              completed={l:v['completed_events'] for l,v in row['completion'].items()})),flush=True)
    I.verify(manifest)
    for period in CFG['periods_ms']:
        with np.load(OUT/f'period_{period:04d}ms.npz') as z:
            assert len(z['time_s']) == period*4+1
            assert all(np.isfinite(z[k]).all() for k in z.files if z[k].dtype.kind in 'fiu')
            assert np.allclose(np.diff(z['time_s']),.001,rtol=0,atol=1e-10)
            assert np.max(np.abs(z['requested_rate_rad_s'])) <= 2.+1e-12
    I.atomic_json(OUT/'RESULTS.json', dict(config=CFG,runs=results,wall_s=time.time()-start,complete=True,
                                        verification='input hashes unchanged; all NPZ loaded finite; 1ms timestamps and 2rad/s limits verified',
                                        native_certificate=False))
    files = {str(p):I.sha256(p) for p in sorted(OUT.iterdir()) if p.is_file()}
    I.atomic_json(OUT/'OUTPUT_HASHES.json',files)
    print('completed verified',round(time.time()-start,3),'seconds',flush=True)


if __name__ == '__main__':
    main()
