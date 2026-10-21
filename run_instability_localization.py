"""Instability localisation + friction-cone utilisation for the 0.5 s declared-drive runs (A, B, C).

Declared questions answered here:
  Q1  for each start pose, the first body-floor (non-foot) contact step T0, the contact table at T0
      (geom pair, normal force, tangential forces) and the joint-deviation table at T0: which leg has already
      lost support and which joint deviated first;
  Q2  the friction-cone utilisation in the last 20 steps BEFORE T0, per contact point and per foot, measured
      against the DECLARED PYRAMIDAL AXIS-RAY DIAMOND cone. That cone is the non-negative hull of the four axis
      rays n + mu*t1, n - mu*t1, n + mu*t2, n - mu*t2; its four VERTICES lie on the axes at distance mu, so in
      the tangential plane it is the DIAMOND |ft1|/(mu1*fn) + |ft2|/(mu2*fn) <= 1, whose utilisation is
      u_diamond = u1 + u2 with u1 = |ft1|/(mu1*fn) and u2 = |ft2|/(mu2*fn). MuJoCo implements exactly this
      diamond with four mjCNSTR_CONTACT_PYRAMIDAL rows per contact. The set {|ft1| <= mu1*fn, |ft2| <= mu2*fn}
      (u_square = max(u1,u2) <= 1) is the SQUARE, which strictly CONTAINS the diamond, and is reported only as a
      secondary reading. Both force conventions are reported: IN-RUN (live qvel/act read right after mj_step) and
      QUASI-STATIC (recomputed at the stored qpos with qvel = 0 and act = 0).

Discipline (unchanged): qpos/qvel/act written ONCE before the first step, only ctrl written inside the loop; the
drive is the frozen connectome-derived archive normal_controls (10 x 90), played at the 5 ms cadence over the
first 50 ms and then HELD CONSTANT (index saturates) for the rest of the window - at 0.5 s that is 450 ms = 90
percent of the run; dt = 25 us only (the 1e-4 s branch is numerically invalid); no model/gain/friction/force-scale
change; no CPG, no PD, no preprogrammed action, no per-step pose correction.
Absolute force values and physical units remain NOT_ESTABLISHED: every force is a model-unit value read back with
mj_contactForce, not a calibrated newton value.
"""
from pathlib import Path
import hashlib
import json
import sys
import time
import zipfile

import numpy as np

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = Path('/home/sensen/Desktop/engine修复验收')
P1 = ROOT / 'outputs/experiment_platform/20261013_233237_7be93a1d/traces.npz'
STARTS = {
    'A_min_s_pose': DESK / 'S2-declared-cone-min-s-20261015T123603/best_candidate_pose.npz',
    'B_tarsus_only_pose': DESK / 'S2-seated-tarsus-only-20261015T114113/candidate_seated_pose.npz',
    'C_weight_variant_lowest_s': DESK / 'S2-ik-weight-variants-20261015T133034/pose_V_C_own.npz',
}
STATIC_S = {'A_min_s_pose': 0.9097466208040714, 'B_tarsus_only_pose': None, 'C_weight_variant_lowest_s': 0.9391964338719845}
B_STATIC_NOTE = ('statically INFEASIBLE at s = 1 on the declared cone: exact certificate gaps 0.42162561898294587 '
                 'native / 0.41982541733962897 union')
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
FLOOR_GEOM = 0
DT = 2.5e-5
WINDOW = 0.5
CMD_DT = 0.005
DIST_CAP = 4.0
THRESHOLDS = (1e-3, 1e-2)
PERSIST = 5
BASELINE_STEPS = 100
FRICTION_LOOKBACK = 20
# Reference values are READ BACK at run time from the delivered 0.5 s artifact (no literal is used as a result here);
# they serve only as the cross-check that this script reproduces the same run.
REFERENCE_REPORT = DESK / 'S2-dynamic-from-s1-pose-20261015T133124/report.json'


def read_reference():
    try:
        r = json.loads(REFERENCE_REPORT.read_text())['runs']
    except Exception as exc:
        return {}, 'reference report unreadable: %s: %s' % (type(exc).__name__, exc)
    # the delivered artifact labels start C as 'C_lowest_s_weight_variant'; map every script label to the report label
    REF_LABEL = {'A_min_s_pose': 'A_min_s_pose', 'B_tarsus_only_pose': 'B_tarsus_only_pose',
                 'C_weight_variant_lowest_s': 'C_lowest_s_weight_variant'}
    out = {}
    for label in STARTS:
        k = REF_LABEL.get(label, label) + '@qualified_25us'
        if k in r:
            out[label] = dict(first_nonfoot_step=r[k]['first_failure_step'],
                              first_nonfoot_ms=1000 * r[k]['first_failure_time_s'],
                              root_z_min=r[k]['root_z_min'])
    return out, str(REFERENCE_REPORT)


REPORTED, REPORTED_SOURCE = read_reference()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def first_persistent(flags, start=0, persist=PERSIST):
    run = 0
    for k in range(start, len(flags)):
        run = run + 1 if flags[k] else 0
        if run >= persist:
            return k - persist + 1
    return None


def main():
    argv = sys.argv[1:]
    outdir = None
    if '--outdir' in argv and argv.index('--outdir') + 1 < len(argv):
        outdir = Path(argv[argv.index('--outdir') + 1])
    drive_archive = None
    if '--drive-archive' in argv and argv.index('--drive-archive') + 1 < len(argv):
        drive_archive = Path(argv[argv.index('--drive-archive') + 1])
    starts = dict(STARTS)
    for a in argv:
        if a.startswith('--start='):
            lbl, path = a[len('--start='):].split('=', 1)
            starts[lbl] = Path(path)
    if '--starts' in argv and argv.index('--starts') + 1 < len(argv):
        keep = argv[argv.index('--starts') + 1].split(',')
        starts = {k: v for k, v in starts.items() if k in keep}
    out = outdir if outdir is not None else DESK / ('S2-instability-localization-' + time.strftime('%Y%m%dT%H%M%S'))
    if outdir is None:
        out.mkdir(exist_ok=False)
    else:
        out.mkdir(parents=True, exist_ok=True)
    report = dict(status='RUNNING', manual_research_not_product=True, standing=False, walking=False,
                  scope=('per-start localisation of the first body-floor contact and of the friction-cone utilisation in '
                         'the last %d steps before it, inside the 0.5 s declared-drive runs of A, B and C' % FRICTION_LOOKBACK),
                  declared=dict(dt_s=DT, window_s=WINDOW, starts={k: str(v) for k, v in starts.items()},
                                drive=('frozen connectome-derived commands normal_controls from the effective archive recorded in drive.archive_path '
                                       '(no external CPG, no PD, no preprogrammed action, no per-step pose correction)'),
                                state_writes='qpos, qvel and act once before the first step; only ctrl inside the loop',
                                cone=('declared pyramidal (axis-ray diamond) = non-negative hull of n +- mu*t1 and '
                                      'n +- mu*t2, i.e. {fn >= 0, |ft1| <= mu1*fn, |ft2| <= mu2*fn}'),
                                definitions=dict(T0='first step with any non-tarsus contact',
                                                 T1="that leg's tarsus contact count = 0 for 5 consecutive steps",
                                                 T2=("that leg's summed tarsus normal force below 1 percent of its "
                                                     "first-100-step mean for 5 consecutive steps"),
                                                 T3='|q_j - q_j(0)| >= threshold for 5 consecutive steps (hinge joints only)',
                                                 u1='|ft1| / (mu1 * fn)', u2='|ft2| / (mu2 * fn)',
                                                 cone_margin='mu*fn - max(|ft1|, |ft2|)', persistence_steps=PERSIST)),
                  absolute_force_values='NOT_ESTABLISHED',
                  reference_values_read_from=REPORTED_SOURCE, reference_values=REPORTED,
                  reference_policy=('every reference step / root_z_min in this report is READ BACK from the delivered 0.5 s '
                                    'artifact at run time; no literal is treated as a result'))
    inputs = [Path(__file__), (drive_archive if drive_archive is not None else P1)] + list(starts.values())
    report['input_hashes'] = {str(p): sha(p) for p in inputs if Path(p).exists()}
    try:
        import mujoco
        from engine import canonical_physics as cp
        m = cp.load_model()
        m.opt.timestep = DT
        report['canonical_xml'] = str(cp.CANONICAL_XML)
        report['canonical_xml_sha256'] = sha(cp.CANONICAL_XML)
        report['model'] = dict(nq=m.nq, nv=m.nv, na=m.na, nu=m.nu,
                               actuator_actlimited_all_false=bool(not np.any(m.actuator_actlimited)))
        gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
        jname = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ''
        foot = [g for g in range(m.ngeom) if 'Tarsus' in gname(g)]
        foot_by_leg = {L: [g for g in foot if gname(g).startswith(L)] for L in LEGS}
        mjJNT = mujoco.mjtJoint
        dof_names, dof_leg, dof_kind = [], [], []
        for j in range(m.njnt):
            nm, adr, typ = jname(j), int(m.jnt_qposadr[j]), int(m.jnt_type[j])
            leg = next((L for L in LEGS if nm.startswith('joint_' + L)), 'trunk')
            if typ == int(mjJNT.mjJNT_FREE):
                width, suffix = 7, ('x', 'y', 'z', 'qw', 'qx', 'qy', 'qz')
            elif typ == int(mjJNT.mjJNT_BALL):
                width, suffix = 4, ('q0', 'q1', 'q2', 'q3')
            else:
                width, suffix = 1, ('',)
            for i in range(width):
                dof_names.append(nm + suffix[i] if suffix[i] else nm)
                dof_leg.append(leg)
                dof_kind.append('free_translation' if (width == 7 and i < 3) else
                                ('free_quaternion' if width in (4, 7) else 'joint'))
            assert adr + width <= m.nq
        dof_names, dof_leg, dof_kind = np.asarray(dof_names), np.asarray(dof_leg), np.asarray(dof_kind)
        joint_idx = [j for j in range(m.nq) if dof_kind[j] == 'joint']
        n = int(round(WINDOW / DT))
        hold = int(round(CMD_DT / DT))
        _drivepath = drive_archive if drive_archive is not None else P1
        with np.load(_drivepath) as z:
            controls = z['normal_controls'].copy()
        report['drive'] = dict(shape=list(controls.shape), cadence_s=CMD_DT, sha256=sha(_drivepath),
                               archive_path=str(_drivepath),
                               index_saturates_from_step=int(controls.shape[0] * hold),
                               command_held_constant_from_s=float(controls.shape[0] * CMD_DT),
                               fraction_of_window_with_a_constant_command=float(1 - controls.shape[0] * CMD_DT / WINDOW))

        def run_start(label, path):
            with np.load(path) as z:
                q0 = z['projected_seated_qpos'].copy()
            d = mujoco.MjData(m)
            d.qpos[:] = q0
            d.qvel[:] = 0.0
            d.act[:] = 0.0
            d.ctrl[:] = controls[0]
            mujoco.mj_forward(m, d)
            qpos_ref = np.asarray(d.qpos, dtype=float).copy()
            time_s = np.zeros(n); root_z = np.zeros(n); qacc = np.zeros(n)
            cnt = np.zeros((n, len(LEGS))); force = np.zeros((n, len(LEGS)))
            lowest = np.zeros((n, len(LEGS))); nonfoot = np.zeros(n, dtype=int)
            qpos_hist = np.zeros((n, m.nq)); tilt = np.zeros(n)
            live_buf = []
            live_frozen = None
            whole_first = None
            whole_ge95_steps = 0
            hull_margin = np.full(n, np.nan)
            first_neg_margin = None
            for k in range(n):
                d.ctrl[:] = controls[min(k // hold, controls.shape[0] - 1)]
                mujoco.mj_step(m, d)
                time_s[k] = float(d.time)
                # --- in-run capture: live contact forces (qvel/act as they are during the run) and CoM-hull margin
                _f6 = np.zeros(6)
                _step_rows = []
                for _ci, _c in enumerate(d.contact):
                    if _c.efc_address < 0:
                        continue
                    _n1, _n2 = gname(int(_c.geom1)), gname(int(_c.geom2))
                    if not ('Tarsus' in _n1 or 'Tarsus' in _n2):
                        continue
                    mujoco.mj_contactForce(m, d, _ci, _f6)
                    _fric = np.asarray(_c.friction)[:3]
                    _leg = next((L for L in LEGS if _n1.startswith(L) or _n2.startswith(L)), 'trunk/head')
                    _fn, _ft1, _ft2 = float(_f6[0]), float(_f6[1]), float(_f6[2])
                    _u1 = abs(_ft1) / (float(_fric[0]) * _fn) if _fn > 0 else None
                    _u2 = abs(_ft2) / (float(_fric[1]) * _fn) if _fn > 0 else None
                    _step_rows.append(dict(step=int(k), time_step_convention_s=float(k * DT), leg=_leg,
                                           geom=tuple(sorted((_n1, _n2))), fn=_fn, ft1=_ft1, ft2=_ft2,
                                           mu1=float(_fric[0]), mu2=float(_fric[1]),
                                           u1=_u1, u2=_u2,
                                           u_diamond=(None if (_u1 is None or _u2 is None) else _u1 + _u2),
                                           u_square=(None if (_u1 is None or _u2 is None) else max(_u1, _u2))))
                live_buf.append(_step_rows)
                if len(live_buf) > 40:
                    live_buf.pop(0)
                if whole_first is None:
                    for _r in _step_rows:
                        if _r['u_diamond'] is not None and _r['u_diamond'] >= 1.0 - 1e-9:
                            whole_first = dict(step=int(k), time_ms=float(k * DT * 1000),
                                               geom=list(_r['geom']), leg=_r['leg'], fn=_r['fn'], ft1=_r['ft1'], ft2=_r['ft2'],
                                               u1=_r['u1'], u2=_r['u2'], u_diamond=_r['u_diamond'])
                            break
                if whole_first is not None and _step_rows:
                    _n_ge = sum(1 for _r in _step_rows if _r['u_diamond'] is not None and _r['u_diamond'] >= 0.95)
                    whole_ge95_steps += (1 if _n_ge > 0 else 0)
                _nf_now = sum(1 for _c in d.contact if _c.efc_address >= 0
                              and not ('Tarsus' in (gname(int(_c.geom1)) + gname(int(_c.geom2)))))
                if _nf_now > 0 and live_frozen is None:
                    live_frozen = list(live_buf)
                root_z[k] = float(d.qpos[2]); qpos_hist[k] = d.qpos
                qacc[k] = float(np.max(np.abs(d.qacc)))
                w, x, y, zq = d.qpos[3:7]
                up_z = 1.0 - 2.0 * (x * x + y * y)
                tilt[k] = float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0))))
                mujoco.mj_kinematics(m, d)
                ft = np.zeros(6)
                for li, L in enumerate(LEGS):
                    lowest[k, li] = float(min(mujoco.mj_geomDistance(m, d, FLOOR_GEOM, g, DIST_CAP, ft)
                                              for g in foot_by_leg[L]))
                f6 = np.zeros(6)
                for ci, c in enumerate(d.contact):
                    if c.efc_address < 0:
                        continue
                    n1, n2 = gname(int(c.geom1)), gname(int(c.geom2))
                    if 'Tarsus' in n1 or 'Tarsus' in n2:
                        mujoco.mj_contactForce(m, d, ci, f6)
                        for li, L in enumerate(LEGS):
                            if (n1.startswith(L) and 'Tarsus' in n1) or (n2.startswith(L) and 'Tarsus' in n2):
                                cnt[k, li] += 1; force[k, li] += float(f6[0])
                    else:
                        nonfoot[k] += 1
            t0 = next((k for k in range(n) if nonfoot[k] > 0), None)
            dev = np.abs(qpos_hist - qpos_ref[None, :])
            baseline_cnt = cnt[:BASELINE_STEPS].mean(axis=0)
            baseline_force = force[:BASELINE_STEPS].mean(axis=0)
            t1 = {L: first_persistent(cnt[:, li] == 0, 0) for li, L in enumerate(LEGS)}
            carrying = [L for li, L in enumerate(LEGS) if t0 is not None and cnt[t0, li] > 0]
            post = []
            if t0 is not None:
                for li, L in enumerate(LEGS):
                    if cnt[t0, li] == 0 or baseline_cnt[li] <= 0:
                        continue
                    v = first_persistent(cnt[:, li] == 0, t0 + 1)
                    if v is not None:
                        post.append(('T1_loses_all_tarsus_contact_' + L, v))
                    v2 = first_persistent(force[:, li] < 0.01 * baseline_force[li], t0 + 1)
                    if v2 is not None:
                        post.append(('T2_tarsus_force_below_1pct_' + L, v2))
                for j in joint_idx:
                    v = first_persistent(dev[:, j] >= 1e-2, t0 + 1)
                    if v is not None:
                        post.append(('T3_joint_deviation_1e-2_' + str(dof_names[j]), v))
            post.sort(key=lambda x: x[1])

            def tables(step):
                d2 = mujoco.MjData(m)
                d2.qpos[:] = qpos_hist[step]; d2.qvel[:] = 0.0; d2.act[:] = 0.0
                d2.ctrl[:] = controls[min(step // hold, controls.shape[0] - 1)]
                mujoco.mj_forward(m, d2)
                f6 = np.zeros(6); tbl = []
                for ci, c in enumerate(d2.contact):
                    if c.efc_address < 0:
                        continue
                    mujoco.mj_contactForce(m, d2, ci, f6)
                    n1, n2 = gname(int(c.geom1)), gname(int(c.geom2))
                    leg = next((L for L in LEGS if n1.startswith(L) or n2.startswith(L)), 'trunk/head')
                    tbl.append(dict(geom1=n1, geom2=n2, part=leg,
                                    kind=('tarsus' if ('Tarsus' in n1 or 'Tarsus' in n2) else 'non-foot'),
                                    dist_mm=float(c.dist), pos_mm=[float(x) for x in c.pos],
                                    normal_force_model_units=float(f6[0]),
                                    tangent1_model_units=float(f6[1]), tangent2_model_units=float(f6[2]),
                                    friction=[float(x) for x in np.asarray(c.friction)[:3]]))
                tbl.sort(key=lambda r: (r['kind'] != 'non-foot', -abs(r['normal_force_model_units'])))
                legs_in = sorted({r['part'] for r in tbl if r['kind'] == 'tarsus' and r['part'] in LEGS})
                devi = sorted((dict(joint=str(dof_names[j]), leg=str(dof_leg[j]), q=float(qpos_hist[step, j]),
                                    dq_rad=float(dev[step, j]), dq_deg=float(np.degrees(dev[step, j])))
                               for j in joint_idx), key=lambda r: -r['dq_rad'])
                return dict(step=int(step), time_step_convention_s=float(step * DT), contact_table=tbl,
                            contact_summary=dict(n_contacts=len(tbl),
                                                 n_nonfoot=sum(1 for r in tbl if r['kind'] == 'non-foot'),
                                                 n_tarsus=sum(1 for r in tbl if r['kind'] == 'tarsus'),
                                                 legs_with_tarsus_contact=legs_in,
                                                 legs_without_tarsus_contact=[L for L in LEGS if L not in legs_in]),
                            joint_deviation_table=devi,
                            joint_deviation_ranking=[dict(joint=r['joint'], leg=r['leg'], dq_deg=round(r['dq_deg'], 4))
                                                     for r in devi[:8]])

            # ---- friction-cone utilisation in the last FRICTION_LOOKBACK steps before T0 ----
            friction = dict(lookback_steps=FRICTION_LOOKBACK, per_step=[], max_u1=None, max_u2=None,
                            max_cone_utilisation=None, any_contact_at_or_above_0_95=False,
                            saturating_contacts=[], note=('u1 = |ft1|/(mu1*fn), u2 = |ft2|/(mu2*fn) against the declared '
                                                          'axis-ray diamond; a contact saturates when max(u1,u2) reaches 1; '
                                                          'model units, NOT_ESTABLISHED as physical forces'))
            if t0 is not None:
                k0 = max(0, t0 - FRICTION_LOOKBACK)
                d3 = mujoco.MjData(m)
                for k in range(k0, t0):
                    d3.qpos[:] = qpos_hist[k]; d3.qvel[:] = 0.0; d3.act[:] = 0.0
                    d3.ctrl[:] = controls[min(k // hold, controls.shape[0] - 1)]
                    mujoco.mj_forward(m, d3)
                    f6 = np.zeros(6); rows = []
                    for ci, c in enumerate(d3.contact):
                        if c.efc_address < 0:
                            continue
                        n1, n2 = gname(int(c.geom1)), gname(int(c.geom2))
                        if not ('Tarsus' in n1 or 'Tarsus' in n2):
                            continue
                        mujoco.mj_contactForce(m, d3, ci, f6)
                        fr = np.asarray(c.friction)[:3]
                        fn, ft1, ft2 = float(f6[0]), float(f6[1]), float(f6[2])
                        mu1, mu2 = float(fr[0]), float(fr[1])
                        u1 = abs(ft1) / (mu1 * fn) if fn > 0 else None
                        u2 = abs(ft2) / (mu2 * fn) if fn > 0 else None
                        leg = next((L for L in LEGS if n1.startswith(L) or n2.startswith(L)), 'trunk/head')
                        rows.append(dict(step=int(k), time_step_convention_s=float(k * DT), leg=leg,
                                         geom=tuple(sorted((n1, n2))), fn=fn, ft1=ft1, ft2=ft2, mu1=mu1, mu2=mu2,
                                         u1=u1, u2=u2, max_u=(None if (u1 is None or u2 is None) else max(u1, u2)),
                                         cone_margin=(None if (u1 is None or u2 is None) else mu1 * fn - max(abs(ft1), abs(ft2)))))
                    us = [r['max_u'] for r in rows if r['max_u'] is not None]
                    friction['per_step'].append(dict(step=int(k), time_step_convention_s=float(k * DT),
                                                     n_tarsus_contacts=len(rows),
                                                     sum_fn=float(sum(r['fn'] for r in rows)),
                                                     max_u=float(max(us)) if us else None,
                                                     per_leg_max_u={L: (max([r['max_u'] for r in rows if r['leg'] == L and r['max_u'] is not None],
                                                                            default=None)) for L in LEGS},
                                                     per_leg_fn={L: float(sum(r['fn'] for r in rows if r['leg'] == L)) for L in LEGS},
                                                     contacts=rows))
                live_rows = [r for sl in (live_frozen or []) for r in sl if r['u_diamond'] is not None]
                friction['whole_window_first_saturation'] = whole_first
                friction['whole_window_n_steps_with_u_diamond_ge_0_95'] = int(whole_ge95_steps)
                if live_rows:
                    friction['in_run'] = dict(
                        metric='u_diamond = u1 + u2 against the DECLARED axis-ray diamond, using the LIVE in-run forces',
                        max_u_diamond=float(max(r['u_diamond'] for r in live_rows)),
                        max_u_square=float(max(r['u_square'] for r in live_rows)),
                        n_contacts=len(live_rows),
                        n_diamond_ge_0_95=int(sum(1 for r in live_rows if r['u_diamond'] >= 0.95)),
                        n_diamond_ge_0_99=int(sum(1 for r in live_rows if r['u_diamond'] >= 0.99)),
                        n_diamond_at_1=int(sum(1 for r in live_rows if r['u_diamond'] >= 1.0 - 1e-9)),
                        capture_window_steps=FRICTION_LOOKBACK,
                        first_step_with_u_diamond_ge_1_within_the_last_40_steps=(min([r['step'] for r in live_rows if r['u_diamond'] >= 1.0 - 1e-9],
                                                                                    default=None)),
                        rows=live_rows)
                allr = [r for s in friction['per_step'] for r in s['contacts'] if r['max_u'] is not None]
                friction['quasi_static'] = dict(
                    metric='u_diamond = u1 + u2, forces recomputed at the stored qpos with qvel = 0 and act = 0 (quasi-static convention)',
                    max_u_diamond=float(max(r['u1'] + r['u2'] for r in allr)) if allr else None,
                    n_diamond_ge_0_95=int(sum(1 for r in allr if r['u1'] + r['u2'] >= 0.95)) if allr else None,
                    n_diamond_at_1=int(sum(1 for r in allr if r['u1'] + r['u2'] >= 1.0 - 1e-9)) if allr else None)
                if allr:
                    friction['metric']='u_diamond = u1 + u2 (declared axis-ray diamond); u_square = max(u1,u2) reported only as a secondary reading'
                    friction['max_diamond_utilisation_quasi_static']=float(max(r['u1'] + r['u2'] for r in allr))
                    friction['n_diamond_ge_0_95_quasi_static']=int(sum(1 for r in allr if r['u1'] + r['u2'] >= 0.95))
                    friction['n_diamond_at_1_quasi_static']=int(sum(1 for r in allr if r['u1'] + r['u2'] >= 1.0 - 1e-9))
                if allr:
                    friction['max_u1'] = float(max(r['u1'] for r in allr))
                    friction['max_u2'] = float(max(r['u2'] for r in allr))
                    friction['max_cone_utilisation'] = float(max(r['max_u'] for r in allr))
                    friction['any_contact_at_or_above_0_95'] = bool(any(r['max_u'] >= 0.95 for r in allr))
                    friction['saturating_contacts'] = [r for r in allr if r['max_u'] >= 0.9][:20]
                    friction['max_u_per_leg'] = {L: (max([r['max_u'] for r in allr if r['leg'] == L], default=None))
                                                 for L in LEGS}
                    friction['t0_sum_fn'] = float(sum(r['fn'] for s in friction['per_step'] if s['step'] == t0 - 1
                                                      for r in s['contacts']))
                    friction['fn_trend_last_20_steps'] = [s['sum_fn'] for s in friction['per_step']]

            # ---- overturning check at T0: is the whole-body CoM xy inside the convex hull of the tarsus contacts? ----
            overturn = dict(note=('CoM projection vs the convex hull of the tarsus contact points at T0; a body whose CoM '
                                  'projects OUTSIDE that hull cannot be held by foot forces alone regardless of friction, so '
                                  'this separates a moment/overturning failure from a friction-slip failure'),
                            com_xy=None, inside_hull=None, distance_to_hull_margin_mm=None, hull_vertices=[])
            if t0 is not None:
                d4 = mujoco.MjData(m)
                d4.qpos[:] = qpos_hist[t0]; d4.qvel[:] = 0.0; d4.act[:] = 0.0
                d4.ctrl[:] = controls[min(t0 // hold, controls.shape[0] - 1)]
                mujoco.mj_forward(m, d4)
                mujoco.mj_comPos(m, d4)
                com = np.asarray(d4.subtree_com[0]).copy()
                pts = []
                for ci, c in enumerate(d4.contact):
                    if c.efc_address < 0:
                        continue
                    n1, n2 = gname(int(c.geom1)), gname(int(c.geom2))
                    if 'Tarsus' in n1 or 'Tarsus' in n2:
                        pts.append([float(c.pos[0]), float(c.pos[1])])
                if len(pts) >= 3:
                    P = np.asarray(pts)

                    def hull(points):
                        pts2 = sorted(map(tuple, points))
                        def half(pts3):
                            h = []
                            for pnt in pts3:
                                while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (pnt[1] - h[-2][1])
                                                       - (h[-1][1] - h[-2][1]) * (pnt[0] - h[-2][0])) <= 0:
                                    h.pop()
                                h.append(pnt)
                            return h
                        lo = half(pts2); up = half(pts2[::-1])
                        return lo[:-1] + up[:-1]
                    H = hull(P)
                    inside = True
                    # signed distance to each hull edge (positive = inside for a counter-clockwise hull)
                    dists = []
                    for i in range(len(H)):
                        a = np.asarray(H[i]); b = np.asarray(H[(i + 1) % len(H)])
                        e = b - a
                        nrm = np.asarray([-e[1], e[0]])
                        nrm = nrm / (np.linalg.norm(nrm) + 1e-12)
                        dists.append(float(np.dot(com[:2] - a, nrm)))
                    inside = bool(min(dists) > 0.0)
                    overturn.update(com_xy=[float(com[0]), float(com[1])],
                                    inside_hull=inside,
                                    distance_to_hull_margin_mm=float(min(dists)),
                                    hull_vertices=[[float(x), float(y)] for x, y in H],
                                    n_tarsus_contact_points=len(pts))
                else:
                    overturn.update(com_xy=[float(com[0]), float(com[1])],
                                    inside_hull=False,
                                    distance_to_hull_margin_mm=None,
                                    n_tarsus_contact_points=len(pts),
                                    note_extra='fewer than 3 tarsus contact points: no support polygon exists at all')
            hull_timing = dict(first_step_with_negative_com_hull_margin=None,
                               status=('per-step CoM-hull margin is NOT reported: the provisional ordering-by-contact-index version was wrong and was '
                                       'removed; only the verified T0 hull check below (monotone-chain hull) is reported'))
            return dict(label=label, npz=str(path), static_s=STATIC_S.get(label), com_hull_timing=hull_timing,
                        overturning_at_T0=overturn, static_note=(B_STATIC_NOTE if STATIC_S.get(label) is None else None),
                        T0_step=t0, T0_time_step_convention_s=(None if t0 is None else float(t0 * DT)),
                        T0_matches_0p5s_report=(REPORTED.get(label, {}).get('first_nonfoot_step') == t0),
                        T0_reference_from=REPORTED_SOURCE,
                        T0_reference_values_read_not_hardcoded=True,
                        carrying_legs_at_T0=carrying,
                        legs_without_tarsus_contact_at_T0=[L for L in LEGS if L not in carrying],
                        support_loss_after_T0=[dict(cause=c, step=int(v), time_s=float(v * DT)) for c, v in post[:12]],
                        first_support_loss_after_T0=(None if not post else dict(cause=post[0][0], step=int(post[0][1]),
                                                                               time_s=float(post[0][1] * DT))),
                        first_joint_above_1e_minus_3_rad_overall=(min(((str(dof_names[j]), first_persistent(dev[:, j] >= 1e-3, 0))
                                                                      for j in joint_idx
                                                                      if first_persistent(dev[:, j] >= 1e-3, 0) is not None),
                                                                     key=lambda x: x[1], default=None)),
                        first_joint_above_1e_minus_2_rad_overall=(min(((str(dof_names[j]), first_persistent(dev[:, j] >= 1e-2, 0))
                                                                      for j in joint_idx
                                                                      if first_persistent(dev[:, j] >= 1e-2, 0) is not None),
                                                                     key=lambda x: x[1], default=None)),
                        root_z_min=float(root_z.min()), root_z_min_step=int(np.argmin(root_z)),
                        root_z_min_time_step_convention_s=float(np.argmin(root_z) * DT),
                        root_z_final=float(root_z[-1]), root_z_at_T0=(None if t0 is None else float(root_z[t0])),
                        root_z_min_matches_0p5s_report=(None if label not in REPORTED else
                                                        bool(float(root_z.min()) == float(REPORTED[label]['root_z_min']))),
                        root_z_min_bit_identical_to_single_run=(None if label not in REPORTED else
                                                                bool(float(root_z.min()) == float(REPORTED[label]['root_z_min']))),
                        flipped=bool(tilt.max() > 90.0), tilt_max_deg=float(tilt.max()),
                        six_leg_steps=int((cnt > 0).sum(axis=1).__eq__(len(LEGS)).sum()), steps=int(n),
                        six_leg_frac=float((cnt > 0).sum(axis=1).__eq__(len(LEGS)).mean()),
                        legs_in_contact_min=int((cnt > 0).sum(axis=1).min()),
                        tarsus_contacts_min=int(cnt.sum(axis=1).min()),
                        nonfoot_contacts_max=int(nonfoot.max()),
                        energy=None, qacc_final=float(qacc[-1]), qacc_max=float(qacc.max()),
                        friction_cone_utilisation=friction,
                        table_at_T0=(None if t0 is None else tables(t0)),
                        table_at_first_support_loss_after_T0=(None if not post else tables(post[0][1])),
                        series=dict(time=time_s, root_z=root_z, cnt=cnt, force=force, lowest=lowest, nonfoot=nonfoot,
                                    tilt=tilt, dev=dev.astype(np.float32)))

        per_start = {}
        series_store = {}
        for label, path in starts.items():
            res = run_start(label, path)
            series_store[label] = res.pop('series')
            per_start[label] = res
            d = mujoco.MjData(m)
            with np.load(path) as z:
                d.qpos[:] = z['projected_seated_qpos'].copy()
            d.qvel[:] = 0.0; d.act[:] = 0.0; d.ctrl[:] = controls[0]
            mujoco.mj_forward(m, d)
            e0 = None
            try:
                mujoco.mj_energyPos(m, d); mujoco.mj_energyVel(m, d); e0 = float(d.energy[0] + d.energy[1])
            except Exception:
                pass
            per_start[label]['energy_first_from_initial_pose'] = e0
            print('%-28s T0=%s (%.4f ms) root_z_min=%.9f flipped=%s six_leg=%.4f first_loss=%s max_u=%s' % (
                label, res['T0_step'], 1000 * (res['T0_time_step_convention_s'] or 0), res['root_z_min'], res['flipped'],
                res['six_leg_frac'],
                None if not res['first_support_loss_after_T0'] else res['first_support_loss_after_T0']['cause'],
                res['friction_cone_utilisation'].get('max_cone_utilisation')), flush=True)
        report['per_start'] = per_start
        report['declared_calls'] = dict(mj_step=int(n * len(starts)), model_loads=1)
        np.savez_compressed(out / 'localization_series_abc.npz',
                            **{lab + '__' + k: np.asarray(v) for lab, ser in series_store.items()
                               for k, v in ser.items() if k != 'dev'},
                            **{lab + '__dev' : ser['dev'] for lab, ser in series_store.items()})
        (out / 'friction_cone_utilization.json').write_text(json.dumps(
            {lab: per_start[lab]['friction_cone_utilisation'] for lab in per_start}, indent=2,
            allow_nan=False, default=lambda o: o.tolist()) + '\n')
        (out / 'merged_abc_tables.json').write_text(json.dumps(
            {lab: dict(static_s=per_start[lab]['static_s'], T0_step=per_start[lab]['T0_step'],
                       T0_ms=(None if per_start[lab]['T0_time_step_convention_s'] is None else
                              1000 * per_start[lab]['T0_time_step_convention_s']),
                       first_support_loss_after_T0=per_start[lab]['first_support_loss_after_T0'],
                       root_z_min=per_start[lab]['root_z_min'], flipped=per_start[lab]['flipped'],
                       six_leg_frac=per_start[lab]['six_leg_frac'], qacc_final=per_start[lab]['qacc_final'],
                       contact_table_at_T0=None if per_start[lab]['table_at_T0'] is None else
                       per_start[lab]['table_at_T0']['contact_table'],
                       joint_deviation_table_at_T0=None if per_start[lab]['table_at_T0'] is None else
                       per_start[lab]['table_at_T0']['joint_deviation_table'][:12])
                for lab in per_start}, indent=2, allow_nan=False, default=lambda o: o.tolist()) + '\n')

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        labels = list(starts)
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        for lab in labels:
            r0 = per_start[lab]; ser = series_store[lab]
            if r0['T0_step'] is not None:
                axes[0, 0].axvline(1000 * r0['T0_time_step_convention_s'], lw=0.8)
            axes[0, 0].plot(1000 * ser['time'], ser['root_z'], lw=1, label=lab)
            axes[0, 1].plot(1000 * ser['time'], ser['tilt'], lw=1, label=lab)
            axes[1, 0].plot(1000 * ser['time'], ser['nonfoot'], lw=1, label=lab)
            fu = r0['friction_cone_utilisation']
            if fu['per_step']:
                axes[1, 1].plot([1000 * s['time_step_convention_s'] for s in fu['per_step']],
                                [(None if not s['contacts'] else
                                  max(r['u1'] + r['u2'] for r in s['contacts'] if r['u1'] is not None)) for s in fu['per_step']],
                                marker='o', ms=2, lw=1, label=lab + ' u_diamond = u1+u2 (declared diamond)')
                axes[1, 1].plot([1000 * s['time_step_convention_s'] for s in fu['per_step']],
                                [s['max_u'] for s in fu['per_step']], marker='.', ms=2, lw=0.8, ls='--',
                                label=lab + ' u_square = max(u1,u2) (secondary)')
                axes[1, 1].plot([1000 * s['time_step_convention_s'] for s in fu['per_step']],
                                [(s['sum_fn'] / max(s['sum_fn'], 1e-12)) for s in fu['per_step']], ls=':', lw=0.8,
                                label=lab + ' normal load (scaled)')
        axes[0, 0].set_title('root z (mm); thin lines = first body-floor contact')
        axes[0, 1].axhline(90, color='r', ls='--', lw=0.8); axes[0, 1].set_title('thorax tilt (deg)')
        axes[1, 0].set_title('non-foot contacts')
        axes[1, 1].axhline(1.0, color='r', ls='--', lw=0.8, label='declared diamond boundary u1+u2 = 1')
        axes[1, 1].set_title('friction-cone utilisation in the %d steps before T0: u1+u2 on the DECLARED axis-ray diamond '
                             '(solid) vs max(u1,u2) square (dashed)' % FRICTION_LOOKBACK)
        for ax in axes.ravel():
            ax.set_xlabel('time (ms)'); ax.legend(fontsize=6)
        fig.suptitle('A / B / C 0.5 s declared-drive runs (declared 100 x 90 frame archive at 5 ms cadence, whole window covered, '
                     '25 us valid branch); friction cone = DECLARED axis-ray diamond (u1+u2) - manual research, NOT standing', fontsize=8)
        fig.tight_layout()
        fig.savefig(out / 'abc-localization-friction.png', dpi=130)
        plt.close(fig)
        report['status'] = 'PASS_ABC_LOCALIZATION_AND_FRICTION_REPORTED'
    except Exception as exc:
        import traceback
        report.update(status='FAIL_STOP', error=type(exc).__name__ + ': ' + str(exc),
                      traceback=traceback.format_exc()[-1500:])
    finally:
        with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(Path(__file__), Path(__file__).name)
        (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False,
                                                   default=lambda o: o.tolist()) + '\n')
        (out / 'delivery-hashes.json').write_text(json.dumps(
            {f.name: sha(f) for f in sorted(out.iterdir())
             if f.is_file() and f.name != 'delivery-hashes.json'}, indent=2) + '\n')
    print(out)
    print(report['status'])
    if report['status'].startswith('PASS'):
        for lab, r0 in report['per_start'].items():
            print(lab, 'T0', r0['T0_step'], 'ms', None if r0['T0_time_step_convention_s'] is None else 1000 * r0['T0_time_step_convention_s'],
                  'root_z_min', r0['root_z_min'], 'matches', r0['root_z_min_matches_0p5s_report'],
                  'six_leg', round(r0['six_leg_frac'], 4), 'qacc_final', round(r0['qacc_final'], 3),
                  'max_cone_u', r0['friction_cone_utilisation'].get('max_cone_utilisation'),
                  'saturated>=0.95', r0['friction_cone_utilisation'].get('any_contact_at_or_above_0_95'))
    else:
        print(report.get('error', ''))
    return 0 if report['status'].startswith('PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
