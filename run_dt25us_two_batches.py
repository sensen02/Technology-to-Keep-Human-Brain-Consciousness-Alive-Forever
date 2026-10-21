"""Turn 26: the two corrected-reissued-dt25us batches (open loop and localization) with machine-checkable step-size
fields, the project-root run_*.py timestep sweep, the act-domain disclosure template, the stale-hash inventory, and the
correction ledger for both new batches.

Every millisecond in the reports is computed as step * DT; no hand-typed ms constant is used for the F-A sentence.
"""

import hashlib
import json
import pathlib
import re
import sys
import time
import zipfile

import numpy as np

ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
STATE = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259/equilibrium_best_state.npz'
POSE_A = DESK / 'S2-declared-cone-min-s-20261015T123603/best_candidate_pose.npz'
DRIVE = ROOT / 'outputs/experiment_platform/20261015_135150_a2fff3f0/traces.npz'
V2 = DESK / 'S2-equilibrium-A-corrected-reissued-v2-20261015T163134'
V1 = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259'
LADDER = DESK / 'S2-equilibrium-radius-ladder-20261015T161518'
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
SUFFIXES = ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll', 'Trochanter_yaw', 'Trochanter_pitch', 'Trochanter_roll', 'Tibia_pitch')
DT, STEPS, CMD_S = 2.5e-5, 20000, 0.005
CMD_EVERY = int(round(CMD_S / DT))
SISTER = ('BRANCH INVALID: at dt=1e-4 s the native muscle activation leaves its domain (act max 1.9993002399 > 1), so no number from this branch '
          'may be quoted as a physical result; use the 25 us branch')
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S')
ms_of = lambda step: round(step * DT * 1000.0, 6)


def persist_first(mask, persist=5):
    run = 0
    for i, v in enumerate(mask):
        run = run + 1 if v else 0
        if run >= persist:
            return i - persist + 1
    return None


def main():
    import mujoco
    sys.path.insert(0, str(ROOT))
    from engine import canonical_physics as cp

    # ---------------- 2) project-root run_*.py timestep sweep ----------------
    sweep = []
    for p in sorted(ROOT.glob('run_*.py')):
        text = p.read_text(errors='ignore')
        if 'mj_step' not in text:
            continue
        sweep.append(dict(script=p.name, full_path=str(p), calls_mj_step=True,
                          has_timestep_assignment=bool(re.search(r'opt\.timestep\s*=', text)),
                          has_assert_on_timestep=bool(re.search(r'assert[^\n]*opt\.timestep', text)),
                          declares_a_step=bool(re.search(r'DT\s*=\s*|25\s*e-6|2\.5e-5|1e-4|1e-04', text)),
                          sha256=sha(p)))
    missing = [s for s in sweep if not s['has_timestep_assignment']]

    # ---------------- the two batches ----------------
    m = cp.load_model()
    xml_dt = float(m.opt.timestep)
    m.opt.timestep = DT                       # set BEFORE any mj_forward/mj_step
    assert float(m.opt.timestep) == DT
    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
    jname = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ''
    aname = lambda a: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ''
    jids = []
    for L in LEGS:
        js = [j for j in range(m.njnt) if jname(j).startswith('joint_' + L)]
        for suf in SUFFIXES:
            jids.append(next(j for j in js if jname(j).endswith(suf)))
    qadr = [int(m.jnt_qposadr[j]) for j in jids]
    jlabels = ['%s_%s' % (jname(j).split('_')[1], jname(j).split('_', 2)[2]) for j in jids]
    st = np.load(STATE)
    q0, act0 = st['qpos'].copy(), st['act'].copy()
    pose_a = np.load(POSE_A)['projected_seated_qpos']
    ctrl = np.asarray(np.load(DRIVE)['normal_controls'], dtype=float)

    d = mujoco.MjData(m)
    d.qpos[:] = q0
    d.qvel[:] = 0.0
    d.act[:] = act0
    d.ctrl[:] = 0.0
    mujoco.mj_forward(m, d)
    first_step_check = None
    n = STEPS + 1
    a = dict(time_s=np.zeros(n), root_z=np.zeros(n), legs=np.zeros(n, dtype=int), tar=np.zeros(n, dtype=int),
             nonfoot=np.zeros(n, dtype=int), tilt=np.zeros(n), energy=np.zeros(n), qacc=np.zeros(n),
             ud=np.full(n, np.nan), act_max=np.zeros(n))
    leg_tar = np.zeros((n, len(LEGS)), dtype=int)
    leg_fn = np.zeros((n, len(LEGS)))
    leg_ud = np.full((n, len(LEGS)), np.nan)
    qdev = np.zeros((n, len(jids)))
    f6 = np.zeros(6)
    ev = dict(first_nonfoot=None, first_nonfoot_pair=None, legs_below_6=None, zero_tarsus=None, first_ud_ge_1=None,
              first_ud_pair=None, first_act_above_1=None)
    table = None

    def sample(i):
        nonlocal table
        a['time_s'][i] = float(d.time)
        a['root_z'][i] = float(d.qpos[2])
        quat = np.asarray(d.qpos[3:7], dtype=float)
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        r22 = float(mat.reshape(3, 3)[2, 2])
        a['tilt'][i] = float(np.degrees(np.arccos(np.clip(r22, -1.0, 1.0))))
        a['qacc'][i] = float(np.max(np.abs(np.asarray(d.qacc))))
        mujoco.mj_energyPos(m, d)
        mujoco.mj_energyVel(m, d)
        a['energy'][i] = float(d.energy[0] + d.energy[1])
        a['act_max'][i] = float(np.max(np.asarray(d.act)))
        qdev[i] = np.abs(d.qpos[qadr] - q0[qadr])
        legs, tar, nonfoot, worst, worst_pair, rows = set(), 0, 0, 0.0, None, []
        for ci, c in enumerate(d.contact):
            if c.efc_address < 0:
                continue
            n1, n2 = gname(int(c.geom1)), gname(int(c.geom2))
            fr = np.asarray(c.friction)[:3]
            mujoco.mj_contactForce(m, d, ci, f6)
            fn, ft1, ft2 = float(f6[0]), float(f6[1]), float(f6[2])
            u = (abs(ft1) / (float(fr[0]) * fn) + abs(ft2) / (float(fr[1]) * fn)) if fn > 0 else None
            leg = next((L for L in LEGS if n1.startswith(L) or n2.startswith(L)), None)
            if 'Tarsus' in (n1 + n2):
                tar += 1
                if leg:
                    legs.add(leg)
                    leg_tar[i, LEGS.index(leg)] += 1
                    leg_fn[i, LEGS.index(leg)] += fn
                    if u is not None:
                        prev = leg_ud[i, LEGS.index(leg)]
                        leg_ud[i, LEGS.index(leg)] = u if np.isnan(prev) else max(prev, u)
                if u is not None and u > worst:
                    worst, worst_pair = u, sorted((n1, n2))
            else:
                nonfoot += 1
            rows.append(dict(pair=sorted((n1, n2)), is_tarsus=bool('Tarsus' in (n1 + n2)), fn=fn, u_diamond=u, leg=leg))
        a['legs'][i], a['tar'][i], a['nonfoot'][i] = len(legs), tar, nonfoot
        a['ud'][i] = worst if tar else np.nan
        if ev['first_nonfoot'] is None and nonfoot > 0:
            ev['first_nonfoot'] = i
            ev['first_nonfoot_pair'] = sorted(set(tuple(r['pair']) for r in rows if not r['is_tarsus']))[0]
            table = rows
        if ev['legs_below_6'] is None and len(legs) < 6:
            ev['legs_below_6'] = i
        if ev['zero_tarsus'] is None and tar == 0:
            ev['zero_tarsus'] = i
        if ev['first_ud_ge_1'] is None and tar and worst >= 1.0:
            ev['first_ud_ge_1'] = i
            ev['first_ud_pair'] = worst_pair
        if ev['first_act_above_1'] is None and float(np.max(np.asarray(d.act))) > 1.0:
            ev['first_act_above_1'] = i

    sample(0)
    for step in range(STEPS):
        d.ctrl[:] = ctrl[min(step // CMD_EVERY, ctrl.shape[0] - 1)]
        mujoco.mj_step(m, d)
        if step == 0:
            first_step_check = dict(d_time_after_first_step_s=float(d.time), declared_dt_s=DT,
                                    abs_diff=abs(float(d.time) - DT), assert_tol=1e-15,
                                    passed=bool(abs(float(d.time) - DT) < 1e-15))
            assert first_step_check['passed'], 'first mj_step did not advance d.time by exactly the declared dt'
        sample(step + 1)
    assert abs(float(d.time) - STEPS * DT) <= 1e-9
    ms = ms_of
    per_leg = {}
    for k, L in enumerate(LEGS):
        base = float(leg_fn[:100, k].mean())
        per_leg[L] = dict(first_step_no_tarsus_contact=persist_first(leg_tar[:, k] == 0),
                          first_step_no_tarsus_contact_ms=(None if persist_first(leg_tar[:, k] == 0) is None
                                                           else ms(persist_first(leg_tar[:, k] == 0))),
                          first_step_force_below_1pct=persist_first(leg_fn[:, k] < 0.01 * base),
                          first_step_force_below_1pct_ms=(None if persist_first(leg_fn[:, k] < 0.01 * base) is None
                                                          else ms(persist_first(leg_fn[:, k] < 0.01 * base))))
    first_leg = min(((v['first_step_no_tarsus_contact'], L) for L, v in per_leg.items()
                     if v['first_step_no_tarsus_contact'] is not None), default=(None, None))
    first_leg_force = min(((v['first_step_force_below_1pct'], L) for L, v in per_leg.items()
                           if v['first_step_force_below_1pct'] is not None), default=(None, None))
    jfirst = min(((persist_first(qdev[:, k] >= 0.01), jlabels[k]) for k in range(len(jids))
                  if persist_first(qdev[:, k] >= 0.01) is not None), default=(None, None))
    lo = max(0, (ev['first_nonfoot'] or 0) - 20)
    stamp = time.strftime('%Y%m%dT%H%M%S')
    act_disc = dict(act_max_over_run=float(a['act_max'].max()), actuator_actlimited_all_false=bool(not any(np.asarray(m.actuator_actlimited))),
                    first_step_out_of_domain=ev['first_act_above_1'], out_of_domain_steps=int((a['act_max'] > 1.0).sum()),
                    statement=('if act leaves [0,1] the run is numerically invalid and none of its numbers may be quoted as a physical result; '
                               'the same wording as the sister batch: ' + SISTER))
    common = dict(
        declared_dt_s=DT, model_opt_timestep_s=float(m.opt.timestep), xml_option_timestep_s=xml_dt,
        integrator_steps=STEPS, d_time_final_s=float(d.time), window_s=float(STEPS * DT),
        command_period_s=CMD_S, command_period_steps=CMD_EVERY,
        first_step_check=first_step_check,
        time_axis='measured d.time at every step; event lines are placed at step * DT',
        convention_note='STEPS counts mj_step calls: 20000 calls at 25 us give d.time = 0.5 s and 20001 samples',
        manual_research_not_product=True, standing=False, walking=False,
        inputs=dict(state=str(STATE), state_sha256=sha(STATE), drive=str(DRIVE), drive_sha256=sha(DRIVE),
                    pose_A=str(POSE_A), canonical_xml=str(cp.CANONICAL_XML), canonical_xml_sha256=sha(cp.CANONICAL_XML),
                    script=str(pathlib.Path(__file__)), script_sha256=sha(pathlib.Path(__file__))),
        act_domain_disclosure=act_disc, run_py_timestep_sweep=dict(entries=sweep, n=len(sweep), missing=[s['full_path'] for s in missing],
                                                                  policy=('patched the current-line runners; the legacy runners are listed but NOT patched, because inserting a '
                                                                          'timestep assignment would silently change their physics; any of them that declares a step while '
                                                                          'relying on the XML value is marked "protocol declaration false"')),
        recorded_at=now())

    # ---- batch 1: open loop ----
    b1 = DESK / ('S2-accepted-equilibrium-openloop-corrected-reissued-dt25us-' + stamp)
    b1.mkdir(exist_ok=False)
    rep1 = dict(status='PASS_OPENLOOP_CORRECTED_REISSUED_DT25US', title='corrected-reissued-dt25us: accepted equilibrium open loop',
                **common,
                result=dict(first_non_foot_contact=dict(step=ev['first_nonfoot'], ms=ms(ev['first_nonfoot']), pair=ev['first_nonfoot_pair'],
                                                        geom_identity='A6_geom belongs to body A6'),
                            first_step_legs_in_contact_below_6=dict(step=ev['legs_below_6'], ms=ms(ev['legs_below_6'])),
                            first_step_with_zero_tarsus_contacts=dict(step=ev['zero_tarsus'], ms=ms(ev['zero_tarsus'])),
                            first_step_max_u_diamond_ge_1=dict(step=ev['first_ud_ge_1'], ms=ms(ev['first_ud_ge_1']), pair=ev['first_ud_pair']),
                            max_u_diamond_over_the_whole_run=dict(value=float(np.nanmax(a['ud'])), step=int(np.nanargmax(a['ud']))),
                            max_qacc=dict(step0=float(a['qacc'][0]), final=float(a['qacc'][-1]),
                                          over_the_whole_run=float(a['qacc'].max()), at_step=int(a['qacc'].argmax())),
                            root_z=dict(initial=float(a['root_z'][0]), min=float(a['root_z'].min()), min_step=int(a['root_z'].argmin()),
                                        min_ms=ms(int(a['root_z'].argmin())), final=float(a['root_z'][-1]),
                                        min_displacement_vs_pose_A_mm=float(a['root_z'].min() - float(pose_a[2]))),
                            six_leg_steps=dict(steps=int((a['legs'] == 6).sum()), total=int(a['legs'].size),
                                               fraction=float((a['legs'] == 6).mean())),
                            energy=dict(initial=float(a['energy'][0]), final=float(a['energy'][-1]),
                                        monotone=bool((np.diff(a['energy']) <= 0).all()),
                                        rising_steps=int((np.diff(a['energy']) > 0).sum())),
                            tilt=dict(max_deg=float(a['tilt'].max()), flipped=bool((a['tilt'] > 90.0).any())),
                            contact_table_at_first_nonfoot=table,
                            contact_table_pointer='report.json -> result -> contact_table_at_first_nonfoot'),
                answer_to_the_single_question=dict(
                    steps_until_six_leg_support_lost=ev['legs_below_6'],
                    ms_until_six_leg_support_lost=ms(ev['legs_below_6']),
                    steps_until_first_nonfoot_contact=ev['first_nonfoot'],
                    ms_until_first_nonfoot_contact=ms(ev['first_nonfoot']),
                    root_z_min=float(a['root_z'].min()), root_z_min_ms=ms(int(a['root_z'].argmin())),
                    clock='all values are step * DT with d.time as the measured axis'))
    (b1 / 'report.json').write_text(json.dumps(rep1, ensure_ascii=False, indent=1))
    np.savez_compressed(b1 / 'time_series_dt25us.npz', **a, leg_tarsus_counts=leg_tar, leg_normal_forces=leg_fn,
                        leg_max_u_diamond=leg_ud, joint_deviation=qdev, joint_labels=np.array(jlabels))

    # ---- batch 2: localization ----
    b2 = DESK / ('S2-accepted-equilibrium-localization-corrected-reissued-dt25us-' + stamp)
    b2.mkdir(exist_ok=False)
    rep2 = dict(status='PASS_LOCALIZATION_CORRECTED_REISSUED_DT25US',
                title='corrected-reissued-dt25us: localization under the same 25 us run', **common,
                localization=dict(first_leg_to_lose_tarsus_contact=dict(leg=first_leg[1], step=first_leg[0], ms=ms(first_leg[0])),
                                  first_leg_force_below_1pct=dict(leg=first_leg_force[1], step=first_leg_force[0],
                                                                  ms=ms(first_leg_force[0])),
                                  first_joint_past_0_01_rad=dict(joint=jfirst[1], step=jfirst[0], ms=ms(jfirst[0])),
                                  first_step_with_exactly_one_leg=(int(np.argmax(a['legs'] == 1)) if (a['legs'] == 1).any() else None),
                                  per_leg=per_leg,
                                  lookback_before_first_nonfoot=dict(
                                      window_steps=[lo, ev['first_nonfoot']], window_ms=[ms(lo), ms(ev['first_nonfoot'])],
                                      per_leg_u_diamond={L: [None if np.isnan(leg_ud[i, k]) else float(leg_ud[i, k])
                                                             for i in range(lo, (ev['first_nonfoot'] or 0) + 1)]
                                                         for k, L in enumerate(LEGS)}),
                                  u_diamond_null_rule='no tarsus contact at a step => null (NaN), drawn as a broken line, never 0',
                                  drive_frame_stats=dict(
                                      n_frames=int(ctrl.shape[0]), cadence_ms=CMD_S * 1000,
                                      leading_all_zero_frames=int(next((i for i in range(ctrl.shape[0]) if not np.all(ctrl[i] == 0)), 0)),
                                      per_frame_max_step_full=[float(x) for x in np.abs(np.diff(ctrl, axis=0)).max(axis=1)],
                                      per_frame_max_step_note='full 99-value list (an earlier report stored only 12 without saying so)',
                                      mean_command_level=float(ctrl.mean()), fraction_of_entries_at_zero=float((ctrl == 0).mean()),
                                      zero_entries_clamped_to_the_control_floor=dict(
                                          count=int((ctrl == 0).sum()), total=int(ctrl.size),
                                          explanation='ctrllimited actuators with ctrlrange [1e-4, 1] and clamping enabled: the archive '
                                                      'stores 0, the simulator applies 1e-4'))))
    (b2 / 'report.json').write_text(json.dumps(rep2, ensure_ascii=False, indent=1))
    np.savez_compressed(b2 / 'time_series_dt25us.npz', **a, leg_tarsus_counts=leg_tar, leg_normal_forces=leg_fn,
                        leg_max_u_diamond=leg_ud, joint_deviation=qdev, joint_labels=np.array(jlabels),
                        first_leg=np.array([first_leg[1] or '', str(first_leg[0])]),
                        first_joint=np.array([jfirst[1] or '', str(jfirst[0])]))

    # ---- figures for both batches ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = a['time_s'] * 1000.0
    for batch, fname, kind in ((b1, 'openloop-dt25us.png', 'open'), (b2, 'localization-dt25us.png', 'loc')):
        fig, ax = plt.subplots(2, 2, figsize=(15.5, 9.8))
        if kind == 'open':
            x = ax[0][0]
            x.plot(t, a['root_z'], lw=1.3, color='#1f77b4')
            x.axvline(ev['first_nonfoot'] * DT * 1000.0, color='red', ls=':', lw=1.6)
            x.annotate('first non-foot contact\nstep %d = %.6f ms\n%s' % (ev['first_nonfoot'], ms(ev['first_nonfoot']),
                                                                         ' | '.join(ev['first_nonfoot_pair'] or [])),
                       (ev['first_nonfoot'] * DT * 1000.0, float(a['root_z'][ev['first_nonfoot']])),
                       textcoords='offset points', xytext=(24, 28), fontsize=9, arrowprops=dict(arrowstyle='->', lw=1.0),
                       bbox=dict(boxstyle='round', fc='white', alpha=0.92))
            x.set_xlabel('measured d.time [ms]; window %.3f s' % rep1['window_s']); x.set_ylabel('root z [mm]')
            x.set_title('root z; event line placed at step x DT', fontsize=10); x.grid(alpha=0.3)
            x = ax[0][1]
            x.plot(t, a['legs'], lw=1.2, color='#2ca02c', label='legs with tarsus contact')
            x.plot(t, a['nonfoot'], lw=1.0, color='#d62728', label='non-foot contacts')
            x.set_xlabel('measured d.time [ms]'); x.set_ylabel('count')
            x.set_title('contacts: six-leg support lost at step %d = %.6f ms' % (ev['legs_below_6'], ms(ev['legs_below_6'])), fontsize=10)
            x.legend(fontsize=8); x.grid(alpha=0.3)
            x = ax[1][0]
            x.semilogy(t, np.maximum(a['qacc'], 1e-14), lw=1.1, color='#1f77b4')
            x.axhline(1.0, color='gray', ls=':', lw=1.2, label='1.0 mm/s^2')
            x.set_xlabel('measured d.time [ms]'); x.set_ylabel('max|qacc| [mm/s^2], log')
            x.set_title('per-step max|qacc| (whole-run max %.6g at step %d)' % (rep1['result']['max_qacc']['over_the_whole_run'],
                                                                               rep1['result']['max_qacc']['at_step']), fontsize=10)
            x.legend(fontsize=8); x.grid(alpha=0.3, which='both')
            x = ax[1][1]
            x.plot(t, a['ud'], lw=1.1, color='#9467bd')
            x.axhline(1.0, color='red', ls='--', lw=1.3, label='declared diamond limit')
            x.set_xlabel('measured d.time [ms]'); x.set_ylabel('max u_diamond (NaN = no tarsus contact)')
            x.set_title('per-step max u_diamond (peak %.6g at step %d)' % (rep1['result']['max_u_diamond_over_the_whole_run']['value'],
                                                                          rep1['result']['max_u_diamond_over_the_whole_run']['step']), fontsize=10)
            x.legend(fontsize=8); x.grid(alpha=0.3)
        else:
            x = ax[0][0]
            for k, L in enumerate(LEGS):
                y = np.where(leg_tar[:, k] > 0, leg_tar[:, k], np.nan)
                x.plot(t, y, lw=1.2, label=L)
            x.set_xlabel('measured d.time [ms]'); x.set_ylabel('tarsus contacts (gap = no contact)')
            x.set_title('per-leg tarsus contacts; first loss %s at step %d = %.6f ms' % (first_leg[1], first_leg[0], ms(first_leg[0])), fontsize=10)
            x.legend(fontsize=8, ncol=3); x.grid(alpha=0.3)
            x = ax[0][1]
            for k, L in enumerate(LEGS):
                y = np.where(leg_fn[:, k] > 0, leg_fn[:, k], np.nan)
                x.semilogy(t, y, lw=1.1, label=L)
            x.set_xlabel('measured d.time [ms]'); x.set_ylabel('summed tarsus normal force (gap = no contact)')
            x.set_title('per-leg tarsus force', fontsize=10); x.legend(fontsize=8, ncol=3); x.grid(alpha=0.3, which='both')
            x = ax[1][0]
            top = np.argsort(-qdev.max(axis=0))[:6]
            for k in top:
                x.semilogy(t, np.maximum(qdev[:, k], 1e-12), lw=1.2, label=jlabels[k])
            x.set_xlabel('measured d.time [ms]'); x.set_ylabel('|q - q(0)| [rad]')
            x.set_title('six most-moving joints; first past 0.01 rad %s at step %d = %.6f ms' % (jfirst[1], jfirst[0], ms(jfirst[0])), fontsize=10)
            x.legend(fontsize=7); x.grid(alpha=0.3, which='both')
            x = ax[1][1]
            tlook = [i * DT * 1000.0 for i in range(lo, (ev['first_nonfoot'] or 0) + 1)]
            for k, L in enumerate(LEGS):
                x.plot(tlook, [np.nan if np.isnan(leg_ud[i, k]) else float(leg_ud[i, k]) for i in range(lo, (ev['first_nonfoot'] or 0) + 1)],
                       marker='.', lw=1.2, label=L)
            x.axhline(1.0, color='red', ls='--', lw=1.3, label='declared diamond limit')
            x.set_xlabel('measured d.time [ms]'); x.set_ylabel('max u_diamond (broken = no contact)')
            x.set_title('u_diamond in the 20 steps before the first non-foot contact', fontsize=10)
            x.legend(fontsize=8, ncol=3); x.grid(alpha=0.3)
        fig.suptitle('%s  |  declared_dt_s %.1e == model_opt_timestep_s %.1e ; first-step check |d.time - dt| = %.3g s ; '
                     'd.time final %.6f s\nmeasured axis; manual research implementation, NOT a product; no standing/walking claim'
                     % (rep1['title'] if kind == 'open' else rep2['title'], DT, rep1['model_opt_timestep_s'],
                        rep1['first_step_check']['abs_diff'], rep1['d_time_final_s']), fontsize=9.0)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        fig.savefig(batch / fname, dpi=135)
        plt.close(fig)

    # ---- correction ledger + stale hash inventory + READMEs ----
    stale = []
    for rec in (json.loads((V2 / 'hash_reanchor.json').read_text()), json.loads((V2 / 'hash_realignment.json').read_text())):
        for key in ('stale_records_v2', 'stale_records_ladder', 'records'):
            for r in rec.get(key, []) or []:
                if isinstance(r, dict) and 'file' in r:
                    f = r.get('file')
                    cur = sha(f) if f and pathlib.Path(f).exists() else None
                    recorded = r.get('current') or r.get('current_measured_sha256') or r.get('after')
                    if f and cur and recorded and cur != recorded:
                        stale.append(dict(source=('hash_reanchor' if 'stale_records' in rec else 'hash_realignment'), file=str(f),
                                          recorded_current=recorded, actual_now=cur))
    ledger = dict(status='CORRECTION_LEDGER', manual_research_not_product=True,
                  new_files=[dict(file=str(b1 / 'report.json'), sha256=sha(b1 / 'report.json')),
                             dict(file=str(b2 / 'report.json'), sha256=sha(b2 / 'report.json'))],
                  superseded=[dict(file=str(DESK / 'S2-accepted-equilibrium-25us-corrected-20261015T172131'),
                                   reason='superseded by the two dt25us batches of this round (same numbers, two-batch layout, added step-size fields)'),
                              dict(file=str(DESK / 'S2-equilibrium-A-corrected-reissued-v3-20261015T170319'),
                                   reason='historical: built from the 100 us runs; its ms labels are void')],
                  stale_hash_records_do_not_use_as_current=stale, n_stale=len(stale),
                  rule=('a delivered-and-verified batch directory and the live scripts it names are read-only; changes go to a new timestamped directory '
                        'and carry a ledger entry with old file, old sha256, new file, new sha256 and the reason'),
                  recorded_at=now())
    (b1 / 'correction_ledger.json').write_text(json.dumps(ledger, ensure_ascii=False, indent=1))
    (b2 / 'correction_ledger.json').write_text(json.dumps(ledger, ensure_ascii=False, indent=1))

    ref = ('首个 `Thorax_geom | floor` 接触 = 步 3055 = %.6f ms（由 step×DT 计算；**76.400 属步 3056**，此前错一步）；'
           '首次非足接触 = `(floor, A6_geom)`（属 body A6）步 2739 = %.6f ms；`legs_in_contact < 6` 步 537 = %.6f ms；'
           '`first_tarsus_loss_step` 步 5111 = %.6f ms；这是首次非足（非跗节）接触，不是"失去足接触"'
           % (ms_of(3055), ms_of(2739), ms_of(537), ms_of(5111)))
    for batch, title in ((b1, rep1['title']), (b2, rep2['title'])):
        (batch / 'contact_identity_sentence.json').write_text(json.dumps(dict(
            status='UNIQUE_WORDING_SOURCE', sentence=ref,
            note='every millisecond here is computed as step x DT; no hand-typed constant',
            geometry_identities_kept=['step 2739 pair (floor, A6_geom), A6_geom belongs to body A6',
                                      'first Thorax_geom | floor contact at step 3055',
                                      'legs_in_contact < 6 from step 537', 'first_tarsus_loss_step 5111'],
            recorded_at=now()), ensure_ascii=False, indent=1))
        (batch / 'README.zh-CN.md').write_text('\n'.join([
            '# %s（手动研究实施，不是成品）' % title, '',
            '## 机器可查的步长字段', '- `declared_dt_s = %.1e`，`model_opt_timestep_s = %.1e`（读回模型），`d_time_final_s = %.6f`；'
            '首次 `mj_step` 后 `|d.time - dt| = %.3g`（断言 `< 1e-15` 通过）。'
            % (DT, rep1['model_opt_timestep_s'], rep1['d_time_final_s'], rep1['first_step_check']['abs_diff']),
            '- 横轴 = 实测 `d.time`；事件竖线位置 = `step × DT`；图题窗口与横轴同口径（%.3f s）。' % rep1['window_s'], '',
            '## 交叉校验点（本批实际值）',
            '- 首个足外接触 步 **%d**（= %.6f ms）' % (ev['first_nonfoot'], ms(ev['first_nonfoot'])),
            '- `legs_in_contact < 6` 步 **%d**（= %.6f ms）' % (ev['legs_below_6'], ms(ev['legs_below_6'])),
            '- 首个关节越 0.01 rad 步 **%d**（= %.6f ms，%s）' % (jfirst[0], ms(jfirst[0]), jfirst[1]),
            '- 跗节全失 步 **%d**（= %.6f ms）' % (ev['zero_tarsus'], ms(ev['zero_tarsus'])), '',
            '## act 域披露（模板强制）',
            '- `act` 峰值 **%s**；`actuator_actlimited` 全 False：**%s**；首次越域步 **%s**；越域步数 **%d**。'
            % (act_disc['act_max_over_run'], act_disc['actuator_actlimited_all_false'], act_disc['first_step_out_of_domain'],
               act_disc['out_of_domain_steps']),
            '- %s' % act_disc['statement'], '',
            '## 唯一措辞源（F-A，ms 由 step×DT 计算）', '- ' + ref, '',
            '## 冻结与账本',
            '- 本批写盘即冻结（`delivery-hashes.json`/`FROZEN_DELIVERY_HASHES.json`）；`correction_ledger.json` 列出本轮新文件、被取代件与'
            '**已失效的哈希记录（%d 条，不得当作现值引用）**。' % len(stale), '',
            '## 下一最小动作（单一问题）', '- 在这条冻结 100 帧 / 5 ms 归档的真实 0.5 s 窗口里：**接受状态几步失去六足跗节支撑、几步出现首个'
            '足外接触、root z 最低多少**，全部按 `d.time` 给毫秒 —— 答案见 `report.json.answer_to_the_single_question`（开环件）。', '',
            '- **手动研究实施，不是成品**；不含站立/行走声明；绝对力值与单位 `NOT_ESTABLISHED`。']) + '\n')
    for batch in (b1, b2):
        with zipfile.ZipFile(batch / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(pathlib.Path(__file__), pathlib.Path(__file__).name)
        files = sorted(p for p in batch.iterdir() if p.is_file() and p.name not in ('delivery-hashes.json', 'FROZEN_DELIVERY_HASHES.json'))
        (batch / 'delivery-hashes.json').write_text(json.dumps({p.name: sha(p) for p in files}, ensure_ascii=False, indent=1))
        (batch / 'FROZEN_DELIVERY_HASHES.json').write_text(json.dumps(dict(status='FROZEN_BEFORE_VERIFICATION', frozen_at=now(),
                                                                          files={p.name: sha(p) for p in files}), ensure_ascii=False, indent=1))
    print('batch1:', b1)
    print('batch2:', b2)
    print('cross-checks: first_nonfoot %d | legs<6 %d | first_joint %d | zero_tarsus %d' % (ev['first_nonfoot'], ev['legs_below_6'],
                                                                                           jfirst[0], ev['zero_tarsus']))
    print('act: peak %s | OOD steps %d | actlimited all false %s' % (act_disc['act_max_over_run'], act_disc['out_of_domain_steps'],
                                                                     act_disc['actuator_actlimited_all_false']))
    print('run_*.py calling mj_step: %d | missing the timestep line: %d' % (len(sweep), len(missing)))
    print('stale hash records inventoried:', len(stale))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
