"""Turn 25, part B: the ONE current batch (25 us, corrected) holding the open loop and the localization from the same run,
the labelled 100 us side branch, the act disclosure, the minor fixes, and two figures drawn on the measured clock.

Minor fixes carried here: contact-table pointer path, amplitude gap measured against the first NON-ZERO frame (with the
degeneracy of the all-zero frame 0 stated), u_diamond = null where a leg has no contact (and a broken line in the figure),
the full per-frame step list with an explicit truncation note, max_u_diamond renamed to its true meaning, and the
ctrlrange clamping of the all-zero archive entries explained.
"""

import hashlib
import json
import pathlib
import sys
import time
import zipfile

import numpy as np

ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
STATE = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259/equilibrium_best_state.npz'
POSE_A = DESK / 'S2-declared-cone-min-s-20261015T123603/best_candidate_pose.npz'
DRIVE = ROOT / 'outputs/experiment_platform/20261015_135150_a2fff3f0/traces.npz'
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
SUFFIXES = ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll', 'Trochanter_yaw', 'Trochanter_pitch', 'Trochanter_roll', 'Tibia_pitch')
DT, STEPS, CMD_S = 2.5e-5, 20000, 0.005
CMD_EVERY = int(round(CMD_S / DT))
SISTER = ('BRANCH INVALID: at dt=1e-4 s the native muscle activation leaves its domain (act max 1.9993002399 > 1), so no number from this branch '
          'may be quoted as a physical result; use the 25 us branch')
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S')


def persist_first(mask, persist=5):
    run = 0
    for i, v in enumerate(mask):
        run = run + 1 if v else 0
        if run >= persist:
            return i - persist + 1
    return None


def simulate(m, mujoco, gname, q0, act0, ctrl, qadr, dt, details):
    m.opt.timestep = dt
    assert abs(float(m.opt.timestep) - dt) <= 0.0
    d = mujoco.MjData(m)
    d.qpos[:] = q0
    d.qvel[:] = 0.0
    d.act[:] = act0
    d.ctrl[:] = 0.0
    mujoco.mj_forward(m, d)
    n = STEPS + 1
    a = dict(time_s=np.zeros(n), root_z=np.zeros(n), legs=np.zeros(n, dtype=int), tar=np.zeros(n, dtype=int),
             nonfoot=np.zeros(n, dtype=int), tilt=np.zeros(n), energy=np.zeros(n), qacc=np.zeros(n),
             ud=np.full(n, np.nan), act_max=np.zeros(n), ctrl_max=np.zeros(n))
    leg_tar = np.zeros((n, len(LEGS)), dtype=int) if details else None
    leg_fn = np.zeros((n, len(LEGS))) if details else None
    leg_ud = np.full((n, len(LEGS)), np.nan) if details else None
    qdev = np.zeros((n, len(qadr))) if details else None
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
        a['ctrl_max'][i] = float(np.max(np.asarray(d.ctrl)))
        if details:
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
                    if details:
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
        a['ud'][i] = worst if tar else np.nan  # no contact -> NaN, never 0
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
        sample(step + 1)
    a['events'] = ev
    a['contact_table_at_first_nonfoot'] = table
    if details:
        a['leg_tar'], a['leg_fn'], a['leg_ud'], a['qdev'] = leg_tar, leg_fn, leg_ud, qdev
    return a


def main():
    import mujoco
    sys.path.insert(0, str(ROOT))
    from engine import canonical_physics as cp
    m = cp.load_model()
    xml_dt = float(m.opt.timestep)
    m.opt.timestep = DT
    assert float(m.opt.timestep) == DT
    assert abs(DT - xml_dt) > 0
    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
    jname = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ''
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

    good = simulate(m, mujoco, gname, q0, act0, ctrl, qadr, DT, details=True)
    m.opt.timestep = DT
    side = simulate(m, mujoco, gname, q0, act0, ctrl, qadr, xml_dt, details=False)
    m.opt.timestep = DT
    assert float(m.opt.timestep) == DT
    assert abs(float(good['time_s'][-1]) - STEPS * DT) <= 1e-9, 'measured final time must equal steps x dt'
    assert abs(float(side['time_s'][-1]) - STEPS * xml_dt) <= 1e-9

    out = DESK / ('S2-accepted-equilibrium-25us-corrected-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    ms = lambda i: (None if i is None else round(i * DT * 1000, 3))
    ev, sev = good['events'], side['events']
    per_leg = {}
    for k, L in enumerate(LEGS):
        base = float(good['leg_fn'][:100, k].mean())
        per_leg[L] = dict(first_step_no_tarsus_contact=persist_first(good['leg_tar'][:, k] == 0),
                          first_step_no_tarsus_contact_ms=ms(persist_first(good['leg_tar'][:, k] == 0)),
                          first_step_force_below_1pct=persist_first(good['leg_fn'][:, k] < 0.01 * base),
                          first_step_force_below_1pct_ms=ms(persist_first(good['leg_fn'][:, k] < 0.01 * base)))
    first_leg = min(((v['first_step_no_tarsus_contact'], L) for L, v in per_leg.items()
                     if v['first_step_no_tarsus_contact'] is not None), default=(None, None))
    first_leg_force = min(((v['first_step_force_below_1pct'], L) for L, v in per_leg.items()
                           if v['first_step_force_below_1pct'] is not None), default=(None, None))
    jfirst = min(((persist_first(good['qdev'][:, k] >= 0.01), jlabels[k]) for k in range(len(jids))
                  if persist_first(good['qdev'][:, k] >= 0.01) is not None), default=(None, None))
    lo = max(0, (ev['first_nonfoot'] or 0) - 20)
    lookback = dict(window_steps=[lo, ev['first_nonfoot']],
                    window_ms=[round(lo * DT * 1000, 3), ms(ev['first_nonfoot'])],
                    per_leg_u_diamond={L: [None if np.isnan(good['leg_ud'][i, k]) else float(good['leg_ud'][i, k])
                                           for i in range(lo, (ev['first_nonfoot'] or 0) + 1)] for k, L in enumerate(LEGS)},
                    root_z=[float(x) for x in good['root_z'][lo:(ev['first_nonfoot'] or 0) + 1]])
    steps_arch = np.abs(np.diff(ctrl, axis=0))
    _act_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i2) or '' for i2 in range(m.nu)]
    _leg_idx = {L: [i2 for i2, n in enumerate(_act_names) if n.startswith(L) or ('_' + L) in n] for L in LEGS}
    zeros_clamped = int((ctrl == 0).sum())

    report = dict(
        status='PASS_25US_CORRECTED_CURRENT', corrected=True, manual_research_not_product=True, standing=False, walking=False,
        title='corrected: accepted equilibrium open loop and localization at 25 us (manual research implementation, not a product)',
        declared=dict(model_timestep_s=float(m.opt.timestep), xml_option_timestep_s=xml_dt, integrator_steps=STEPS,
                      window_s=float(STEPS * DT), command_period_s=CMD_S, command_period_steps=CMD_EVERY,
                      time_axis='measured d.time at every step; the final value is asserted to equal steps x dt',
                      convention_note=('STEPS counts mj_step calls: 20000 calls at 25 us give a final d.time of 0.5 s and 20001 samples. '
                                       'A literal (steps-1)*dt assertion would be wrong by one step; the assertion used here is '
                                       'final d.time == STEPS * DT'),
                      actuator_actlimited_all_false=bool(not any(np.asarray(m.actuator_actlimited)))),
        mandatory_act_disclosure=dict(act_max_over_run=float(good['act_max'].max()),
                                      act_out_of_domain_steps=int((good['act_max'] > 1.0).sum()),
                                      side_branch_100us_act_max=float(side['act_max'].max()),
                                      side_branch_100us_out_of_domain_steps=int((side['act_max'] > 1.0).sum()),
                                      statement=SISTER),
        inputs=dict(state=str(STATE), state_sha256=sha(STATE), pose_A=str(POSE_A), pose_A_sha256=sha(POSE_A),
                    drive=str(DRIVE), drive_sha256=sha(DRIVE), canonical_xml=str(cp.CANONICAL_XML),
                    canonical_xml_sha256=sha(cp.CANONICAL_XML), script=str(pathlib.Path(__file__)),
                    script_sha256=sha(pathlib.Path(__file__))),
        open_loop=dict(first_non_foot_contact=dict(step=ev['first_nonfoot'], ms=ms(ev['first_nonfoot']), pair=ev['first_nonfoot_pair']),
                       first_step_legs_in_contact_below_6=dict(step=ev['legs_below_6'], ms=ms(ev['legs_below_6'])),
                       first_step_with_zero_tarsus_contacts=dict(step=ev['zero_tarsus'], ms=ms(ev['zero_tarsus'])),
                       first_step_max_u_diamond_ge_1=dict(step=ev['first_ud_ge_1'], ms=ms(ev['first_ud_ge_1']), pair=ev['first_ud_pair']),
                       max_u_diamond_over_the_whole_run=dict(value=float(np.nanmax(good['ud'])),
                                                             step=int(np.nanargmax(good['ud'])),
                                                             note='renamed from max_u_diamond_window, which wrongly suggested a window'),
                       max_qacc=dict(step0=float(good['qacc'][0]), final=float(good['qacc'][-1]),
                                     over_the_whole_run=float(good['qacc'].max()), at_step=int(good['qacc'].argmax()),
                                     first_above_1e_minus_3_step=int(np.argmax(good['qacc'] > 1e-3)),
                                     first_above_1e_minus_3_ms=ms(int(np.argmax(good['qacc'] > 1e-3)))),
                       root_z=dict(initial=float(good['root_z'][0]), min=float(good['root_z'].min()),
                                   min_step=int(good['root_z'].argmin()), min_ms=ms(int(good['root_z'].argmin())),
                                   final=float(good['root_z'][-1]),
                                   min_displacement_vs_pose_A_mm=float(good['root_z'].min() - float(pose_a[2]))),
                       six_leg_steps=dict(steps=int((good['legs'] == 6).sum()), total=int(good['legs'].size),
                                          fraction=float((good['legs'] == 6).mean())),
                       energy=dict(initial=float(good['energy'][0]), final=float(good['energy'][-1]),
                                   monotone=bool((np.diff(good['energy']) <= 0).all()),
                                   rising_steps=int((np.diff(good['energy']) > 0).sum())),
                       tilt=dict(max_deg=float(good['tilt'].max()), final_deg=float(good['tilt'][-1]),
                                 flipped=bool((good['tilt'] > 90.0).any())),
                       contact_table_at_first_nonfoot=good['contact_table_at_first_nonfoot'],
                       contact_table_pointer='report.json -> open_loop -> contact_table_at_first_nonfoot (top level, not under Q1)'),
        localization=dict(first_leg_to_lose_tarsus_contact=dict(leg=first_leg[1], step=first_leg[0], ms=ms(first_leg[0])),
                          first_leg_force_below_1pct=dict(leg=first_leg_force[1], step=first_leg_force[0], ms=ms(first_leg_force[0])),
                          first_joint_past_0_01_rad=dict(joint=jfirst[1], step=jfirst[0], ms=ms(jfirst[0])),
                          first_step_with_exactly_one_leg=int(np.argmax(good['legs'] == 1)) if (good['legs'] == 1).any() else None,
                          first_step_with_exactly_one_leg_ms=(ms(int(np.argmax(good['legs'] == 1)))
                                                              if (good['legs'] == 1).any() else None),
                          per_leg=per_leg, lookback_before_first_nonfoot=lookback,
                          u_diamond_null_rule='where a leg has no tarsus contact at a step the value is null (NaN), never 0, and the figure '
                                              'draws a broken line there',
                          drive_amplitude_gap=dict(
                              reference_frame='the first NON-ZERO frame (index 2)',
                              degeneracy_note=('measuring the gap against frame 0 is degenerate: frame 0 is all-zero, so the gap would equal the '
                                               'equilibrium activation exactly'),
                              per_leg={L: dict(mean=float(np.abs(act0[_idx] - ctrl[2][_idx]).mean()),
                                               max=float(np.abs(act0[_idx] - ctrl[2][_idx]).max()),
                                               n_actuators=len(_idx))
                                       for L, _idx in _leg_idx.items()}),
                          drive_frame_stats=dict(n_frames=int(ctrl.shape[0]), cadence_ms=CMD_S * 1000,
                                                 leading_all_zero_frames=int(next((i for i in range(ctrl.shape[0])
                                                                                   if not np.all(ctrl[i] == 0)), 0)),
                                                 per_frame_max_step_full=[float(x) for x in steps_arch.max(axis=1)],
                                                 per_frame_max_step_note='the full 99-value list; an earlier report stored only the first 12 '
                                                                         'without saying so',
                                                 mean_step_between_frames=float(steps_arch.mean()),
                                                 max_step_between_frames=float(steps_arch.max()),
                                                 mean_command_level=float(ctrl.mean()),
                                                 fraction_of_entries_at_zero=float((ctrl == 0).mean()),
                                                 zero_entries_clamped_to_the_control_floor=dict(
                                                     count=zeros_clamped, total=int(ctrl.size),
                                                     explanation=('the model has ctrllimited actuators with ctrlrange [1e-4, 1] and clamping is '
                                                                  'enabled, so every stored 0 is applied as 1e-4; the archive stores 0, the '
                                                                  'simulator applies 1e-4'))),
                          ),
        side_branch_100us=dict(dt_s=xml_dt, final_time_s=float(side['time_s'][-1]), act_max=float(side['act_max'].max()),
                               out_of_domain_steps=int((side['act_max'] > 1.0).sum()),
                               first_nonfoot_step=sev['first_nonfoot'], labelled='numerically_invalid', statement=SISTER,
                               rule='kept for reference only; no number from this branch may be quoted as a physical result'),
        comparison=dict(accepted_equilibrium_first_non_foot_ms=ms(ev['first_nonfoot']), pose_A_first_non_foot_ms=59.950),
        not_established='absolute force values, physical units, biological validity; no standing, no walking',
        recorded_at=now())
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    np.savez_compressed(out / 'time_series_25us.npz',
                        **{'physical__' + k: v for k, v in good.items() if isinstance(v, np.ndarray)},
                        **{'numerically_invalid_100us__' + k: v for k, v in side.items() if isinstance(v, np.ndarray)},
                        joint_labels=np.array(jlabels),
                        note='physical__* is the 25 us branch (final time 0.5 s); numerically_invalid_100us__* is the XML-step branch '
                             '(final time 2.0 s, act out of [0,1])')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = good['time_s'] * 1000.0
    fig, ax = plt.subplots(2, 2, figsize=(15.5, 9.8))
    a = ax[0][0]
    a.plot(t, good['root_z'], lw=1.3, color='#1f77b4')
    a.axvline(t[ev['first_nonfoot']], color='red', ls=':', lw=1.6)
    a.annotate('first non-foot contact\nstep %d = %.3f ms\n%s' % (ev['first_nonfoot'], ms(ev['first_nonfoot']),
                                                                 ' | '.join(ev['first_nonfoot_pair'] or [])),
               (t[ev['first_nonfoot']], float(good['root_z'][ev['first_nonfoot']])), textcoords='offset points',
               xytext=(24, 28), fontsize=9, arrowprops=dict(arrowstyle='->', lw=1.0),
               bbox=dict(boxstyle='round', fc='white', alpha=0.92))
    a.set_xlabel('measured time [ms] (true window %.3f s)' % report['declared']['window_s']); a.set_ylabel('root z [mm]')
    a.set_title('root z; event line and label share the measured clock', fontsize=10); a.grid(alpha=0.3)
    a = ax[0][1]
    a.plot(t, good['legs'], lw=1.2, color='#2ca02c', label='legs with tarsus contact')
    a.plot(t, good['nonfoot'], lw=1.0, color='#d62728', label='non-foot contacts')
    a.set_xlabel('measured time [ms]'); a.set_ylabel('count')
    a.set_title('contacts (legs<6 at step %d = %s ms)' % (ev['legs_below_6'], ms(ev['legs_below_6'])), fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.3)
    a = ax[1][0]
    a.semilogy(t, np.maximum(good['qacc'], 1e-14), lw=1.1, color='#1f77b4', label='max|qacc| (25 us, physical)')
    a.semilogy(side['time_s'] * 1000.0, np.maximum(side['qacc'], 1e-14), lw=1.0, color='gray', ls='--',
               label='max|qacc| (100 us, numerically invalid)')
    a.axhline(1.0, color='gray', ls=':', lw=1.2, label='1.0 mm/s^2')
    a.set_xlabel('time [ms] (note the side branch spans 2000 ms)'); a.set_ylabel('max|qacc| [mm/s^2], log')
    a.set_title('per-step max|qacc|: physical branch vs the invalid side branch', fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.3, which='both')
    a = ax[1][1]
    a.plot(t, good['ud'], lw=1.1, color='#9467bd', label='max u_diamond (NaN where no tarsus contact)')
    a.axhline(1.0, color='red', ls='--', lw=1.3, label='declared diamond limit')
    a.set_xlabel('measured time [ms]'); a.set_ylabel('max u_diamond')
    a.set_title('per-step max u_diamond (whole-run peak %.6g at step %d)'
                % (report['open_loop']['max_u_diamond_over_the_whole_run']['value'],
                   report['open_loop']['max_u_diamond_over_the_whole_run']['step']), fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.suptitle('corrected: accepted declared-cone static equilibrium, open loop at the enforced 25 us step\n'
                 'model_timestep_s %.1e read back from the model; 20000 steps; measured window %.3f s; command held %d steps = %.4f s\n'
                 'manual research implementation, NOT a product; no standing/walking claim'
                 % (report['declared']['model_timestep_s'], report['declared']['window_s'], CMD_EVERY, CMD_EVERY * DT), fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out / 'openloop-25us-corrected.png', dpi=135)
    plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(15.5, 9.8))
    a = ax[0][0]
    for k, L in enumerate(LEGS):
        y = np.where(good['leg_tar'][:, k] > 0, good['leg_tar'][:, k], np.nan)  # no contact -> gap, not a zero line
        a.plot(t, y, lw=1.2, label=L)
    a.set_xlabel('measured time [ms]'); a.set_ylabel('tarsus contacts of that leg (gap = no contact)')
    a.set_title('per-leg tarsus contacts (first loss: %s at step %s = %s ms)'
                % (first_leg[1], first_leg[0], ms(first_leg[0])), fontsize=10)
    a.legend(fontsize=8, ncol=3); a.grid(alpha=0.3)
    a = ax[0][1]
    for k, L in enumerate(LEGS):
        y = np.where(good['leg_fn'][:, k] > 0, good['leg_fn'][:, k], np.nan)
        a.semilogy(t, y, lw=1.1, label=L)
    a.set_xlabel('measured time [ms]'); a.set_ylabel('summed tarsus normal force [model units]')
    a.set_title('per-leg tarsus force (gap = no tarsus contact)', fontsize=10)
    a.legend(fontsize=8, ncol=3); a.grid(alpha=0.3, which='both')
    a = ax[1][0]
    top = np.argsort(-good['qdev'].max(axis=0))[:6]
    for k in top:
        a.semilogy(t, np.maximum(good['qdev'][:, k], 1e-12), lw=1.2, label=jlabels[k])
    a.set_xlabel('measured time [ms]'); a.set_ylabel('|q - q(0)| [rad]')
    a.set_title('six most-moving joints (first past 0.01 rad: %s at step %s = %s ms)'
                % (jfirst[1], jfirst[0], ms(jfirst[0])), fontsize=10)
    a.legend(fontsize=7); a.grid(alpha=0.3, which='both')
    a = ax[1][1]
    tlook = [i * DT * 1000 for i in range(lo, (ev['first_nonfoot'] or 0) + 1)]
    for k, L in enumerate(LEGS):
        a.plot(tlook, [np.nan if v is None else v for v in lookback['per_leg_u_diamond'][L]], marker='.', lw=1.2, label=L)
    a.axhline(1.0, color='red', ls='--', lw=1.3, label='declared diamond limit')
    a.set_xlabel('measured time [ms]'); a.set_ylabel('max u_diamond of that leg (broken = no contact)')
    a.set_title('u_diamond in the 20 steps before the first non-foot contact', fontsize=10)
    a.legend(fontsize=8, ncol=3); a.grid(alpha=0.3)
    fig.suptitle('corrected localization at the enforced 25 us step, same run as the open loop\n'
                 'measured clock; nulls are drawn as broken lines because a leg with no tarsus contact has no u_diamond\n'
                 'manual research implementation, NOT a product; no standing/walking claim', fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out / 'localization-25us-corrected.png', dpi=135)
    plt.close(fig)

    readme = '\n'.join([
        '# corrected：25 µs（强制并断言）下的被接受平衡态开环 + 同批定位（手动研究实施，不是成品）', '',
        '## 为什么还有这一批', '- 之前声明的 25 µs 是假的（脚本未设 `m.opt.timestep`，实跑 XML 的 1e-04），'
        '两个 VOID 目录（`…164559`、`…170058`）的物理口径已作废；本批是**唯一当前有效**的 25 µs 重跑件。', '',
        '## 时钟（真修复，不是换算法）', '- `m.opt.timestep = 2.5e-5` 写死并断言（且断言它与 XML 值不同）；'
        '`time_s` **每步取自实测 `d.time`**，并用 `final d.time == steps × dt` 断言（约定：20000 次 `mj_step` ⇒ 0.5 s、20001 个采样）；',
        '- 声明的 `(steps-1)*dt` 写法会在本约定下差一步，故本批使用 `steps × dt` 并在此写明。',
        '- **不再用 `arange*25e-6` 造时钟**：本轮已把 `run_accepted_equilibrium_localization.py`、`measure_openloop_step_qacc.py`、'
        '`measure_openloop_step_u_diamond.py` 的假时钟改成实测 `d.time`（改动 before/after 见 '
        '`/home/sensen/Desktop/engine修复验收/S2-25us-clock-patches-and-events-20261015T172015/clock_patch_ledger.json`）。', '',
        '## 开环（25 µs，物理口径）',
        '- **首个非足接触：步 %d = %s ms**，接触对 `%s`' % (ev['first_nonfoot'], ms(ev['first_nonfoot']), ev['first_nonfoot_pair']),
        '- `legs_in_contact < 6` 步 %d = %s ms；跗节全失 步 %s = %s ms' % (ev['legs_below_6'], ms(ev['legs_below_6']),
                                                                          ev['zero_tarsus'], ms(ev['zero_tarsus'])),
        '- root z：初值 %s、最小 %s（步 %d = %s ms）、末值 %s' % (report['open_loop']['root_z']['initial'],
                                                                 report['open_loop']['root_z']['min'],
                                                                 report['open_loop']['root_z']['min_step'],
                                                                 report['open_loop']['root_z']['min_ms'],
                                                                 report['open_loop']['root_z']['final']),
        '- **act 峰值 %s（越域步数 %d）**；100 µs 旁证支峰值 %s（越域 %d 步，标 `numerically_invalid`）'
        % (report['mandatory_act_disclosure']['act_max_over_run'], report['mandatory_act_disclosure']['act_out_of_domain_steps'],
           report['mandatory_act_disclosure']['side_branch_100us_act_max'],
           report['mandatory_act_disclosure']['side_branch_100us_out_of_domain_steps']),
        '- `max|qacc|` 首步 %s、末步 %s、全程最大 %s（步 %d）；`max u_diamond` 全程峰值 %s（步 %d）'
        % (report['open_loop']['max_qacc']['step0'], report['open_loop']['max_qacc']['final'],
           report['open_loop']['max_qacc']['over_the_whole_run'], report['open_loop']['max_qacc']['at_step'],
           report['open_loop']['max_u_diamond_over_the_whole_run']['value'],
           report['open_loop']['max_u_diamond_over_the_whole_run']['step']),
        '- 六足同触 %d/%d；能量 %s → %s（单调 %s，上升步 %d）；倾角最大 %.6f°、翻转 %s'
        % (report['open_loop']['six_leg_steps']['steps'], report['open_loop']['six_leg_steps']['total'],
           report['open_loop']['energy']['initial'], report['open_loop']['energy']['final'],
           report['open_loop']['energy']['monotone'], report['open_loop']['energy']['rising_steps'],
           report['open_loop']['tilt']['max_deg'], report['open_loop']['tilt']['flipped']), '',
        '## 同批定位（真 ms）',
        '- 先失去跗节接触的腿 **`%s` 步 %d = %s ms**；力先跌破自身前 100 步均值 1%% 的腿 **`%s` 步 %d = %s ms**'
        % (first_leg[1], first_leg[0], ms(first_leg[0]), first_leg_force[1], first_leg_force[0], ms(first_leg_force[0])),
        '- 先离开平衡的关节 **`%s` 步 %d = %s ms**；只剩 1 条腿 步 %s = %s ms'
        % (jfirst[1], jfirst[0], ms(jfirst[0]), report['localization']['first_step_with_exactly_one_leg'],
           report['localization']['first_step_with_exactly_one_leg_ms']), '',
        '## 本轮 minor 就地修',
        '- contact table 指针改为 `open_loop.contact_table_at_first_nonfoot`（此前写的 `Q1/…` 路径不存在）。',
        '- 幅度差改为对**第一个非零帧**（索引 2）计算，并写明对帧 0 计算是退化的（帧 0 全零）。',
        '- 无跗节接触处 `u_diamond` 记为 **null/NaN**，图上画**断线**（此前记 0 会读成"锥内"）。',
        '- `per_frame_max_step` 给出**完整 99 项**并注明旧件只存前 12 项。',
        '- `max_u_diamond_window` 更名为 `max_u_diamond_over_the_whole_run`（原名字暗示窗口，实为全程最大）。',
        '- 归档中 **%d / %d** 个条目为 0：模型 `ctrllimited` 且 `ctrlrange=[1e-4,1]`、启用夹取，故**存 0、实际按 1e-4 施加**，已在报告写明。'
        % (zeros_clamped, int(ctrl.size)), '',
        '## 附加事件（不改冻结件）',
        '- 见 `/home/sensen/Desktop/engine修复验收/S2-25us-clock-patches-and-events-20261015T172015/appended_events.json`：'
        'EV-01 装配脚本哈希失配（`5c421276…` → `9271e578…`，16:14:08）、EV-02 冻结件在核验窗口内被改写（17:01:26，`INVALID → WITHDRAWN`，'
        'before 哈希**未记录⇒不可追溯**，after `d1aa1412…`），均按追加记录处理，**不用改哈希去迎合现值**。', '',
        '## 披露', '- **手动研究实施，不是成品**；不含站立/行走结论；绝对力值与单位 `NOT_ESTABLISHED`。'])
    (out / 'README.zh-CN.md').write_text(readme)
    with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(pathlib.Path(__file__), pathlib.Path(__file__).name)
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name not in ('delivery-hashes.json', 'FROZEN_DELIVERY_HASHES.json'))
    (out / 'delivery-hashes.json').write_text(json.dumps({p.name: sha(p) for p in files}, ensure_ascii=False, indent=1))
    (out / 'FROZEN_DELIVERY_HASHES.json').write_text(json.dumps(dict(status='FROZEN_BEFORE_VERIFICATION', frozen_at=now(),
                                                                    files={p.name: sha(p) for p in files}), ensure_ascii=False, indent=1))
    print(report['status'])
    print('out:', out)
    print('first non-foot step %d = %s ms (%s) | legs<6 %d = %s ms | zero tarsus %d = %s ms'
          % (ev['first_nonfoot'], ms(ev['first_nonfoot']), ev['first_nonfoot_pair'], ev['legs_below_6'], ms(ev['legs_below_6']),
             ev['zero_tarsus'], ms(ev['zero_tarsus'])))
    print('act 25us %s (OOD %d) | 100us %s (OOD %d)' % (report['mandatory_act_disclosure']['act_max_over_run'],
                                                        report['mandatory_act_disclosure']['act_out_of_domain_steps'],
                                                        report['mandatory_act_disclosure']['side_branch_100us_act_max'],
                                                        report['mandatory_act_disclosure']['side_branch_100us_out_of_domain_steps']))
    print('root z min %s at step %d | max qacc %s | max u_diamond %s at step %d'
          % (report['open_loop']['root_z']['min'], report['open_loop']['root_z']['min_step'],
             report['open_loop']['max_qacc']['over_the_whole_run'], report['open_loop']['max_u_diamond_over_the_whole_run']['value'],
             report['open_loop']['max_u_diamond_over_the_whole_run']['step']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
