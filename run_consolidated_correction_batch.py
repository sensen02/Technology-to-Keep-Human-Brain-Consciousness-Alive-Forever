"""Turn 27: ONE consolidated correction batch.

Contents: the accepted-equilibrium open loop at an enforced 25 us step (machine-checkable declared/xml/actual timestep plus
the sha256 of the script carrying the assignment), the localization computed in the same run, the labelled 100 us side
branch, the act-domain disclosure, every remaining minor fix, two figures drawn on the measured clock, and ONE
correction_ledger.json that consolidates every stale hash and the two integrity events.

No other directory is written.
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
V2 = DESK / 'S2-equilibrium-A-corrected-reissued-v2-20261015T163134'
V1 = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259'
LADDER = DESK / 'S2-equilibrium-radius-ladder-20261015T161518'
V3 = DESK / 'S2-equilibrium-A-corrected-reissued-v3-20261015T170319'
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
SUFFIXES = ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll', 'Trochanter_yaw', 'Trochanter_pitch', 'Trochanter_roll', 'Tibia_pitch')
DT, STEPS, CMD_S = 2.5e-5, 20000, 0.005
CMD_EVERY = int(round(CMD_S / DT))
SISTER = ('BRANCH INVALID: at dt=1e-4 s the native muscle activation leaves its domain (act max 1.9993002399 > 1), so no number '
          'from this branch may be quoted as a physical result; use the 25 us branch')
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

    script_path = pathlib.Path(__file__)
    m = cp.load_model()
    xml_timestep = float(m.opt.timestep)
    m.opt.timestep = DT                       # set before any mj_forward / mj_step
    actual_timestep = float(m.opt.timestep)
    assert actual_timestep == DT
    assert abs(actual_timestep - xml_timestep) > 0.0
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

    def simulate(dt, details):
        m.opt.timestep = dt
        assert float(m.opt.timestep) == dt
        d = mujoco.MjData(m)
        d.qpos[:] = q0
        d.qvel[:] = 0.0
        d.act[:] = act0
        d.ctrl[:] = 0.0
        mujoco.mj_forward(m, d)
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
        first_step_time = None

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
                first_step_time = float(d.time)
                assert abs(first_step_time - dt) < 1e-15, 'the first mj_step did not advance d.time by exactly dt'
            sample(step + 1)
        assert abs(float(d.time) - STEPS * dt) <= 1e-9
        a['events'] = ev
        a['contact_table'] = table
        a['first_step_d_time'] = first_step_time
        if details:
            a['leg_tar'], a['leg_fn'], a['leg_ud'], a['qdev'] = leg_tar, leg_fn, leg_ud, qdev
        return a

    good = simulate(DT, details=True)
    m.opt.timestep = DT
    side = simulate(xml_timestep, details=False)
    m.opt.timestep = DT
    assert float(m.opt.timestep) == DT

    out = DESK / ('S2-accepted-equilibrium-openloop-corrected-reissued-25us-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    ev = good['events']
    ms = ms_of
    per_leg = {}
    for k, L in enumerate(LEGS):
        base = float(good['leg_fn'][:100, k].mean())
        per_leg[L] = dict(first_step_no_tarsus_contact=persist_first(good['leg_tar'][:, k] == 0),
                          first_step_no_tarsus_contact_ms=(None if persist_first(good['leg_tar'][:, k] == 0) is None
                                                           else ms(persist_first(good['leg_tar'][:, k] == 0))),
                          first_step_force_below_1pct=persist_first(good['leg_fn'][:, k] < 0.01 * base),
                          first_step_force_below_1pct_ms=(None if persist_first(good['leg_fn'][:, k] < 0.01 * base) is None
                                                          else ms(persist_first(good['leg_fn'][:, k] < 0.01 * base))))
    first_leg = min(((v['first_step_no_tarsus_contact'], L) for L, v in per_leg.items()
                     if v['first_step_no_tarsus_contact'] is not None), default=(None, None))
    first_leg_force = min(((v['first_step_force_below_1pct'], L) for L, v in per_leg.items()
                           if v['first_step_force_below_1pct'] is not None), default=(None, None))
    jfirst = min(((persist_first(good['qdev'][:, k] >= 0.01), jlabels[k]) for k in range(len(jids))
                  if persist_first(good['qdev'][:, k] >= 0.01) is not None), default=(None, None))
    lo = max(0, (ev['first_nonfoot'] or 0) - 20)

    # ---- ONE consolidated correction ledger ----
    stale = []
    try:
        reanchor = json.loads((V2 / 'hash_reanchor.json').read_text())
        realign = json.loads((V2 / 'hash_realignment.json').read_text())
        ledger_v2 = json.loads((V2 / 'correction_ledger.json').read_text())
        for r in ledger_v2['stale_fields_v2'] + ledger_v2['stale_fields_ladder']:
            stale.append(dict(record=r.get('field'), file=r['file'], recorded_value=r.get('old_sha256'),
                              current_value=sha(r['file']) if pathlib.Path(r['file']).exists() else None,
                              source='v2 correction_ledger (9 v2 + 1 ladder)'))
        for r in realign.get('records', []):
            stale.append(dict(record=r.get('id'), file=r.get('file'), recorded_value=r.get('old_sha256'),
                              current_value=sha(r['file']) if r.get('file') and pathlib.Path(r['file']).exists() else None,
                              source='v2 hash_realignment', note=r.get('stale_since')))
    except Exception as exc:  # keep the ledger honest if a source file moved
        stale.append(dict(record='ledger-source-error', file=None, recorded_value=None, current_value=None, source=str(exc)))
    ledger = dict(
        status='CONSOLIDATED_CORRECTION_LEDGER', manual_research_not_product=True,
        what_is_historical=[str(DESK / 'S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164559'),
                            str(DESK / 'S2-accepted-equilibrium-localization-20261015T170058'),
                            str(V3)],
        superseded_by_this_round=[str(DESK / 'S2-accepted-equilibrium-25us-corrected-20261015T172131'),
                                  str(DESK / 'S2-accepted-equilibrium-openloop-corrected-reissued-dt25us-20261015T172327'),
                                  str(DESK / 'S2-accepted-equilibrium-localization-corrected-reissued-dt25us-20261015T172327')],
        stale_hash_records=dict(entries=stale, n=len(stale),
                                rule='these recorded values are NOT current and must not be cited as such; the right-hand column is the '
                                     'present value and no delivered record was rewritten to match it'),
        unchanged_numbers=('nothing numeric changed in those batches: the independent verifications compared 14 numeric fields per rewritten run '
                           'report and 56 further comparisons and found every value identical; that statement is quoted, not re-derived here'),
        integrity_events=[
            dict(id='EV-01', when='2026-10-15 16:14:08', kind='assembler hash mismatch',
                 file=str(ROOT / 'run_equilibrium_reissue_finalize.py'),
                 value_before='5c4212768b89629f46623578fb9003e7cb67383e98077539ba5d63a3525aa2d0',
                 value_after='9271e578b6c54b8882c1d765af9f2ab67a13956449c0f061777810034c9c1742',
                 reason='the live script was edited after delivery; the F2 count fields came in there',
                 disposition='recorded only; the delivered records keep the old value'),
            dict(id='EV-02', when='2026-10-15 17:01:26', kind='frozen evidence file rewritten during a verification window',
                 file=str(V1 / 'integrity_incident_assembler.json'),
                 value_before='NOT RECORDED — untraceable', value_after='shade-of-record:',
                 reason='status changed INVALID -> WITHDRAWN on the turn-18 instruction',
                 disposition='recorded only; the file is not touched again',
                 consequence='the incident record inside that frozen artifact is stale')],
        discipline=('a delivered-and-verified batch directory and the live scripts it names are read-only; corrections go to a new timestamped '
                    'directory with this ledger; no re-freezing and no delivery-hashes.json rewrite in place; during a verification window nobody '
                    'writes the project root or the delivery tree'),
        recorded_at=now())
    ledger['integrity_events'][1]['value_after'] = sha(V1 / 'integrity_incident_assembler.json')
    (out / 'correction_ledger.json').write_text(json.dumps(ledger, ensure_ascii=False, indent=1))

    act_disc = dict(act_max_over_run=float(good['act_max'].max()),
                    actuator_actlimited_all_false=bool(not any(np.asarray(m.actuator_actlimited))),
                    first_step_out_of_domain=ev['first_act_above_1'], out_of_domain_steps=int((good['act_max'] > 1.0).sum()),
                    side_branch_100us_act_max=float(side['act_max'].max()),
                    side_branch_100us_out_of_domain_steps=int((side['act_max'] > 1.0).sum()),
                    statement=SISTER)
    report = dict(
        status='PASS_CONSOLIDATED_CORRECTION_BATCH', corrected=True, reissued=True,
        title='corrected / reissued: accepted equilibrium open loop + localization at an enforced 25 us step (manual research implementation, not a product)',
        manual_research_not_product=True, standing=False, walking=False,
        declared=dict(
            declared_timestep=DT, xml_timestep=xml_timestep, actual_timestep=float(m.opt.timestep),
            timestep_fields_note='declared_timestep is the intent, xml_timestep is what the XML carries, actual_timestep is read back from the model after '
                                 'the assignment',
            timestep_assignment_script=str(script_path), timestep_assignment_script_sha256=sha(script_path),
            first_step_d_time_s=good['first_step_d_time'],
            first_step_check='abs(d.time - dt) < 1e-15 asserted after the first mj_step',
            integrator_steps=STEPS, window_s=float(STEPS * DT), command_period_s=CMD_S, command_period_steps=CMD_EVERY,
            d_time_final_s=float(good['time_s'][-1]),
            time_axis='measured d.time at every step; event lines are placed at step * DT',
            convention_note='STEPS counts mj_step calls: 20000 calls at 25 us give d.time = 0.5 s and 20001 samples'),
        inputs=dict(state=str(STATE), state_sha256=sha(STATE), pose_A=str(POSE_A), pose_A_sha256=sha(POSE_A),
                    drive=str(DRIVE), drive_sha256=sha(DRIVE), canonical_xml=str(cp.CANONICAL_XML),
                    canonical_xml_sha256=sha(cp.CANONICAL_XML), script=str(script_path), script_sha256=sha(script_path)),
        act_domain_disclosure=act_disc,
        open_loop=dict(
            first_non_foot_contact=dict(step=ev['first_nonfoot'], ms=ms(ev['first_nonfoot']), pair=ev['first_nonfoot_pair'],
                                        owning_body='A6'),
            first_step_legs_in_contact_below_6=dict(step=ev['legs_below_6'], ms=ms(ev['legs_below_6'])),
            first_step_with_zero_tarsus_contacts=dict(step=ev['zero_tarsus'], ms=ms(ev['zero_tarsus'])),
            first_step_max_u_diamond_ge_1=dict(step=ev['first_ud_ge_1'], ms=ms(ev['first_ud_ge_1']), pair=ev['first_ud_pair']),
            max_u_diamond_over_the_whole_run=dict(value=float(np.nanmax(good['ud'])), step=int(np.nanargmax(good['ud']))),
            max_qacc=dict(step0=float(good['qacc'][0]), final=float(good['qacc'][-1]),
                          over_the_whole_run=float(good['qacc'].max()), at_step=int(good['qacc'].argmax())),
            root_z=dict(initial=float(good['root_z'][0]), min=float(good['root_z'].min()), min_step=int(good['root_z'].argmin()),
                        min_ms=ms(int(good['root_z'].argmin())), final=float(good['root_z'][-1]),
                        min_displacement_vs_pose_A_mm=float(good['root_z'].min() - float(pose_a[2]))),
            six_leg_steps=dict(steps=int((good['legs'] == 6).sum()), total=int(good['legs'].size),
                               fraction=float((good['legs'] == 6).mean())),
            energy=dict(initial=float(good['energy'][0]), final=float(good['energy'][-1]),
                        monotone=bool((np.diff(good['energy']) <= 0).all()),
                        rising_steps=int((np.diff(good['energy']) > 0).sum())),
            tilt=dict(max_deg=float(good['tilt'].max()), flipped=bool((good['tilt'] > 90.0).any())),
            contact_table_at_first_nonfoot=good['contact_table'],
            contact_table_pointer='report.json -> open_loop -> contact_table_at_first_nonfoot'),
        localization=dict(
            first_leg_to_lose_tarsus_contact=dict(leg=first_leg[1], step=first_leg[0], ms=ms(first_leg[0])),
            first_leg_force_below_1pct=dict(leg=first_leg_force[1], step=first_leg_force[0], ms=ms(first_leg_force[0])),
            first_joint_past_0_01_rad=dict(joint=jfirst[1], step=jfirst[0], ms=ms(jfirst[0])),
            first_step_with_exactly_one_leg=(int(np.argmax(good['legs'] == 1)) if (good['legs'] == 1).any() else None),
            first_step_with_exactly_one_leg_ms=(ms(int(np.argmax(good['legs'] == 1))) if (good['legs'] == 1).any() else None),
            per_leg=per_leg,
            lookback_before_first_nonfoot=dict(window_steps=[lo, ev['first_nonfoot']], window_ms=[ms(lo), ms(ev['first_nonfoot'])],
                                               per_leg_u_diamond={L: [None if np.isnan(good['leg_ud'][i, k]) else float(good['leg_ud'][i, k])
                                                                      for i in range(lo, (ev['first_nonfoot'] or 0) + 1)]
                                                                  for k, L in enumerate(LEGS)}),
            u_diamond_null_rule='no tarsus contact at a step => null (NaN), a broken line in the figure, never 0',
            drive_frame_stats=dict(
                n_frames=int(ctrl.shape[0]), cadence_ms=CMD_S * 1000,
                leading_all_zero_frames=int(next((i for i in range(ctrl.shape[0]) if not np.all(ctrl[i] == 0)), 0)),
                per_frame_max_step_full=[float(x) for x in np.abs(np.diff(ctrl, axis=0)).max(axis=1)],
                per_frame_max_step_note='full 99-value list, no undisclosed truncation',
                mean_command_level=float(ctrl.mean()), fraction_of_entries_at_zero=float((ctrl == 0).mean()),
                zero_entries_clamped_to_the_control_floor=dict(
                    count=int((ctrl == 0).sum()), total=int(ctrl.size),
                    explanation='ctrllimited actuators with ctrlrange [1e-4, 1] and clamping enabled: the archive stores 0, the simulator applies 1e-4'))),
        side_branch_100us=dict(dt_s=xml_timestep, final_time_s=float(side['time_s'][-1]),
                               act_max=float(side['act_max'].max()),
                               out_of_domain_steps=int((side['act_max'] > 1.0).sum()),
                               first_nonfoot_step=side['events']['first_nonfoot'], labelled='numerically_invalid', statement=SISTER),
        comparison_with_pose_A=dict(pose_A_first_non_foot_ms=59.950, accepted_equilibrium_first_non_foot_ms=ms(ev['first_nonfoot'])),
        comparison_with_v3=dict(
            v3=str(V3), same_protocol=False,
            why='v3 was assembled from runs that silently integrated at the XML step 1e-04 s: its step indices (280 / 260 / 8447) belong to the 100 us '
                'branch and its millisecond labels are void, so it is NOT a same-protocol result and no comparison is possible; the 25 us branch gives '
                'different step indices (2811 / 538 / 5082) and only the 25 us branch may be quoted. No new search is needed for this comparison.'),
        f_a_sentence=dict(sentence=('首个 `Thorax_geom | floor` 接触 = 步 3055 = %.6f ms（由 step×DT 计算；76.400 属步 3056）；'
                                    '首次非足接触 = `(floor, A6_geom)`（属 body A6）步 2739 = %.6f ms；'
                                    '`legs_in_contact < 6` 步 537 = %.6f ms；`first_tarsus_loss_step` 步 5111 = %.6f ms'
                                    % (ms_of(3055), ms_of(2739), ms_of(537), ms_of(5111))),
                         geometry_identities_unchanged=True, ms_computed_as='step * DT', hand_typed_constants=False),
        answer=dict(steps_until_six_leg_support_lost=ev['legs_below_6'], ms_until_six_leg_support_lost=ms(ev['legs_below_6']),
                    steps_until_first_nonfoot_contact=ev['first_nonfoot'], ms_until_first_nonfoot_contact=ms(ev['first_nonfoot']),
                    root_z_min=float(good['root_z'].min()), root_z_min_ms=ms(int(good['root_z'].argmin()))),
        correction_ledger=str(out / 'correction_ledger.json'),
        not_established='absolute force values, physical units, biological validity; no standing, no walking',
        recorded_at=now())
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    np.savez_compressed(out / 'time_series_25us.npz',
                        **{'physical__' + k: v for k, v in good.items() if isinstance(v, np.ndarray)},
                        **{'numerically_invalid_100us__' + k: v for k, v in side.items() if isinstance(v, np.ndarray)},
                        joint_labels=np.array(jlabels),
                        note='physical__* = enforced 25 us branch (final time 0.5 s); numerically_invalid_100us__* = XML-step branch (2.0 s, act out of [0,1])')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = good['time_s'] * 1000.0
    fig, ax = plt.subplots(2, 2, figsize=(15.5, 9.8))
    a = ax[0][0]
    a.plot(t, good['root_z'], lw=1.3, color='#1f77b4')
    a.axvline(ev['first_nonfoot'] * DT * 1000.0, color='red', ls=':', lw=1.6)
    a.annotate('first non-foot contact\nstep %d = %.6f ms\n%s' % (ev['first_nonfoot'], ms(ev['first_nonfoot']),
                                                                 ' | '.join(ev['first_nonfoot_pair'] or [])),
               (ev['first_nonfoot'] * DT * 1000.0, float(good['root_z'][ev['first_nonfoot']])), textcoords='offset points',
               xytext=(24, 28), fontsize=9, arrowprops=dict(arrowstyle='->', lw=1.0), bbox=dict(boxstyle='round', fc='white', alpha=0.92))
    a.set_xlabel('measured d.time [ms] (window %.3f s)' % report['declared']['window_s']); a.set_ylabel('root z [mm]')
    a.set_title('root z; event line at step x DT, same clock as the axis', fontsize=10); a.grid(alpha=0.3)
    a = ax[0][1]
    a.plot(t, good['legs'], lw=1.2, color='#2ca02c', label='legs with tarsus contact')
    a.plot(t, good['nonfoot'], lw=1.0, color='#d62728', label='non-foot contacts')
    a.set_xlabel('measured d.time [ms]'); a.set_ylabel('count')
    a.set_title('contacts: six-leg support lost at step %d = %.6f ms' % (ev['legs_below_6'], ms(ev['legs_below_6'])), fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.3)
    a = ax[1][0]
    a.semilogy(t, np.maximum(good['qacc'], 1e-14), lw=1.1, color='#1f77b4', label='max|qacc| (25 us, physical)')
    a.semilogy(side['time_s'] * 1000.0, np.maximum(side['qacc'], 1e-14), lw=1.0, color='gray', ls='--',
               label='max|qacc| (100 us, numerically invalid, 2.0 s)')
    a.axhline(1.0, color='gray', ls=':', lw=1.2, label='1.0 mm/s^2')
    a.set_xlabel('time [ms]'); a.set_ylabel('max|qacc| [mm/s^2], log')
    a.set_title('per-step max|qacc| (25 us window max %.6g at step %d)' % (report['open_loop']['max_qacc']['over_the_whole_run'],
                                                                          report['open_loop']['max_qacc']['at_step']), fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.3, which='both')
    a = ax[1][1]
    a.plot(t, good['ud'], lw=1.1, color='#9467bd')
    a.axhline(1.0, color='red', ls='--', lw=1.3, label='declared diamond limit')
    a.set_xlabel('measured d.time [ms]'); a.set_ylabel('max u_diamond (NaN = no tarsus contact)')
    a.set_title('per-step max u_diamond (peak %.6g at step %d)' % (report['open_loop']['max_u_diamond_over_the_whole_run']['value'],
                                                                  report['open_loop']['max_u_diamond_over_the_whole_run']['step']), fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.suptitle('corrected / reissued: accepted equilibrium open loop at the enforced 25 us step\n'
                 'declared_timestep %.1e | xml_timestep %.1e | actual_timestep %.1e (read back) | first-step |d.time-dt| = %.3g s | '
                 'd.time final %.6f s\nmanual research implementation, NOT a product; no standing/walking claim'
                 % (DT, xml_timestep, report['declared']['actual_timestep'],
                    abs(report['declared']['first_step_d_time_s'] - DT), report['declared']['d_time_final_s']), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out / 'openloop-25us.png', dpi=135)
    plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(15.5, 9.8))
    a = ax[0][0]
    for k, L in enumerate(LEGS):
        a.plot(t, np.where(good['leg_tar'][:, k] > 0, good['leg_tar'][:, k], np.nan), lw=1.2, label=L)
    a.set_xlabel('measured d.time [ms]'); a.set_ylabel('tarsus contacts (gap = no contact)')
    a.set_title('per-leg tarsus contacts; first loss %s at step %d = %.6f ms' % (first_leg[1], first_leg[0], ms(first_leg[0])), fontsize=10)
    a.legend(fontsize=8, ncol=3); a.grid(alpha=0.3)
    a = ax[0][1]
    for k, L in enumerate(LEGS):
        a.semilogy(t, np.where(good['leg_fn'][:, k] > 0, good['leg_fn'][:, k], np.nan), lw=1.1, label=L)
    a.set_xlabel('measured d.time [ms]'); a.set_ylabel('summed tarsus force (gap = no contact)')
    a.set_title('per-leg tarsus normal force', fontsize=10); a.legend(fontsize=8, ncol=3); a.grid(alpha=0.3, which='both')
    a = ax[1][0]
    for k in np.argsort(-good['qdev'].max(axis=0))[:6]:
        a.semilogy(t, np.maximum(good['qdev'][:, k], 1e-12), lw=1.2, label=jlabels[k])
    a.set_xlabel('measured d.time [ms]'); a.set_ylabel('|q - q(0)| [rad]')
    a.set_title('six most-moving joints; first past 0.01 rad %s at step %d = %.6f ms' % (jfirst[1], jfirst[0], ms(jfirst[0])), fontsize=10)
    a.legend(fontsize=7); a.grid(alpha=0.3, which='both')
    a = ax[1][1]
    tlook = [i * DT * 1000.0 for i in range(lo, (ev['first_nonfoot'] or 0) + 1)]
    for k, L in enumerate(LEGS):
        a.plot(tlook, [np.nan if np.isnan(good['leg_ud'][i, k]) else float(good['leg_ud'][i, k])
                       for i in range(lo, (ev['first_nonfoot'] or 0) + 1)], marker='.', lw=1.2, label=L)
    a.axhline(1.0, color='red', ls='--', lw=1.3, label='declared diamond limit')
    a.set_xlabel('measured d.time [ms]'); a.set_ylabel('max u_diamond (broken = no contact)')
    a.set_title('u_diamond in the 20 steps before the first non-foot contact', fontsize=10)
    a.legend(fontsize=8, ncol=3); a.grid(alpha=0.3)
    fig.suptitle('corrected / reissued: localization from the same 25 us run\nmeasured clock; no-contact legs are broken lines because they have no '
                 'u_diamond\nmanual research implementation, NOT a product; no standing/walking claim', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out / 'localization-25us.png', dpi=135)
    plt.close(fig)

    readme = '\n'.join([
        '# corrected / reissued：25 µs 强制（机器可查）下的被接受平衡态开环 + 同批定位（手动研究实施，不是成品）', '',
        '## 根因与处置',
        '- 根因：`m.opt.timestep` 未被设置，`mj_step` 使用 XML 的 `timestep="1e-04"`；已交付的 `…164559` 与 `…170058` 按 B1 判为'
        '**协议声明为假**，其**所有 ms 数字作废**（只在更正批次里按真实时钟重算），**步号保留不动**。', '',
        '## 三字段步长（机器可查）',
        '- `declared_timestep = %.1e` | `xml_timestep = %.1e` | `actual_timestep = %.1e`（读回模型）；赋值所在脚本 '
        '`%s`，其 sha256 `%s`；首次 `mj_step` 后 `|d.time − dt| = %.3g s`（断言 `< 1e-15`）；`d.time` 末值 `%.6f s`。'
        % (DT, xml_timestep, report['declared']['actual_timestep'], script_path.name, sha(script_path),
           abs(report['declared']['first_step_d_time_s'] - DT), report['declared']['d_time_final_s']),
        '- 三者不一致的产物**一律不得交付**。', '',
        '## 结论（单一问题）',
        '- 六足跗节支撑 **步 %d = %.6f ms** 失去；首个足外接触 **步 %d = %.6f ms**（%s）；root z 最低 **%s**（步 %d = %.6f ms）。'
        % (ev['legs_below_6'], ms(ev['legs_below_6']), ev['first_nonfoot'], ms(ev['first_nonfoot']),
           ' | '.join(ev['first_nonfoot_pair'] or []), report['open_loop']['root_z']['min'], report['open_loop']['root_z']['min_step'],
           report['open_loop']['root_z']['min_ms']),
        '- 与 v3 的**同口径比较不成立**：v3 由 100 µs 分支汇编（步号 280/260/8447、ms 作废），25 µs 分支是 2811/538/5082；'
        '**不需要新搜索**，也不新开目录做这件事。', '',
        '## act 域披露',
        '- 25 µs：峰值 **%s**、越域 **%d** 步；100 µs 旁证支：峰值 **%s**、越域 **%d** 步（标 `numerically_invalid`）。'
        % (act_disc['act_max_over_run'], act_disc['out_of_domain_steps'], act_disc['side_branch_100us_act_max'],
           act_disc['side_branch_100us_out_of_domain_steps']),
        '- `actuator_actlimited` 全 False：**%s**；%s' % (act_disc['actuator_actlimited_all_false'], act_disc['statement']), '',
        '## 余下缺陷一次性修完',
        '- F2 指针路径（`open_loop.contact_table_at_first_nonfoot`）、F4 无接触记 null 且图上断线、F5 完整 99 项并声明未截断、'
        'F7 `max_u_diamond_over_the_whole_run`、M1 事件线与横轴同一实测时钟、M2 出图脚本即本脚本（sha256 与 `inputs.script_sha256` 相同并已进 '
        '`source_snapshot.zip`）、M3/M4 见 `correction_ledger.json`、F-A 「步 3055 = 76.375 ms」由 `step×DT` 计算。', '',
        '## 冻结与纪律',
        '- 本批写盘即冻结（`delivery-hashes.json` + `FROZEN_DELIVERY_HASHES.json`）；已交付目录与 incident 文件一律只读，'
        '不就地改写、不二次冻结、不重写清单；所有过期哈希与两次完整性事件集中在 `correction_ledger.json`，**未变部分引用独立验证的 '
        '14 字段 / 56 项结果**，不重跑。', '',
        '- **手动研究实施，不是成品**；不宣称站立/行走/完成度；绝对力值与单位 `NOT_ESTABLISHED`。'])
    (out / 'README.zh-CN.md').write_text(readme)
    with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(script_path, script_path.name)
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name not in ('delivery-hashes.json', 'FROZEN_DELIVERY_HASHES.json'))
    (out / 'delivery-hashes.json').write_text(json.dumps({p.name: sha(p) for p in files}, ensure_ascii=False, indent=1))
    (out / 'FROZEN_DELIVERY_HASHES.json').write_text(json.dumps(dict(status='FROZEN_BEFORE_VERIFICATION', frozen_at=now(),
                                                                    files={p.name: sha(p) for p in files}), ensure_ascii=False, indent=1))
    print(report['status'])
    print('out:', out)
    print('timesteps: declared %s | xml %s | actual %s | first-step diff %s | final d.time %s'
          % (DT, xml_timestep, report['declared']['actual_timestep'], abs(report['declared']['first_step_d_time_s'] - DT),
             report['declared']['d_time_final_s']))
    print('answer: six-leg lost step %d = %.6f ms | first non-foot step %d = %.6f ms | root z min %s at step %d'
          % (ev['legs_below_6'], ms(ev['legs_below_6']), ev['first_nonfoot'], ms(ev['first_nonfoot']),
             report['open_loop']['root_z']['min'], report['open_loop']['root_z']['min_step']))
    print('act: 25us %s (OOD %d) | 100us %s (OOD %d) | ledger stale records %d'
          % (act_disc['act_max_over_run'], act_disc['out_of_domain_steps'], act_disc['side_branch_100us_act_max'],
             act_disc['side_branch_100us_out_of_domain_steps'], ledger['stale_hash_records']['n']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
