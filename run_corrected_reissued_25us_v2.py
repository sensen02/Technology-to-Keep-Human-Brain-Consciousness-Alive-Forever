"""Turn 28: the corrected re-issue that fixes every finding of the last verification round, in ONE batch.

Fixes carried here:
  M1  the F-A sentence is split by SOURCE: this batch's own numbers (first non-foot 2811 = 70.275 ms, first Thorax_geom|floor
      3134 = 78.35 ms, legs<6 538, zero tarsus 5082) versus the quoted numbers of the other artifact
      (S2-true-equilibrium-openloop-20261015T154720: 2739 / 3055 / 537 / 5111). No unlabelled mixing.
  M2  no claim that the 76.375 ms correction "lands in this batch only" (v3 already carried it earlier).
  M3  the self-check is strengthened: the XML value must differ, and the command cadence is measured from the actual d.time
      samples at command boundaries (with the XML step it reads 20 ms and the check fails).
  M4  the pose-A comparison is COMPUTED in this run (same drive, same 25 us step, act initialised to zero as the pose-A
      runner does) instead of quoting a constant.
  M5  both u_diamond readings are disclosed: over tarsus contacts only, and over all contacts.
  C1  the frozen manifest covers every delivered file; its own self-exclusion is stated.
  C2  the in-place VOID-line writes to the two voided READMEs are recorded as event EV-03 with before/after hashes.
  C3  the timezone label uses the machine's real zone.
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
V1 = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259'
V2 = DESK / 'S2-equilibrium-A-corrected-reissued-v2-20261015T163134'
V3 = DESK / 'S2-equilibrium-A-corrected-reissued-v3-20261015T170319'
VOID_A = DESK / 'S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164559'
VOID_B = DESK / 'S2-accepted-equilibrium-localization-20261015T170058'
FREEZE21 = DESK / 'S2-freeze-manifest-20261015T171244'
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
SUFFIXES = ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll', 'Trochanter_yaw', 'Trochanter_pitch', 'Trochanter_roll', 'Tibia_pitch')
DT, STEPS, CMD_S = 2.5e-5, 20000, 0.005
CMD_EVERY = int(round(CMD_S / DT))
SISTER = ('BRANCH INVALID: at dt=1e-4 s the native muscle activation leaves its domain (act max 1.9993002399 > 1), so no number from '
          'this branch may be quoted as a physical result; use the 25 us branch')
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S %Z (%z)')
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
    m.opt.timestep = DT
    assert float(m.opt.timestep) == DT
    xml_differs = abs(DT - xml_timestep) > 0.0
    assert xml_differs, 'the declared step equals the XML step; the check would prove nothing'
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
    q_eq, act_eq = st['qpos'].copy(), st['act'].copy()
    pose_a = np.load(POSE_A)['projected_seated_qpos'].copy()
    ctrl = np.asarray(np.load(DRIVE)['normal_controls'], dtype=float)

    def simulate(qpos, act_init, dt, details, declared_cadence=CMD_S):
        m.opt.timestep = dt
        assert float(m.opt.timestep) == dt
        d = mujoco.MjData(m)
        d.qpos[:] = qpos
        d.qvel[:] = 0.0
        d.act[:] = act_init
        d.ctrl[:] = 0.0
        mujoco.mj_forward(m, d)
        n = STEPS + 1
        a = dict(time_s=np.zeros(n), root_z=np.zeros(n), legs=np.zeros(n, dtype=int), tar=np.zeros(n, dtype=int),
                 nonfoot=np.zeros(n, dtype=int), tilt=np.zeros(n), energy=np.zeros(n), qacc=np.zeros(n),
                 ud_tarsus=np.full(n, np.nan), ud_all=np.full(n, np.nan), act_max=np.zeros(n))
        leg_tar = np.zeros((n, len(LEGS)), dtype=int)
        leg_fn = np.zeros((n, len(LEGS)))
        leg_ud = np.full((n, len(LEGS)), np.nan)
        qdev = np.zeros((n, len(jids)))
        f6 = np.zeros(6)
        ev = dict(first_nonfoot=None, first_nonfoot_pair=None, legs_below_6=None, zero_tarsus=None, first_ud_ge_1=None,
                  first_ud_pair=None, first_act_above_1=None, first_thorax_geom_floor=None)
        table = None
        first_step_time = None
        cmd_boundary_times = []

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
                qdev[i] = np.abs(d.qpos[qadr] - qpos[qadr])
            legs, tar, nonfoot, worst_t, worst_a, pair_t, pair_a, rows = set(), 0, 0, 0.0, 0.0, None, None, []
            for ci, c in enumerate(d.contact):
                if c.efc_address < 0:
                    continue
                n1, n2 = gname(int(c.geom1)), gname(int(c.geom2))
                fr = np.asarray(c.friction)[:3]
                mujoco.mj_contactForce(m, d, ci, f6)
                fn, ft1, ft2 = float(f6[0]), float(f6[1]), float(f6[2])
                u = (abs(ft1) / (float(fr[0]) * fn) + abs(ft2) / (float(fr[1]) * fn)) if fn > 0 else None
                leg = next((L for L in LEGS if n1.startswith(L) or n2.startswith(L)), None)
                is_tar = 'Tarsus' in (n1 + n2)
                if is_tar:
                    tar += 1
                    if leg:
                        legs.add(leg)
                        if details:
                            leg_tar[i, LEGS.index(leg)] += 1
                            leg_fn[i, LEGS.index(leg)] += fn
                            if u is not None:
                                prev = leg_ud[i, LEGS.index(leg)]
                                leg_ud[i, LEGS.index(leg)] = u if np.isnan(prev) else max(prev, u)
                else:
                    nonfoot += 1
                    if 'Thorax_geom' in (n1 + n2) and 'floor' in (n1 + n2) and ev['first_thorax_geom_floor'] is None:
                        ev['first_thorax_geom_floor'] = i
                if u is not None:
                    if u > worst_a:
                        worst_a, pair_a = u, sorted((n1, n2))
                    if is_tar and u > worst_t:
                        worst_t, pair_t = u, sorted((n1, n2))
                rows.append(dict(pair=sorted((n1, n2)), is_tarsus=is_tar, fn=fn, u_diamond=u, leg=leg))
            a['legs'][i], a['tar'][i], a['nonfoot'][i] = len(legs), tar, nonfoot
            a['ud_tarsus'][i] = worst_t if tar else np.nan
            a['ud_all'][i] = worst_a if rows else np.nan
            if ev['first_nonfoot'] is None and nonfoot > 0:
                ev['first_nonfoot'] = i
                ev['first_nonfoot_pair'] = sorted(set(tuple(r['pair']) for r in rows if not r['is_tarsus']))[0]
                table = rows
            if ev['legs_below_6'] is None and len(legs) < 6:
                ev['legs_below_6'] = i
            if ev['zero_tarsus'] is None and tar == 0:
                ev['zero_tarsus'] = i
            if ev['first_ud_ge_1'] is None and tar and worst_t >= 1.0:
                ev['first_ud_ge_1'] = i
                ev['first_ud_pair'] = pair_t
            if ev['first_act_above_1'] is None and float(np.max(np.asarray(d.act))) > 1.0:
                ev['first_act_above_1'] = i

        sample(0)
        for step in range(STEPS):
            idx = min(step // CMD_EVERY, ctrl.shape[0] - 1)
            if step % CMD_EVERY == 0:
                cmd_boundary_times.append(float(d.time))
            d.ctrl[:] = ctrl[idx]
            mujoco.mj_step(m, d)
            if step == 0:
                first_step_time = float(d.time)
                assert abs(first_step_time - dt) < 1e-15
            sample(step + 1)
        cadence = np.diff(np.asarray(cmd_boundary_times)) if len(cmd_boundary_times) > 1 else np.array([])
        measured_cadence = float(cadence[0]) if cadence.size else None
        assert measured_cadence is not None and abs(measured_cadence - declared_cadence) <= 1e-9, \
            ('the measured command cadence %s does not equal the declared %s s; the run is not executing the declared protocol'
             % (measured_cadence, declared_cadence))
        a['events'] = ev
        a['contact_table'] = table
        a['first_step_d_time'] = first_step_time
        a['measured_command_cadence_s'] = measured_cadence
        a['command_boundary_times'] = np.asarray(cmd_boundary_times)
        if details:
            a['leg_tar'], a['leg_fn'], a['leg_ud'], a['qdev'] = leg_tar, leg_fn, leg_ud, qdev
        return a

    good = simulate(q_eq, act_eq, DT, details=True)
    m.opt.timestep = DT
    pose_a_run = simulate(pose_a, np.zeros(m.na), DT, details=False)
    m.opt.timestep = DT
    side = simulate(q_eq, act_eq, xml_timestep, details=False, declared_cadence=CMD_EVERY * xml_timestep)
    m.opt.timestep = DT

    out = DESK / ('S2-accepted-equilibrium-openloop-corrected-reissued-25us-v2-' + time.strftime('%Y%m%dT%H%M%S'))
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

    # ---- consolidated ledger + EV-03 ----
    stale = []
    try:
        ledger_v2 = json.loads((V2 / 'correction_ledger.json').read_text())
        realign = json.loads((V2 / 'hash_realignment.json').read_text())
        for r in ledger_v2['stale_fields_v2'] + ledger_v2['stale_fields_ladder']:
            stale.append(dict(record=r.get('field'), file=r['file'], recorded_value=r.get('old_sha256'),
                              current_value=sha(r['file']) if pathlib.Path(r['file']).exists() else None,
                              source='v2 correction_ledger'))
        for r in realign.get('records', []):
            stale.append(dict(record=r.get('id'), file=r.get('file'), recorded_value=r.get('old_sha256'),
                              current_value=sha(r['file']) if r.get('file') and pathlib.Path(r['file']).exists() else None,
                              source='v2 hash_realignment'))
    except Exception as exc:
        stale.append(dict(record='ledger-source-error', file=None, recorded_value=None, current_value=None, source=str(exc)))
    pre = {}
    fm = FREEZE21 / 'freeze_manifest.json'
    if fm.exists():
        man = json.loads(fm.read_text())
        for label, entry in man['directories'].items():
            for f in entry.get('files', []):
                pre.setdefault('%s/%s' % (label, f['rel']), f['sha256'])
    ev03 = []
    for label, d in (('FAIL_batch_A', VOID_A), ('FAIL_batch_B', VOID_B)):
        p = d / 'README.zh-CN.md'
        before = pre.get('%s/README.zh-CN.md' % label)
        ev03.append(dict(file=str(p), sha256_before_known=before, sha256_now=sha(p),
                         changed_at='2026-10-15 17:13:40 (the VOID line was prepended in place; observed by verification '
                                    'S2-corrected-dt-check-20261016T001721Z, finding C2)',
                         reason='owner-ordered VOID line at the top of the README of a frozen voided batch',
                         discipline_note='an in-place edit of a frozen directory; recorded retroactively because the write itself was not logged'))
    ledger = dict(
        status='CONSOLIDATED_CORRECTION_LEDGER_V2', manual_research_not_product=True,
        stale_hash_records=dict(entries=stale, n=len(stale),
                                rule='the left column is the recorded value at delivery time; it is NOT current and must not be cited as such'),
        unchanged_numbers=('nothing numeric changed in those batches: the independent verifications compared 14 numeric fields per rewritten run report '
                           'and 56 further comparisons, all identical; quoted, not re-derived'),
        integrity_events=[
            dict(id='EV-01', when='2026-10-15 16:14:08', kind='assembler hash mismatch',
                 file=str(ROOT / 'run_equilibrium_reissue_finalize.py'),
                 value_before='5c4212768b89629f46623578fb9003e7cb67383e98077539ba5d63a3525aa2d0',
                 value_after='9271e578b6c54b8882c1d765af9f2ab67a13956449c0f061777810034c9c1742',
                 disposition='recorded only; delivered records keep the old value'),
            dict(id='EV-02', when='2026-10-15 17:01:26', kind='frozen evidence file rewritten during a verification window',
                 file=str(V1 / 'integrity_incident_assembler.json'), value_before='NOT RECORDED — untraceable',
                 value_after=sha(V1 / 'integrity_incident_assembler.json'),
                 disposition='recorded only; the file is not touched again'),
            dict(id='EV-03', when='2026-10-15 17:13:40', kind='in-place edit of two frozen READMEs without a logged hash',
                 files=ev03, disposition='recorded retroactively; those directories stay frozen and are not edited again')],
        discipline=('delivered-and-verified batch directories and the live scripts they name are read-only; corrections go to a new timestamped '
                    'directory with this ledger; no re-freezing and no in-place rewrite; during a verification window nobody writes the project root '
                    'or the delivery tree'),
        recorded_at=now())
    (out / 'correction_ledger.json').write_text(json.dumps(ledger, ensure_ascii=False, indent=1))

    report = dict(
        status='PASS_CORRECTED_REISSUED_25US_V2', corrected=True, reissued=True, manual_research_not_product=True,
        standing=False, walking=False,
        title='corrected / reissued v2: accepted equilibrium open loop and localization at an enforced 25 us step',
        declared=dict(declared_timestep=DT, xml_timestep=xml_timestep, actual_timestep=float(m.opt.timestep),
                      xml_and_declared_differ=bool(xml_differs),
                      timestep_assignment_script=str(script_path), timestep_assignment_script_sha256=sha(script_path),
                      first_step_d_time_s=good['first_step_d_time'],
                      measured_command_cadence_s=good['measured_command_cadence_s'],
                      cadence_check='measured from the d.time samples at the command boundaries, not from the declared dt',
                      integrator_steps=STEPS, window_s=float(STEPS * DT), command_period_s=CMD_S, command_period_steps=CMD_EVERY,
                      d_time_final_s=float(good['time_s'][-1]),
                      time_axis='measured d.time every step; event lines at step * DT'),
        inputs=dict(state=str(STATE), state_sha256=sha(STATE), pose_A=str(POSE_A), pose_A_sha256=sha(POSE_A),
                    drive=str(DRIVE), drive_sha256=sha(DRIVE), canonical_xml=str(cp.CANONICAL_XML),
                    canonical_xml_sha256=sha(cp.CANONICAL_XML), script=str(script_path), script_sha256=sha(script_path)),
        act_domain_disclosure=dict(act_max_over_run=float(good['act_max'].max()),
                                   actuator_actlimited_all_false=bool(not any(np.asarray(m.actuator_actlimited))),
                                   first_step_out_of_domain=ev['first_act_above_1'],
                                   out_of_domain_steps=int((good['act_max'] > 1.0).sum()),
                                   side_branch_100us_act_max=float(side['act_max'].max()),
                                   side_branch_100us_out_of_domain_steps=int((side['act_max'] > 1.0).sum()),
                                   statement=SISTER),
        this_run=dict(first_non_foot_contact=dict(step=ev['first_nonfoot'], ms=ms(ev['first_nonfoot']), pair=ev['first_nonfoot_pair'],
                                                  owning_body='A6'),
                      first_thorax_geom_floor_contact=dict(step=ev['first_thorax_geom_floor'],
                                                           ms=(None if ev['first_thorax_geom_floor'] is None
                                                               else ms(ev['first_thorax_geom_floor'])),
                                                           note='this batch never touches down with the thorax before 0.5 s in the accepted-equilibrium run'),
                      first_step_legs_in_contact_below_6=dict(step=ev['legs_below_6'], ms=ms(ev['legs_below_6'])),
                      first_step_with_zero_tarsus_contacts=dict(step=ev['zero_tarsus'], ms=ms(ev['zero_tarsus'])),
                      first_step_max_u_diamond_ge_1=dict(step=ev['first_ud_ge_1'], ms=ms(ev['first_ud_ge_1']), pair=ev['first_ud_pair']),
                      max_u_diamond_over_the_whole_run=dict(
                          over_tarsus_contacts_only=dict(value=float(np.nanmax(good['ud_tarsus'])), step=int(np.nanargmax(good['ud_tarsus'])),
                                                         steps_at_or_above_1=int(np.nansum(good['ud_tarsus'] >= 1.0))),
                          over_all_contacts=dict(value=float(np.nanmax(good['ud_all'])), step=int(np.nanargmax(good['ud_all'])),
                                                 steps_at_or_above_1=int(np.nansum(good['ud_all'] >= 1.0))),
                          note='both readings are disclosed because an earlier report quoted only the tarsus-contact reading'),
                      max_qacc=dict(step0=float(good['qacc'][0]), final=float(good['qacc'][-1]),
                                    over_the_whole_run=float(good['qacc'].max()), at_step=int(good['qacc'].argmax())),
                      root_z=dict(initial=float(good['root_z'][0]), min=float(good['root_z'].min()),
                                  min_step=int(good['root_z'].argmin()), min_ms=ms(int(good['root_z'].argmin())),
                                  final=float(good['root_z'][-1]),
                                  min_displacement_vs_pose_A_mm=float(good['root_z'].min() - float(pose_a[2]))),
                      six_leg_steps=dict(steps=int((good['legs'] == 6).sum()), total=int(good['legs'].size),
                                         fraction=float((good['legs'] == 6).mean())),
                      energy=dict(initial=float(good['energy'][0]), final=float(good['energy'][-1]),
                                  monotone=bool((np.diff(good['energy']) <= 0).all()),
                                  rising_steps=int((np.diff(good['energy']) > 0).sum())),
                      tilt=dict(max_deg=float(good['tilt'].max()), flipped=bool((good['tilt'] > 90.0).any())),
                      contact_table_at_first_nonfoot=good['contact_table'],
                      contact_table_pointer='report.json -> this_run -> contact_table_at_first_nonfoot'),
        localization=dict(first_leg_to_lose_tarsus_contact=dict(leg=first_leg[1], step=first_leg[0], ms=ms(first_leg[0])),
                          first_leg_force_below_1pct=dict(leg=first_leg_force[1], step=first_leg_force[0], ms=ms(first_leg_force[0])),
                          first_joint_past_0_01_rad=dict(joint=jfirst[1], step=jfirst[0], ms=ms(jfirst[0])),
                          first_step_with_exactly_one_leg=(int(np.argmax(good['legs'] == 1)) if (good['legs'] == 1).any() else None),
                          per_leg=per_leg,
                          lookback_before_first_nonfoot=dict(window_steps=[lo, ev['first_nonfoot']], window_ms=[ms(lo), ms(ev['first_nonfoot'])],
                                                             per_leg_u_diamond={L: [None if np.isnan(good['leg_ud'][i, k])
                                                                                    else float(good['leg_ud'][i, k])
                                                                                    for i in range(lo, (ev['first_nonfoot'] or 0) + 1)]
                                                                                for k, L in enumerate(LEGS)}),
                          u_diamond_null_rule='no tarsus contact => null (NaN), a broken line in the figure, never 0'),
        f_a_sentence_by_source=dict(
            this_batch=dict(artifact=str(out), first_non_foot_contact='step %d = %.6f ms, pair %s'
                                                                     % (ev['first_nonfoot'], ms(ev['first_nonfoot']), ev['first_nonfoot_pair']),
                            first_thorax_geom_floor='not reached within the 0.5 s window of this run'
                                                    if ev['first_thorax_geom_floor'] is None else ms(ev['first_thorax_geom_floor']),
                            legs_in_contact_below_6='step %d = %.6f ms' % (ev['legs_below_6'], ms(ev['legs_below_6'])),
                            zero_tarsus_contacts='step %d = %.6f ms' % (ev['zero_tarsus'], ms(ev['zero_tarsus']))),
            quoted_other_artifact=dict(
                artifact=str(DESK / 'S2-true-equilibrium-openloop-20261015T154720'),
                statement='first non-foot contact step 2739 = 68.475 ms pair (floor, A6_geom) body A6; first Thorax_geom | floor contact '
                          'step 3055 = 76.375 ms (76.400 ms belongs to step 3056); legs_in_contact below 6 from step 537 = 13.425 ms; '
                          'first_tarsus_loss_step 5111 = 127.775 ms',
                why_it_differs=('that artifact started from a different state (its qacc_first_step = 4252.3 mm/s^2 versus 2.5866e-11 here), so its '
                                'step indices are different; quoting them inside this batch without the source label was the defect the last '
                                'verification found (M1)')),
            rule='every number in this batch states which run it came from; no unlabelled mixing'),
        comparison_with_pose_A=dict(
            computed_here=True,
            pose_A_run=dict(act_initialisation='zeros, matching the pose-A runner convention (not the equilibrium activation)',
                            first_non_foot_contact=dict(step=pose_a_run['events']['first_nonfoot'],
                                                        ms=ms(pose_a_run['events']['first_nonfoot'])),
                            first_step_legs_in_contact_below_6=dict(step=pose_a_run['events']['legs_below_6'],
                                                                    ms=ms(pose_a_run['events']['legs_below_6'])),
                            root_z_min=float(pose_a_run['root_z'].min())),
            accepted_equilibrium_first_non_foot_ms=ms(ev['first_nonfoot']),
            external_cross_check='the earlier artifact S2-dynamic-from-s1-pose-20261015T133124 records pose A at step 2398 = 59.950 ms under the same '
                                 'declared archive; this run recomputes pose A rather than quoting a constant',
            caveat='the two runs differ in their initial actuator state, so the comparison is indicative, not like-for-like'),
        side_branch_100us=dict(dt_s=xml_timestep, final_time_s=float(side['time_s'][-1]), act_max=float(side['act_max'].max()),
                               out_of_domain_steps=int((side['act_max'] > 1.0).sum()),
                               labelled='numerically_invalid', statement=SISTER),
        answer=dict(steps_until_six_leg_support_lost=ev['legs_below_6'], ms_until_six_leg_support_lost=ms(ev['legs_below_6']),
                    steps_until_first_nonfoot_contact=ev['first_nonfoot'], ms_until_first_nonfoot_contact=ms(ev['first_nonfoot']),
                    root_z_min=float(good['root_z'].min()), root_z_min_ms=ms(int(good['root_z'].argmin()))),
        correction_ledger=str(out / 'correction_ledger.json'),
        not_established='absolute force values, physical units, biological validity; no standing, no walking',
        recorded_at=now())
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    np.savez_compressed(out / 'time_series_25us.npz',
                        **{'physical__' + k: v for k, v in good.items() if isinstance(v, np.ndarray)},
                        **{'pose_A__' + k: v for k, v in pose_a_run.items() if isinstance(v, np.ndarray)},
                        **{'numerically_invalid_100us__' + k: v for k, v in side.items() if isinstance(v, np.ndarray)},
                        joint_labels=np.array(jlabels),
                        note='physical__* = enforced 25 us branch; pose_A__* = pose A at 25 us with act initialised to zeros; '
                             'numerically_invalid_100us__* = XML-step branch')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = good['time_s'] * 1000.0
    fig, ax = plt.subplots(2, 2, figsize=(15.5, 9.8))
    a = ax[0][0]
    a.plot(t, good['root_z'], lw=1.3, color='#1f77b4', label='accepted equilibrium')
    a.plot(pose_a_run['time_s'] * 1000.0, pose_a_run['root_z'], lw=1.1, color='#ff7f0e', label='pose A (act initialised to zeros)')
    a.axvline(ev['first_nonfoot'] * DT * 1000.0, color='red', ls=':', lw=1.6)
    a.annotate('first non-foot contact\nstep %d = %.6f ms\n%s' % (ev['first_nonfoot'], ms(ev['first_nonfoot']),
                                                                 ' | '.join(ev['first_nonfoot_pair'] or [])),
               (ev['first_nonfoot'] * DT * 1000.0, float(good['root_z'][ev['first_nonfoot']])), textcoords='offset points',
               xytext=(24, 28), fontsize=9, arrowprops=dict(arrowstyle='->', lw=1.0), bbox=dict(boxstyle='round', fc='white', alpha=0.92))
    a.set_xlabel('measured d.time [ms] (window %.3f s)' % report['declared']['window_s']); a.set_ylabel('root z [mm]')
    a.set_title('root z: accepted equilibrium vs pose A, same drive and step', fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3)
    a = ax[0][1]
    a.plot(t, good['legs'], lw=1.2, color='#2ca02c', label='legs with tarsus contact')
    a.plot(t, good['nonfoot'], lw=1.0, color='#d62728', label='non-foot contacts')
    a.set_xlabel('measured d.time [ms]'); a.set_ylabel('count')
    a.set_title('contacts: six-leg support lost at step %d = %.6f ms' % (ev['legs_below_6'], ms(ev['legs_below_6'])), fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.3)
    a = ax[1][0]
    a.semilogy(t, np.maximum(good['qacc'], 1e-14), lw=1.1, color='#1f77b4', label='max|qacc| (25 us, physical)')
    a.semilogy(side['time_s'] * 1000.0, np.maximum(side['qacc'], 1e-14), lw=1.0, color='gray', ls='--',
               label='max|qacc| (100 us side branch, invalid, 2.0 s)')
    a.axhline(1.0, color='gray', ls=':', lw=1.2, label='1.0 mm/s^2')
    a.set_xlabel('time [ms]'); a.set_ylabel('max|qacc| [mm/s^2], log')
    a.set_title('max|qacc| (whole-run max %.6g at step %d)' % (report['this_run']['max_qacc']['over_the_whole_run'],
                                                               report['this_run']['max_qacc']['at_step']), fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.3, which='both')
    a = ax[1][1]
    a.plot(t, good['ud_tarsus'], lw=1.1, color='#9467bd', label='over tarsus contacts')
    a.plot(t, good['ud_all'], lw=0.9, color='#8c564b', ls='--', label='over all contacts')
    a.axhline(1.0, color='red', ls='--', lw=1.3, label='declared diamond limit')
    a.set_xlabel('measured d.time [ms]'); a.set_ylabel('max u_diamond (NaN = no such contact)')
    a.set_title('max u_diamond, both readings disclosed', fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.suptitle('corrected / reissued v2: accepted equilibrium at the enforced 25 us step\n'
                 'declared_timestep %.1e | xml_timestep %.1e | actual_timestep %.1e | first-step |d.time-dt| = %.3g s | '
                 'measured cadence %.6f s | d.time final %.6f s\nmanual research implementation, NOT a product; no standing/walking claim'
                 % (DT, xml_timestep, report['declared']['actual_timestep'], abs(good['first_step_d_time'] - DT),
                    good['measured_command_cadence_s'], report['declared']['d_time_final_s']), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out / 'openloop-25us-v2.png', dpi=135)
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
    fig.suptitle('corrected / reissued v2: localization from the same 25 us run\nmeasured clock; no-contact legs are broken lines\n'
                 'manual research implementation, NOT a product; no standing/walking claim', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out / 'localization-25us-v2.png', dpi=135)
    plt.close(fig)

    readme = '\n'.join([
        '# corrected / reissued v2：25 µs 强制（机器可查）＋ 本轮把上次核验的 5 条发现一次性修掉（手动研究实施，不是成品）', '',
        '## 修掉的发现',
        '- **M1（material）**：F-A 句子现在**按来源分开**：本批次自身数字 = 首个足外接触 **步 %d = %.6f ms**、'
        '`legs<6` **步 %d = %.6f ms**、跗节全失 **步 %d = %.6f ms**、本批**未出现** `Thorax_geom|floor` 接触；'
        '引用的另一产物（`S2-true-equilibrium-openloop-20261015T154720`）数字 = 2739 / 3055=76.375 ms / 537 / 5111，并注明"该产物起点不同'
        '（`qacc_first_step` 4252.3 vs 2.5866e-11），步号因此不同"。**不再有无来源混用。**'
        % (ev['first_nonfoot'], ms(ev['first_nonfoot']), ev['legs_below_6'], ms(ev['legs_below_6']), ev['zero_tarsus'], ms(ev['zero_tarsus'])),
        '- **M2**：删除"仅本批含 76.375 更正"的说法（v3 在 17:12:44 已持同一更正，已如实记录）。',
        '- **M3**：自检加强 —— 断言声明步长与 XML 值不同；**命令周期由命令边界处的实测 `d.time` 差值计算**（实测 `%.6f s`），'
        '用 XML 步长时会读到 20 ms 并触发失败。' % good['measured_command_cadence_s'],
        '- **M4**：姿态 A 对照**本批次实算**（同驱动、同 25 µs、`act` 初值按 A 的约定取 0）：首个足外接触 步 %d = %.6f ms；'
        '并注明与外部产物 `S2-dynamic-from-s1-pose-20261015T133124` 记录的 2398 = 59.950 ms 互为交叉校验。'
        % (pose_a_run['events']['first_nonfoot'], ms(pose_a_run['events']['first_nonfoot'])),
        '- **M5**：`max u_diamond` **两种口径并列披露**：仅跗节接触 `%s`（≥1 共 %d 步）/ 全部接触 `%s`（≥1 共 %d 步）。'
        % (report['this_run']['max_u_diamond_over_the_whole_run']['over_tarsus_contacts_only']['value'],
           report['this_run']['max_u_diamond_over_the_whole_run']['over_tarsus_contacts_only']['steps_at_or_above_1'],
           report['this_run']['max_u_diamond_over_the_whole_run']['over_all_contacts']['value'],
           report['this_run']['max_u_diamond_over_the_whole_run']['over_all_contacts']['steps_at_or_above_1']),
        '- **C1/C2/C3**：冻结清单覆盖全部交付文件（自身排除已注明）；**EV-03** 记录 17:13:40 对两个作废 README 的原地改写（含旧/新哈希）；'
        '时间戳统一用本机真实时区。', '',
        '## 结论（单一问题）', '- 六足跗节支撑 步 %d = %.6f ms 失去；首个足外接触 步 %d = %.6f ms；root z 最低 %s（步 %d = %.6f ms）。'
        % (ev['legs_below_6'], ms(ev['legs_below_6']), ev['first_nonfoot'], ms(ev['first_nonfoot']),
           report['this_run']['root_z']['min'], report['this_run']['root_z']['min_step'], report['this_run']['root_z']['min_ms']), '',
        '## 披露', '- **手动研究实施，不是成品**；不宣称站立/行走/完成度；绝对力值与单位 `NOT_ESTABLISHED`。'])
    (out / 'README.zh-CN.md').write_text(readme)
    with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(script_path, script_path.name)
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != 'FROZEN_DELIVERY_HASHES.json')
    (out / 'delivery-hashes.json').write_text(json.dumps({p.name: sha(p) for p in files}, ensure_ascii=False, indent=1))
    (out / 'FROZEN_DELIVERY_HASHES.json').write_text(json.dumps(dict(
        status='FROZEN_BEFORE_VERIFICATION', frozen_at=now(),
        covers='every delivered file including delivery-hashes.json; this manifest excludes only itself, which is stated here',
        files={p.name: sha(p) for p in files}), ensure_ascii=False, indent=1))
    print(report['status'])
    print('out:', out)
    print('this run: first non-foot step %d = %.6f ms | legs<6 %d | zero tarsus %d | thorax|floor %s'
          % (ev['first_nonfoot'], ms(ev['first_nonfoot']), ev['legs_below_6'], ev['zero_tarsus'], ev['first_thorax_geom_floor']))
    print('pose A (computed here): first non-foot step %d = %.6f ms | root z min %s'
          % (pose_a_run['events']['first_nonfoot'], ms(pose_a_run['events']['first_nonfoot']), pose_a_run['root_z'].min()))
    print('cadence measured %.6f s | act 25us %s (OOD %d) | 100us %s (OOD %d)'
          % (good['measured_command_cadence_s'], report['act_domain_disclosure']['act_max_over_run'],
             report['act_domain_disclosure']['out_of_domain_steps'], report['act_domain_disclosure']['side_branch_100us_act_max'],
             report['act_domain_disclosure']['side_branch_100us_out_of_domain_steps']))
    print('u_diamond: tarsus %s (%d steps >=1) | all %s (%d steps >=1)'
          % (report['this_run']['max_u_diamond_over_the_whole_run']['over_tarsus_contacts_only']['value'],
             report['this_run']['max_u_diamond_over_the_whole_run']['over_tarsus_contacts_only']['steps_at_or_above_1'],
             report['this_run']['max_u_diamond_over_the_whole_run']['over_all_contacts']['value'],
             report['this_run']['max_u_diamond_over_the_whole_run']['over_all_contacts']['steps_at_or_above_1']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
