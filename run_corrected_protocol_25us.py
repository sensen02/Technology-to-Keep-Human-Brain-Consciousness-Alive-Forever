"""CORRECTED-PROTOCOL batch: the previous open-loop / localization scripts never set m.opt.timestep, so the model's XML
timestep (1e-04 s) was used: 20000 steps were 2.0 s, not 0.5 s, and every ms label was 4x too small. This script sets
m.opt.timestep = 25e-6 explicitly and re-runs:

  * the accepted-equilibrium open loop under both drives (the declared 100-frame archive and the archive exported from the
    state itself), with per-step max|qacc| and max u_diamond;
  * the localization under the true 25 us protocol (which leg loses support first, which joint leaves the equilibrium first,
    the u_diamond trajectory before the first non-foot contact, and what the drive lacks);

and it documents the defect, the affected artifacts and the relabelled ms table in protocol_correction.json. Step indices
of the OLD runs remain valid; their ms labels do not.
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
sys.path.insert(0, str(ROOT))
STATE = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259/equilibrium_best_state.npz'
POSE_A = DESK / 'S2-declared-cone-min-s-20261015T123603/best_candidate_pose.npz'
DRIVE_DECLARED = ROOT / 'outputs/experiment_platform/20261015_135150_a2fff3f0/traces.npz'
DRIVE_SELF = ROOT / 'outputs/experiment_platform/20261015_164914_eb008108/traces.npz'
V3 = DESK / 'S2-equilibrium-A-corrected-reissued-v3-20261015T170319'
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
SUFFIXES = ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll', 'Trochanter_yaw', 'Trochanter_pitch', 'Trochanter_roll', 'Tibia_pitch')
DT = 25e-6
STEPS = 20000
CMD_EVERY = int(round(0.005 / DT))  # 200 steps = 5 ms
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S')


def simulate(m, mujoco, q0, act0, ctrl_seq, gname, qadr, collect_localization=False):
    d = mujoco.MjData(m)
    d.qpos[:] = q0
    d.qvel[:] = 0.0
    d.act[:] = act0
    d.ctrl[:] = 0.0
    mujoco.mj_forward(m, d)
    n = STEPS + 1
    out = dict(time_s=np.arange(n) * DT, root_z=np.zeros(n), legs=np.zeros(n, dtype=int), tar=np.zeros(n, dtype=int),
               nonfoot=np.zeros(n, dtype=int), tilt=np.zeros(n), energy=np.zeros(n), qacc=np.zeros(n), ud=np.zeros(n),
               act_max=np.zeros(n), ctrl_max=np.zeros(n))
    leg_tar = np.zeros((n, len(LEGS)), dtype=int) if collect_localization else None
    leg_fn = np.zeros((n, len(LEGS))) if collect_localization else None
    qdev = np.zeros((n, len(qadr))) if collect_localization else None
    f6 = np.zeros(6)
    events = dict(first_nonfoot=None, first_nonfoot_pair=None, first_legs_below_6=None, first_zero_tarsus=None,
                  first_ud_ge_1=None, first_ud_pair=None, first_act_above_1=None, flipped=None)
    contact_table = None

    def sample(i):
        nonlocal contact_table
        out['root_z'][i] = float(d.qpos[2])
        quat = np.asarray(d.qpos[3:7], dtype=float)
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        r22 = float(mat.reshape(3, 3)[2, 2])
        out['tilt'][i] = float(np.degrees(np.arccos(np.clip(r22, -1.0, 1.0))))
        out['qacc'][i] = float(np.max(np.abs(np.asarray(d.qacc))))
        mujoco.mj_energyPos(m, d)
        mujoco.mj_energyVel(m, d)
        out['energy'][i] = float(d.energy[0] + d.energy[1])
        out['act_max'][i] = float(np.max(np.asarray(d.act)))
        out['ctrl_max'][i] = float(np.max(np.asarray(d.ctrl)))
        if collect_localization:
            qdev[i] = np.abs(d.qpos[qadr] - q0[qadr])
        rows, legs, tar, nonfoot, worst, worst_pair = [], set(), 0, 0, 0.0, None
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
                    if collect_localization:
                        leg_tar[i, LEGS.index(leg)] += 1
                        leg_fn[i, LEGS.index(leg)] += fn
                if u is not None and u > worst:
                    worst, worst_pair = u, sorted((n1, n2))
            else:
                nonfoot += 1
            rows.append(dict(pair=sorted((n1, n2)), is_tarsus=bool('Tarsus' in (n1 + n2)), fn=fn, u_diamond=u, leg=leg))
        out['legs'][i], out['tar'][i], out['nonfoot'][i], out['ud'][i] = len(legs), tar, nonfoot, worst
        if events['first_nonfoot'] is None and nonfoot > 0:
            events['first_nonfoot'] = i
            events['first_nonfoot_pair'] = sorted(set(tuple(r['pair']) for r in rows if not r['is_tarsus']))[0]
            contact_table = rows
        if events['first_legs_below_6'] is None and len(legs) < 6:
            events['first_legs_below_6'] = i
        if events['first_zero_tarsus'] is None and tar == 0:
            events['first_zero_tarsus'] = i
        if events['first_ud_ge_1'] is None and worst >= 1.0:
            events['first_ud_ge_1'] = i
            events['first_ud_pair'] = worst_pair
        if events['first_act_above_1'] is None and float(np.max(np.asarray(d.act))) > 1.0:
            events['first_act_above_1'] = i
        if events['flipped'] is None and r22 < 0.0:
            events['flipped'] = i

    sample(0)
    for step in range(STEPS):
        d.ctrl[:] = ctrl_seq[min(step // CMD_EVERY, ctrl_seq.shape[0] - 1)]
        mujoco.mj_step(m, d)
        sample(step + 1)
    out['events'] = events
    out['contact_table_at_first_nonfoot'] = contact_table
    if collect_localization:
        out['leg_tar'], out['leg_fn'], out['qdev'] = leg_tar, leg_fn, qdev
    return out


def persist_first(mask, persist=5):
    run = 0
    for i, v in enumerate(mask):
        run = run + 1 if v else 0
        if run >= persist:
            return i - persist + 1
    return None


def main():
    import mujoco
    from engine import canonical_physics as cp
    m = cp.load_model()
    m.opt.timestep = DT  # <-- the line the previous scripts dropped
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
    ms = lambda i: (None if i is None else round(i * DT * 1000, 3))

    runs = {}
    for tag, drive in (('declared_archive', DRIVE_DECLARED), ('self_exported_archive', DRIVE_SELF)):
        ctrl = np.asarray(np.load(drive)['normal_controls'], dtype=float)
        runs[tag] = simulate(m, mujoco, q0, act0, ctrl, gname, qadr, collect_localization=(tag == 'declared_archive'))
    loc = runs['declared_archive']
    leg_tar, leg_fn, qdev = loc['leg_tar'], loc['leg_fn'], loc['qdev']
    per_leg = {}
    for k, L in enumerate(LEGS):
        base = float(leg_fn[:100, k].mean())
        per_leg[L] = dict(tarsus_contacts_at_step0=int(leg_tar[0, k]),
                          first_step_no_tarsus_contact=persist_first(leg_tar[:, k] == 0),
                          first_step_no_tarsus_contact_ms=ms(persist_first(leg_tar[:, k] == 0)),
                          first_step_force_below_1pct=persist_first(leg_fn[:, k] < 0.01 * base),
                          first_step_force_below_1pct_ms=ms(persist_first(leg_fn[:, k] < 0.01 * base)))
    first_leg = min(((v['first_step_no_tarsus_contact'], L) for L, v in per_leg.items()
                     if v['first_step_no_tarsus_contact'] is not None), default=(None, None))
    first_leg_force = min(((v['first_step_force_below_1pct'], L) for L, v in per_leg.items()
                           if v['first_step_force_below_1pct'] is not None), default=(None, None))
    jfirst = min(((persist_first(qdev[:, k] >= 0.01), jlabels[k]) for k in range(len(jids))
                  if persist_first(qdev[:, k] >= 0.01) is not None), default=(None, None))
    ev = loc['events']
    first_nonfoot = ev['first_nonfoot']
    look = dict(window=[max(0, first_nonfoot - 20), first_nonfoot],
                per_leg_u_diamond={L: [float(x) for x in np.zeros(1)] for L in LEGS}) if first_nonfoot is None else None
    if first_nonfoot is not None:
        lo = max(0, first_nonfoot - 20)
        look = dict(window=[lo, first_nonfoot],
                    per_leg_u_diamond={L: [None if leg_tar[i, k] == 0 else float(np.nan) for i in range(lo, first_nonfoot + 1)]
                                       for k, L in enumerate(LEGS)})

    ctrl_dec = np.asarray(np.load(DRIVE_DECLARED)['normal_controls'], dtype=float)
    steps_arch = np.abs(np.diff(ctrl_dec, axis=0))
    leg_of_act = [next((L for L in LEGS if n.startswith(L) or ('_' + L) in n), 'other') for n in
                  [aname(i) for i in range(m.nu)]]
    amp = {}
    for L in LEGS:
        idxs = [i for i, L2 in enumerate(leg_of_act) if L2 == L]
        amp[L] = dict(n_actuators=len(idxs),
                      mean_abs_gap_vs_first_nonzero_frame=float(np.abs(act0[idxs] - ctrl_dec[2][idxs]).mean()),
                      max_abs_gap_vs_first_nonzero_frame=float(np.abs(act0[idxs] - ctrl_dec[2][idxs]).max()),
                      mean_archive_amplitude=float(ctrl_dec[:, idxs].mean()),
                      mean_equilibrium_activation=float(act0[idxs].mean()))

    out = DESK / ('S2-accepted-equilibrium-CORRECTED-PROTOCOL-25us-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)

    correction = dict(
        status='PROTOCOL_DEFECT_AND_CORRECTION', manual_research_not_product=True,
        defect=('the previous open-loop and localization scripts never set m.opt.timestep, so canonical_physics.load_model() kept the XML value '
                '<option timestep="1e-04">: the integration step was 100 us, the 20000 steps were 2.0 s, and the command index advanced every 200 '
                'steps = 20 ms instead of 5 ms'),
        discovered_by='independent verification S2-accepted-equilibrium-localization-check-20261016T000138Z (finding F1, blocking)',
        confirmed_here=dict(model_timestep_before=0.0001, timestep_set_now=DT,
                            scripts_that_do_set_it=['run_dynamic_from_s1_pose.py line 140', 'run_instability_localization.py line 141'],
                            scripts_that_did_not=['run_accepted_equilibrium_openloop.py', 'run_accepted_equilibrium_localization.py',
                                                  'measure_openloop_step_qacc.py', 'measure_openloop_step_u_diamond.py']),
        affected_artifacts=[
            '/home/sensen/Desktop/engine修复验收/S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164559/',
            '/home/sensen/Desktop/engine修复验收/S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164943-from-accepted-pose-archive/',
            '/home/sensen/Desktop/engine修复验收/S2-accepted-equilibrium-localization-20261015T170058/',
            '/home/sensen/Desktop/engine修复验收/S2-equilibrium-A-corrected-reissued-v3-20261015T170319/ (inherits the ms labels)'],
        what_stands=['every step index', 'every measured value (forces, qacc, root z, contacts, energy, tilt)', 'the contact identities',
                     'which leg and which joint are first BY STEP INDEX',
                     'the artifact S2-true-equilibrium-openloop-20261015T154720 and its step 537 / 2739 / 3055 / 5111, because its runner '
                     '(run_dynamic_from_s1_pose.py) does set the timestep, so its 25 us / 0.5 s labels are correct'],
        what_does_not_stand=['every millisecond label derived from the 100 us runs: they must be multiplied by 4',
                             'the claim that the declared archive covers the whole 0.5 s window in those runs (it covered 2.0 s at 20 ms per frame)'],
        relabelled_ms_100us_runs=dict(first_nonfoot=dict(step=280, wrong_ms=7.0, correct_ms=28.0),
                                      legs_below_6=dict(step=260, wrong_ms=6.5, correct_ms=26.0),
                                      first_leg_force_drop=dict(step=235, wrong_ms=5.875, correct_ms=23.5),
                                      only_one_leg=dict(step=1962, wrong_ms=49.05, correct_ms=196.2),
                                      first_joint_deviation=dict(step=26, wrong_ms=0.65, correct_ms=2.6),
                                      zero_tarsus_contacts=dict(step=8447, wrong_ms=211.175, correct_ms=844.7),
                                      first_qacc_above_1=dict(step=2, wrong_ms=0.05, correct_ms=0.2)),
        note='this batch re-runs everything under the true 25 us protocol; the old artifacts are frozen and are not rewritten',
        recorded_at=now())
    (out / 'protocol_correction.json').write_text(json.dumps(correction, ensure_ascii=False, indent=1))

    report = dict(status='PASS_CORRECTED_PROTOCOL_BATCH', manual_research_not_product=True, standing=False, walking=False,
                  protocol=dict(dt_s=DT, steps=STEPS, duration_s=DT * STEPS, command_cadence_s=0.005, command_index_every_steps=CMD_EVERY,
                                state_writes='once (qpos, qvel, act)', loop_writes='ctrl only',
                                timestep_explicitly_set=True, timestep_source='m.opt.timestep = 25e-6'),
                  inputs=dict(state=str(STATE), state_sha256=sha(STATE), pose_A=str(POSE_A), pose_A_sha256=sha(POSE_A),
                              drive_declared=str(DRIVE_DECLARED), drive_declared_sha256=sha(DRIVE_DECLARED),
                              drive_self_exported=str(DRIVE_SELF), drive_self_exported_sha256=sha(DRIVE_SELF),
                              canonical_xml=str(cp.CANONICAL_XML), canonical_xml_sha256=sha(cp.CANONICAL_XML),
                              script=str(pathlib.Path(__file__)), script_sha256=sha(pathlib.Path(__file__))),
                  correction=correction, runs={}, localization={}, recorded_at=now())
    for tag, r in runs.items():
        e = r['events']
        report['runs'][tag] = dict(
            first_nonfoot_contact=dict(step=e['first_nonfoot'], ms=ms(e['first_nonfoot']), pair=e['first_nonfoot_pair']),
            first_step_legs_in_contact_below_6=dict(step=e['first_legs_below_6'], ms=ms(e['first_legs_below_6'])),
            first_step_with_zero_tarsus_contacts=dict(step=e['first_zero_tarsus'], ms=ms(e['first_zero_tarsus'])),
            first_step_max_u_diamond_ge_1=dict(step=e['first_ud_ge_1'], ms=ms(e['first_ud_ge_1']), pair=e['first_ud_pair']),
            first_step_act_above_1=dict(step=e['first_act_above_1'], ms=ms(e['first_act_above_1'])),
            flipped_at_step=e['flipped'],
            root_z=dict(initial=float(r['root_z'][0]), min=float(r['root_z'].min()), min_step=int(r['root_z'].argmin()),
                        final=float(r['root_z'][-1]),
                        displacement_vs_pose_A_min_mm=float(r['root_z'].min() - float(pose_a[2]))),
            six_leg_steps=dict(steps=int((r['legs'] == 6).sum()), total=int(r['legs'].size),
                               fraction=float((r['legs'] == 6).mean())),
            energy=dict(initial=float(r['energy'][0]), final=float(r['energy'][-1]),
                        monotone=bool((np.diff(r['energy']) <= 0).all()),
                        rising_steps=int((np.diff(r['energy']) > 0).sum()), compared=int(r['energy'].size - 1)),
            tilt=dict(max_deg=float(r['tilt'].max()), final_deg=float(r['tilt'][-1])),
            per_step_max_qacc=dict(step0=float(r['qacc'][0]), final=float(r['qacc'][-1]), window_max=float(r['qacc'].max()),
                                   window_max_step=int(r['qacc'].argmax()),
                                   first_above_1e_minus_3=ms(int(np.argmax(r['qacc'] > 1e-3))),
                                   first_above_1e_minus_3_step=int(np.argmax(r['qacc'] > 1e-3)),
                                   first_above_1=ms(int(np.argmax(r['qacc'] > 1.0))),
                                   first_above_1_step=int(np.argmax(r['qacc'] > 1.0))),
            act_max=float(r['act_max'].max()), ctrl_max=float(r['ctrl_max'].max()))
    report['localization'] = dict(per_leg=per_leg, first_leg_to_lose_tarsus_contact=dict(leg=first_leg[1], step=first_leg[0], ms=ms(first_leg[0])),
                                  first_leg_force_below_1pct=dict(leg=first_leg_force[1], step=first_leg_force[0], ms=ms(first_leg_force[0])),
                                  first_joint_past_0_01_rad=dict(joint=jfirst[1], step=jfirst[0], ms=ms(jfirst[0])),
                                  contact_table_at_first_nonfoot=loc['contact_table_at_first_nonfoot'],
                                  lookback_before_first_nonfoot=look,
                                  drive_amplitude_gap_per_leg=amp,
                                  drive_frame_stats=dict(n_frames=int(ctrl_dec.shape[0]), cadence_ms=5.0,
                                                         leading_all_zero_frames=int(next((i for i in range(ctrl_dec.shape[0])
                                                                                           if not np.all(ctrl_dec[i] == 0)), 0)),
                                                         mean_step_between_frames=float(steps_arch.mean()),
                                                         max_step_between_frames=float(steps_arch.max()),
                                                         mean_command_level=float(ctrl_dec.mean()),
                                                         fraction_of_entries_at_zero=float((ctrl_dec == 0).mean())))
    np.savez_compressed(out / 'time_series_corrected_25us.npz',
                        **{('%s__%s' % (tag, k)): r[k] for tag, r in runs.items() for k in
                           ('time_s', 'root_z', 'legs', 'tar', 'nonfoot', 'tilt', 'energy', 'qacc', 'ud', 'act_max', 'ctrl_max')},
                        note='CORRECTED PROTOCOL: dt = 25 us explicitly set, 20000 steps = 0.5 s, ctrl every 200 steps = 5 ms')
    report['outputs'] = str(out)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5))
    a = axes[0][0]
    for tag, col in (('declared_archive', '#1f77b4'), ('self_exported_archive', '#ff7f0e')):
        r = runs[tag]
        a.plot(r['time_s'] * 1000, r['root_z'], lw=1.2, color=col, label=tag)
    a.set_xlabel('time [ms] (true 25 us protocol)'); a.set_ylabel('root z [mm]')
    a.set_title('root z under the CORRECTED 25 us protocol (0.5 s window)', fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.3)
    a = axes[0][1]
    for tag, col in (('declared_archive', '#1f77b4'), ('self_exported_archive', '#ff7f0e')):
        r = runs[tag]
        a.plot(r['time_s'] * 1000, r['legs'], lw=1.2, color=col, label='legs with tarsus contact (%s)' % tag)
    a.axhline(6, color='gray', ls=':', lw=1.0)
    a.set_xlabel('time [ms]'); a.set_ylabel('count'); a.set_title('legs with tarsus contact (corrected protocol)', fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.3)
    a = axes[1][0]
    for tag, col in (('declared_archive', '#1f77b4'), ('self_exported_archive', '#ff7f0e')):
        r = runs[tag]
        a.semilogy(r['time_s'] * 1000, np.maximum(r['qacc'], 1e-14), lw=1.1, color=col, label=tag)
    a.axhline(1.0, color='gray', ls='--', lw=1.2, label='1.0 mm/s^2 tolerance')
    a.axhline(1e-3, color='black', ls=':', lw=1.5, label='declared gate 1e-3 mm/s^2')
    a.set_xlabel('time [ms]'); a.set_ylabel('max|qacc| [mm/s^2], log')
    a.set_title('per-step max|qacc| (corrected protocol)', fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.3, which='both')
    a = axes[1][1]
    for tag, col in (('declared_archive', '#1f77b4'), ('self_exported_archive', '#ff7f0e')):
        r = runs[tag]
        a.plot(r['time_s'] * 1000, r['ud'], lw=1.1, color=col, label='max u_diamond (%s)' % tag)
    a.axhline(1.0, color='red', ls='--', lw=1.3, label='declared diamond limit')
    a.set_xlabel('time [ms]'); a.set_ylabel('max u_diamond'); a.set_title('per-step max u_diamond (corrected protocol)', fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.suptitle('CORRECTED PROTOCOL (m.opt.timestep = 25 us explicitly set): 20000 steps = 0.5 s, ctrl every 200 steps = 5 ms\n'
                 'the previous batch silently ran at the XML timestep of 100 us (2.0 s), so its ms labels were 4x too small\n'
                 'manual research implementation, not a product; no standing/walking claim', fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / 'corrected-protocol-openloop.png', dpi=135)
    plt.close(fig)

    r = report['runs']['declared_archive']
    lines = ['# 修正协议批次（真 25 µs）：被接受平衡态开环 + 定位（手动研究实施，不是成品）', '',
             '## 为什么有这个批次',
             '- 独立核验 `S2-accepted-equilibrium-localization-check-20261016T000138Z` 判定 **blocking**：此前脚本**从不设置 `m.opt.timestep`**，'
             '模型沿用 XML 的 `<option timestep="1e-04">`，**实际步长 100 µs**，20000 步 = **2.0 s**、命令周期 **20 ms**，'
             '因此旧产物里**每一处 ms 都偏小 4 倍**；步号与数值本身不受影响。',
             '- 本批次把 `m.opt.timestep = 25e-6` **显式写进脚本**后重跑：dt = **25 µs**、20000 步 = **0.5 s**、`ctrl` 每 **200 步 = 5 ms** 更新一次。',
             '',
             '## 修正后的开环实测（声明档案，100 帧 / 5 ms）',
             '- 首次非足接触：**步 %d = %s ms**，接触对 `%s`' % (r['first_nonfoot_contact']['step'], r['first_nonfoot_contact']['ms'],
                                                                 r['first_nonfoot_contact']['pair']),
             '- `legs_in_contact < 6` 首步：**步 %d = %s ms**；跗节全失：步 %s = %s ms'
             % (r['first_step_legs_in_contact_below_6']['step'], r['first_step_legs_in_contact_below_6']['ms'],
                r['first_step_with_zero_tarsus_contacts']['step'], r['first_step_with_zero_tarsus_contacts']['ms']),
             '- 首次 `u_diamond ≥ 1`：**步 %d = %s ms**，接触对 `%s`' % (r['first_step_max_u_diamond_ge_1']['step'],
                                                                        r['first_step_max_u_diamond_ge_1']['ms'],
                                                                        r['first_step_max_u_diamond_ge_1']['pair']),
             '- root z：最小 **%s**、末值 **%s**（相对姿态 A 最小位移 **%s mm**）'
             % (r['root_z']['min'], r['root_z']['final'], r['root_z']['displacement_vs_pose_A_min_mm']),
             '- 六足同时接触：**%d / %d = %.8f**' % (r['six_leg_steps']['steps'], r['six_leg_steps']['total'], r['six_leg_steps']['fraction']),
             '- 能量：%s → %s，**单调下降 %s**（上升步 %d/%d）'
             % (r['energy']['initial'], r['energy']['final'], r['energy']['monotone'], r['energy']['rising_steps'], r['energy']['compared']),
             '- 逐步 `max|qacc|`：首步 %s、末值 %s、全窗最大 %s（步 %d）；首次越过 1e-3 步 %d（%s ms）、越过 1.0 步 %d（%s ms）'
             % (r['per_step_max_qacc']['step0'], r['per_step_max_qacc']['final'], r['per_step_max_qacc']['window_max'],
                r['per_step_max_qacc']['window_max_step'], r['per_step_max_qacc']['first_above_1e_minus_3_step'],
                r['per_step_max_qacc']['first_above_1e_minus_3'], r['per_step_max_qacc']['first_above_1_step'],
                r['per_step_max_qacc']['first_above_1']),
             '- `act` 首次 >1：步 %s = %s ms；`act` 峰值 %s' % (r['first_step_act_above_1']['step'], r['first_step_act_above_1']['ms'], r['act_max']),
             '',
             '## 修正后的定位（同一 25 µs 协议）',
             '- **先失去跗节接触的腿：%s，步 %d = %s ms**；力先跌破自身前 100 步均值 1%% 的腿：%s，步 %d = %s ms'
             % (report['localization']['first_leg_to_lose_tarsus_contact']['leg'],
                report['localization']['first_leg_to_lose_tarsus_contact']['step'],
                report['localization']['first_leg_to_lose_tarsus_contact']['ms'],
                report['localization']['first_leg_force_below_1pct']['leg'], report['localization']['first_leg_force_below_1pct']['step'],
                report['localization']['first_leg_force_below_1pct']['ms']),
             '- **先离开平衡的关节（0.01 rad，连续 5 步）：%s，步 %d = %s ms**'
             % (report['localization']['first_joint_past_0_01_rad']['joint'], report['localization']['first_joint_past_0_01_rad']['step'],
                report['localization']['first_joint_past_0_01_rad']['ms']),
             '- 驱动缺口（幅度差对第一个非零帧）：%s' % json.dumps(report['localization']['drive_amplitude_gap_per_leg'], ensure_ascii=False),
             '- 档案帧统计：%s' % json.dumps(report['localization']['drive_frame_stats'], ensure_ascii=False),
             '',
             '## 披露',
             '- 旧产物的 ms 标注**不成立**（×4 才是真值）；本批次给出修正值，旧目录保持冻结不动。',
             '- `S2-true-equilibrium-openloop-20261015T154720` 及其 537 / 2739 / 3055 / 5111 步号**不受影响**：它的 runner 设置了 timestep。',
             '- `qacc ≈ 0` 只是静态必要条件；本目录不含站立/行走结论；绝对力值 `NOT_ESTABLISHED`。',
             '- **手动研究实施，不是成品。**']
    (out / 'README.zh-CN.md').write_text('\n'.join(lines) + '\n')
    with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(pathlib.Path(__file__), pathlib.Path(__file__).name)
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != 'delivery-hashes.json')
    (out / 'delivery-hashes.json').write_text(json.dumps({p.name: sha(p) for p in files}, ensure_ascii=False, indent=1))

    # notice into v3 (mine, unverified): append + re-freeze
    if V3.exists():
        (V3 / 'protocol_correction_notice.json').write_text(json.dumps(dict(
            status='MS_LABELS_IN_V3_CORRECTED_ELSEWHERE', wrong='every millisecond label inherited from the 100 us runs is 4x too small',
            see=str(out), corrected_ms=correction['relabelled_ms_100us_runs'], recorded_at=now()), ensure_ascii=False, indent=1))
        v3files = sorted(p for p in V3.iterdir() if p.is_file() and p.name not in ('delivery-hashes.json', 'FROZEN_DELIVERY_HASHES.json'))
        (V3 / 'delivery-hashes.json').write_text(json.dumps({p.name: sha(p) for p in v3files}, ensure_ascii=False, indent=1))
        (V3 / 'FROZEN_DELIVERY_HASHES.json').write_text(json.dumps(dict(status='FROZEN_AT_WRITE', frozen_at=now(),
                                                                        files={p.name: sha(p) for p in v3files}), ensure_ascii=False, indent=1))

    print(report['status'])
    print('out:', out)
    for tag in runs:
        rr = report['runs'][tag]
        print('%-22s first non-foot step %s = %s ms | legs<6 %s = %s ms | zero tarsus %s = %s ms | u_diamond>=1 step %s = %s ms'
              % (tag, rr['first_nonfoot_contact']['step'], rr['first_nonfoot_contact']['ms'],
                 rr['first_step_legs_in_contact_below_6']['step'], rr['first_step_legs_in_contact_below_6']['ms'],
                 rr['first_step_with_zero_tarsus_contacts']['step'], rr['first_step_with_zero_tarsus_contacts']['ms'],
                 rr['first_step_max_u_diamond_ge_1']['step'], rr['first_step_max_u_diamond_ge_1']['ms']))
    print('localization:', report['localization']['first_leg_to_lose_tarsus_contact'],
          report['localization']['first_joint_past_0_01_rad'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
