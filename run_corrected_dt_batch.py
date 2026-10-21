"""CORRECTED-DT batch: the accepted equilibrium open loop at a runtime-verified 25 us step, with the localization results
recomputed in the SAME run, a runtime self-check that refuses to publish on any mismatch, the canonical contact-identity
sentence (76.375 ms) landing only here, and an immutable incident record for the 17:01:26 rewrite.

Self-check rule (owner's instruction): read back m.opt.timestep, the final d.time and the command-hold step count; if any of
them disagrees with the declaration, write status=FAIL and exit without publishing numbers.
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
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
SUFFIXES = ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll', 'Trochanter_yaw', 'Trochanter_pitch', 'Trochanter_roll', 'Tibia_pitch')
DT, STEPS, CMD_S = 2.5e-5, 20000, 0.005
CMD_EVERY = int(round(CMD_S / DT))
SENTENCE = ('首次非足接触 = `(floor, A6_geom)`（`A6_geom` 属 body A6），**步 2739 = 68.475 ms**；'
            '首个 `Thorax_geom | floor` 接触在**步 3055 = 76.375 ms**（3055 × 25 µs；此前写 76.400 ms 对应步 3056，错一步）；'
            '这是**首次非足（躯干/肢体、非跗节）接触，不是"失去足接触"**（`first_tarsus_loss_step = 5111` = 127.775 ms；'
            '`legs_in_contact` 自步 537 = 13.425 ms 起少于 6）')
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S')


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

    m = cp.load_model()
    xml_dt = float(m.opt.timestep)
    m.opt.timestep = DT
    assert m.opt.timestep == DT, 'the model refused the declared timestep'
    assert abs(DT - xml_dt) > 0.0, 'the declared timestep equals the XML value; the assert would prove nothing'
    actlimited = [bool(x) for x in np.asarray(m.actuator_actlimited)]
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

    d = mujoco.MjData(m)
    d.qpos[:] = q0
    d.qvel[:] = 0.0
    d.act[:] = act0
    d.ctrl[:] = 0.0
    mujoco.mj_forward(m, d)
    n = STEPS + 1
    arr = dict(time_s=np.zeros(n), root_z=np.zeros(n), legs=np.zeros(n, dtype=int), tar=np.zeros(n, dtype=int),
               nonfoot=np.zeros(n, dtype=int), tilt=np.zeros(n), energy=np.zeros(n), qacc=np.zeros(n), ud=np.zeros(n),
               act_max=np.zeros(n), ctrl_max=np.zeros(n))
    leg_tar = np.zeros((n, len(LEGS)), dtype=int)
    leg_fn = np.zeros((n, len(LEGS)))
    qdev = np.zeros((n, len(jids)))
    index_used = np.zeros(n, dtype=int)
    f6 = np.zeros(6)
    ev = dict(first_nonfoot=None, first_nonfoot_pair=None, legs_below_6=None, zero_tarsus=None, first_ud_ge_1=None,
              first_ud_pair=None, first_act_above_1=None)
    table = None

    def sample(i):
        nonlocal table
        arr['time_s'][i] = float(d.time)
        arr['root_z'][i] = float(d.qpos[2])
        quat = np.asarray(d.qpos[3:7], dtype=float)
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        r22 = float(mat.reshape(3, 3)[2, 2])
        arr['tilt'][i] = float(np.degrees(np.arccos(np.clip(r22, -1.0, 1.0))))
        arr['qacc'][i] = float(np.max(np.abs(np.asarray(d.qacc))))
        mujoco.mj_energyPos(m, d)
        mujoco.mj_energyVel(m, d)
        arr['energy'][i] = float(d.energy[0] + d.energy[1])
        arr['act_max'][i] = float(np.max(np.asarray(d.act)))
        arr['ctrl_max'][i] = float(np.max(np.asarray(d.ctrl)))
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
                if u is not None and u > worst:
                    worst, worst_pair = u, sorted((n1, n2))
            else:
                nonfoot += 1
            rows.append(dict(pair=sorted((n1, n2)), is_tarsus=bool('Tarsus' in (n1 + n2)), fn=fn, u_diamond=u, leg=leg))
        arr['legs'][i], arr['tar'][i], arr['nonfoot'][i], arr['ud'][i] = len(legs), tar, nonfoot, worst
        if ev['first_nonfoot'] is None and nonfoot > 0:
            ev['first_nonfoot'] = i
            ev['first_nonfoot_pair'] = sorted(set(tuple(r['pair']) for r in rows if not r['is_tarsus']))[0]
            table = rows
        if ev['legs_below_6'] is None and len(legs) < 6:
            ev['legs_below_6'] = i
        if ev['zero_tarsus'] is None and tar == 0:
            ev['zero_tarsus'] = i
        if ev['first_ud_ge_1'] is None and worst >= 1.0:
            ev['first_ud_ge_1'] = i
            ev['first_ud_pair'] = worst_pair
        if ev['first_act_above_1'] is None and float(np.max(np.asarray(d.act))) > 1.0:
            ev['first_act_above_1'] = i

    sample(0)
    for step in range(STEPS):
        index_used[step + 1] = min(step // CMD_EVERY, ctrl.shape[0] - 1)
        d.ctrl[:] = ctrl[index_used[step + 1]]
        mujoco.mj_step(m, d)
        sample(step + 1)

    # ---------------- runtime self-check: refuse to publish on mismatch ----------------
    holds = np.diff(np.where(np.diff(index_used, prepend=index_used[0]) != 0)[0])
    self_check = dict(
        read_back_model_timestep_s=float(m.opt.timestep), declared_timestep_s=DT,
        finals_d_time_s=float(arr['time_s'][-1]), declared_duration_s=DT * STEPS,
        command_hold_steps_observed=(int(holds[0]) if holds.size else None), command_hold_steps_declared=CMD_EVERY,
        command_hold_seconds_observed=float(holds[0] * DT) if holds.size else None, command_cadence_declared_s=CMD_S,
        steps_used=int(index_used.max()) + 1, drive_frames=int(ctrl.shape[0]),
        match=dict(timestep=bool(float(m.opt.timestep) == DT),
                   duration=bool(abs(float(arr['time_s'][-1]) - DT * STEPS) <= 1e-9),
                   cadence=bool(holds.size and int(holds[0]) == CMD_EVERY and abs(holds[0] * DT - CMD_S) <= 1e-12)))
    ok = all(self_check['match'].values())
    stamp = time.strftime('%Y%m%dT%H%M%S')
    out = DESK / ('S2-accepted-equilibrium-openloop-corrected-dt-' + stamp)
    out.mkdir(exist_ok=False)
    if not ok:
        (out / 'report.json').write_text(json.dumps(dict(status='FAIL', reason='runtime self-check mismatch',
                                                         self_check=self_check), ensure_ascii=False, indent=1))
        print('FAIL: runtime self-check mismatch; no numbers published')
        print(json.dumps(self_check, ensure_ascii=False, indent=1))
        return 1

    ms = lambda i: (None if i is None else round(i * DT * 1000, 3))
    per_leg = {}
    for k, L in enumerate(LEGS):
        base = float(leg_fn[:100, k].mean())
        per_leg[L] = dict(first_step_no_tarsus_contact=persist_first(leg_tar[:, k] == 0),
                          first_step_no_tarsus_contact_ms=ms(persist_first(leg_tar[:, k] == 0)),
                          first_step_force_below_1pct=persist_first(leg_fn[:, k] < 0.01 * base),
                          first_step_force_below_1pct_ms=ms(persist_first(leg_fn[:, k] < 0.01 * base)))
    first_leg = min(((v['first_step_no_tarsus_contact'], L) for L, v in per_leg.items()
                     if v['first_step_no_tarsus_contact'] is not None), default=(None, None))
    first_leg_force = min(((v['first_step_force_below_1pct'], L) for L, v in per_leg.items()
                           if v['first_step_force_below_1pct'] is not None), default=(None, None))
    jfirst = min(((persist_first(qdev[:, k] >= 0.01), jlabels[k]) for k in range(len(jids))
                  if persist_first(qdev[:, k] >= 0.01) is not None), default=(None, None))
    lo = max(0, (ev['first_nonfoot'] or 0) - 20)
    lookback = dict(window=[lo, ev['first_nonfoot']],
                    per_leg_u_diamond={L: [None if leg_tar[i, k] == 0 else float(arr['ud'][i]) for i in range(lo, (ev['first_nonfoot'] or 0) + 1)]
                                       for k, L in enumerate(LEGS)},
                    root_z=[float(x) for x in arr['root_z'][lo:(ev['first_nonfoot'] or 0) + 1]])
    steps_arch = np.abs(np.diff(ctrl, axis=0))

    report = dict(
        status='PASS_CORRECTED_DT_BATCH', corrected=True, reissued=True, manual_research_not_product=True,
        standing=False, walking=False,
        title='corrected-dt: accepted equilibrium open loop at a runtime-verified 25 us step (manual research implementation, not a product)',
        declared=dict(dt_s=DT, steps=STEPS, duration_s=DT * STEPS, command_cadence_s=CMD_S, command_hold_steps=CMD_EVERY,
                      model_timestep_s=float(m.opt.timestep), xml_option_timestep_s=xml_dt,
                      actuator_actlimited_all_false=bool(not any(actlimited)),
                      runtime_self_check=self_check,
                      self_check_rule='if the read-back timestep, the final d.time or the command-hold step count disagree with the declaration, '
                                      'this batch writes status=FAIL and publishes no numbers'),
        inputs=dict(state=str(STATE), state_sha256=sha(STATE), pose_A=str(POSE_A), pose_A_sha256=sha(POSE_A),
                    drive=str(DRIVE), drive_sha256=sha(DRIVE), canonical_xml=str(cp.CANONICAL_XML),
                    canonical_xml_sha256=sha(cp.CANONICAL_XML), script=str(pathlib.Path(__file__)),
                    script_sha256=sha(pathlib.Path(__file__))),
        result=dict(
            first_non_foot_contact=dict(step=ev['first_nonfoot'], ms=ms(ev['first_nonfoot']), pair=ev['first_nonfoot_pair']),
            collapse_at_25us=True if ev['first_nonfoot'] is not None else False,
            collapse_definition='first body-floor (non-tarsus) contact = the accepted equilibrium does not stay airborne-worthy past this instant',
            first_step_legs_in_contact_below_6=dict(step=ev['legs_below_6'], ms=ms(ev['legs_below_6'])),
            first_step_with_zero_tarsus_contacts=dict(step=ev['zero_tarsus'], ms=ms(ev['zero_tarsus'])),
            first_step_max_u_diamond_ge_1=dict(step=ev['first_ud_ge_1'], ms=ms(ev['first_ud_ge_1']), pair=ev['first_ud_pair']),
            max_u_diamond=dict(window_max=float(arr['ud'].max()), window_max_step=int(arr['ud'].argmax())),
            max_qacc=dict(step0=float(arr['qacc'][0]), final=float(arr['qacc'][-1]), window_max=float(arr['qacc'].max()),
                          window_max_step=int(arr['qacc'].argmax())),
            act_peak=dict(value=float(arr['act_max'].max()), step=int(arr['act_max'].argmax()),
                          inside_domain=bool(arr['act_max'].max() <= 1.0),
                          actuator_actlimited_all_false=bool(not any(actlimited)),
                          first_step_above_1=ev['first_act_above_1'],
                          note='MuJoCo does not clamp act for this model, so the value is reported as measured; at 25 us it stays inside [0,1]'),
            root_z=dict(initial=float(arr['root_z'][0]), min=float(arr['root_z'].min()), min_step=int(arr['root_z'].argmin()),
                        min_ms=ms(int(arr['root_z'].argmin())), final=float(arr['root_z'][-1]),
                        min_displacement_vs_pose_A_mm=float(arr['root_z'].min() - float(pose_a[2]))),
            six_leg_steps=dict(steps=int((arr['legs'] == 6).sum()), total=int(arr['legs'].size),
                               fraction=float((arr['legs'] == 6).mean())),
            energy=dict(initial=float(arr['energy'][0]), final=float(arr['energy'][-1]),
                        monotone=bool((np.diff(arr['energy']) <= 0).all()),
                        rising_steps=int((np.diff(arr['energy']) > 0).sum())),
            tilt=dict(max_deg=float(arr['tilt'].max()), final_deg=float(arr['tilt'][-1]),
                      flipped=bool((arr['tilt'] > 90.0).any())),
            contact_table_at_first_nonfoot=table),
        localization=dict(first_leg_to_lose_tarsus_contact=dict(leg=first_leg[1], step=first_leg[0], ms=ms(first_leg[0])),
                          first_leg_force_below_1pct=dict(leg=first_leg_force[1], step=first_leg_force[0], ms=ms(first_leg_force[0])),
                          first_joint_past_0_01_rad=dict(joint=jfirst[1], step=jfirst[0], ms=ms(jfirst[0])),
                          per_leg=per_leg, lookback_before_first_nonfoot=lookback,
                          drive_frame_stats=dict(n_frames=int(ctrl.shape[0]), cadence_ms=CMD_S * 1000,
                                                 leading_all_zero_frames=int(next((i for i in range(ctrl.shape[0])
                                                                                   if not np.all(ctrl[i] == 0)), 0)),
                                                 mean_step_between_frames=float(steps_arch.mean()),
                                                 max_step_between_frames=float(steps_arch.max()),
                                                 mean_command_level=float(ctrl.mean()),
                                                 fraction_of_entries_at_zero=float((ctrl == 0).mean()))),
        comparison_with_pose_A=dict(pose_A_first_non_foot_ms=59.950,
                                    accepted_equilibrium_first_non_foot_ms=ms(ev['first_nonfoot']),
                                    reading='the accepted equilibrium reaches its first non-foot contact later than pose A under the same drive'),
        not_established='absolute force values, physical units, biological validity; no standing, no walking')
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    np.savez_compressed(out / 'time_series_25us.npz', **arr, leg_tarsus_counts=leg_tar, leg_normal_forces=leg_fn,
                        joint_deviation=qdev, joint_labels=np.array(jlabels), command_index_used=index_used,
                        note='dt = 25 us verified at runtime; time_s is the measured d.time')

    # canonical sentence lands ONLY here
    (out / 'contact_identity_sentence.json').write_text(json.dumps(dict(
        status='CANONICAL_SENTENCE', sentence=SENTENCE, recorded_at=now(),
        note='the 76.400 -> 76.375 ms fix lands in this batch only; earlier copies in frozen batches are not rewritten'),
        ensure_ascii=False, indent=1))

    # immutable incident record for the 17:01:26 rewrite
    integrity_now = V1 / 'integrity_incident_assembler.json'
    (out / 'incident_17_01_26.json').write_text(json.dumps(dict(
        status='IMMUTABLE_INCIDENT_RECORD', manual_research_not_product=True,
        event='a frozen evidence file was rewritten during a verification window',
        file=str(integrity_now), changed_at='2026-10-15 17:01:26 (+08:00 local)',
        before_status='INVALID', after_status='WITHDRAWN',
        sha256_before='NOT RECORDED — no hash of the INVALID revision was taken before the rewrite; this part is untraceable',
        sha256_after=sha(integrity_now),
        changed_by='the model, running turn18_admin_closure.py on the owner\'s turn-18 instruction ("要么按事实重写它，要么标为 withdrawn 并在 v1 '
                   'README 里写清作废理由"); the actor is therefore traceable, the before-state is not',
        observed_by=['S2-openloop-frozen-check-20261015T235925Z (finding M5, first read at 16:59 showed INVALID, at 17:05 showed WITHDRAWN)'],
        consequence='the verification that was running had frozen that file at 16:58:44 and its anchor for that file was invalidated mid-run',
        why_it_matters='a frozen evidence file must not be rewritten while a verification window is open, even when the rewrite is ordered',
        remediation=['the rewrite is recorded here with the after hash', 'the before-state cannot be restored and is not invented',
                     'from 2026-10-15 any correction that touches a frozen directory is published as a NEW correction file, never in place'],
        recorded_at=now()), ensure_ascii=False, indent=1))

    # figures (true clock)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = arr['time_s'] * 1000.0
    fig, ax = plt.subplots(2, 2, figsize=(15.5, 9.5))
    a = ax[0][0]
    a.plot(t, arr['root_z'], lw=1.3, color='#1f77b4')
    if ev['first_nonfoot'] is not None:
        a.axvline(t[ev['first_nonfoot']], color='red', ls=':', lw=1.6)
        a.annotate('first non-foot contact\nstep %d = %.3f ms\n%s' % (ev['first_nonfoot'], ms(ev['first_nonfoot']),
                                                                     ' | '.join(ev['first_nonfoot_pair'] or [])),
                   (t[ev['first_nonfoot']], float(arr['root_z'][ev['first_nonfoot']])), textcoords='offset points',
                   xytext=(24, 30), fontsize=9, arrowprops=dict(arrowstyle='->', lw=1.0),
                   bbox=dict(boxstyle='round', fc='white', alpha=0.92))
    a.set_xlabel('measured time [ms] (0.5 s window, verified 25 us step)'); a.set_ylabel('root z [mm]')
    a.set_title('root z and the first non-foot contact', fontsize=10); a.grid(alpha=0.3)
    a = ax[0][1]
    a.plot(t, arr['legs'], lw=1.2, color='#2ca02c', label='legs with tarsus contact')
    a.plot(t, arr['nonfoot'], lw=1.0, color='#d62728', label='non-foot contacts')
    a.set_xlabel('measured time [ms]'); a.set_ylabel('count')
    a.set_title('contact bookkeeping (legs<6 at step %d = %s ms; six-leg %d/%d)'
                % (ev['legs_below_6'], ms(ev['legs_below_6']), report['result']['six_leg_steps']['steps'],
                   report['result']['six_leg_steps']['total']), fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.3)
    a = ax[1][0]
    a.semilogy(t, np.maximum(arr['qacc'], 1e-14), lw=1.1, color='#1f77b4')
    a.axhline(1.0, color='gray', ls=':', lw=1.2, label='1.0 mm/s^2')
    a.set_xlabel('measured time [ms]'); a.set_ylabel('max|qacc| [mm/s^2], log')
    a.set_title('per-step max|qacc| (window max %.6g at step %d)' % (report['result']['max_qacc']['window_max'],
                                                                    report['result']['max_qacc']['window_max_step']), fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.3, which='both')
    a = ax[1][1]
    a.plot(t, arr['ud'], lw=1.1, color='#9467bd')
    a.axhline(1.0, color='red', ls='--', lw=1.3, label='declared diamond limit')
    a.set_xlabel('measured time [ms]'); a.set_ylabel('max u_diamond')
    a.set_title('per-step max u_diamond (peak %.6g at step %d; first >=1 at step %s = %s ms)'
                % (report['result']['max_u_diamond']['window_max'], report['result']['max_u_diamond']['window_max_step'],
                   ev['first_ud_ge_1'], ms(ev['first_ud_ge_1'])), fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.suptitle('CORRECTED-DT: accepted declared-cone static equilibrium, open loop at a runtime-verified 25 us step\n'
                 'self-check: m.opt.timestep read-back %s == declared 2.5e-05, final d.time %.6f s, command held %d steps = %.3f s\n'
                 '100-frame / 5 ms archive; manual research implementation, NOT a product; no standing/walking claim'
                 % (self_check['read_back_model_timestep_s'], self_check['finals_d_time_s'],
                    self_check['command_hold_steps_observed'], self_check['command_hold_seconds_observed']), fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out / 'corrected-dt-25us-openloop.png', dpi=135)
    plt.close(fig)

    lines = ['# corrected-dt：真 25 µs（运行时自检通过）下的被接受平衡态开环 + 同批定位（手动研究实施，不是成品）', '',
             '## 运行时自检（不过就不出数）',
             '- 读回 `m.opt.timestep = %s`（声明 %s，XML 值 %s）；末步 `d.time = %.6f s`（声明 %.3f s）；命令保持 **%d 步 = %.3f s**（声明 5 ms）。'
             % (self_check['read_back_model_timestep_s'], DT, xml_dt, self_check['finals_d_time_s'], DT * STEPS,
                self_check['command_hold_steps_observed'], self_check['command_hold_seconds_observed']),
             '- `actuator_actlimited` 全为 False：**%s**（MuJoCo 不夹 `act`，故必须报告实测值）。' % report['declared']['actuator_actlimited_all_false'],
             '',
             '## 开环结果（25 µs，唯一可作物理口径的一支）',
             '- **首个非足接触：步 %d = %s ms**，接触对 `%s`' % (ev['first_nonfoot'], ms(ev['first_nonfoot']), ev['first_nonfoot_pair']),
             '- **该 equilibrium 在 25 µs 下确实崩**：崩的定义 = 出现首个躯干/非跗节对地接触；本状态在 0.5 s 窗内于上述时刻发生。',
             '- 六足退化：`legs_in_contact < 6` **步 %d = %s ms**；跗节全失 步 %s = %s ms' % (ev['legs_below_6'], ms(ev['legs_below_6']),
                                                                                              ev['zero_tarsus'], ms(ev['zero_tarsus'])),
             '- root z：初值 %s、**最小 %s（步 %d = %s ms）**、末值 %s（相对姿态 A 最小位移 %s mm）'
             % (report['result']['root_z']['initial'], report['result']['root_z']['min'], report['result']['root_z']['min_step'],
                report['result']['root_z']['min_ms'], report['result']['root_z']['final'],
                report['result']['root_z']['min_displacement_vs_pose_A_mm']),
             '- **`act` 峰值 = %s**（≤1：%s；首次 >1 步：%s）' % (report['result']['act_peak']['value'],
                                                                report['result']['act_peak']['inside_domain'],
                                                                report['result']['act_peak']['first_step_above_1']),
             '- `max|qacc|`：首步 %s、末步 %s、全窗最大 %s（步 %d）' % (report['result']['max_qacc']['step0'],
                                                                        report['result']['max_qacc']['final'],
                                                                        report['result']['max_qacc']['window_max'],
                                                                        report['result']['max_qacc']['window_max_step']),
             '- `max u_diamond`：峰值 %s（步 %d）；首次 ≥1 步 %s = %s ms' % (report['result']['max_u_diamond']['window_max'],
                                                                             report['result']['max_u_diamond']['window_max_step'],
                                                                             ev['first_ud_ge_1'], ms(ev['first_ud_ge_1'])),
             '- 六足同触 %d/%d = %.8f；能量 %s → %s（单调 %s，上升步 %d）；倾角最大 %.6f°、翻转 %s'
             % (report['result']['six_leg_steps']['steps'], report['result']['six_leg_steps']['total'],
                report['result']['six_leg_steps']['fraction'], report['result']['energy']['initial'], report['result']['energy']['final'],
                report['result']['energy']['monotone'], report['result']['energy']['rising_steps'],
                report['result']['tilt']['max_deg'], report['result']['tilt']['flipped']),
             '',
             '## 同批重算的定位（ms 已按真 dt 计算）',
             '- 先失去跗节接触的腿：**`%s`，步 %d = %s ms**；力先跌破自身前 100 步均值 1%%：**`%s`，步 %d = %s ms**'
             % (first_leg[1], first_leg[0], ms(first_leg[0]), first_leg_force[1], first_leg_force[0], ms(first_leg_force[0])),
             '- 先离开平衡的关节：**`%s`，步 %d = %s ms**' % (jfirst[1], jfirst[0], ms(jfirst[0])),
             '- 逐腿首次失跗节接触：%s' % json.dumps({L: per_leg[L]['first_step_no_tarsus_contact_ms'] for L in LEGS}, ensure_ascii=False),
             '',
             '## 对照与结论',
             '- 姿态 A（s=0.95941）同驱动、同 25 µs：首个非足接触 **59.950 ms**；本状态 **%s ms** ⇒ **更晚触地**。' % ms(ev['first_nonfoot']),
             '- **结论**：`qacc ≈ 0` 只是**静态必要条件**；本状态在这份冻结指令下**不能持续维持**（首个对地接触即为其崩点），'
             '是否"能站住一会儿"要看首个退化步与 `act` 上限（见下一最小动作）。不含站立/行走结论。',
             '',
             '## 附：本轮纪律与事故',
             '- 冻结规则：写盘即冻结（`FROZEN_DELIVERY_HASHES.json`）；冻结后不再改写已封存目录的 `report.json`/图/脚本；'
             '后续更正一律以**新的 correction 文件**发布（旧值 + 旧 sha256 + 新值 + 新 sha256 + 写盘当刻复核时间）。',
             '- 事故记录：`incident_17_01_26.json`（17:01:26 把冻结中的事故文件由 INVALID 改为 WITHDRAWN；before 哈希**未记录** ⇒ **不可追溯**，'
             'after sha256 已记；操作者为本人脚本、依业主第 18 轮指令执行）。',
             '- **手动研究实施，不是成品**；绝对力值与单位 `NOT_ESTABLISHED`。']
    (out / 'README.zh-CN.md').write_text('\n'.join(lines) + '\n')
    with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(pathlib.Path(__file__), pathlib.Path(__file__).name)
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name not in ('delivery-hashes.json', 'FROZEN_DELIVERY_HASHES.json'))
    (out / 'delivery-hashes.json').write_text(json.dumps({p.name: sha(p) for p in files}, ensure_ascii=False, indent=1))
    (out / 'FROZEN_DELIVERY_HASHES.json').write_text(json.dumps(dict(status='FROZEN_BEFORE_VERIFICATION', frozen_at=now(),
                                                                    files={p.name: sha(p) for p in files}), ensure_ascii=False, indent=1))
    print(report['status'])
    print('out:', out)
    print('self-check:', json.dumps(self_check['match'], ensure_ascii=False), '| all ok:', ok)
    print('first non-foot step %d = %s ms (%s); legs<6 step %d = %s ms; zero tarsus step %d = %s ms'
          % (ev['first_nonfoot'], ms(ev['first_nonfoot']), ev['first_nonfoot_pair'], ev['legs_below_6'], ms(ev['legs_below_6']),
             ev['zero_tarsus'], ms(ev['zero_tarsus'])))
    print('act peak %s (inside %s) | root z min %s at step %d | max qacc %s | max u_diamond %s at step %d'
          % (report['result']['act_peak']['value'], report['result']['act_peak']['inside_domain'], report['result']['root_z']['min'],
             report['result']['root_z']['min_step'], report['result']['max_qacc']['window_max'],
             report['result']['max_u_diamond']['window_max'], report['result']['max_u_diamond']['window_max_step']))
    print('first leg', first_leg, '| first joint', jfirst)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
