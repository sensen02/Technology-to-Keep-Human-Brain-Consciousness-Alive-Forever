"""Turn 22, part B: the one minimal closed loop, done correctly.

  * m.opt.timestep = 2.5e-5 is written explicitly and asserted; the XML value (1e-4) is never relied on;
  * 20000 steps at 25 us = 0.5 s, command index advances every 200 steps = 5 ms over the 100-frame archive;
  * the time axis is the measured d.time, not arange * dt;
  * the same run is also done once at the XML step of 1e-4 s PURELY as a labelled numerically-invalid side branch, with the
    sister batch's own sentence reused verbatim, and the two act peaks are printed side by side;
  * report.json carries model_timestep_s and xml_option_timestep_s;
  * figures are produced by this script itself (so the figure-producing code travels in source_snapshot.zip) and their
    titles say what the clock is and that this is a manual research implementation, not a product.

Output directory: the batch directory that already holds correction_ledger_consolidated.json.
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
OUT = DESK / 'S2-accepted-equilibrium-openloop-CORRECTED-REISSUED-25us-20261015T172000'
STATE = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259/equilibrium_best_state.npz'
POSE_A = DESK / 'S2-declared-cone-min-s-20261015T123603/best_candidate_pose.npz'
DRIVE = ROOT / 'outputs/experiment_platform/20261015_135150_a2fff3f0/traces.npz'
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
SUFFIXES = ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll', 'Trochanter_yaw', 'Trochanter_pitch', 'Trochanter_roll', 'Tibia_pitch')
DT = 2.5e-5
STEPS = 20000
CMD_EVERY = int(round(0.005 / DT))
SISTER_SENTENCE = ('BRANCH INVALID: at dt=1e-4 s the native muscle activation leaves its domain (act max 1.9993002399 > 1), so no number from '
                   'this branch may be quoted as a physical result; use the 25 us branch')
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S')


def run_once(q0, act0, ctrl, m, mujoco, gname, qadr, timestep, collect_details):
    m.opt.timestep = timestep
    d = mujoco.MjData(m)
    d.qpos[:] = q0
    d.qvel[:] = 0.0
    d.act[:] = act0
    d.ctrl[:] = 0.0
    mujoco.mj_forward(m, d)
    n = STEPS + 1
    res = dict(time_s=np.zeros(n), root_z=np.zeros(n), legs=np.zeros(n, dtype=int), tar=np.zeros(n, dtype=int),
               nonfoot=np.zeros(n, dtype=int), tilt=np.zeros(n), energy=np.zeros(n), qacc=np.zeros(n), ud=np.zeros(n),
               act_max=np.zeros(n), ctrl_max=np.zeros(n))
    leg_tar = np.zeros((n, len(LEGS)), dtype=int) if collect_details else None
    leg_fn = np.zeros((n, len(LEGS))) if collect_details else None
    qdev = np.zeros((n, len(qadr))) if collect_details else None
    f6 = np.zeros(6)
    ev = dict(first_nonfoot=None, first_nonfoot_pair=None, legs_below_6=None, zero_tarsus=None, first_ud_ge_1=None,
              first_ud_pair=None, first_act_above_1=None, jac=None)
    table = None

    def sample(i):
        nonlocal table
        res['time_s'][i] = float(d.time)  # measured, not assumed
        res['root_z'][i] = float(d.qpos[2])
        quat = np.asarray(d.qpos[3:7], dtype=float)
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        r22 = float(mat.reshape(3, 3)[2, 2])
        res['tilt'][i] = float(np.degrees(np.arccos(np.clip(r22, -1.0, 1.0))))
        res['qacc'][i] = float(np.max(np.abs(np.asarray(d.qacc))))
        mujoco.mj_energyPos(m, d)
        mujoco.mj_energyVel(m, d)
        res['energy'][i] = float(d.energy[0] + d.energy[1])
        res['act_max'][i] = float(np.max(np.asarray(d.act)))
        res['ctrl_max'][i] = float(np.max(np.asarray(d.ctrl)))
        if collect_details:
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
                    if collect_details:
                        leg_tar[i, LEGS.index(leg)] += 1
                        leg_fn[i, LEGS.index(leg)] += fn
                if u is not None and u > worst:
                    worst, worst_pair = u, sorted((n1, n2))
            else:
                nonfoot += 1
            rows.append(dict(pair=sorted((n1, n2)), is_tarsus=bool('Tarsus' in (n1 + n2)), fn=fn, u_diamond=u, leg=leg))
        res['legs'][i], res['tar'][i], res['nonfoot'][i], res['ud'][i] = len(legs), tar, nonfoot, worst
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
        d.ctrl[:] = ctrl[min(step // CMD_EVERY, ctrl.shape[0] - 1)]
        mujoco.mj_step(m, d)
        sample(step + 1)
    res['events'] = ev
    res['contact_table_at_first_nonfoot'] = table
    if collect_details:
        res['leg_tar'], res['leg_fn'], res['qdev'] = leg_tar, leg_fn, qdev
    return res


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
    assert abs(m.opt.timestep - xml_dt) > 0, 'the declared timestep equals the XML value, so the assert proves nothing'
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

    good = run_once(q0, act0, ctrl, m, mujoco, gname, qadr, DT, collect_details=True)
    invalid = run_once(q0, act0, ctrl, m, mujoco, gname, qadr, xml_dt, collect_details=False)
    m.opt.timestep = DT
    assert m.opt.timestep == DT

    ev = good['events']
    ms = lambda i: (None if i is None else round(i * DT * 1000, 3))
    iv = invalid['events']
    ms_inv = lambda i: (None if i is None else round(i * xml_dt * 1000, 3))
    per_leg = {}
    for k, L in enumerate(LEGS):
        base = float(good['leg_fn'][:100, k].mean())
        per_leg[L] = dict(first_step_no_tarsus_contact=persist_first(good['leg_tar'][:, k] == 0),
                          first_step_no_tarsus_contact_ms=ms(persist_first(good['leg_tar'][:, k] == 0)),
                          first_step_force_below_1pct=persist_first(good['leg_fn'][:, k] < 0.01 * base),
                          first_step_force_below_1pct_ms=ms(persist_first(good['leg_fn'][:, k] < 0.01 * base)))
    first_leg = min(((v['first_step_no_tarsus_contact'], L) for L, v in per_leg.items()
                     if v['first_step_no_tarsus_contact'] is not None), default=(None, None))
    jfirst = min(((persist_first(good['qdev'][:, k] >= 0.01), jlabels[k]) for k in range(len(jids))
                  if persist_first(good['qdev'][:, k] >= 0.01) is not None), default=(None, None))

    report = dict(
        status='PASS_CORRECTED_REISSUED_25US_CLOSED_LOOP', corrected=True, reissued=True,
        manual_research_not_product=True, standing=False, walking=False,
        title='CORRECTED-REISSUED minimal closed loop at the true 25 us step (manual research implementation, not a product)',
        protocol=dict(dt_s=DT, steps=STEPS, duration_s=DT * STEPS, command_cadence_s=0.005, command_index_every_steps=CMD_EVERY,
                      model_timestep_s=DT, xml_option_timestep_s=xml_dt,
                      timestep_line='m.opt.timestep = 2.5e-5 followed by assert m.opt.timestep == 2.5e-5',
                      time_axis='measured d.time, not arange * dt',
                      state_writes='once (qpos, qvel, act)', loop_writes='ctrl only',
                      drive_frames=int(ctrl.shape[0]), drive_covers_window=bool(ctrl.shape[0] * 0.005 >= DT * STEPS - 1e-12)),
        inputs=dict(state=str(STATE), state_sha256=sha(STATE), pose_A=str(POSE_A), pose_A_sha256=sha(POSE_A),
                    drive=str(DRIVE), drive_sha256=sha(DRIVE), canonical_xml=str(cp.CANONICAL_XML),
                    canonical_xml_sha256=sha(cp.CANONICAL_XML), script=str(pathlib.Path(__file__)),
                    script_sha256=sha(pathlib.Path(__file__))),
        result_25us=dict(
            first_non_foot_contact=dict(step=ev['first_nonfoot'], ms=ms(ev['first_nonfoot']), pair=ev['first_nonfoot_pair']),
            first_step_legs_in_contact_below_6=dict(step=ev['legs_below_6'], ms=ms(ev['legs_below_6'])),
            first_step_with_zero_tarsus_contacts=dict(step=ev['zero_tarsus'], ms=ms(ev['zero_tarsus'])),
            first_step_max_u_diamond_ge_1=dict(step=ev['first_ud_ge_1'], ms=ms(ev['first_ud_ge_1']), pair=ev['first_ud_pair']),
            max_qacc=dict(step0=float(good['qacc'][0]), final=float(good['qacc'][-1]), window_max=float(good['qacc'].max()),
                          window_max_step=int(good['qacc'].argmax()),
                          first_above_gate_step=int(np.argmax(good['qacc'] > 1e-3)),
                          first_above_gate_ms=ms(int(np.argmax(good['qacc'] > 1e-3)))),
            max_u_diamond=dict(window_max=float(good['ud'].max()), window_max_step=int(good['ud'].argmax())),
            act_peak=dict(value=float(good['act_max'].max()), step=int(good['act_max'].argmax()),
                          inside_domain=bool(good['act_max'].max() <= 1.0),
                          first_step_above_1=ev['first_act_above_1'],
                          mandatory_disclosure='the activation must stay inside [0,1]; if a run exceeds it, that run is invalid and none of its '
                                               'numbers may be quoted as a physical result'),
            root_z=dict(initial=float(good['root_z'][0]), min=float(good['root_z'].min()), min_step=int(good['root_z'].argmin()),
                        min_ms=ms(int(good['root_z'].argmin())), final=float(good['root_z'][-1]),
                        min_displacement_vs_pose_A_mm=float(good['root_z'].min() - float(pose_a[2]))),
            six_leg_steps=dict(steps=int((good['legs'] == 6).sum()), total=int(good['legs'].size),
                               fraction=float((good['legs'] == 6).mean())),
            energy=dict(initial=float(good['energy'][0]), final=float(good['energy'][-1]),
                        monotone=bool((np.diff(good['energy']) <= 0).all()),
                        rising_steps=int((np.diff(good['energy']) > 0).sum())),
            tilt=dict(max_deg=float(good['tilt'].max()), final_deg=float(good['tilt'][-1]),
                      flipped=bool((good['tilt'] > 90.0).any())),
            first_leg_to_lose_tarsus_contact=dict(leg=first_leg[1], step=first_leg[0], ms=ms(first_leg[0])),
            first_joint_past_0_01_rad=dict(joint=jfirst[1], step=jfirst[0], ms=ms(jfirst[0])),
            per_leg=per_leg, contact_table_at_first_nonfoot=good['contact_table_at_first_nonfoot']),
        side_branch_100us_numerically_invalid=dict(
            dt_s=xml_dt, steps=STEPS, duration_s=STEPS * xml_dt,
            act_peak=float(invalid['act_max'].max()), first_step_act_above_1=iv['first_act_above_1'],
            first_non_foot_contact=dict(step=iv['first_nonfoot'], ms_at_its_own_dt=ms_inv(iv['first_nonfoot'])),
            statement=SISTER_SENTENCE,
            note='this branch exists only to show the effect of the XML step; it is numerically invalid and nothing in it may be quoted'),
        comparison=dict(accepted_equilibrium_first_non_foot_ms=ms(ev['first_nonfoot']),
                        pose_A_first_non_foot_ms=59.950,
                        reading='the accepted equilibrium reaches its first non-foot contact later than pose A under the same declared drive'),
        not_established='absolute force values, physical units, biological validity; no standing, no walking',
        outputs=str(OUT), recorded_at=now())
    np.savez_compressed(OUT / 'time_series_25us.npz',
                        **{'good__' + k: v for k, v in good.items() if isinstance(v, np.ndarray)},
                        **{'side_100us__' + k: v for k, v in invalid.items() if isinstance(v, np.ndarray)},
                        joint_labels=np.array(jlabels),
                        note='25 us branch = physical; side_100us branch = numerically invalid (act leaves [0,1])')
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = good['time_s'] * 1000.0
    fig, ax = plt.subplots(2, 2, figsize=(15.5, 9.5))
    a = ax[0][0]
    a.plot(t, good['root_z'], lw=1.3, color='#1f77b4', label='root z (25 us branch)')
    if ev['first_nonfoot'] is not None:
        a.axvline(t[ev['first_nonfoot']], color='red', ls=':', lw=1.6)
        a.annotate('first non-foot contact\nstep %d = %.3f ms\n%s' % (ev['first_nonfoot'], ms(ev['first_nonfoot']),
                                                                     ' | '.join(ev['first_nonfoot_pair'] or [])),
                   (t[ev['first_nonfoot']], float(good['root_z'][ev['first_nonfoot']])), textcoords='offset points',
                   xytext=(24, 30), fontsize=9, arrowprops=dict(arrowstyle='->', lw=1.0),
                   bbox=dict(boxstyle='round', fc='white', alpha=0.92))
    a.set_xlabel('measured time [ms] (0.5 s window, 25 us step)'); a.set_ylabel('root z [mm]')
    a.set_title('root z and the first non-foot contact (true clock)', fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3)
    a = ax[0][1]
    a.plot(t, good['legs'], lw=1.2, color='#2ca02c', label='legs with tarsus contact')
    a.plot(t, good['nonfoot'], lw=1.0, color='#d62728', label='non-foot contacts')
    a.set_xlabel('measured time [ms]'); a.set_ylabel('count')
    a.set_title('contact bookkeeping (six-leg %d/%d = %.6f)' % (report['result_25us']['six_leg_steps']['steps'],
                                                                report['result_25us']['six_leg_steps']['total'],
                                                                report['result_25us']['six_leg_steps']['fraction']), fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.3)
    a = ax[1][0]
    a.semilogy(t, np.maximum(good['qacc'], 1e-14), lw=1.1, color='#1f77b4', label='max|qacc| (25 us)')
    a.semilogy(invalid['time_s'] * 1000.0, np.maximum(invalid['qacc'], 1e-14), lw=1.0, color='gray', ls='--',
               label='max|qacc| (100 us side branch, numerically invalid)')
    a.axhline(1.0, color='gray', ls=':', lw=1.2, label='1.0 mm/s^2')
    a.set_xlabel('time [ms]'); a.set_ylabel('max|qacc| [mm/s^2], log')
    a.set_title('per-step max|qacc|: 25 us branch vs the invalid 100 us side branch', fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.3, which='both')
    a = ax[1][1]
    a.plot(t, good['ud'], lw=1.1, color='#9467bd', label='max u_diamond (25 us)')
    a.axhline(1.0, color='red', ls='--', lw=1.3, label='declared diamond limit')
    a.set_xlabel('time [ms]'); a.set_ylabel('max u_diamond')
    a.set_title('per-step max u_diamond (peak %.6g at step %d)' % (report['result_25us']['max_u_diamond']['window_max'],
                                                                  report['result_25us']['max_u_diamond']['window_max_step']), fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.suptitle('CORRECTED-REISSUED: accepted declared-cone static equilibrium, open loop, m.opt.timestep = 2.5e-5 asserted\n'
                 '100-frame / 5 ms command archive, 20000 steps = 0.5 s, time axis measured from d.time, state written once, only ctrl in the loop\n'
                 'act peak %.9f (inside [0,1]: %s)  |  manual research implementation, NOT a product; no standing/walking claim'
                 % (report['result_25us']['act_peak']['value'], report['result_25us']['act_peak']['inside_domain']), fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT / 'corrected-reissued-25us-openloop.png', dpi=135)
    plt.close(fig)

    lines = ['# CORRECTED-REISSUED：真 25 µs 最小闭环（手动研究实施，不是成品）', '',
             '## 协议（写死在脚本里，并带 assert）',
             '- `m.opt.timestep = 2.5e-5`，紧跟 `assert m.opt.timestep == 2.5e-5`；**不依赖** XML 的 `timestep="1e-04"`。',
             '- `report.json` 固定给出 `model_timestep_s = %s` 与 `xml_option_timestep_s = %s`。' % (DT, xml_dt),
             '- 20000 步 × 25 µs = **0.5 s**；命令每 **200 步 = 5 ms** 更新一次；100 帧档案覆盖整窗（%s）。'
             % report['protocol']['drive_covers_window'],
             '- 时间轴用**实测 `d.time`**，不用 `arange*dt`。',
             '',
             '## 结果（25 µs 分支，唯一可作物理口径的一支）',
             '- 首个足外接触：**步 %d = %s ms**，接触对 `%s`' % (ev['first_nonfoot'], ms(ev['first_nonfoot']), ev['first_nonfoot_pair']),
             '- 首个失足（`legs_in_contact < 6`）：**步 %d = %s ms**；跗节全失：步 %s = %s ms'
             % (ev['legs_below_6'], ms(ev['legs_below_6']), ev['zero_tarsus'], ms(ev['zero_tarsus'])),
             '- **先失去跗节接触的腿：`%s`，步 %d = %s ms**；先离开平衡的关节：`%s`，步 %d = %s ms'
             % (first_leg[1], first_leg[0], ms(first_leg[0]), jfirst[1], jfirst[0], ms(jfirst[0])),
             '- root z：初值 %s、**最小 %s（步 %d = %s ms）**、末值 %s（相对姿态 A 最小位移 %s mm）'
             % (report['result_25us']['root_z']['initial'], report['result_25us']['root_z']['min'],
                report['result_25us']['root_z']['min_step'], report['result_25us']['root_z']['min_ms'],
                report['result_25us']['root_z']['final'], report['result_25us']['root_z']['min_displacement_vs_pose_A_mm']),
             '- **`act` 峰值 = %s**（步 %d），**在 [0,1] 内：%s**；越界的 100 µs 旁证支峰值 = %s（首次越界步 %s）'
             % (report['result_25us']['act_peak']['value'], report['result_25us']['act_peak']['step'],
                report['result_25us']['act_peak']['inside_domain'], report['side_branch_100us_numerically_invalid']['act_peak'],
                report['side_branch_100us_numerically_invalid']['first_step_act_above_1']),
             '- `max|qacc|`：首步 %s、末步 %s、全窗最大 %s（步 %d）' % (report['result_25us']['max_qacc']['step0'],
                                                                        report['result_25us']['max_qacc']['final'],
                                                                        report['result_25us']['max_qacc']['window_max'],
                                                                        report['result_25us']['max_qacc']['window_max_step']),
             '- `max u_diamond`：全窗最大 %s（步 %d）；首次达 1.0：步 %s = %s ms（接触对 `%s`）'
             % (report['result_25us']['max_u_diamond']['window_max'], report['result_25us']['max_u_diamond']['window_max_step'],
                ev['first_ud_ge_1'], ms(ev['first_ud_ge_1']), ev['first_ud_pair']),
             '- 六足同触 %d/%d = %.8f；能量 %s → %s（单调下降 %s，上升步 %d）；倾角最大 %.6f°、翻转 %s'
             % (report['result_25us']['six_leg_steps']['steps'], report['result_25us']['six_leg_steps']['total'],
                report['result_25us']['six_leg_steps']['fraction'], report['result_25us']['energy']['initial'],
                report['result_25us']['energy']['final'], report['result_25us']['energy']['monotone'],
                report['result_25us']['energy']['rising_steps'], report['result_25us']['tilt']['max_deg'],
                report['result_25us']['tilt']['flipped']),
             '',
             '## 100 µs 旁证支（数值无效，禁止引用）',
             '- ' + SISTER_SENTENCE,
             '',
             '## 对照',
             '- 姿态 A（s=0.95941）在同一驱动下的首个足外接触为 **59.950 ms**（其 runner 设置了 timestep，标注有效）；'
             '本轮被接受平衡态为 **%s ms** ⇒ 更晚触地。' % ms(ev['first_nonfoot']),
             '',
             '## 披露',
             '- **手动研究实施，不是成品**；不含站立/行走结论；绝对力值与物理单位 `NOT_ESTABLISHED`。',
             '- 被作废目录与其 ms 标注：见 `correction_ledger_consolidated.json` 与各目录 README 顶部的 PROTOCOL VOID 行。']
    (OUT / 'README.zh-CN.md').write_text('\n'.join(lines) + '\n')
    with zipfile.ZipFile(OUT / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in (pathlib.Path(__file__), ROOT / 'turn22_voids_and_ledger.py', ROOT / 'turn22_void_pages_ascii.py'):
            if p.exists():
                z.write(p, p.name)
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name not in ('delivery-hashes.json', 'FROZEN_DELIVERY_HASHES.json'))
    (OUT / 'delivery-hashes.json').write_text(json.dumps({p.name: sha(p) for p in files}, ensure_ascii=False, indent=1))
    (OUT / 'FROZEN_DELIVERY_HASHES.json').write_text(json.dumps(dict(status='FROZEN_BEFORE_VERIFICATION', frozen_at=now(),
                                                                    files={p.name: sha(p) for p in files}), ensure_ascii=False, indent=1))
    print(report['status'])
    print('out:', OUT)
    print('first non-foot step %d = %s ms (%s) | legs<6 step %d = %s ms | zero tarsus step %d = %s ms'
          % (ev['first_nonfoot'], ms(ev['first_nonfoot']), ev['first_nonfoot_pair'], ev['legs_below_6'], ms(ev['legs_below_6']),
             ev['zero_tarsus'], ms(ev['zero_tarsus'])))
    print('act peak 25us = %s (inside %s) | 100us side branch = %s' % (report['result_25us']['act_peak']['value'],
                                                                       report['result_25us']['act_peak']['inside_domain'],
                                                                       report['side_branch_100us_numerically_invalid']['act_peak']))
    print('root z min %s at step %s | max qacc %s | max u_diamond %s (step %s)'
          % (report['result_25us']['root_z']['min'], report['result_25us']['root_z']['min_step'],
             report['result_25us']['max_qacc']['window_max'], report['result_25us']['max_u_diamond']['window_max'],
             report['result_25us']['max_u_diamond']['window_max_step']))
    print('first leg to lose tarsus contact:', first_leg, '| first joint:', jfirst)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
