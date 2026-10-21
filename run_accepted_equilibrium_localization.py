"""Localisation of the accepted declared-cone static equilibrium under the frozen declared drive:

  Q1  which leg loses its tarsus support first, and when (step / ms), and what the full contact table looks like at the
      first body-floor (non-foot) contact;
  Q2  which leg joint leaves the equilibrium first, and by how much;
  Q3  the u_diamond trajectory per leg in the last 20 steps before that first non-foot contact;
  Q4  what the frozen drive lacks: the per-leg amplitude gap between the archive commands and the equilibrium activation,
      and the frame-to-frame step size of the archive (phase/stepiness).

Discipline: qpos/qvel/act written once before the first step (act taken FROM the state, not zeroed), only ctrl inside the
loop, dt = 25 us, 20000 steps = 0.5 s, no pose correction, no PD, no gain change.

Manual research implementation, not a product. No standing/walking claim; absolute force values NOT_ESTABLISHED.
"""

import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = Path('/home/sensen/Desktop/engine修复验收')
STATE = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259/equilibrium_best_state.npz'
DRIVE = ROOT / 'outputs/experiment_platform/20261015_135150_a2fff3f0/traces.npz'
POSE_A = DESK / 'S2-declared-cone-min-s-20261015T123603/best_candidate_pose.npz'
CANONICAL_XML = ROOT / 'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml'
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
SUFFIXES = ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll', 'Trochanter_yaw', 'Trochanter_pitch', 'Trochanter_roll', 'Tibia_pitch')
DT, STEPS, CMD_DT = 25e-6, 20000, 0.005
PERSIST = 5
DEV_THRESHOLDS = (0.01, 0.05)
FORCE_DROP_FRACTION = 0.01
LOOKBACK = 20
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import mujoco
    sys.path.insert(0, str(ROOT))
    from engine import canonical_physics as cp

    m = cp.load_model()
    m.opt.timestep = DT
    assert float(m.opt.timestep) == DT, 'the model refused the declared timestep'
    assert abs(float(m.opt.timestep) - float(cp.load_model().opt.timestep)) > 0.0, ('the declared timestep equals the XML value; the assert '
                                                                                    'would prove nothing')
    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
    aname = lambda a: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ''
    jname = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ''
    jids = []
    for L in LEGS:
        js = [j for j in range(m.njnt) if jname(j).startswith('joint_' + L)]
        for suf in SUFFIXES:
            jids.append(next(j for j in js if jname(j).endswith(suf)))
    jlabels = ['%s_%s' % (jname(j).split('_')[1], jname(j).split('_', 2)[2]) for j in jids]
    qadr = [int(m.jnt_qposadr[j]) for j in jids]

    st = np.load(STATE)
    q0, act0 = st['qpos'].copy(), st['act'].copy()
    ctrl_seq = np.asarray(np.load(DRIVE)['normal_controls'], dtype=float)
    pose_a = np.load(POSE_A)['projected_seated_qpos']

    d = mujoco.MjData(m)
    d.qpos[:] = q0
    d.qvel[:] = 0.0
    d.act[:] = act0
    d.ctrl[:] = 0.0
    mujoco.mj_forward(m, d)
    qacc0 = float(np.max(np.abs(np.asarray(d.qacc))))

    leg_tar = np.zeros((STEPS + 1, len(LEGS)), dtype=int)
    leg_fn = np.zeros((STEPS + 1, len(LEGS)))
    leg_u = np.zeros((STEPS + 1, len(LEGS)))
    qdev = np.zeros((STEPS + 1, len(jids)))
    root_z = np.zeros(STEPS + 1)
    time_s_measured = np.zeros(STEPS + 1)
    n_nonfoot = np.zeros(STEPS + 1, dtype=int)
    f6 = np.zeros(6)
    contact_table_T0 = None
    first_nonfoot = None

    def sample(i):
        nonlocal contact_table_T0, first_nonfoot
        time_s_measured[i] = float(d.time)
        root_z[i] = float(d.qpos[2])
        qdev[i] = np.abs(d.qpos[qadr] - q0[qadr])
        rows = []
        nonfoot = 0
        for ci, c in enumerate(d.contact):
            if c.efc_address < 0:
                continue
            n1, n2 = gname(int(c.geom1)), gname(int(c.geom2))
            muc = np.asarray(c.friction)[:3]
            mujoco.mj_contactForce(m, d, ci, f6)
            fn, ft1, ft2 = float(f6[0]), float(f6[1]), float(f6[2])
            u = (abs(ft1) / (float(muc[0]) * fn) + abs(ft2) / (float(muc[1]) * fn)) if fn > 0 else None
            leg = next((L for L in LEGS if n1.startswith(L) or n2.startswith(L)), None)
            if 'Tarsus' in (n1 + n2) and leg:
                k = LEGS.index(leg)
                leg_tar[i, k] += 1
                leg_fn[i, k] += fn
                if u is not None:
                    leg_u[i, k] = max(leg_u[i, k], u)
            else:
                nonfoot += 1
            rows.append(dict(pair=sorted((n1, n2)), leg=leg or 'trunk', is_tarsus=bool('Tarsus' in (n1 + n2)),
                             fn=fn, ft1=ft1, ft2=ft2, u_diamond=u))
        n_nonfoot[i] = nonfoot
        if first_nonfoot is None and nonfoot > 0:
            first_nonfoot = i
            contact_table_T0 = rows

    sample(0)
    t0 = time.time()
    for step in range(STEPS):
        idx = min(step // int(round(CMD_DT / DT)), ctrl_seq.shape[0] - 1)
        d.ctrl[:] = ctrl_seq[idx]
        mujoco.mj_step(m, d)
        sample(step + 1)
    wall = time.time() - t0

    # ---- Q1: per-leg loss ----
    def persist_first(mask, persist=PERSIST):
        run = 0
        for i, v in enumerate(mask):
            run = run + 1 if v else 0
            if run >= persist:
                return i - persist + 1
        return None

    assert float(m.opt.timestep) == DT, 'runtime timestep mismatch'
    assert abs(float(d.time) - STEPS * DT) <= 1e-9, ('the measured final d.time must equal steps x dt; note the convention: STEPS mj_step calls '
                                                     'give d.time = STEPS * DT = 0.5 s and STEPS + 1 samples')
    assert abs(float(d.time) - DT * STEPS) <= 1e-9, 'runtime window does not match steps x dt'
    ms = lambda i: (None if i is None else round(i * DT * 1000, 3))
    legs_in_contact = (leg_tar > 0).sum(axis=1)
    per_leg = {}
    for k, L in enumerate(LEGS):
        base = float(leg_fn[:100, k].mean())
        per_leg[L] = dict(
            tarsus_contacts_at_step0=int(leg_tar[0, k]),
            first_step_with_no_tarsus_contact=persist_first(leg_tar[:, k] == 0),
            first_step_with_no_tarsus_contact_ms=ms(persist_first(leg_tar[:, k] == 0)),
            first_step_force_below_1pct_of_the_first_100_steps=persist_first(leg_fn[:, k] < FORCE_DROP_FRACTION * base),
            first_step_force_below_1pct_ms=ms(persist_first(leg_fn[:, k] < FORCE_DROP_FRACTION * base)),
            mean_tarsus_normal_force_first_100_steps=base,
            max_u_diamond_window=float(leg_u[:, k].max()),
            first_step_u_diamond_ge_1=next((i for i in range(leg_u.shape[0]) if leg_u[i, k] >= 1.0), None))
    first_leg_to_lose = min(((v['first_step_with_no_tarsus_contact'], L) for L, v in per_leg.items()
                             if v['first_step_with_no_tarsus_contact'] is not None), default=(None, None))
    first_leg_force_drop = min(((v['first_step_force_below_1pct_of_the_first_100_steps'], L) for L, v in per_leg.items()
                                if v['first_step_force_below_1pct_of_the_first_100_steps'] is not None), default=(None, None))

    # ---- Q2: per-joint deviation ----
    joints = {}
    for k, lab in enumerate(jlabels):
        joints[lab] = {('first_step_abs_dev_ge_%.2f' % th): persist_first(qdev[:, k] >= th) for th in DEV_THRESHOLDS}
        joints[lab]['dev_at_step_100'] = float(qdev[100, k])
        joints[lab]['dev_at_step_1000'] = float(qdev[1000, k])
        joints[lab]['dev_max'] = float(qdev[:, k].max())
    first_joint = min(((min(v for kk, v in joints[lab].items() if kk.startswith('first_step_abs_dev_ge_0.01')
                            and v is not None), lab) for lab in joints
                       if any(v is not None for kk, v in joints[lab].items() if kk.startswith('first_step_abs_dev_ge_0.01'))),
                      default=(None, None))

    # ---- Q3: u_diamond in the lookback window before T0 ----
    lookback = None
    if first_nonfoot is not None:
        lo = max(0, first_nonfoot - LOOKBACK)
        lookback = dict(window=[lo, first_nonfoot],
                        per_leg_max_u_diamond={L: [float(x) for x in leg_u[lo:first_nonfoot + 1, k]] for k, L in enumerate(LEGS)},
                        root_z=[float(x) for x in root_z[lo:first_nonfoot + 1]])

    # ---- Q4: what the drive lacks ----
    act_names = [aname(i) for i in range(m.nu)]
    leg_of_act = [next((L for L in LEGS if n.startswith(L) or ('_' + L) in n), 'other') for n in act_names]
    gap_first = np.abs(act0 - ctrl_seq[0])
    gap_nonzero = np.abs(act0 - ctrl_seq[2])
    amplitude = {}
    for L in LEGS + ('other',):
        idxs = [i for i, L2 in enumerate(leg_of_act) if L2 == L]
        if not idxs:
            continue
        amplitude[L] = dict(n_actuators=len(idxs),
                            mean_abs_gap_vs_frame0=float(gap_first[idxs].mean()),
                            max_abs_gap_vs_frame0=float(gap_first[idxs].max()),
                            mean_abs_gap_vs_frame2=float(gap_nonzero[idxs].mean()),
                            max_abs_gap_vs_frame2=float(gap_nonzero[idxs].max()),
                            mean_archive_amplitude=float(ctrl_seq[:, idxs].mean()),
                            mean_equilibrium_activation=float(act0[idxs].mean()))
    steps_archive = np.abs(np.diff(ctrl_seq, axis=0))
    drive_shape = dict(
        n_frames=int(ctrl_seq.shape[0]), cadence_ms=CMD_DT * 1000,
        leading_all_zero_frames=int(next((i for i in range(ctrl_seq.shape[0]) if not np.all(ctrl_seq[i] == 0)),
                                         ctrl_seq.shape[0])),
        per_frame_max_step=[float(x) for x in steps_archive.max(axis=1)][:12],
        mean_step_between_frames=float(steps_archive.mean()),
        max_step_between_frames=float(steps_archive.max()),
        mean_command_level=float(ctrl_seq.mean()),
        fraction_of_entries_at_zero=float((ctrl_seq == 0).mean()))

    out = DESK / ('S2-accepted-equilibrium-localization-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    report = dict(
        status='PASS_ACCEPTED_EQUILIBRIUM_LOCALIZATION', manual_research_not_product=True, standing=False, walking=False,
        scope=('localisation of the accepted declared-cone static equilibrium under the frozen declared 100-frame drive: which leg loses '
               'support first, which joint leaves the equilibrium first, the u_diamond trajectory before the first non-foot contact, and '
               'what the drive lacks (amplitude per leg, frame-to-frame step size)'),
        declared_runtime_fields=dict(model_timestep_s=float(m.opt.timestep), integrator_steps=int(STEPS),
                                     window_s=float(STEPS * float(m.opt.timestep)),
                                     cmd_period_s=float(int(round(CMD_DT / float(m.opt.timestep))) * float(m.opt.timestep)),
                                     cmd_period_steps=int(round(CMD_DT / float(m.opt.timestep))),
                                     read_back_not_copied=True),
        mandatory_act_disclosure=dict(act_max_over_run=float(np.max(np.asarray(d.act))),
                                      actuator_actlimited_all_false=bool(not any(np.asarray(m.actuator_actlimited))),
                                      rule='if act leaves [0,1] the run is numerically invalid and no number from it may be quoted as a '
                                           'physical result (same wording as the sister batch)'),
        protocol=dict(dt_s=DT, steps=STEPS, duration_s=DT * STEPS, command_cadence_s=CMD_DT,
                      state_writes='qpos, qvel and act once before the first step (act FROM the state, not zeroed)',
                      loop_writes='ctrl only', pose_correction='none', explicit_difference_from_run_instability_localization=(
                          'that script zeroes act before the run; this one keeps the equilibrium activation, which is what the frozen-drive '
                          'question needs')),
        input_hashes=dict(state=str(STATE), state_sha256=sha(STATE), drive=str(DRIVE), drive_sha256=sha(DRIVE),
                          pose_A=str(POSE_A), pose_A_sha256=sha(POSE_A), canonical_xml=str(CANONICAL_XML),
                          canonical_xml_sha256=sha(CANONICAL_XML), script=str(Path(__file__)), script_sha256=sha(Path(__file__))),
        initial_condition=dict(max_abs_qacc_mm_s2=qacc0, act_min=float(act0.min()), act_max=float(act0.max()),
                               root_z=float(q0[2]), root_z_pose_A=float(pose_a[2]),
                               max_abs_delta_q_vs_pose_A=float(np.max(np.abs(q0[qadr] - pose_a[qadr])))),
        Q1_leg_loss=dict(first_leg_to_lose_tarsus_contact=first_leg_to_lose[1],
                         first_step_with_no_tarsus_contact=first_leg_to_lose[0],
                         first_step_with_no_tarsus_contact_ms=ms(first_leg_to_lose[0]),
                         first_leg_whose_force_drops_below_1pct=first_leg_force_drop[1],
                         first_step_force_below_1pct=first_leg_force_drop[0],
                         first_step_force_below_1pct_ms=ms(first_leg_force_drop[0]),
                         first_step_legs_in_contact_below_6=int(np.argmax(legs_in_contact < 6)) if (legs_in_contact < 6).any() else None,
                         first_step_legs_in_contact_below_6_ms=ms(int(np.argmax(legs_in_contact < 6)) if (legs_in_contact < 6).any() else None),
                         first_step_legs_in_contact_equals_1=(next((i for i in range(legs_in_contact.size)
                                                                    if legs_in_contact[i] == 1), None)),
                         first_step_legs_in_contact_equals_1_ms=ms(next((i for i in range(legs_in_contact.size)
                                                                         if legs_in_contact[i] == 1), None)),
                         per_leg=per_leg),
        Q2_joint_deviation=dict(first_joint_to_leave_the_equilibrium=first_joint[1],
                                first_step_joint_deviation_ge_0_01_rad=first_joint[0],
                                first_step_joint_deviation_ge_0_01_rad_ms=ms(first_joint[0]),
                                thresholds_rad=list(DEV_THRESHOLDS), persistence_steps=PERSIST, per_joint=joints),
        Q3_lookback_before_the_first_nonfoot_contact=lookback,
        Q4_what_the_drive_lacks=dict(frame_to_frame=drive_shape, amplitude_gap_per_leg=amplitude,
                                     reading='amplitude gap = |archive command - equilibrium activation| per actuator, grouped by leg; '
                                             'frame-to-frame step = |u[k+1] - u[k]| of the archive, i.e. how steppy the command sequence is'),
        first_nonfoot_contact=dict(step=first_nonfoot, ms=ms(first_nonfoot),
                                   contact_table=contact_table_T0),
        wall_seconds=round(wall, 1),
        not_established='absolute force values, physical units, biological validity; no standing, no walking')
    np.savez_compressed(out / 'time_series.npz', leg_tarsus_counts=leg_tar, leg_normal_forces=leg_fn, leg_max_u_diamond=leg_u,
                        joint_deviation=qdev, joint_labels=np.array(jlabels), root_z=root_z,
                        n_nonfoot_contacts=n_nonfoot, legs_in_contact=legs_in_contact,
                        time_s=time_s_measured  # taken from d.time during the run, never arange*DT, act_equilibrium=act0, ctrl_sequence=ctrl_seq,
                        actuator_names=np.array(act_names), actuator_leg=np.array(leg_of_act))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(15.5, 9.5))
    a = ax[0][0]
    for k, L in enumerate(LEGS):
        a.plot(np.arange(root_z.size) * DT * 1000, leg_tar[:, k], lw=1.2, label=L)
    a.set_xlabel('time [ms]'); a.set_ylabel('tarsus contacts of that leg')
    a.set_title('per-leg tarsus contacts (first leg to lose support: %s at step %s = %s ms)'
                % (first_leg_to_lose[1], first_leg_to_lose[0], ms(first_leg_to_lose[0])), fontsize=9)
    a.legend(fontsize=8, ncol=3); a.grid(alpha=0.3)
    a = ax[0][1]
    for k, L in enumerate(LEGS):
        a.semilogy(np.arange(root_z.size) * DT * 1000, np.maximum(leg_fn[:, k], 1e-12), lw=1.1, label=L)
    a.set_xlabel('time [ms]'); a.set_ylabel('summed tarsus normal force [model units]')
    a.set_title('per-leg tarsus normal force (first leg below 1%% of its own first-100-step mean: %s at step %s)'
                % (first_leg_force_drop[1], first_leg_force_drop[0]), fontsize=9)
    a.legend(fontsize=8, ncol=3); a.grid(alpha=0.3, which='both')
    a = ax[1][0]
    top = np.argsort(-qdev.max(axis=0))[:6]
    for k in top:
        a.semilogy(np.arange(root_z.size) * DT * 1000, np.maximum(qdev[:, k], 1e-12), lw=1.2, label=jlabels[k])
    a.set_xlabel('time [ms]'); a.set_ylabel('|q - q(0)| [rad]')
    a.set_title('the six joints that move most (first joint past 0.01 rad: %s at step %s)'
                % (first_joint[1], first_joint[0]), fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=0.3, which='both')
    a = ax[1][1]
    if lookback is not None:
        for k, L in enumerate(LEGS):
            a.plot(np.arange(lookback['window'][0], lookback['window'][1] + 1) * DT * 1000,
                   lookback['per_leg_max_u_diamond'][L], marker='.', lw=1.2, label=L)
        a.axhline(1.0, color='red', ls='--', lw=1.3, label='declared diamond limit')
    a.set_xlabel('time [ms]'); a.set_ylabel('max u_diamond of that leg')
    a.set_title('u_diamond in the %d steps before the first non-foot contact' % LOOKBACK, fontsize=9)
    a.legend(fontsize=8, ncol=3); a.grid(alpha=0.3)
    fig.suptitle('localisation of the accepted declared-cone static equilibrium under the frozen declared 100-frame drive\n'
                 'dt 25 us, 20000 steps, state written once (act from the state), only ctrl in the loop, no pose correction, no PD\n'
                 'manual research implementation, not a product; no standing/walking claim', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / 'accepted-equilibrium-localization.png', dpi=135)
    plt.close(fig)

    readme = '\n'.join([
        '# 被接受"声明锥静态平衡"在冻结声明驱动下的定位（手动研究实施，不是成品）',
        '',
        '## 协议（与本项目其他开环产物一致）',
        '- 状态：`%s`（sha256 `%s`）；`qpos` = 该状态，`qvel = 0`，**`act` 取自状态本身**（不像 `run_instability_localization.py` 那样把 act 清零 —— 这是本次刻意的差别，因为问的是"冻结驱动能不能维持这个激活状态"）。'
        % (STATE, sha(STATE)),
        '- 驱动：`%s`（sha256 `%s`，100 帧 × 90、5 ms 间距）。' % (DRIVE, sha(DRIVE)),
        '- dt 25 µs、20000 步 = 0.5 s；状态只写一次，循环内只写 `ctrl`；无姿态修正、无 PD、无增益改动。',
        '- 初值：`max|qacc| = %.6g mm/s²`（静态必要条件），`act ∈ [%.6g, %.6g]`，`max|Δq| vs 姿态 A = %.6f rad`。'
        % (qacc0, act0.min(), act0.max(), report['initial_condition']['max_abs_delta_q_vs_pose_A']),
        '',
        '## Q1 哪条腿先失载',
        '- **先失去跗节接触的腿：%s，步 %s = %s ms**（连续 %d 步无跗节接触为准）。' % (first_leg_to_lose[1], first_leg_to_lose[0], ms(first_leg_to_lose[0]), PERSIST),
        '- **先跌破自身前 100 步均值 1%% 法向力的腿：%s，步 %s = %s ms**。' % (first_leg_force_drop[1], first_leg_force_drop[0], ms(first_leg_force_drop[0])),
        '- 六足条件破坏（`legs_in_contact < 6`）：步 **%s = %s ms**；只剩 1 条腿的步：**%s = %s ms**。'
        % (report['Q1_leg_loss']['first_step_legs_in_contact_below_6'], report['Q1_leg_loss']['first_step_legs_in_contact_below_6_ms'],
           report['Q1_leg_loss']['first_step_legs_in_contact_equals_1'], report['Q1_leg_loss']['first_step_legs_in_contact_equals_1_ms']),
        '- 首次足外接触：步 **%s = %s ms**，接触表见 `report.json.Q1/Q_first_nonfoot_contact.contact_table`。'
        % (first_nonfoot, ms(first_nonfoot)),
        '',
        '## Q2 哪个关节先离开平衡',
        '- **先偏离的关节：%s，步 %s = %s ms**（阈值 0.01 rad，连续 %d 步）。' % (first_joint[1], first_joint[0], ms(first_joint[0]), PERSIST),
        '- 每个关节的 `first_step_abs_dev_ge_0.01 / 0.05`、步 100 与步 1000 的偏差、全窗最大偏差见 `report.json.Q2_joint_deviation.per_joint`。',
        '',
        '## Q3 触地前 %d 步的 u_diamond' % LOOKBACK,
        '- 逐腿轨迹在 `report.json.Q3_lookback_before_the_first_nonfoot_contact` 与 `time_series.npz`。',
        '',
        '## Q4 冻结驱动缺什么',
        '- 逐腿"幅度差"= |归档命令 − 平衡态 act|；相位/步进 = 归档帧间差。见 `Q4_what_the_drive_lacks`。',
        '- 归档前 %d 帧全零；帧间最大步进 %.6g、平均步进 %.6g；命令均值 %.6g，零元素占比 %.6g。'
        % (drive_shape['leading_all_zero_frames'], drive_shape['max_step_between_frames'],
           drive_shape['mean_step_between_frames'], drive_shape['mean_command_level'], drive_shape['fraction_of_entries_at_zero']),
        '',
        '## 披露',
        '- `qacc ≈ 0` 只是**静态必要条件**；本目录只报**实测**，不推断稳定性，也不含站立/行走结论。',
        '- 绝对力值与物理单位 `NOT_ESTABLISHED`（力数值为**模型单位**的相对量）。',
        '- **手动研究实施，不是成品。**'])
    (out / 'README.zh-CN.md').write_text(readme)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(Path(__file__), Path(__file__).name)
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != 'delivery-hashes.json')
    (out / 'delivery-hashes.json').write_text(json.dumps({p.name: sha(p) for p in files}, ensure_ascii=False, indent=1))
    (out / 'FROZEN_DELIVERY_HASHES.json').write_text(json.dumps(dict(
        status='FROZEN_BEFORE_VERIFICATION', frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'),
        files={p.name: sha(p) for p in files}), ensure_ascii=False, indent=1))
    print(report['status'])
    print('out:', out)
    print('first leg to lose tarsus contact:', first_leg_to_lose, '| first leg with force < 1%%:', first_leg_force_drop)
    print('first joint past 0.01 rad:', first_joint)
    print('legs<6 at step', report['Q1_leg_loss']['first_step_legs_in_contact_below_6'],
          '| legs==1 at step', report['Q1_leg_loss']['first_step_legs_in_contact_equals_1'],
          '| first non-foot at step', first_nonfoot)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
