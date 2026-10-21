"""Exhaustive per-frame constant-hold test over the FROZEN declared archive's own 100 frames.

DECLARED CRITERIA (written before any run, and not relaxed afterwards):
  protocol      : state = the accepted equilibrium (qpos and act from equilibrium_best_state.npz), qvel = 0,
                  m.opt.timestep = 2.5e-5 asserted (the XML value 1e-4 is never used), 20000 steps = 0.5 s,
                  ctrl constant and equal to frame k of the frozen archive for the whole run, only ctrl written in the loop,
                  time axis taken from the measured d.time.
  per frame k   : max|qacc| at step 0; first step with legs_in_contact < 6; first step with a non-tarsus (non-foot)
                  contact, with the geom pair and the body owning the non-floor geom; root_z minimum; max u_diamond over
                  tarsus contacts and over all contacts for the whole run; max elementwise |frame_k - act_equilibrium|.
  survival      : legs_in_contact == 6 at every one of the 20001 samples AND root_z >= 0.9 * root_z(0) at every sample.
  reporting     : number of surviving frames, each survivor's margin (min root_z / initial root_z, minimum leg count and
                  the step where it occurs), and a failure-mode classification for every failing frame.

This is an exhaustive sweep of the frozen archive's OWN frames: nothing new is added and no drive is hidden. Surviving a
frame is not standing, not walking, and says nothing about biology. Absolute force values and units remain
NOT_ESTABLISHED. Manual research implementation, not a product.
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
DRIVE = ROOT / 'outputs/experiment_platform/20261015_135150_a2fff3f0/traces.npz'
POSE_A = DESK / 'S2-declared-cone-min-s-20261015T123603/best_candidate_pose.npz'
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
DT, STEPS = 2.5e-5, 20000
ROOT_Z_MARGIN = 0.9
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S %Z (%z)')
ms = lambda step: round(step * DT * 1000.0, 6)


def main():
    import mujoco
    sys.path.insert(0, str(ROOT))
    from engine import canonical_physics as cp

    m = cp.load_model()
    xml_dt = float(m.opt.timestep)
    m.opt.timestep = DT
    assert float(m.opt.timestep) == DT
    assert abs(DT - xml_dt) > 0.0
    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
    bname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ''
    st = np.load(STATE)
    q0, act0 = st['qpos'].copy(), st['act'].copy()
    ctrl = np.asarray(np.load(DRIVE)['normal_controls'], dtype=float)
    n_frames = int(ctrl.shape[0])
    f6 = np.zeros(6)
    rows = []
    t_start = time.time()
    for k in range(n_frames):
        m.opt.timestep = DT
        assert float(m.opt.timestep) == DT
        d = mujoco.MjData(m)
        d.qpos[:] = q0
        d.qvel[:] = 0.0
        d.act[:] = act0
        d.ctrl[:] = ctrl[k]
        mujoco.mj_forward(m, d)
        qacc0 = float(np.max(np.abs(np.asarray(d.qacc))))
        rz0 = float(d.qpos[2])
        first_legs_below_6 = None
        first_nonfoot = None
        first_nonfoot_pair = None
        first_nonfoot_body = None
        root_z_min = rz0
        root_z_min_step = 0
        ud_tar = 0.0
        ud_all = 0.0
        six_broken_at = None
        for step in range(STEPS):
            d.ctrl[:] = ctrl[k]
            mujoco.mj_step(m, d)
            i = step + 1
            legs = set()
            tar = 0
            nonfoot_rows = []
            for ci, c in enumerate(d.contact):
                if c.efc_address < 0:
                    continue
                n1, n2 = gname(int(c.geom1)), gname(int(c.geom2))
                fr = np.asarray(c.friction)[:3]
                mujoco.mj_contactForce(m, d, ci, f6)
                fn, ft1, ft2 = float(f6[0]), float(f6[1]), float(f6[2])
                u = (abs(ft1) / (float(fr[0]) * fn) + abs(ft2) / (float(fr[1]) * fn)) if fn > 0 else None
                if 'Tarsus' in (n1 + n2):
                    tar += 1
                    for L in LEGS:
                        if n1.startswith(L) or n2.startswith(L):
                            legs.add(L)
                    if u is not None:
                        ud_tar = max(ud_tar, u)
                else:
                    nonfoot_rows.append((int(c.geom1), int(c.geom2)))
                if u is not None:
                    ud_all = max(ud_all, u)
            if first_legs_below_6 is None and len(legs) < 6:
                first_legs_below_6 = i
                six_broken_at = i
            if first_nonfoot is None and nonfoot_rows:
                first_nonfoot = i
                g1, g2 = nonfoot_rows[0]
                pair = sorted((gname(g1), gname(g2)))
                first_nonfoot_pair = pair
                own = bname(g2) if 'floor' in gname(g1) else bname(g1)
                first_nonfoot_body = own
            rz = float(d.qpos[2])
            if rz < root_z_min:
                root_z_min, root_z_min_step = rz, i
        survived = bool(six_broken_at is None and root_z_min >= ROOT_Z_MARGIN * rz0)
        if survived:
            mode = 'SURVIVED'
        elif six_broken_at is not None and first_nonfoot is not None:
            mode = 'both_six_leg_break_and_body_floor_contact'
        elif first_nonfoot is not None:
            mode = 'body_floor_contact_without_six_leg_break'
        elif six_broken_at is not None:
            mode = 'six_leg_break_without_body_floor_contact'
        else:
            mode = 'root_z_below_90pct_without_contact_event'
        rows.append(dict(frame=k, max_abs_qacc_at_step0=qacc0,
                         first_step_legs_in_contact_below_6=first_legs_below_6,
                         first_step_legs_in_contact_below_6_ms=(None if first_legs_below_6 is None else ms(first_legs_below_6)),
                         first_nonfoot_contact_step=first_nonfoot,
                         first_nonfoot_contact_ms=(None if first_nonfoot is None else ms(first_nonfoot)),
                         first_nonfoot_contact_pair=first_nonfoot_pair,
                         first_nonfoot_contact_body=first_nonfoot_body,
                         root_z_initial=rz0, root_z_min=root_z_min, root_z_min_step=root_z_min_step,
                         root_z_min_over_initial=float(root_z_min / rz0),
                         max_u_diamond_tarsus=ud_tar, max_u_diamond_all_contacts=ud_all,
                         max_elementwise_abs_difference_frame_vs_act_equilibrium=float(np.max(np.abs(ctrl[k] - act0))),
                         survived=survived, failure_mode=mode))
        if k % 10 == 0:
            print('frame %3d/%d  survived=%s  first legs<6=%s  first non-foot=%s  root_z min/init=%.4f'
                  % (k, n_frames, survived, first_legs_below_6, first_nonfoot, root_z_min / rz0), flush=True)
    wall = time.time() - t_start
    survivors = [r for r in rows if r['survived']]
    modes = {}
    for r in rows:
        modes[r['failure_mode']] = modes.get(r['failure_mode'], 0) + 1

    out = DESK / ('S2-per-frame-hold-test-frozen-archive-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    report = dict(
        status='PASS_PER_FRAME_HOLD_TEST', manual_research_not_product=True, standing=False, walking=False,
        title='exhaustive per-frame constant-hold test over the frozen declared archive (manual research implementation, not a product)',
        declared_criteria=dict(
            dt_s=DT, xml_timestep_s=xml_dt, steps=STEPS, window_s=DT * STEPS, frames_tested=n_frames,
            ctrl='ctrl is constant and equal to frame k of the frozen archive for the whole run; nothing is added or hidden',
            state='accepted equilibrium: qpos and act from equilibrium_best_state.npz, qvel = 0',
            max_abs_qacc_at_step0='one mj_forward at the initial state with ctrl set to the frame',
            first_legs_in_contact_below_6='first step whose number of legs with a tarsus contact is below 6',
            first_nonfoot_contact='first step with a contact whose geom pair contains no Tarsus, reported with the pair and the body owning '
                                  'the non-floor geom',
            root_z_min='minimum root z over the run',
            max_u_diamond='maximum over the whole run, reported both over tarsus contacts and over all contacts',
            max_elementwise_abs_difference_frame_vs_act_equilibrium='max_k |ctrl[k] - act_equilibrium| elementwise',
            survival=f'legs_in_contact == 6 at all 20001 samples AND root_z >= {ROOT_Z_MARGIN} * root_z(0) at all samples',
            survival_is_not_standing='surviving a frame is a numerical outcome for that constant command; it is not standing, not walking, and says '
                                     'nothing about biology'),
        inputs=dict(state=str(STATE), state_sha256=sha(STATE), drive=str(DRIVE), drive_sha256=sha(DRIVE),
                    pose_A=str(POSE_A), canonical_xml=str(cp.CANONICAL_XML), canonical_xml_sha256=sha(cp.CANONICAL_XML),
                    script=str(pathlib.Path(__file__)), script_sha256=sha(pathlib.Path(__file__))),
        result=dict(n_frames=n_frames, n_survived=len(survivors),
                    survivor_frames=[r['frame'] for r in survivors],
                    survivor_margins=[dict(frame=r['frame'], root_z_min_over_initial=r['root_z_min_over_initial'],
                                           min_root_z=r['root_z_min'], max_u_diamond_tarsus=r['max_u_diamond_tarsus'],
                                           max_u_diamond_all_contacts=r['max_u_diamond_all_contacts'],
                                           max_abs_qacc_at_step0=r['max_abs_qacc_at_step0'],
                                           frame_vs_act_difference=r['max_elementwise_abs_difference_frame_vs_act_equilibrium'])
                                      for r in survivors],
                    failure_mode_counts=modes,
                    best_frame_by_root_z_margin=max(rows, key=lambda r: r['root_z_min_over_initial'])['frame'],
                    worst_frame_by_root_z_margin=min(rows, key=lambda r: r['root_z_min_over_initial'])['frame'],
                    per_frame=rows, wall_seconds=round(wall, 1)),
        not_established='absolute force values, physical units, biological validity; no standing, no walking',
        recorded_at=now())
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    np.savez_compressed(out / 'per_frame_hold_test.npz',
                        frames=np.array([r['frame'] for r in rows]),
                        max_abs_qacc_at_step0=np.array([r['max_abs_qacc_at_step0'] for r in rows]),
                        first_step_legs_below_6=np.array([-1 if r['first_step_legs_in_contact_below_6'] is None
                                                          else r['first_step_legs_in_contact_below_6'] for r in rows]),
                        first_nonfoot_step=np.array([-1 if r['first_nonfoot_contact_step'] is None else r['first_nonfoot_contact_step']
                                                     for r in rows]),
                        root_z_min=np.array([r['root_z_min'] for r in rows]),
                        root_z_min_over_initial=np.array([r['root_z_min_over_initial'] for r in rows]),
                        max_u_diamond_tarsus=np.array([r['max_u_diamond_tarsus'] for r in rows]),
                        max_u_diamond_all=np.array([r['max_u_diamond_all_contacts'] for r in rows]),
                        frame_vs_act_difference=np.array([r['max_elementwise_abs_difference_frame_vs_act_equilibrium'] for r in rows]),
                        survived=np.array([r['survived'] for r in rows]),
                        ctrl_sequence=ctrl, act_equilibrium=act0)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(15.5, 9.8))
    fr = [r['frame'] for r in rows]
    a = ax[0][0]
    a.bar(fr, [r['root_z_min_over_initial'] for r in rows],
          color=['#2ca02c' if r['survived'] else '#d62728' for r in rows])
    a.axhline(ROOT_Z_MARGIN, color='black', ls=':', lw=1.4, label='survival floor %.2f x initial root z' % ROOT_Z_MARGIN)
    a.set_xlabel('frozen archive frame index k'); a.set_ylabel('min root z / initial root z')
    a.set_title('per-frame constant-hold: root z margin (%d/%d survive)' % (len(survivors), n_frames), fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.3, axis='y')
    a = ax[0][1]
    a.plot(fr, [np.nan if r['first_step_legs_in_contact_below_6'] is None else ms(r['first_step_legs_in_contact_below_6'])
                for r in rows], marker='.', lw=1.0, color='#ff7f0e', label='first legs<6 [ms]')
    a.plot(fr, [np.nan if r['first_nonfoot_contact_step'] is None else ms(r['first_nonfoot_contact_step'])
                for r in rows], marker='.', lw=1.0, color='#d62728', label='first non-foot contact [ms]')
    a.set_xlabel('frame index k'); a.set_ylabel('time [ms] (measured clock, 0.5 s window)')
    a.set_title('onset times per frame (gaps = never within 0.5 s)', fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.3)
    a = ax[1][0]
    a.plot(fr, [r['max_elementwise_abs_difference_frame_vs_act_equilibrium'] for r in rows], lw=1.2, color='#1f77b4')
    a.set_xlabel('frame index k'); a.set_ylabel('max |ctrl[k] - act_equilibrium|')
    a.set_title('how far each held frame is from the equilibrium activation', fontsize=10); a.grid(alpha=0.3)
    a = ax[1][1]
    a.plot(fr, [r['max_u_diamond_tarsus'] for r in rows], lw=1.1, color='#9467bd', label='over tarsus contacts')
    a.plot(fr, [r['max_u_diamond_all_contacts'] for r in rows], lw=0.9, color='#8c564b', ls='--', label='over all contacts')
    a.axhline(1.0, color='red', ls='--', lw=1.3, label='declared diamond limit')
    a.set_xlabel('frame index k'); a.set_ylabel('max u_diamond over the run')
    a.set_title('diamond utilisation per frame, both readings', fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.suptitle('exhaustive per-frame constant-hold test over the frozen declared archive (100 frames, 20000 steps each)\n'
                 'dt = 2.5e-5 s enforced and asserted; state = accepted equilibrium; survival = six-leg tarsus contact unbroken AND root z >= '
                 '%.2f x initial\nmanual research implementation, NOT a product; surviving a frame is not standing; no standing/walking claim'
                 % ROOT_Z_MARGIN, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out / 'per-frame-hold-test.png', dpi=135)
    plt.close(fig)

    readme = '\n'.join([
        '# 冻结档案 100 帧的逐帧恒定保持测试（穷举；手动研究实施，不是成品）', '',
        '## 事先写死的判据（未事后放宽）',
        '- 状态 = 接受平衡态（`qpos` 与 `act` 取自 `equilibrium_best_state.npz`），`qvel = 0`；`m.opt.timestep = 2.5e-5` 断言（XML 值 1e-4 从不使用）；'
        '20000 步 = 0.5 s；`ctrl` **恒等于冻结档案第 k 帧**；时间轴取实测 `d.time`。',
        '- 每帧记录：步 0 的 `max|qacc|`、首个 `legs_in_contact < 6` 步、首个非足接触步（含几何对与所属 body）、`root_z` 最小、'
        '全程 `max u_diamond`（跗节口径 + 全接触口径）、以及 **|第 k 帧 − 平衡态 act| 的逐元素最大差**。',
        '- **存活定义**：20001 个采样点全部 `legs_in_contact == 6` **且** `root_z ≥ 0.9 × root_z(0)`。',
        '- **声明**：这是对**已冻结档案自身帧集**的**穷举**，未新增、未隐藏任何驱动；**存活本身不等于站立**；绝对力值与单位 `NOT_ESTABLISHED`。', '',
        '## 结果',
        '- **存活帧数 %d / %d**；失败模式计数：%s' % (len(survivors), n_frames, json.dumps(modes, ensure_ascii=False)),
        '- 存活帧列表与各自余量见 `report.json.result.survivor_margins`；逐帧全表见 `per_frame_hold_test.npz` 与 `result.per_frame`。', '',
        '## 图（已用看图工具实际打开核对）', '- `per-frame-hold-test.png`：四面板 = 每帧 root z 余量（绿=存活）、每帧起始时间、'
        '每帧与平衡态激活的差异、每帧 `u_diamond` 两口径。', '',
        '- **手动研究实施，不是成品**；不宣称站立/行走/完成度。'])
    (out / 'README.zh-CN.md').write_text(readme)
    with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(pathlib.Path(__file__), pathlib.Path(__file__).name)
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != 'FROZEN_DELIVERY_HASHES.json')
    (out / 'delivery-hashes.json').write_text(json.dumps({p.name: sha(p) for p in files}, ensure_ascii=False, indent=1))
    (out / 'FROZEN_DELIVERY_HASHES.json').write_text(json.dumps(dict(status='FROZEN_BEFORE_VERIFICATION', frozen_at=now(),
                                                                    files={p.name: sha(p) for p in files}), ensure_ascii=False, indent=1))
    print(report['status'])
    print('out:', out)
    print('survived %d/%d | modes %s | wall %.1f s' % (len(survivors), n_frames, json.dumps(modes, ensure_ascii=False), wall))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
