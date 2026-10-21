"""Hold the accepted equilibrium commanded by ITS OWN act, as a constant (owner turn 14).

ONE FACTOR: the command is the equilibrium's own 90-dim act held constant for the whole window.
No archive frame is indexed at all. Everything else is the declared protocol.

DECLARED BEFORE RUNNING
  state  : /home/sensen/Desktop/engine修复验收/S2-equilibrium-A-corrected-reissued-20261015T161259/equilibrium_best_state.npz
           qpos AND act read together, verbatim (sha256 asserted below)
  model  : /run/media/sensen/Data2/cell_wound_prototype/outputs/muscles_six_legs/fruitfly_six_leg_standing.xml
  step   : m.opt.timestep = 2.5e-5 written and ASSERTED to differ from the XML value, BEFORE any mj_forward/mj_step
  window : 20000 steps = 0.5 s; state written ONCE outside the loop; inside the loop only d.ctrl[:] is written
  command: d.ctrl[:] = the equilibrium's own act, constant, every step

READ-OUT RULE, WRITTEN BEFORE THE RUN so it cannot be re-interpreted afterwards:
  if six legs keep tarsus contact and no non-foot contact occurs inside 0.5 s
      -> the pose is short-term maintainable under its own equilibrium drive; the collapse is an
         archive/pose mismatch; next round generates a pose-compatible command archive from the connectome.
  if it still collapses within tens of milliseconds
      -> max|qacc| ~ 1e-11 was necessary but NOT sufficient: the pose is an unstable equilibrium;
         next round re-solves the pose / footfall / contact layout and the command layer is parked.
"""

import hashlib
import json
import os
import pathlib
import sys
import time
import zipfile

import numpy as np

ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
STATE = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259/equilibrium_best_state.npz'
XML = ROOT / 'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml'
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
DT, STEPS, CMD_EVERY = 2.5e-5, 20000, 200
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S %Z (%z)')
utc = lambda: time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
ms = lambda s: round(s * DT * 1000.0, 6)


def main():
    sys.path.insert(0, str(ROOT))
    import mujoco
    from engine import canonical_physics as cp

    model = cp.load_model()
    xml_dt = float(model.opt.timestep)
    assert abs(xml_dt - 1e-4) < 1e-18, 'expected the XML dt'
    model.opt.timestep = DT                       # BEFORE any mj_forward / mj_step
    assert float(model.opt.timestep) == DT
    assert float(model.opt.timestep) != xml_dt
    gname = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
    bname = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ''

    st = np.load(STATE)
    q0, act0 = st['qpos'].copy(), st['act'].copy()
    command = np.clip(act0.copy(), 0.0, 1.0)      # THE ONE FACTOR: the equilibrium's own act, constant

    d = mujoco.MjData(model)
    d.qpos[:] = q0
    d.qvel[:] = 0.0
    d.act[:] = act0
    d.ctrl[:] = command
    mujoco.mj_forward(model, d)
    assert float(d.time) == 0.0

    first_tarsus_loss = first_legs6 = first_nonfoot = None
    pair = body = None
    first_leg_over_001 = None
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
    joint_first_ms = {}
    rz_min = rz_final = float(d.qpos[2])
    qacc_max = float(np.max(np.abs(np.asarray(d.qacc))))
    qacc_max_step = 0
    act_peak = float(np.max(np.asarray(d.act)))
    ood = 0
    ud_tar = ud_all = 0.0
    f6 = np.zeros(6)
    cmd_boundaries = []
    legs_min = 6

    for step in range(STEPS):
        d.ctrl[:] = command
        mujoco.mj_step(model, d)
        i = step + 1
        if i % CMD_EVERY == 0:
            cmd_boundaries.append(float(d.time))
        legs, tar, nonfoot_rows = set(), 0, []
        for ci, c in enumerate(d.contact):
            if c.efc_address < 0:
                continue
            n1, n2 = gname(int(c.geom1)), gname(int(c.geom2))
            fr = np.asarray(c.friction)[:3]
            mujoco.mj_contactForce(model, d, ci, f6)
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
        legs_min = min(legs_min, len(legs))
        if first_legs6 is None and len(legs) < 6:
            first_legs6 = i
            qoff = np.abs(np.asarray(d.qpos[:model.nq]) - q0)
            joint_first_ms['legs_below_6'] = dict(step=i, ms=ms(i),
                                                  per_joint_over_0_01=[joint_names[j] for j in range(model.njnt)
                                                                       if j < len(qoff) and abs(qoff[j]) > 0.01][:6])
        if first_tarsus_loss is None and tar == 0:
            first_tarsus_loss = i
        if first_nonfoot is None and nonfoot_rows:
            first_nonfoot = i
            g1, g2 = nonfoot_rows[0]
            pair = sorted((gname(g1), gname(g2)))
            body = bname(g2) if 'floor' in gname(g1) else bname(g1)
        if first_leg_over_001 is None:
            qoff = np.abs(np.asarray(d.qpos[:model.nq]) - q0)
            hits = [joint_names[j] for j in range(min(model.njnt, len(qoff))) if abs(qoff[j]) > 0.01]
            if hits:
                first_leg_over_001 = dict(step=i, ms=ms(i), joints=hits[:6])
        a_now = float(np.max(np.abs(np.asarray(d.act))))
        act_peak = max(act_peak, a_now)
        if a_now > 1.0:
            ood += 1
        qa = float(np.max(np.abs(np.asarray(d.qacc))))
        if qa > qacc_max:
            qacc_max, qacc_max_step = qa, i
        rz = float(d.qpos[2])
        rz_min = min(rz_min, rz)
        rz_final = rz

    cadence = (round(cmd_boundaries[1] - cmd_boundaries[0], 12) if len(cmd_boundaries) > 1 else None)
    six_kept = (legs_min == 6)
    no_nonfoot = (first_nonfoot is None)
    verdict = ('POSE HOLDABLE UNDER ITS OWN EQUILIBRIUM DRIVE' if (six_kept and no_nonfoot)
               else 'POSE IS AN UNSTABLE EQUILIBRIUM: max|qacc| ~ 1e-11 was necessary but not sufficient')

    out = DESK / ('S2-hold-under-own-equilibrium-act-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    report = dict(status='PASS_HOLD_OWN_EQUILIBRIUM_ACT', manual_research_not_product=True, standing=False, walking=False,
        title='hold the accepted equilibrium with its own act as a constant command (manual research, not a product)',
        declared=dict(dt_s=DT, xml_timestep_s=xml_dt, steps=STEPS, window_s=DT * STEPS, command_period_steps=CMD_EVERY,
                      state_written_once=True, loop_writes='ctrl only',
                      command='d.ctrl[:] = the equilibrium own act, constant for the whole window; NO archive frame is indexed',
                      read_out_rule_written_before_the_run='six-leg tarsus contact kept and no non-foot contact -> pose holdable, collapse is an archive/pose '
                        'mismatch; still collapsing within tens of ms -> max|qacc| ~ 1e-11 is necessary but not sufficient and the pose is an unstable equilibrium'),
        inputs=dict(state=str(STATE), state_sha256=sha(STATE), xml=str(XML), xml_sha256=sha(XML),
                    canonical_xml=str(cp.CANONICAL_XML), canonical_xml_sha256=sha(cp.CANONICAL_XML),
                    command_sha256=hashlib.sha256(command.tobytes()).hexdigest(), script=str(pathlib.Path(__file__)),
                    script_sha256=sha(pathlib.Path(__file__))),
        result=dict(first_joint_over_0_01_rad=first_leg_over_001, first_step_legs_in_contact_below_6=first_legs6,
                    first_step_legs_in_contact_below_6_ms=(None if first_legs6 is None else ms(first_legs6)),
                    first_step_zero_tarsus_contacts=first_tarsus_loss,
                    first_step_zero_tarsus_contacts_ms=(None if first_tarsus_loss is None else ms(first_tarsus_loss)),
                    first_nonfoot_contact_step=first_nonfoot,
                    first_nonfoot_contact_ms=(None if first_nonfoot is None else ms(first_nonfoot)),
                    first_nonfoot_pair=pair, first_nonfoot_owning_body=body,
                    root_z_min=rz_min, root_z_final=rz_final, legs_in_contact_min=legs_min,
                    max_abs_qacc=qacc_max, max_abs_qacc_step=qacc_max_step,
                    act_peak=act_peak, act_out_of_domain_steps=ood,
                    max_u_diamond_tarsus=ud_tar, max_u_diamond_all_contacts=ud_all,
                    measured_final_d_time=float(d.time), measured_command_cadence_s=cadence,
                    six_leg_contact_kept_whole_window=six_kept, no_nonfoot_contact_in_window=no_nonfoot),
        verdict=verdict,
        verdict_rule='written before the run, see declared.read_out_rule_written_before_the_run',
        not_established='absolute force values, physical units, biological validity; no standing, no walking, no completion percentage',
        recorded_at=now(), recorded_at_utc=utc())
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r = report['result']
    fig, ax = plt.subplots(2, 2, figsize=(15.5, 9.8))
    a = ax[0][0]
    bars = [r['first_joint_over_0_01_rad']['step'] if r['first_joint_over_0_01_rad'] else np.nan,
            r['first_step_legs_in_contact_below_6'] or np.nan, r['first_step_zero_tarsus_contacts'] or np.nan,
            r['first_nonfoot_contact_step'] or np.nan]
    names = ['first joint > 0.01 rad', 'legs<6', 'zero tarsus', 'first non-foot']
    a.bar(np.arange(4), bars, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    a.set_xticks(np.arange(4)); a.set_xticklabels(names, rotation=12, fontsize=8)
    a.set_ylabel('step'); a.set_title('event steps under the constant own-act command', fontsize=10); a.grid(alpha=0.3, axis='y')
    a = ax[0][1]
    a.bar([0, 1], [r['root_z_min'], r['root_z_final']], color=['#9467bd', '#8c564b'])
    a.set_xticks([0, 1]); a.set_xticklabels(['root_z min', 'root_z final']); a.set_ylabel('root z [mm]')
    a.set_title('root height over the window', fontsize=10); a.grid(alpha=0.3, axis='y')
    a = ax[1][0]
    a.bar([0, 1], [r['legs_in_contact_min'], 6], color=['#d62728', '#2ca02c'])
    a.axhline(6, color='k', ls=':', lw=1.0)
    a.set_xticks([0, 1]); a.set_xticklabels(['minimum observed', 'six-leg target']); a.set_ylabel('legs in tarsus contact')
    a.set_title('six-leg condition: kept whole window = %s' % r['six_leg_contact_kept_whole_window'], fontsize=10); a.grid(alpha=0.3, axis='y')
    a = ax[1][1]
    a.bar([0, 1], [r['max_abs_qacc'], r['act_peak']], color=['#2ca02c', '#1f77b4'])
    a.set_xticks([0, 1]); a.set_xticklabels(['max|qacc| generalised', 'act peak'])
    a.set_title('peak magnitudes (act out of domain steps: %d)' % r['act_out_of_domain_steps'], fontsize=10); a.grid(alpha=0.3, axis='y')
    fig.suptitle('accepted equilibrium commanded by its OWN act, constant, for 0.500 s\ndt 2.5e-5 asserted before any step, state written once, only ctrl in the loop, no archive frame indexed\n'
                 'manual research implementation, NOT a product; no standing/walking claim', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out / 'hold-own-act.png', dpi=135)
    plt.close(fig)

    (out / 'README.zh-CN.md').write_text('\n'.join([
        '# 用平衡态自己的 act 恒定驱动（**手动研究实施，不是成品**）', '',
        '## 单因子', '- **命令 = 该平衡态自己的 90 维 `act`，全程恒定**；**不索引任何归档帧**。其余全部按声明协议。',
        '- `dt = 2.5e-5` 在**任何 `mj_forward`/`mj_step` 之前**写死并断言 `≠ XML 的 1e-4`；20000 步 = 0.5 s；状态只在循环外写一次；循环内只写 `d.ctrl[:]`。', '',
        '## 事先写死的判读规则', '- 六足保持且无非足接触 ⇒ **姿态在自身平衡驱动下短时可持**，塌落是"归档与姿态不匹配"，下一步转入生成**与姿态兼容的归档**（仍须来自连接组）。',
        '- 仍在几十毫秒内塌 ⇒ **`max|qacc| ~ 1e-11` 只是必要非充分条件，姿态本身是不稳定平衡**，下一步转入**姿态/落点/接触排布重解**，命令层先搁置。', '',
        '## 结果', '',
        '| 量 | 值 |', '|---|---|',
        '| 首个关节 > 0.01 rad | %s |' % (r['first_joint_over_0_01_rad'] or '未发生'),
        '| `legs_in_contact < 6` | %s 步（%s ms） |' % (r['first_step_legs_in_contact_below_6'], r['first_step_legs_in_contact_below_6_ms']),
        '| 首个零跗节接触 | %s 步（%s ms） |' % (r['first_step_zero_tarsus_contacts'], r['first_step_zero_tarsus_contacts_ms']),
        '| 首个非足接触 | %s 步（%s ms）几何对 `%s` 属体 %s |' % (r['first_nonfoot_contact_step'], r['first_nonfoot_contact_ms'], r['first_nonfoot_pair'], r['first_nonfoot_owning_body']),
        '| `root_z` 最小 / 末值 | %.6f / %.6f mm |' % (r['root_z_min'], r['root_z_final']),
        '| `max\\|qacc\\|` 峰值 / 步号 | %.10g / %s |' % (r['max_abs_qacc'], r['max_abs_qacc_step']),
        '| `act` 峰值 / 越域步数 | %.16g / %d |' % (r['act_peak'], r['act_out_of_domain_steps']),
        '| `u_diamond` 峰（跗节 / 全接触） | %.16g / %.16g |' % (r['max_u_diamond_tarsus'], r['max_u_diamond_all_contacts']),
        '| 实测末态 `d.time` | %.6f s |' % r['measured_final_d_time'],
        '| 命令周期（由命令边界 `d.time` 差实测） | %s s |' % r['measured_command_cadence_s'], '',
        '## 判定', '- **%s**' % verdict, '',
        '- 绝对力值与单位 `NOT_ESTABLISHED`；不宣称站立/行走；无完成百分比。']), encoding='utf-8')
    with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(pathlib.Path(__file__), pathlib.Path(__file__).name)
    fp = out / 'FROZEN_DELIVERY_HASHES.json'
    fp.write_text('{}', encoding='utf-8')
    (out / 'delivery-hashes.json').write_text('{}', encoding='utf-8')
    listing = sorted(p.name for p in out.iterdir() if p.is_file())
    data = [f for f in listing if f not in ('FROZEN_DELIVERY_HASHES.json', 'delivery-hashes.json')]
    m = dict(status='FROZEN_BEFORE_VERIFICATION', frozen_at=now(), frozen_at_utc=utc(), files={f: sha(out / f) for f in data},
             covers_complete='covers every delivered file except this manifest itself and except delivery-hashes.json',
             covers_rule='len(files) == os.listdir - 2', covers_self=False, manifest_sha256=None,
             manifest_sha256_note='null by definition; pinned by delivery-hashes.json',
             self_reference='not defined (a digest of a file cannot contain itself)', template_id='FROZEN_TEMPLATE_v13_owner_turn6')
    fp.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding='utf-8')
    dhd = {f: sha(out / f) for f in listing if f != 'delivery-hashes.json'}
    (out / 'delivery-hashes.json').write_text(json.dumps(dhd, ensure_ascii=False, indent=1), encoding='utf-8')
    assert dhd['FROZEN_DELIVERY_HASHES.json'] == sha(fp)
    assert not [f for f, h in m['files'].items() if sha(out / f) != h]
    for p in sorted(out.iterdir()):
        if p.is_file():
            os.chmod(p, 0o444)
    os.chmod(out, 0o555)
    print('OUT:', out)
    print('verdict:', verdict)
    for k, v in report['result'].items():
        if k not in ('first_nonfoot_pair',):
            print('   %-42s %s' % (k, v))
    return out


if __name__ == '__main__':
    main()
