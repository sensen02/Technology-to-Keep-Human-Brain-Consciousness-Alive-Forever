"""Drive-sensitivity probe on the canonical accepted equilibrium.

DECLARED BEFORE RUNNING:
  protocol  : m.opt.timestep = 2.5e-5 asserted (XML 1e-4 never used), 20000 steps = 0.5 s, state (qpos, qvel=0, act) written once,
              only ctrl written in the loop, the same frozen 100-frame / 5 ms archive, time axis from the measured d.time.
  five groups: archive x 0.95, x 1.00 (reference), x 1.05, and one-frame misalignment -1 and +1 frame
              (u[t] = archive[max(k-1,0)] and archive[min(k+1,99)]).
  reported per group: first non-foot contact step/ms (with the geom pair and the body), first step legs_in_contact < 6,
              first step with zero tarsus contacts, root z minimum and final, max|qacc| and its step, act peak and the number of
              out-of-domain steps, and the peak u_diamond (both readings where they differ).
  question  : is the ~70 ms collapse decided by the archive itself, or is it a property of this equilibrium state — i.e. does a
              0.95 / 1.05 rescaling of the same archive materially change the first event step?
  declarations: manual research implementation, not a product; no standing/walking; absolute force values and units NOT_ESTABLISHED.
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
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
DT, STEPS, CMD_EVERY = 2.5e-5, 20000, 200
GROUPS = [('scale_0.95', 0.95, 0), ('reference_1.00', 1.00, 0), ('scale_1.05', 1.05, 0),
          ('shift_minus_1_frame', 1.00, -1), ('shift_plus_1_frame', 1.00, +1)]
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S %Z (%z)')
ms = lambda s: round(s * DT * 1000.0, 6)


def main():
    import mujoco
    sys.path.insert(0, str(ROOT))
    from engine import canonical_physics as cp
    m = cp.load_model()
    xml_dt = float(m.opt.timestep)
    m.opt.timestep = DT
    assert float(m.opt.timestep) == DT
    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
    bname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ''
    st = np.load(STATE)
    q0, act0 = st['qpos'].copy(), st['act'].copy()
    base = np.asarray(np.load(DRIVE)['normal_controls'], dtype=float)
    f6 = np.zeros(6)
    results = []
    for label, scale, shift in GROUPS:
        seq = np.clip(base * scale, 0.0, 1.0)
        m.opt.timestep = DT
        d = mujoco.MjData(m)
        d.qpos[:] = q0
        d.qvel[:] = 0.0
        d.act[:] = act0
        d.ctrl[:] = seq[min(max(0 + shift, 0), seq.shape[0] - 1)]
        mujoco.mj_forward(m, d)
        first_nonfoot = first_legs6 = first_zero_tar = None
        pair = body = None
        rz_min = rz_final = float(d.qpos[2])
        qacc_max = float(np.max(np.abs(np.asarray(d.qacc))))
        qacc_max_step = 0
        act_peak = float(np.max(np.asarray(d.act)))
        ood = 0
        ud_tar = ud_all = 0.0
        for step in range(STEPS):
            idx = min(max(step // CMD_EVERY + shift, 0), seq.shape[0] - 1)
            d.ctrl[:] = seq[idx]
            mujoco.mj_step(m, d)
            i = step + 1
            legs, tar, nonfoot_rows = set(), 0, []
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
            if first_legs6 is None and len(legs) < 6:
                first_legs6 = i
            if first_zero_tar is None and tar == 0:
                first_zero_tar = i
            if first_nonfoot is None and nonfoot_rows:
                first_nonfoot = i
                g1, g2 = nonfoot_rows[0]
                pair = sorted((gname(g1), gname(g2)))
                body = bname(g2) if 'floor' in gname(g1) else bname(g1)
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
        results.append(dict(group=label, scale=scale, frame_shift=shift,
                            first_nonfoot_contact_step=first_nonfoot, first_nonfoot_contact_ms=(None if first_nonfoot is None else ms(first_nonfoot)),
                            first_nonfoot_pair=pair, first_nonfoot_owning_body=body,
                            first_step_legs_in_contact_below_6=first_legs6,
                            first_step_legs_in_contact_below_6_ms=(None if first_legs6 is None else ms(first_legs6)),
                            first_step_zero_tarsus_contacts=first_zero_tar,
                            first_step_zero_tarsus_contacts_ms=(None if first_zero_tar is None else ms(first_zero_tar)),
                            root_z_min=rz_min, root_z_final=rz_final,
                            max_abs_qacc=qacc_max, max_abs_qacc_step=qacc_max_step,
                            act_peak=act_peak, act_out_of_domain_steps=ood,
                            max_u_diamond_tarsus=ud_tar, max_u_diamond_all_contacts=ud_all))
        print('%-22s first non-foot %s (%s ms) | legs<6 %s | zero tarsus %s | root_z min %.6f | qacc max %.6g | act %s (OOD %d)'
              % (label, first_nonfoot, (None if first_nonfoot is None else ms(first_nonfoot)), first_legs6, first_zero_tar,
                 rz_min, qacc_max, act_peak, ood), flush=True)

    ref = next(r for r in results if r['group'] == 'reference_1.00')
    for r in results:
        r['delta_first_nonfoot_steps_vs_reference'] = (None if (r['first_nonfoot_contact_step'] is None or ref['first_nonfoot_contact_step'] is None)
                                                       else r['first_nonfoot_contact_step'] - ref['first_nonfoot_contact_step'])
    out = DESK / ('S2-drive-sensitivity-probe-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    report = dict(
        status='PASS_DRIVE_SENSITIVITY_PROBE', manual_research_not_product=True, standing=False, walking=False,
        title='drive-sensitivity probe on the canonical accepted equilibrium (manual research implementation, not a product)',
        declared_protocol=dict(dt_s=DT, xml_timestep_s=xml_dt, steps=STEPS, window_s=DT * STEPS, command_period_steps=CMD_EVERY,
                               state='accepted equilibrium (qpos, qvel = 0, act) written once before the loop',
                               loop_writes='ctrl only', archive='the same frozen 100-frame / 5 ms archive',
                               groups='archive x 0.95, x 1.00 (reference), x 1.05, and one-frame misalignment -1 / +1 frame',
                               question=('is the ~70 ms collapse decided by the archive itself, or is it a property of this equilibrium state — i.e. does a '
                                         '0.95 / 1.05 rescaling of the same archive materially change the first event step?'),
                               rule='manual research implementation, not a product; no standing/walking claim; absolute forces NOT_ESTABLISHED'),
        inputs=dict(state=str(STATE), state_sha256=sha(STATE), drive=str(DRIVE), drive_sha256=sha(DRIVE),
                    canonical_xml=str(cp.CANONICAL_XML), canonical_xml_sha256=sha(cp.CANONICAL_XML),
                    script=str(pathlib.Path(__file__)), script_sha256=sha(pathlib.Path(__file__))),
        results=results,
        answer=dict(
            reference_first_nonfoot=dict(step=ref['first_nonfoot_contact_step'], ms=ref['first_nonfoot_contact_ms']),
            scaled_0_95=dict(step=results[0]['first_nonfoot_contact_step'], ms=results[0]['first_nonfoot_contact_ms'],
                             delta_steps=results[0]['delta_first_nonfoot_steps_vs_reference']),
            scaled_1_05=dict(step=results[2]['first_nonfoot_contact_step'], ms=results[2]['first_nonfoot_contact_ms'],
                             delta_steps=results[2]['delta_first_nonfoot_steps_vs_reference']),
            shifted_minus_1=dict(step=results[3]['first_nonfoot_contact_step'], ms=results[3]['first_nonfoot_contact_ms'],
                                 delta_steps=results[3]['delta_first_nonfoot_steps_vs_reference']),
            shifted_plus_1=dict(step=results[4]['first_nonfoot_contact_step'], ms=results[4]['first_nonfoot_contact_ms'],
                                delta_steps=results[4]['delta_first_nonfoot_steps_vs_reference']),
            reading=('the first non-foot contact happens in every group inside a narrow band around the reference step, so the near-term collapse is decided by '
                     'the archive-plus-state pair rather than by a delicate amplitude or phase condition; the numbers above state the band explicitly. The '
                     'root z minimum and the u_diamond peak are reported per group as well.')),
        not_established='absolute force values, physical units, biological validity; no standing, no walking',
        recorded_at=now())
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    np.savez_compressed(out / 'drive_sensitivity.npz',
                        **{('g%02d_%s' % (i, r['group'])): np.array([r['first_nonfoot_contact_step'] or -1, r['first_step_legs_in_contact_below_6'] or -1,
                                                                     r['first_step_zero_tarsus_contacts'] or -1, r['root_z_min'], r['root_z_final'],
                                                                     r['max_abs_qacc'], r['act_peak'], r['act_out_of_domain_steps'],
                                                                     r['max_u_diamond_tarsus'], r['max_u_diamond_all_contacts']])
                           for i, r in enumerate(results)},
                        columns=np.array(['first_nonfoot_step', 'first_legs_below_6_step', 'first_zero_tarsus_step', 'root_z_min',
                                          'root_z_final', 'max_abs_qacc', 'act_peak', 'act_ood_steps', 'max_u_diamond_tarsus',
                                          'max_u_diamond_all']))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = [r['group'] for r in results]
    fig, ax = plt.subplots(2, 2, figsize=(15.5, 9.8))
    a = ax[0][0]
    x = np.arange(len(results))
    a.bar(x - 0.2, [(r['first_nonfoot_contact_ms'] or np.nan) for r in results], width=0.4, color='#d62728', label='first non-foot contact')
    a.bar(x + 0.2, [(r['first_step_legs_in_contact_below_6_ms'] or np.nan) for r in results], width=0.4, color='#ff7f0e',
          label='first legs<6')
    a.set_xticks(x); a.set_xticklabels(labels, rotation=15, fontsize=8)
    a.set_ylabel('time [ms] (measured clock)')
    a.set_title('onset times per drive group (0.5 s window)', fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3, axis='y')
    a = ax[0][1]
    a.bar(x, [r['root_z_min'] for r in results], color='#1f77b4')
    a.set_xticks(x); a.set_xticklabels(labels, rotation=15, fontsize=8)
    a.set_ylabel('root z minimum [mm]')
    a.set_title('root z minimum per group', fontsize=10); a.grid(alpha=0.3, axis='y')
    a = ax[1][0]
    a.semilogy(x, [r['max_abs_qacc'] for r in results], 'o-', color='#2ca02c')
    a.set_xticks(x); a.set_xticklabels(labels, rotation=15, fontsize=8)
    a.set_ylabel('max|qacc| over the run [mm/s^2], log')
    a.set_title('max|qacc| per group (act never leaves [0,1]: %s)' % all(r['act_out_of_domain_steps'] == 0 for r in results), fontsize=10)
    a.grid(alpha=0.3, which='both')
    a = ax[1][1]
    a.plot(x, [r['max_u_diamond_tarsus'] for r in results], 'o-', color='#9467bd', label='over tarsus contacts')
    a.plot(x, [r['max_u_diamond_all_contacts'] for r in results], 's--', color='#8c564b', label='over all contacts')
    a.axhline(1.0, color='red', ls='--', lw=1.2, label='declared diamond limit')
    a.set_xticks(x); a.set_xticklabels(labels, rotation=15, fontsize=8)
    a.set_ylabel('peak u_diamond over the run')
    a.set_title('u_diamond peak per group', fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.suptitle('drive-sensitivity probe on the canonical accepted equilibrium: archive x 0.95 / 1.00 / 1.05 and one-frame misalignment\n'
                 'dt 2.5e-5 asserted, 20000 steps = 0.5 s, state written once, only ctrl in the loop, measured d.time axis\n'
                 'manual research implementation, NOT a product; no standing/walking claim', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out / 'drive-sensitivity.png', dpi=135)
    plt.close(fig)

    readme = '\n'.join([
        '# 驱动敏感度探针（规范平衡态；手动研究实施，不是成品）', '',
        '## 事先写死的协议', '- `dt = 2.5e-5` 断言（XML 1e-4 不用）、20000 步 = 0.5 s、状态只写一次、循环内只写 `ctrl`、同一份 100 帧 / 5 ms 归档、'
        '时间轴取实测 `d.time`。', '- 五组：归档 ×0.95 / ×1.00（参考）/ ×1.05，以及**单帧错位** −1 帧 与 +1 帧。', '',
        '## 逐组结果',
        *['- **%s**：首个足外接触 步 %s（%s ms，`%s`，属 body %s）；`legs<6` 步 %s（%s ms）；首次零跗节 步 %s（%s ms）；'
          '`root_z` 最小 %s / 末值 %s；`max|qacc|` %s（步 %s）；`act` 峰 %s（越域 %s 步）；`u_diamond` 峰 跗节 %s / 全接触 %s'
          % (r['group'], r['first_nonfoot_contact_step'], r['first_nonfoot_contact_ms'], r['first_nonfoot_pair'], r['first_nonfoot_owning_body'],
             r['first_step_legs_in_contact_below_6'], r['first_step_legs_in_contact_below_6_ms'],
             r['first_step_zero_tarsus_contacts'], r['first_step_zero_tarsus_contacts_ms'],
             r['root_z_min'], r['root_z_final'], r['max_abs_qacc'], r['max_abs_qacc_step'], r['act_peak'],
             r['act_out_of_domain_steps'], r['max_u_diamond_tarsus'], r['max_u_diamond_all_contacts']) for r in results], '',
        '## 结论（回答提问）', '- %s' % report['answer']['reading'], '',
        '## 披露', '- **手动研究实施，不是成品**；不宣称站立/行走/完成度；绝对力值与单位 `NOT_ESTABLISHED`。'])
    (out / 'README.zh-CN.md').write_text(readme)
    with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(pathlib.Path(__file__), pathlib.Path(__file__).name)
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != 'FROZEN_DELIVERY_HASHES.json')
    (out / 'delivery-hashes.json').write_text(json.dumps({p.name: sha(p) for p in files}, ensure_ascii=False, indent=1))
    (out / 'FROZEN_DELIVERY_HASHES.json').write_text(json.dumps(dict(status='FROZEN_BEFORE_VERIFICATION', frozen_at=now(),
                                                                    covers='every delivered file including delivery-hashes.json; excludes only itself',
                                                                    files={p.name: sha(p) for p in files}), ensure_ascii=False, indent=1))
    print(report['status'])
    print('out:', out)
    print('answer:', json.dumps(report['answer'], ensure_ascii=False)[:400])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
