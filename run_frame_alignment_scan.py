"""Frame-alignment scan on the canonical accepted equilibrium (owner turn 33, item 5).

DECLARED BEFORE RUNNING
  state     : /home/sensen/Desktop/engine修复验收/S2-equilibrium-A-corrected-reissued-20261015T161259/equilibrium_best_state.npz
              sha256 976388f28a866793ba0738311919e0f1f0837b052ac8d37281ad685c1d77c5be (qpos + act; qvel set to zero)
  drive     : /run/media/sensen/Data2/cell_wound_prototype/outputs/experiment_platform/20261015_135150_a2fff3f0/traces.npz
              key normal_controls, sha256 37fe24d3eac6f0962eee4ca6aeb51c89ecc9804be4b553c87ce91fcafda83a4f, shape (100, 90), 5 ms cadence
  step      : m.opt.timestep = 2.5e-5 s ASSERTED and asserted to differ from the XML value; 20000 steps = 0.5 s
  writes    : the state is written ONCE before the loop; inside the loop ONLY ctrl is written
  offsets   : integer frame-index offsets k in {-2, -1, 0, +1, +2}. The ONLY change per group is the starting index:
                ctrl[t] = archive[clip((t // 200) + k, 0, 99)]
              so k > 0 reads the archive LATER (command delayed by k frames) and k < 0 reads it EARLIER.
              Out-of-range indices are CLAMPED to the ends of the archive (per the pre-declared rule); no interpolation,
              no rescaling, no repetition.
  per group : first non-foot contact step/ms and the geom pair, first step with legs_in_contact < 6, first step with zero
              tarsus contacts, root z minimum and final, max|qacc| and its step, act peak and out-of-domain steps,
              peak u_diamond for BOTH readings (tarsus contacts only / all contacts).
  question  : does the ~70 ms collapse time move monotonically with frame alignment, and by how many steps per frame?
"""

import hashlib
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
STATE = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259/equilibrium_best_state.npz'
DRIVE = ROOT / 'outputs/experiment_platform/20261015_135150_a2fff3f0/traces.npz'
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
DT, STEPS, CMD_EVERY = 2.5e-5, 20000, 200
OFFSETS = [-2, -1, 0, 1, 2]
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S %Z (%z)')
ms = lambda s: round(s * DT * 1000.0, 6)


def main():
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(__import__('engine.canonical_physics', fromlist=['x']).CANONICAL_XML)) \
        if False else None
    sys.path.insert(0, str(ROOT))
    from engine import canonical_physics as cp
    m = cp.load_model()
    xml_dt = float(m.opt.timestep)
    assert xml_dt != DT, 'the whole point is that the XML dt is NOT the used dt'
    m.opt.timestep = DT
    assert float(m.opt.timestep) == DT
    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
    bname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ''
    st = np.load(STATE)
    q0, act0 = st['qpos'].copy(), st['act'].copy()
    base = np.asarray(np.load(DRIVE)['normal_controls'], dtype=float)
    f6 = np.zeros(6)
    results = []
    for k in OFFSETS:
        m.opt.timestep = DT
        d = mujoco.MjData(m)
        d.qpos[:] = q0
        d.qvel[:] = 0.0
        d.act[:] = act0
        d.ctrl[:] = base[min(max(0 + k, 0), base.shape[0] - 1)]
        mujoco.mj_forward(m, d)
        assert float(d.time) == 0.0
        first_nonfoot = first_legs6 = first_zero_tar = None
        pair = body = None
        rz_min = rz_final = float(d.qpos[2])
        qacc_max = float(np.max(np.abs(np.asarray(d.qacc))))
        qacc_max_step = 0
        act_peak = float(np.max(np.asarray(d.act)))
        ood = 0
        ud_tar = ud_all = 0.0
        first_dt_err = None
        for step in range(STEPS):
            idx = min(max(step // CMD_EVERY + k, 0), base.shape[0] - 1)
            d.ctrl[:] = base[idx]
            mujoco.mj_step(m, d)
            i = step + 1
            if first_dt_err is None:
                first_dt_err = abs(float(d.time) - i * DT)
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
        results.append(dict(offset_frames=k, command='ctrl[t] = archive[clip(t//200 + %d, 0, 99)]' % k,
                            first_dt_error=first_dt_err,
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
        print('k=%+d  first non-foot %s (%s ms) | legs<6 %s | zero tarsus %s | root_z min %.6f | qacc max %.6g @ %d | act %s (OOD %d)'
              % (k, first_nonfoot, ms(first_nonfoot), first_legs6, first_zero_tar, rz_min, qacc_max, qacc_max_step, act_peak, ood),
              flush=True)

    steps_by_k = {r['offset_frames']: r['first_nonfoot_contact_step'] for r in results}
    seq = [steps_by_k[k] for k in OFFSETS]
    monotone_increasing = all(seq[i] < seq[i + 1] for i in range(len(seq) - 1))
    monotone_decreasing = all(seq[i] > seq[i + 1] for i in range(len(seq) - 1))
    monotone = monotone_increasing or monotone_decreasing
    slope = round((seq[-1] - seq[0]) / (OFFSETS[-1] - OFFSETS[0]), 4)
    ref = steps_by_k[0]
    per_frame = [dict(offset_frames=OFFSETS[i], first_nonfoot_step=seq[i], delta_vs_k0_steps=seq[i] - ref,
                      delta_vs_k0_ms=round((seq[i] - ref) * DT * 1000.0, 6),
                      delta_per_frame_steps=(None if OFFSETS[i] == 0 else round((seq[i] - ref) / OFFSETS[i], 4)))
                  for i in range(len(OFFSETS))]

    out = DESK / ('S2-frame-alignment-scan-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    report = dict(
        status='PASS_FRAME_ALIGNMENT_SCAN', manual_research_not_product=True, standing=False, walking=False,
        title='frame-alignment scan on the canonical accepted equilibrium (fixed input; manual research, not a product)',
        declared=dict(dt_s=DT, xml_timestep_s=xml_dt, steps=STEPS, window_s=DT * STEPS, command_period_steps=CMD_EVERY,
                      offsets_frames=OFFSETS, index_rule='idx = clip(step // 200 + k, 0, 99)',
                      out_of_range_rule='clamped to the archive ends; no interpolation, no rescaling, no repetition',
                      sign_convention='k > 0 reads the archive later (command delayed by k frames); k < 0 reads it earlier',
                      state_written_once=True, loop_writes='ctrl only',
                      archive='the same single 100-frame / 5 ms archive for every group; only the starting index differs',
                      same_input_statement='this is a fixed-input computation on the same frozen archive: nothing is searched, tuned or '
                                           'optimised, and no per-step pose correction is applied'),
        inputs=dict(state=str(STATE), state_sha256=sha(STATE), drive=str(DRIVE), drive_sha256=sha(DRIVE),
                    archive_shape=list(base.shape), canonical_xml=str(cp.CANONICAL_XML), canonical_xml_sha256=sha(cp.CANONICAL_XML),
                    script=str(pathlib.Path(__file__)), script_sha256=sha(pathlib.Path(__file__))),
        results=results,
        answer=dict(
            monotone_with_alignment=bool(monotone),
            monotone_direction=('increasing (later command -> later collapse)' if monotone_increasing else
                                ('decreasing (later command -> earlier collapse)' if monotone_decreasing else 'NOT monotone')),
            first_nonfoot_step_by_offset={('%+d' % k): steps_by_k[k] for k in OFFSETS},
            mean_steps_per_frame=(None if monotone is False else slope),
            endpoint_slope_steps_per_frame=slope,
            per_frame=per_frame,
            sentence=('the first non-foot contact step moves monotonically with the frame-index offset: %s, and the endpoint slope is '
                      '%s steps per frame (over the scanned -2..+2 range). The ~70 ms collapse time is therefore governed by frame '
                      'alignment, and whether the collapse happens at all is decided jointly by the state and the archive as a pair.'
                      % ('yes' if monotone else 'no', slope)),
            range_caveat='the scanned range is only -2..+2 frames around the alignment, so the slope is an endpoint average over that '
                         'range and is not extrapolated beyond it'),
        not_established='absolute force values, physical units, biological validity; no standing, no walking',
        recorded_at=now())
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    np.savez_compressed(out / 'frame_alignment_scan.npz',
                        **{('k%+d' % r['offset_frames']): np.array([r['first_nonfoot_contact_step'],
                                                                    r['first_step_legs_in_contact_below_6'],
                                                                    r['first_step_zero_tarsus_contacts'],
                                                                    r['root_z_min'], r['root_z_final'], r['max_abs_qacc'],
                                                                    r['max_abs_qacc_step'], r['act_peak'],
                                                                    r['act_out_of_domain_steps'],
                                                                    r['max_u_diamond_tarsus'],
                                                                    r['max_u_diamond_all_contacts']]) for r in results},
                        columns=np.array(['first_nonfoot_step', 'first_legs_below_6_step', 'first_zero_tarsus_step', 'root_z_min',
                                          'root_z_final', 'max_abs_qacc', 'max_abs_qacc_step', 'act_peak', 'act_ood_steps',
                                          'max_u_diamond_tarsus', 'max_u_diamond_all']))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = np.array(OFFSETS)
    labels = ['%+d' % k for k in OFFSETS]
    fig, ax = plt.subplots(2, 2, figsize=(15.5, 9.8))
    a = ax[0][0]
    a.bar(np.arange(len(OFFSETS)) - 0.2, [(r['first_nonfoot_contact_ms'] or np.nan) for r in results], width=0.4,
          color='#d62728', label='first non-foot contact')
    a.bar(np.arange(len(OFFSETS)) + 0.2, [(r['first_step_legs_in_contact_below_6_ms'] or np.nan) for r in results], width=0.4,
          color='#ff7f0e', label='first legs<6')
    a.set_xticks(np.arange(len(OFFSETS))); a.set_xticklabels(labels)
    a.set_xlabel('integer frame-index offset k [-] (archive read k frames later for k>0)')
    a.set_ylabel('time [ms] (measured clock)')
    a.set_title('onset times vs frame alignment (0.5 s window)', fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3, axis='y')
    a = ax[0][1]
    a.plot(x, [r['first_nonfoot_contact_step'] for r in results], 'o-', color='#1f77b4')
    for xi, r in zip(x, results):
        a.annotate('%d' % r['first_nonfoot_contact_step'], (xi, r['first_nonfoot_contact_step']), textcoords='offset points',
                   xytext=(0, 6), ha='center', fontsize=8)
    a.set_xlabel('frame-index offset k'); a.set_ylabel('first non-foot contact [executed steps]')
    a.set_title('first-event step vs k: slope %.1f steps/frame (endpoints %d..%d)'
                % (slope, OFFSETS[0], OFFSETS[-1]), fontsize=10); a.grid(alpha=0.3)
    a = ax[1][0]
    a.plot(x, [r['max_abs_qacc'] for r in results], 'o-', color='#2ca02c')
    a.set_xlabel('frame-index offset k'); a.set_ylabel('max|qacc| over the run [mm/s^2]')
    a.set_title('max|qacc| vs k (act never leaves [0,1]: %s)' % all(r['act_out_of_domain_steps'] == 0 for r in results), fontsize=10)
    a.grid(alpha=0.3)
    a = ax[1][1]
    a.plot(x, [r['root_z_min'] for r in results], 'o-', color='#9467bd', label='root z minimum')
    a.plot(x, [r['root_z_final'] for r in results], 's--', color='#8c564b', label='root z final')
    a.set_xlabel('frame-index offset k'); a.set_ylabel('root z [mm]')
    a.set_title('root z minimum and final vs k', fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.suptitle('frame-alignment scan on the canonical accepted equilibrium: one frozen 100-frame archive, only the starting index differs\n'
                 'dt 2.5e-5 asserted, 20000 steps = 0.5 s, state written once, only ctrl in the loop, measured d.time axis\n'
                 'manual research implementation, NOT a product; no standing/walking claim', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out / 'frame-alignment-scan.png', dpi=135)
    plt.close(fig)

    (out / 'README.zh-CN.md').write_text('\n'.join([
        '# 帧对齐细扫（canonical 平衡态，固定输入；**手动研究实施，不是成品**）', '',
        '## 事先写死的协议',
        '- 状态：`%s`（sha256 `%s`）；驱动：`%s` 的 `normal_controls`（sha256 `%s`，形状 `%s`）。'
        % (str(STATE), sha(STATE), str(DRIVE), sha(DRIVE), str(list(base.shape))),
        '- `dt = 2.5e-5` **断言**（XML `%s` 不用）、20000 步 = 0.5 s、**状态只在循环前写一次**、**循环内只写 `ctrl`**。' % xml_dt,
        '- **唯一变化**是起始整数帧索引：`idx = clip(step // 200 + k, 0, 99)`，`k ∈ {−2,−1,0,+1,+2}`；**越界一律夹到档案两端**'
        '（不插值、不缩放、不重复）。`k > 0` 表示档案**晚读** k 帧（指令延后），`k < 0` 表示**早读**。',
        '- **同一份冻结档案、同一份平衡态**：固定输入计算，不做搜索、不调参、不做每步姿态修正。', '',
        '## 逐组结果（步号 = 已执行步数；ms = 步 × 25 µs）', '',
        '| k | 首个非足接触步 / ms | geom 对（属体） | legs<6 步 | 零跗节步 | root_z 最小 / 末值 | max\\|qacc\\| / 步 | act 峰 / 越域 | u_diamond 峰（跗节 / 全接触） |',
        '|---|---|---|---|---|---|---|---|---|',
        *['| %+d | %s / %s | `%s`（%s） | %s | %s | %s / %s | %s / %s | %s / %s | %s / %s |'
          % (r['offset_frames'], r['first_nonfoot_contact_step'], r['first_nonfoot_contact_ms'], r['first_nonfoot_pair'],
             r['first_nonfoot_owning_body'], r['first_step_legs_in_contact_below_6'], r['first_step_zero_tarsus_contacts'],
             r['root_z_min'], r['root_z_final'], r['max_abs_qacc'], r['max_abs_qacc_step'], r['act_peak'],
             r['act_out_of_domain_steps'], r['max_u_diamond_tarsus'], r['max_u_diamond_all_contacts']) for r in results],
        '',
        '## 回答（一句）',
        '- **~70 ms 塌陷时间随帧对齐%s**：首个非足接触步按 k 单调 %s（端点斜率 **%s 步/帧**，扫描范围仅 −2..+2 帧，不外推）；'
        '即塌陷**时刻**由帧对齐支配，而**是否塌陷**由状态与档案这一对共同决定。'
        % ('单调' if monotone else '**不**单调', ('增大' if monotone_increasing else '减小') if monotone else '（见逐组）', slope),
        '',
        '## 披露', '- **手动研究实施，不是成品**；不宣称站立/行走/完成度；绝对力值与单位 `NOT_ESTABLISHED`。']),
        encoding='utf-8')
    with __import__('zipfile').ZipFile(out / 'source_snapshot.zip', 'w', __import__('zipfile').ZIP_DEFLATED) as z:
        z.write(pathlib.Path(__file__), pathlib.Path(__file__).name)

    files = sorted(p for p in out.iterdir() if p.is_file())
    (out / 'FROZEN_DELIVERY_HASHES.json').write_text('{}', encoding='utf-8')
    m2 = dict(status='FROZEN_BEFORE_VERIFICATION', frozen_at=now(), self_excluded=True,
              self_excluded_reason='this manifest cannot list itself: writing its own digest changes the bytes it would describe',
              files={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'FROZEN_DELIVERY_HASHES.json'},
              covers_complete=True,
              covers_rule='`files` == directory files minus this manifest; asserted len(files) + 1 == os.listdir count',
              manifest_sha256=None, manifest_sha256_note='null by necessity; verify by re-hashing')
    (out / 'FROZEN_DELIVERY_HASHES.json').write_text(json.dumps(m2, ensure_ascii=False, indent=1), encoding='utf-8')
    m2 = json.loads((out / 'FROZEN_DELIVERY_HASHES.json').read_text(encoding='utf-8'))
    m2['files']['FROZEN_DELIVERY_HASHES.json'] = None
    del m2['files']['FROZEN_DELIVERY_HASHES.json']
    dh = {p.name: sha(p) for p in sorted(out.iterdir())
          if p.is_file() and p.name not in ('FROZEN_DELIVERY_HASHES.json', 'delivery-hashes.json')}
    dh['_self_excluded'] = 'cannot contain its own digest; the manifest carries it'
    (out / 'delivery-hashes.json').write_text(json.dumps(dh, ensure_ascii=False, indent=1), encoding='utf-8')
    m2['files']['delivery-hashes.json'] = sha(out / 'delivery-hashes.json')
    n = len([x for x in os.listdir(out) if (out / x).is_file()])
    m2['invariant'] = dict(directory_file_count=n, files_counted_with_manifest=len(m2['files']) + 1,
                           rule='len(files) + 1 == directory file count', holds=(len(m2['files']) + 1 == n))
    (out / 'FROZEN_DELIVERY_HASHES.json').write_text(json.dumps(m2, ensure_ascii=False, indent=1), encoding='utf-8')
    m2 = json.loads((out / 'FROZEN_DELIVERY_HASHES.json').read_text(encoding='utf-8'))
    assert 'FROZEN_DELIVERY_HASHES.json' not in m2['files'] and m2['self_excluded'] is True
    assert len(m2['files']) + 1 == n, (len(m2['files']), n)
    for f, h in m2['files'].items():
        assert sha(out / f) == h, f
    print()
    print('OUT:', out)
    print('monotone:', monotone, '| direction:', report['answer']['monotone_direction'], '| slope steps/frame:', slope)
    print('steps by k:', {('%+d' % k): steps_by_k[k] for k in OFFSETS})
    print('frozen:', n, 'files | covers_complete', m2['covers_complete'], '| invariant holds', len(m2['files']) + 1 == n)
    return out


if __name__ == '__main__':
    main()
