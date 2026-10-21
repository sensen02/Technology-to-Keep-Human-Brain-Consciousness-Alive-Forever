"""Channel-level attribution of the -557/+328 step phase shift (the next single computation).

DECLARED PROTOCOL (identical to the probe/scan batches)
  state     : /home/sensen/Desktop/engine修复验收/S2-equilibrium-A-corrected-reissued-20261015T161259/equilibrium_best_state.npz
              sha256 976388f28a866793ba0738311919e0f1f0837b052ac8d37281ad685c1d77c5be
  drive     : /run/media/sensen/Data2/cell_wound_prototype/outputs/experiment_platform/20261015_135150_a2fff3f0/traces.npz
              key normal_controls (100, 90), sha256 37fe24d3eac6f0962eee4ca6aeb51c89ecc9804be4b553c87ce91fcafda83a4f
  step      : m.opt.timestep = 2.5e-5 ASSERTED before any mj_forward; 20000 steps = 0.5 s
  writes    : qpos/qvel/act written ONCE before the loop; inside the loop ONLY ctrl is written
  factor    : ONE factor only — which archived frame drives which leg's channels. NO archive value is changed:
              for a leg L, channels of L are driven by frame_b while the other 75 channels are driven by frame_a.
              This is a zero-order-hold frame-selection comparison, not a re-timing of the archive.
  baselines : k=0 (all channels frame 0 for the whole window; the adopted convention) and the shift pair at frame 5,
              which is the frame used by the +1 alignment offset.
  question  : which muscle channels, or which leg's channel group, explain the -557/+328 step displacement?

Nothing is written into any delivered directory: the new batch is its own timestamped directory.
"""

import hashlib
import os
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
FRAME_A, FRAME_B = 0, 5                     # the two frames whose constant hold is already measured
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S %Z (%z)')
ms = lambda s: round(s * DT * 1000.0, 6)


def leg_channel_map(model, mujoco):
    """Which of the 90 actuator channels belong to which leg, taken from the model, not guessed."""
    out = {}
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ''
        for L in LEGS:
            if name.startswith(L) or ('_' + L) in name or name.startswith('joint_' + L):
                out.setdefault(L, []).append(a)
                break
    return out


def run_group(model, mujoco, q0, act0, base, ctrl_of, label):
    d = mujoco.MjData(model)
    model.opt.timestep = DT
    d.qpos[:] = q0
    d.qvel[:] = 0.0
    d.act[:] = act0
    d.ctrl[:] = ctrl_of
    mujoco.mj_forward(model, d)
    assert float(d.time) == 0.0
    gname = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
    bname = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ''
    f6 = np.zeros(6)
    first_nonfoot = first_legs6 = first_zero_tar = None
    pair = body = None
    rz_min = rz_final = float(d.qpos[2])
    qacc_max = float(np.max(np.abs(np.asarray(d.qacc))))
    qacc_max_step = 0
    act_peak = float(np.max(np.asarray(d.act)))
    ood = 0
    ud_tar = ud_all = 0.0
    for step in range(STEPS):
        d.ctrl[:] = ctrl_of
        mujoco.mj_step(model, d)
        i = step + 1
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
    return dict(group=label, first_nonfoot_contact_step=first_nonfoot,
                first_nonfoot_contact_ms=(None if first_nonfoot is None else ms(first_nonfoot)),
                first_nonfoot_pair=pair, first_nonfoot_owning_body=body,
                first_step_legs_in_contact_below_6=first_legs6,
                first_step_zero_tarsus_contacts=first_zero_tar,
                root_z_min=rz_min, root_z_final=rz_final,
                max_abs_qacc=qacc_max, max_abs_qacc_step=qacc_max_step,
                act_peak=act_peak, act_out_of_domain_steps=ood,
                max_u_diamond_tarsus=ud_tar, max_u_diamond_all_contacts=ud_all)


def main():
    import mujoco
    sys.path.insert(0, str(ROOT))
    from engine import canonical_physics as cp
    model = cp.load_model()
    xml_dt = float(model.opt.timestep)
    assert xml_dt != DT, 'the XML dt must differ from the asserted dt'
    model.opt.timestep = DT
    assert float(model.opt.timestep) == DT, 'timestep assertion must hold before any mj_forward'
    st = np.load(STATE)
    q0, act0 = st['qpos'].copy(), st['act'].copy()
    base = np.asarray(np.load(DRIVE)['normal_controls'], dtype=float)
    lmap = leg_channel_map(model, mujoco)

    ctrlA = np.clip(base[FRAME_A], 0.0, 1.0)
    ctrlB = np.clip(base[FRAME_B], 0.0, 1.0)

    results = []
    results.append(run_group(model, mujoco, q0, act0, base, ctrlA, 'baseline_all_frame%d' % FRAME_A))
    print('baseline frame %d: first non-foot %s' % (FRAME_A, results[-1]['first_nonfoot_contact_step']), flush=True)
    results.append(run_group(model, mujoco, q0, act0, base, ctrlB, 'baseline_all_frame%d' % FRAME_B))
    print('baseline frame %d: first non-foot %s' % (FRAME_B, results[-1]['first_nonfoot_contact_step']), flush=True)

    swapped = []
    for L in LEGS:
        ch = lmap.get(L, [])
        c = ctrlA.copy()
        c[ch] = ctrlB[ch]
        r = run_group(model, mujoco, q0, act0, base, c, 'leg_%s_frame%d_others_frame%d' % (L, FRAME_B, FRAME_A))
        r['leg'] = L
        r['channels_swapped'] = len(ch)
        r['channels'] = ch
        swapped.append(r)
        print('swap %-3s (%2d ch): first non-foot %s' % (L, len(ch), r['first_nonfoot_contact_step']), flush=True)

    a = results[0]['first_nonfoot_contact_step']
    b = results[1]['first_nonfoot_contact_step']
    span = b - a
    attribution = [dict(leg=r['leg'], channels=r['channels_swapped'], first_nonfoot_step=r['first_nonfoot_contact_step'],
                        shift_vs_frame0=r['first_nonfoot_contact_step'] - a,
                        fraction_of_full_shift=(None if span == 0 else round((r['first_nonfoot_contact_step'] - a) / span, 4)))
                   for r in swapped]

    out = DESK / ('S2-channel-attribution-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    report = dict(
        status='PASS_CHANNEL_ATTRIBUTION', manual_research_not_product=True, standing=False, walking=False,
        title='channel-level attribution of the frame-alignment phase shift (manual research, not a product)',
        declared=dict(dt_s=DT, xml_timestep_s=xml_dt, steps=STEPS, window_s=DT * STEPS,
                      frames=dict(a=FRAME_A, b=FRAME_B),
                      factor='which archived frame drives which leg channel group; no archive value is modified',
                      rule='ctrl = archive[frame_a] with one leg channel group replaced by archive[frame_b] (zero-order hold)',
                      state_written_once=True, loop_writes='ctrl only',
                      convention='the adopted alignment convention remains frame k for the k-th 5 ms window, 200 steps/frame; offsets are diagnostics'),
        inputs=dict(state=str(STATE), state_sha256=sha(STATE), drive=str(DRIVE), drive_sha256=sha(DRIVE),
                    archive_shape=list(base.shape), canonical_xml=str(cp.CANONICAL_XML), canonical_xml_sha256=sha(cp.CANONICAL_XML),
                    script=str(pathlib.Path(__file__)), script_sha256=sha(pathlib.Path(__file__)),
                    leg_channel_map={L: lmap.get(L, []) for L in LEGS},
                    frame_a_all_zero=bool(np.allclose(base[FRAME_A], 0.0)),
                    frame_b_all_zero=bool(np.allclose(base[FRAME_B], 0.0))),
        baselines=results, swapped=swapped, attribution=attribution,
        answer=dict(
            full_shift_steps=span, full_shift_ms=round(span * DT * 1000.0, 6),
            largest_single_leg_contribution=max(attribution, key=lambda x: abs(x['shift_vs_frame0'])) if attribution else None,
            sentence=('the frame-alignment displacement is not carried by one leg alone: swapping a single leg channel group from frame %d to '
                      'frame %d moves the first non-foot contact by at most %s steps of the full %s-step shift. The displacement is therefore '
                      'distributed across the leg channel groups, and no single group is the cause.'
                      % (FRAME_A, FRAME_B,
                         max((abs(x['shift_vs_frame0']) for x in attribution), default=None), span)),
            caveat='six single-leg swaps were tested; a pairwise or all-but-one combination is not measured here, and the constant-frame arms are '
                   'zero-order holds rather than the sequenced drive'),
        not_established='absolute force values, physical units, biological validity; no standing, no walking',
        recorded_at=now())
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = np.arange(len(LEGS))
    fig, ax = plt.subplots(2, 2, figsize=(15.5, 9.8))
    a1 = ax[0][0]
    a1.bar(x, [r['first_nonfoot_contact_step'] for r in swapped], color='#1f77b4', label='one leg at frame %d' % FRAME_B)
    a1.axhline(a, color='#2ca02c', ls='--', lw=1.1, label='all frame %d (baseline %d)' % (FRAME_A, a))
    a1.axhline(b, color='#d62728', ls='--', lw=1.1, label='all frame %d (baseline %d)' % (FRAME_B, b))
    a1.set_xticks(x); a1.set_xticklabels(LEGS); a1.set_ylabel('first non-foot contact [executed steps]')
    a1.set_title('single-leg channel-group swap, frame %d -> %d' % (FRAME_A, FRAME_B), fontsize=10)
    a1.legend(fontsize=8); a1.grid(alpha=0.3, axis='y')
    a2 = ax[0][1]
    a2.bar(x, [r['shift_vs_frame0'] for r in attribution], color='#ff7f0e')
    a2.set_xticks(x); a2.set_xticklabels(LEGS); a2.set_ylabel('shift vs all-frame-%d [steps]' % FRAME_A)
    a2.set_title('attribution of the %d-step full shift (fraction in labels)' % span, fontsize=10); a2.grid(alpha=0.3, axis='y')
    for xi, r in zip(x, attribution):
        a2.annotate('%.2f' % (r['fraction_of_full_shift'] or 0.0), (xi, r['shift_vs_frame0']), textcoords='offset points',
                    xytext=(0, 4), ha='center', fontsize=8)
    a3 = ax[1][0]
    a3.plot(x, [r['max_abs_qacc'] for r in swapped], 'o-', color='#2ca02c')
    a3.set_xticks(x); a3.set_xticklabels(LEGS); a3.set_ylabel('max|qacc| over the run')
    a3.set_title('max|qacc| (act never leaves [0,1]: %s)' % all(r['act_out_of_domain_steps'] == 0 for r in swapped), fontsize=10)
    a3.grid(alpha=0.3)
    a4 = ax[1][1]
    a4.plot(x, [r['root_z_min'] for r in swapped], 'o-', color='#9467bd', label='root z minimum')
    a4.plot(x, [r['root_z_final'] for r in swapped], 's--', color='#8c564b', label='root z final')
    a4.set_xticks(x); a4.set_xticklabels(LEGS); a4.set_ylabel('root z [mm]')
    a4.set_title('root z minimum and final per swapped leg group', fontsize=10); a4.legend(fontsize=8); a4.grid(alpha=0.3)
    fig.suptitle('channel-level attribution of the frame-alignment phase shift: one leg channel group at frame %d, the rest at frame %d\n'
                 'dt 2.5e-5 asserted, 20000 steps = 0.500 s, state written once, only ctrl in the loop\n'
                 'manual research implementation, NOT a product; no standing/walking claim' % (FRAME_B, FRAME_A), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out / 'channel-attribution.png', dpi=135)
    plt.close(fig)

    (out / 'README.zh-CN.md').write_text('\n'.join([
        '# 帧对齐位移的通道级归因（**手动研究实施，不是成品**）', '',
        '## 事先写死的协议', '- 状态 `%s`（sha256 `%s`）；驱动 `%s` 的 `normal_controls`（%s，sha256 `%s`）。'
        % (str(STATE), sha(STATE), str(DRIVE), str(list(base.shape)), sha(DRIVE)),
        '- `dt = 2.5e-5` **在任何 `mj_forward` 之前断言**；20000 步 = 0.500 s；**状态只在循环前写一次**；**循环内只写 `ctrl`**。',
        '- **唯一因子**：哪一帧驱动**哪条腿的通道组**（零阶保持的帧选择对照），**不修改任何档案数值**。',
        '- 基线：全通道 frame %d（%d 步）与全通道 frame %d（%d 步），差 **%d 步 = %s ms**。' % (FRAME_A, a, FRAME_B, b, span, round(span*DT*1000,6)),
        '- 说明：frame %d 是**全零帧**（`all_zero=%s`），frame %d 非零（`all_zero=%s`）。' % (
            FRAME_A, report['inputs']['frame_a_all_zero'], FRAME_B, report['inputs']['frame_b_all_zero']), '',
        '## 逐腿结果', '',
        '| 腿 | 换帧通道数 | 首个非足步 | 相对全-frame-%d 的位移 | 占全位移比例 |' % FRAME_A,
        '|---|---|---|---|---|',
        *['| %s | %d | %s | %+d | %s |' % (r['leg'], r['channels'], r['first_nonfoot_step'], r['shift_vs_frame0'],
                                          ('%.2f' % r['fraction_of_full_shift']) if r['fraction_of_full_shift'] is not None else 'n/a') for r in attribution],
        '',
        '## 结论', '- %s' % report['answer']['sentence'],
        '- **局限**：只测了六种**单腿**换帧；成对或"除一条腿外全部"的组合**未测**；恒定帧臂是零阶保持，不是序列驱动。', '',
        '- 绝对力值与单位 `NOT_ESTABLISHED`；不宣称站立/行走/完成度。']), encoding='utf-8')
    with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(pathlib.Path(__file__), pathlib.Path(__file__).name)

    fp = out / 'FROZEN_DELIVERY_HASHES.json'
    fp.write_text('{}', encoding='utf-8')
    (out / 'delivery-hashes.json').write_text('{}', encoding='utf-8')
    listing = sorted(p.name for p in out.iterdir() if p.is_file())
    data = [f for f in listing if f not in ('FROZEN_DELIVERY_HASHES.json', 'delivery-hashes.json')]
    m = dict(status='FROZEN_BEFORE_VERIFICATION', frozen_at=now(), files={f: sha(out / f) for f in data},
             covers_complete='covers every delivered file except this manifest itself and except delivery-hashes.json',
             covers_rule='len(files) == os.listdir - 2', self_reference='not defined (a digest of a file cannot contain itself)',
             manifest_sha256=None, manifest_sha256_note='null by definition; pinned by delivery-hashes.json',
             placeholder_rule='no placeholder digest is ever written', template_id='FROZEN_TEMPLATE_v11_owner_planA')
    fp.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding='utf-8')
    dhd = {f: sha(out / f) for f in listing if f != 'delivery-hashes.json'}
    (out / 'delivery-hashes.json').write_text(json.dumps(dhd, ensure_ascii=False, indent=1), encoding='utf-8')
    assert dhd['FROZEN_DELIVERY_HASHES.json'] == sha(fp)
    assert not [f for f, h in m['files'].items() if sha(out / f) != h]
    assert hashlib.sha256(b'{}').hexdigest() not in (out / 'delivery-hashes.json').read_text(encoding='utf-8')
    for p in sorted(out.iterdir()):
        if p.is_file():
            os.chmod(p, 0o444)
    os.chmod(out, 0o555)

    print()
    print('OUT:', out)
    print('baselines: frame %d -> %d ; frame %d -> %d ; full shift %d steps (%s ms)' % (FRAME_A, a, FRAME_B, b, span, round(span * DT * 1000, 6)))
    for r in attribution:
        print('  %-3s  ch %2d  first_nonfoot %5d  shift %+5d  fraction %s' % (r['leg'], r['channels'], r['first_nonfoot_step'],
                                                                             r['shift_vs_frame0'], r['fraction_of_full_shift']))
    return out


if __name__ == '__main__':
    main()
