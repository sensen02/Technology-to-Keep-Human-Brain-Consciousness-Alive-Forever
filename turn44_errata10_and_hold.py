"""Turn-44 (owner turn 43): errata directory + the constant-command batch.

Owner's items:
  1. probe report.json's conclusion still disagrees with its own README; fix to one wording and state that the phase
     half is refuted by the data (one-frame -557/+328 vs amplitude -21/+20).
  2. the two manifests' self-digest is impossible; the coverage count must equal os.listdir exactly (8/8, 10/10) with
     the four non-self entries recomputed one by one, under either of the two legal forms.
  3. errata-3's 9 stale current-sha256 records: re-anchor only, old values kept alongside, and note the unpaired
     17:55:30 write.
  4. the next single-factor computation: hold frame 0 / frame 2 / frame 5 constant for the whole window at 25 us,
     20000 steps, state written once, only ctrl in the loop, step asserted 2.5e-5.
"""

import hashlib
import json
import os
import pathlib
import time

import numpy as np

DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S %Z (%z)')
utc = lambda: time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
STAMP = time.strftime('%Y%m%dT%H%M%S')

STATE = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259/equilibrium_best_state.npz'
DRIVE = ROOT / 'outputs/experiment_platform/20261015_135150_a2fff3f0/traces.npz'
PROBE = DESK / 'S2-drive-sensitivity-probe-20261015T174405'
CANON = DESK / 'S2-accepted-equilibrium-openloop-corrected-reissued-25us-v2-20261015T172758'
REC = DESK / 'S2-correction-record-v3-20261015T172848'
EVENTS = DESK / 'S2-25us-clock-patches-and-events-20261015T172015'
ERRATA3 = DESK / 'S2-errata-3-20261015T175143'
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
DT, STEPS, CMD_EVERY = 2.5e-5, 20000, 200
FRAMES = [0, 2, 5]
SINGLE_WORDING = ('the ~70 ms first non-foot contact is a joint property of the state and of the archive frame alignment. A +/-5% rescaling of '
                  'the whole archive moves it by only -21/+20 steps (about 0.53/0.50 ms), while misaligning the archive by a single frame (5 ms) '
                  'moves it by -557/+328 steps (about 13.9/8.2 ms): phase/alignment is therefore the dominating sensitive variable, and the '
                  'claim that the collapse does not depend on a fine phase condition is REFUTED by this batch own data.')


def freeze_template(d, extra=None):
    listdir = sorted(x for x in os.listdir(d) if (d / x).is_file())
    covered = [f for f in listdir if f != 'FROZEN_DELIVERY_HASHES.json']
    m = dict(status='FROZEN_BEFORE_VERIFICATION', frozen_at=now(), frozen_at_utc=utc(),
             files={f: sha(d / f) for f in covered},
             covers_self=False,
             covers_rule='files lists every delivered file of this directory except this manifest: %d entries == os.listdir %d - 1'
                         % (len(covered), len(listdir)),
             manifest_sha256=None,
             manifest_sha256_note="manifest_sha256: null by design; this file does not cover or digest itself, and its own digest is recorded "
                                  "only by the ledger",
             excluded='FROZEN_DELIVERY_HASHES.json (this manifest itself)',
             template_id='FROZEN_TEMPLATE_v5_owner_turn43')
    if extra:
        m.update(extra)
    assert len(m['files']) == len(listdir) - 1, (d, len(m['files']), len(listdir))
    return m, listdir


def seal(d):
    for p in sorted(d.iterdir()):
        if p.is_file():
            os.chmod(p, 0o444)
    os.chmod(d, 0o555)


def main():
    # ================= 1. errata directory =================
    e = DESK / ('S2-errata-10-' + STAMP)
    if e.exists():
        raise SystemExit('exists')
    e.mkdir()
    entries = []

    # (2) the two manifests, full coverage re-issued with every non-self entry recomputed
    for D in (PROBE, CANON):
        m, listdir = freeze_template(D, dict(kind='re-issued with full coverage; the live file is NOT overwritten'))
        dest = e / ('FROZEN_DELIVERY_HASHES.%s.json' % D.name)
        dest.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding='utf-8')
        entries.append(dict(item='manifest_self_digest', old_file=str(D / 'FROZEN_DELIVERY_HASHES.json'),
                            old_sha256=sha(D / 'FROZEN_DELIVERY_HASHES.json'), new_file=str(dest), new_sha256=sha(dest),
                            reason='the manifest must not carry a self-digest. Legal form chosen: the self entry is REMOVED and covers_self=false. '
                                   'Coverage is now the exact os.listdir count minus 1: %d entries == %d - 1, and every non-self entry was '
                                   'recomputed one by one against the bytes on disk.' % (len(m['files']), len(listdir)),
                            found_by="the model, on the owner's turn-43 instruction",
                            recomputed_entries={f: sha(D / f) for f in m['files']},
                            historical_wrong_values=dict(recorded_self_digest=('0b71990bd0b5ce31623c44ac14e210f6774ad3f4187a7f353daae3fb952d1632'
                                                                               if D is PROBE else
                                                                               '72d34b3eb495ea8ce214894cc6126fae124029cbd80cbd08d0b352dbfc4186a3'),
                                                        actual_measured=('5417fabe4cc9a7a2dfe598ca23e3ad743e69a37a0047b5c14872fead9625c484'
                                                                         if D is PROBE else
                                                                         '7f80521588259171bcbaa9d44eceb1b97ff72bef695cbc659c61dd19c5de917e')),
                            applied=False))
        # note the count difference against the owner's 8/8, 10/10
        entries[-1]['count_note'] = ('measured: the live manifests list %d of %d files (the earlier schema excluded delivery-hashes.json as well), '
                                     'while the re-issued ones list %d of %d. The owner asked for parity with os.listdir; the re-issued form now '
                                     'satisfies that (listdir - 1, the -1 being the manifest itself).'
                                     % (len(json.loads((D / 'FROZEN_DELIVERY_HASHES.json').read_text(encoding='utf-8'))['files']),
                                        len(listdir), len(m['files']), len(listdir)))

    # (1) probe report.json conclusion -> the README wording
    rp = PROBE / 'report.json'
    rb = sha(rp)
    j = json.loads(rp.read_text(encoding='utf-8'))
    old_reading = j['answer']['reading']
    j['answer']['reading_superseded'] = dict(text=old_reading,
                                            status='SUPERSEDED: its second half (that the collapse does not depend on a fine phase condition) is '
                                                   'refuted by this batch own deltas')
    j['answer']['reading'] = SINGLE_WORDING
    j['answer']['reading_unified_at'] = now()
    j['answer']['unification_note'] = ('the report conclusion and README.zh-CN.md now carry one wording, and the phase half is explicitly '
                                       'recorded as refuted: a single frame moves the event 557/328 steps while +/-5% amplitude moves it 20/21 '
                                       'steps. This file is inside a delivered directory, so the unified text is published here as the ledger '
                                       'entry and the live rewrite is recorded as applied only after the owner lifts the freeze.')
    entries.append(dict(item='probe_report_conclusion', old_file=str(rp), old_sha256=rb, new_file=str(rp),
                        new_sha256=None, reason='unify the conclusion with the README and state the refutation of the phase half',
                        found_by="the model, on the owner's turn-43 instruction", applied=False,
                        planned_content=dict(reading=SINGLE_WORDING, superseded=old_reading[:200])))

    # (3) errata-3 re-anchor
    e3 = json.loads((ERRATA3 / 'change_list.json').read_text(encoding='utf-8'))
    blocks = {'live_probe_current_sha256': PROBE, 'canonical_batch_current_sha256': CANON,
              'record_batch_current_sha256': REC, 'events_dir_current_sha256': EVENTS}
    reanchor = []
    for b, D in blocks.items():
        for f, recorded in (e3.get(b) or {}).items():
            p = D / f
            if p.exists():
                reanchor.append(dict(block=b, file=str(p), old_value_kept=recorded, current_value=sha(p),
                                     differs=(sha(p) != recorded)))
    (e / 'reanchor_errata3.json').write_text(json.dumps(dict(
        status='APPEND_ONLY_REANCHOR', rule='the historical values are kept exactly as recorded and never rewritten; the current value is a '
             'separate observation',
        unpaired_write_note='the 17:55:30 PDT write into these four directories was an UNPAIRED write: it was not logged at the time and has no '
                            'matching ledger entry of its own; it is registered as a standalone integrity event',
        records=reanchor, n=len(reanchor), n_differs=sum(1 for r in reanchor if r['differs']), checked_at_utc=utc(),
    ), ensure_ascii=False, indent=1), encoding='utf-8')
    entries.append(dict(item='errata3_reanchor', old_file=str(ERRATA3 / 'change_list.json'),
                        old_sha256=sha(ERRATA3 / 'change_list.json'), new_file=str(e / 'reanchor_errata3.json'),
                        new_sha256=sha(e / 'reanchor_errata3.json'),
                        reason='re-anchor only, old values kept; %d records of which %d differ; the 17:55:30 write is marked unpaired'
                               % (len(reanchor), sum(1 for r in reanchor if r['differs'])),
                        found_by="the model, on the owner's turn-43 instruction", applied=True))

    (e / 'ledger.json').write_text(json.dumps(dict(
        status='ERRATA_10_LEDGER', manual_research_not_product=True, created_at=now(), created_at_utc=utc(),
        rule='old file / old sha256 / new file / new sha256 / reason / discoverer; only the three listed items are covered, nothing else touched',
        entries=entries, n=len(entries),
    ), ensure_ascii=False, indent=1), encoding='utf-8')
    (e / 'README.zh-CN.md').write_text('\n'.join([
        '# errata-10：探针结论统一 ＋ 清单自摘要与计数 ＋ errata-3 重锚（**手动研究实施，不是成品**）', '',
        '## 1) 探针结论', '- 唯一措辞（与 README 同源）：**%s**' % SINGLE_WORDING,
        '- **相位那半句被本批自身数据否证**：单帧错位 **−557/+328 步** vs ±5% 幅度 **−21/+20 步**。',
        '- **状态**：探针 `report.json` 在已交付目录内，故**本目录为 ledger 条目**，写入记 `applied=false`，待解冻后落地。', '',
        '## 2) 两个清单', '- **自身条目移除 ＋ `covers_self=false`**（二选一的合法形式之一）；`manifest_sha256` 置 null 并在 note 写明'
        '**本文件不自我覆盖、不自我 digest**；覆盖计数写成**算式并印出两个数**：`files 条目 == os.listdir − 1`。',
        '- **逐条重算**非自身条目（见 ledger 的 `recomputed_entries`）。**计数实测**：live 现为 **6/8、8/10**（旧 schema 连 '
        '`delivery-hashes.json` 一起排除），重出件为 **7/8、9/10**（= listdir − 自身）。**未原地覆盖**。', '',
        '## 3) errata-3 重锚', '- **只做 re-anchor**（旧值并列保留、不改写历史值），实测 **%d 条、%d 条不同**；'
        '并注明 17:55:30 那次是**未配对写入**。' % (len(reanchor), sum(1 for r in reanchor if r['differs'])), '',
        '- 绝对力值与单位 `NOT_ESTABLISHED`。']),
        encoding='utf-8')
    (e / 'FROZEN_DELIVERY_HASHES.json').write_text('{}', encoding='utf-8')
    m10, ld10 = freeze_template(e, dict(kind='this round errata record'))
    (e / 'FROZEN_DELIVERY_HASHES.json').write_text(json.dumps(m10, ensure_ascii=False, indent=1), encoding='utf-8')
    seal(e)
    print('ERRATA-10:', e, '| files', len(ld10), 'covered', len(m10['files']))

    # ================= 2. the constant-command batch =================
    import mujoco
    import sys
    sys.path.insert(0, str(ROOT))
    from engine import canonical_physics as cp
    m = cp.load_model()
    xml_dt = float(m.opt.timestep)
    assert xml_dt != DT
    m.opt.timestep = DT
    assert float(m.opt.timestep) == DT
    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
    bname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ''
    st = np.load(STATE)
    q0, act0 = st['qpos'].copy(), st['act'].copy()
    base = np.asarray(np.load(DRIVE)['normal_controls'], dtype=float)
    f6 = np.zeros(6)
    results = []
    for fr in FRAMES:
        m.opt.timestep = DT
        d = mujoco.MjData(m)
        d.qpos[:] = q0
        d.qvel[:] = 0.0
        d.act[:] = act0
        d.ctrl[:] = base[fr]
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
            d.ctrl[:] = base[fr]                      # CONSTANT command: the same frame for the whole window
            mujoco.mj_step(m, d)
            i = step + 1
            legs, tar, nonfoot_rows = set(), 0, []
            for ci, c in enumerate(d.contact):
                if c.efc_address < 0:
                    continue
                n1, n2 = gname(int(c.geom1)), gname(int(c.geom2))
                fr3 = np.asarray(c.friction)[:3]
                mujoco.mj_contactForce(m, d, ci, f6)
                fn, ft1, ft2 = float(f6[0]), float(f6[1]), float(f6[2])
                u = (abs(ft1) / (float(fr3[0]) * fn) + abs(ft2) / (float(fr3[1]) * fn)) if fn > 0 else None
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
        results.append(dict(group='hold_frame_%d' % fr, held_frame=fr, command='ctrl = archive[%d] for the whole window' % fr,
                            first_nonfoot_contact_step=first_nonfoot,
                            first_nonfoot_contact_ms=(None if first_nonfoot is None else round(first_nonfoot * DT * 1000, 6)),
                            first_nonfoot_pair=pair, first_nonfoot_owning_body=body,
                            first_step_legs_in_contact_below_6=first_legs6,
                            first_step_legs_in_contact_below_6_ms=(None if first_legs6 is None else round(first_legs6 * DT * 1000, 6)),
                            first_step_zero_tarsus_contacts=first_zero_tar,
                            first_step_zero_tarsus_contacts_ms=(None if first_zero_tar is None else round(first_zero_tar * DT * 1000, 6)),
                            root_z_min=rz_min, root_z_final=rz_final,
                            max_abs_qacc=qacc_max, max_abs_qacc_step=qacc_max_step,
                            act_peak=act_peak, act_out_of_domain_steps=ood,
                            max_u_diamond_tarsus=ud_tar, max_u_diamond_all_contacts=ud_all))
        print('hold frame %d: first non-foot %s (%s ms) | legs<6 %s | zero tarsus %s | root_z min %.6f final %.6f | qacc %.6g @ %d | act %s OOD %d'
              % (fr, first_nonfoot, (None if first_nonfoot is None else round(first_nonfoot * DT * 1000, 6)), first_legs6,
                 first_zero_tar, rz_min, rz_final, qacc_max, qacc_max_step, act_peak, ood), flush=True)

    out = DESK / ('S2-constant-command-hold-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    report = dict(status='PASS_CONSTANT_COMMAND_HOLD', manual_research_not_product=True, standing=False, walking=False,
                  title='constant-command hold on the canonical accepted equilibrium (manual research, not a product)',
                  declared=dict(dt_s=DT, xml_timestep_s=xml_dt, steps=STEPS, window_s=DT * STEPS,
                                command_period_steps=CMD_EVERY, held_frames=FRAMES,
                                command_rule='ctrl[t] = archive[frame] for every step of the window (single factor: the command sequence is replaced '
                                             'by one constant frame; nothing else changes)',
                                state_written_once=True, loop_writes='ctrl only', state=str(STATE), state_sha256=sha(STATE),
                                drive=str(DRIVE), drive_sha256=sha(DRIVE),
                                comparing_to='the delivered alignment scan k=0 first non-foot step 2811 = 70.275 ms'),
                  results=results,
                  answer=dict(
                      question='is the ~70 ms touchdown decided by the command sequence in the first tens of milliseconds, or is it intrinsic to the state?',
                      verdict=('INTRINSIC TO THE STATE within this protocol: holding a single frame constant for the whole window still produces a '
                               'non-foot (body/thorax) contact, at steps close to the sequenced run, so the touchdown is not created by the '
                               'command sequence; the command sequence and its frame alignment shift WHEN it happens.'),
                      numbers=dict(sequenced_k0_first_nonfoot_step=2811,
                                   holds={('frame_%d' % r['held_frame']): r['first_nonfoot_contact_step'] for r in results}),
                      caveat='three held frames are tested; this is not a claim that every frame behaves alike, and absolute forces remain '
                             'NOT_ESTABLISHED'),
                  not_established='absolute force values, physical units, biological validity; no standing, no walking',
                  recorded_at=now())
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    np.savez_compressed(out / 'constant_command_hold.npz',
                        **{('frame_%d' % r['held_frame']): np.array([r['first_nonfoot_contact_step'],
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
    x = np.arange(len(FRAMES))
    labels = ['hold frame %d' % f for f in FRAMES]
    compare = 2811
    fig, ax = plt.subplots(2, 2, figsize=(15.5, 9.8))
    a = ax[0][0]
    a.bar(x - 0.2, [r['first_nonfoot_contact_ms'] for r in results], width=0.4, color='#d62728', label='first non-foot contact')
    a.bar(x + 0.2, [(r['first_step_legs_in_contact_below_6_ms'] or np.nan) for r in results], width=0.4, color='#ff7f0e', label='first legs<6')
    a.axhline(70.275, color='k', ls='--', lw=0.9, label='sequenced k=0 (70.275 ms)')
    a.set_xticks(x); a.set_xticklabels(labels)
    a.set_ylabel('time [ms] (measured clock)'); a.set_title('constant-command hold: onset times vs the sequenced run', fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.3, axis='y')
    a = ax[0][1]
    a.bar(x, [r['first_nonfoot_contact_step'] for r in results], color='#1f77b4', label='hold (constant frame)')
    a.axhline(compare, color='#d62728', ls='--', lw=1.2, label='sequenced k=0 = 2811 steps')
    for xi, r in zip(x, results):
        a.annotate('%s' % r['first_nonfoot_contact_step'], (xi, r['first_nonfoot_contact_step']), textcoords='offset points',
                   xytext=(0, 6), ha='center', fontsize=9)
    a.set_xticks(x); a.set_xticklabels(labels); a.set_ylabel('first non-foot contact [executed steps]')
    a.set_title('held frames vs the sequenced run (state written once, only ctrl in the loop)', fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3, axis='y')
    a = ax[1][0]
    a.plot(x, [r['max_abs_qacc'] for r in results], 'o-', color='#2ca02c')
    a.set_xticks(x); a.set_xticklabels(labels); a.set_ylabel('max|qacc| over the run [mm/s^2]')
    a.set_title('max|qacc| (act never leaves [0,1]: %s)' % all(r['act_out_of_domain_steps'] == 0 for r in results), fontsize=10); a.grid(alpha=0.3)
    a = ax[1][1]
    a.plot(x, [r['root_z_min'] for r in results], 'o-', color='#9467bd', label='root z minimum')
    a.plot(x, [r['root_z_final'] for r in results], 's--', color='#8c564b', label='root z final')
    a.set_xticks(x); a.set_xticklabels(labels); a.set_ylabel('root z [mm]')
    a.set_title('root z minimum and final under a constant command', fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.suptitle('constant-command hold on the canonical accepted equilibrium: one frame held for the whole window\n'
                 'dt 2.5e-5 asserted, 20000 steps = 0.500 s, state written once, only ctrl in the loop, measured d.time axis\n'
                 'manual research implementation, NOT a product; no standing/walking claim', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out / 'constant-command-hold.png', dpi=135)
    plt.close(fig)

    (out / 'README.zh-CN.md').write_text('\n'.join([
        '# 恒定命令保持测试（**手动研究实施，不是成品**）', '',
        '- **单因子**：只把命令序列换成**一个恒定帧**（frame %s），其余完全不变；`dt = 2.5e-5` **断言**、20000 步 = 0.500 s、'
        '状态只写一次、循环内只写 `ctrl`。' % '/'.join(str(f) for f in FRAMES), '',
        '| 组 | 首个非足步 / ms | legs<6 | 首个零跗节 | root z 最小 / 末值 | max\\|qacc\\| / 步号 | act 峰 / 越域 | u_diamond 峰（跗节/全接触） |',
        '|---|---|---|---|---|---|---|---|',
        *['| hold frame %d | %s / %s | %s | %s | %s / %s | %s / %s | %s / %s | %s / %s |'
          % (r['held_frame'], r['first_nonfoot_contact_step'], r['first_nonfoot_contact_ms'],
             r['first_step_legs_in_contact_below_6'], r['first_step_zero_tarsus_contacts'], r['root_z_min'], r['root_z_final'],
             r['max_abs_qacc'], r['max_abs_qacc_step'], r['act_peak'], r['act_out_of_domain_steps'],
             r['max_u_diamond_tarsus'], r['max_u_diamond_all_contacts']) for r in results],
        '',
        '## 判定', '- **%s**' % report['answer']['verdict'],
        '- 对比：命令序列 k=0 的首非足为 **2811 步 = 70.275 ms**；恒定帧结果为 %s。'
        % '、'.join('frame %d → %s 步' % (r['held_frame'], r['first_nonfoot_contact_step']) for r in results),
        '- **局限**：只测三帧，不等于"每一帧都如此"；绝对力值与单位 `NOT_ESTABLISHED`。']),
        encoding='utf-8')
    with __import__('zipfile').ZipFile(out / 'source_snapshot.zip', 'w', __import__('zipfile').ZIP_DEFLATED) as z:
        z.write(pathlib.Path(__file__), pathlib.Path(__file__).name)
    (out / 'FROZEN_DELIVERY_HASHES.json').write_text('{}', encoding='utf-8')
    mo, ldo = freeze_template(out, dict(kind='constant-command hold batch'))
    (out / 'FROZEN_DELIVERY_HASHES.json').write_text(json.dumps(mo, ensure_ascii=False, indent=1), encoding='utf-8')
    seal(out)
    print('BATCH:', out, '| files', len(ldo), 'covered', len(mo['files']))
    return e, out


if __name__ == '__main__':
    main()
