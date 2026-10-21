"""Turn-40 (owner turn 39): three settlement items + the phase-sensitivity curve (new computation).

Owner's items:
  1. the probe batch physics stands and the e736139b early alert IS its single independent verification: no full
     report, no second verifier, no re-run of any search. This file only records that, it does not re-verify.
  2. (a) the self-digest entries are removed and covers_complete is limited to non-self files (done; old wrong
     values kept as history only); (b) the probe report's blanket sentence is replaced by a TWO-PARAGRAPH wording
     (amplitude +/-5% -> -21/+20 steps, insensitive; one-frame misalignment -> -557/+328 steps, sensitive), the old
     sentence may appear only as a withdrawn quotation; (c) the errata-3 stale "current sha256" records become
     as-of anchored historical records: as-of time + value then + value now + why they differ, never rewriting the
     "now" field to masquerade as the then value.
  3. the 17:55:30 in-window write is registered once more as a discipline event with time, files, before/after
     hashes and the writer's own account; recorded in the ledger and in the tracker discipline section.
  4. the phase-sensitivity curve: built from the already-delivered scan data as a NEW timestamped directory.
"""

import hashlib
import json
import os
import pathlib
import shutil
import stat
import time

DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S %Z (%z)')
utc = lambda: time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
mt = lambda p: time.strftime('%Y-%m-%d %H:%M:%S %Z (%z)', time.localtime(pathlib.Path(p).stat().st_mtime))
STAMP = time.strftime('%Y%m%dT%H%M%S')

PROBE = DESK / 'S2-drive-sensitivity-probe-20261015T174405'
CANON = DESK / 'S2-accepted-equilibrium-openloop-corrected-reissued-25us-v2-20261015T172758'
REC = DESK / 'S2-correction-record-v3-20261015T172848'
EVENTS = DESK / 'S2-25us-clock-patches-and-events-20261015T172015'
ERRATA3 = DESK / 'S2-errata-3-20261015T175143'
SCAN = DESK / 'S2-frame-alignment-scan-20261015T180345'
BLOCKS = {'live_probe_current_sha256': PROBE, 'canonical_batch_current_sha256': CANON,
          'record_batch_current_sha256': REC, 'events_dir_current_sha256': EVENTS}

PARA_1 = ('amplitude is NOT the governing variable: rescaling the whole archive by +/-5% moves the first non-foot contact by only -21/+20 '
          'steps (2790/2831 against the reference 2811).')
PARA_2 = ('command timing IS the governing variable: shifting the same archive by a single frame (5 ms) moves the first non-foot contact by '
          '-557/+328 steps (2254/3139). Frame alignment / phase therefore dominates, by about one order of magnitude.')
WITHDRAWN = ('the first non-foot contact happens in every group inside a narrow band around the reference step, so the near-term collapse is '
             'decided by the archive-plus-state pair rather than by a delicate amplitude or phase condition (WITHDRAWN: refuted by this batch '
             'own deltas - amplitude moves the event by 20-21 steps, a one-frame shift by 328-557 steps).')


def freeze_by_template(d, extra=None):
    listdir = sorted(x for x in os.listdir(d) if (d / x).is_file())
    covered = [f for f in listdir if f != 'FROZEN_DELIVERY_HASHES.json']
    m = dict(status='FROZEN_BEFORE_VERIFICATION', frozen_at=now(),
             files={f: sha(d / f) for f in covered},
             self_listed=False, self_excluded=True, self_sha256=None,
             manifest_sha256=None,
             manifest_sha256_reason=('a manifest cannot contain its own digest: writing the digest changes the bytes it would describe, and '
                                     'no other file may carry it either because a mutual digest pair has no fixed point'),
             covers_complete=True,
             covers_complete_meaning='`files` covers every delivered file except this manifest itself; no "recompute once to align" loop is allowed',
             assert_rule='len(files) == len([f for f in os.listdir() if isfile]) - 1',
             template_id='FROZEN_TEMPLATE_v2_owner_turn38')
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
    # ============ part 1: settlement notes in one new errata directory ============
    e = DESK / ('S2-errata-9-' + STAMP)
    if e.exists():
        raise SystemExit('exists')
    e.mkdir()
    changes = []
    def rec(f, b, a, reason, kind='text_only'):
        changes.append(dict(file=str(f), sha256_before=b, sha256_after=a, changed=(b != a),
                            reason=reason, changed_by="the model, on the owner's turn-39 instruction",
                            at=now(), change_kind=kind))

    # (2c) as-of anchored historical records: as-of time + value then + value now + why they differ
    e3 = json.loads((ERRATA3 / 'change_list.json').read_text(encoding='utf-8'))
    fz3 = json.loads((ERRATA3 / 'FROZEN_DELIVERY_HASHES.json').read_text(encoding='utf-8'))
    asof_records = []
    for b, D in BLOCKS.items():
        for f, then_value in (e3.get(b) or {}).items():
            p = D / f
            if not p.exists():
                continue
            asof_records.append(dict(
                record_id='%s/%s' % (b, f), file=str(p),
                as_of=dict(when='2026-10-15 17:51:43 PDT (-0700) (the moment errata-3 was recorded)',
                           which_revision='the revision that existed when errata-3 wrote this record',
                           value_then=then_value),
                value_now=dict(when=now(), value=sha(p), mtime=mt(p)),
                differs=(sha(p) != then_value),
                why=('the file was rewritten by this work line inside an open verification window at 17:55:30 PDT (continuing 17:54:07 and '
                     '17:58:17), so the errata-3 record describes a revision that no longer exists on disk. The record is anchored to the time '
                     'it was made; the `value_now` field is a separate observation and was NOT written back into the historical field.'),
                action='APPEND-ONLY: the historical value is preserved exactly as recorded; no field is restated to make old and current agree',
            ))
    # the freeze manifest self-coverage record, handled by rule 1
    asof_records.append(dict(
        record_id='freeze_manifest_self_coverage',
        file=str(PROBE / 'FROZEN_DELIVERY_HASHES.json'),
        as_of=dict(when=now(), which_revision='the revision generated at 17:55:30 under the old schema',
                   value_then='the old revision listed the manifest itself in `files` with a self-digest that cannot be valid'),
        value_now=dict(when=now(), value=sha(PROBE / 'FROZEN_DELIVERY_HASHES.json'), mtime=mt(PROBE / 'FROZEN_DELIVERY_HASHES.json')),
        differs=True,
        why=('a manifest cannot contain its own digest (writing the digest changes the bytes), and the same file also carried '
             'manifest_sha256: null, which contradicted the self entry. The entries were removed and covers_complete was redefined over the '
             'non-self files only; the wrong values are kept as history, and no second self-hash is ever computed to "align" it.'),
        action='APPEND-ONLY'))
    (e / 'asof_anchored_records.json').write_text(json.dumps(dict(
        status='AS_OF_ANCHORED_HISTORICAL_RECORDS', rule='as-of time + value then + value now + why they differ; historical fields never rewritten',
        records=asof_records, n=len(asof_records), checked_at_utc=utc(),
        measured_totals=dict(records=len(asof_records), differing=sum(1 for r in asof_records if r['differs']))),
        ensure_ascii=False, indent=1), encoding='utf-8')
    rec(e / 'asof_anchored_records.json', None, sha(e / 'asof_anchored_records.json'),
        'errata-3 records re-expressed as as-of anchored history (value then / value now / why), append-only', kind='new_file')

    # (2b) two-paragraph conclusion, with the old sentence only as a withdrawn quotation
    (e / 'conclusion_two_paragraphs.json').write_text(json.dumps(dict(
        status='CONCLUSION_TWO_PARAGRAPHS', at=now(),
        applies_to=[str(PROBE), str(CANON)],
        paragraph_1_amplitude=PARA_1, paragraph_2_timing=PARA_2,
        measured=dict(amplitude_steps=[-21, 20], amplitude_ms=[-0.525, 0.525], phase_steps=[-557, 328], phase_ms=[-13.925, 8.2],
                      reference_step=2811, reference_ms=70.275),
        withdrawn_old_sentence=dict(text=WITHDRAWN, status='WITHDRAWN - quoted only as superseded text, must not appear as a current conclusion',
                                    old_file_sha256='6aef8a92b22ef9046f179c6ea83c690ac07d6b974d0e52fdef8f0526226b2bb2',
                                    note='that sha256 is the probe report.json revision that carried the sentence; the current revision is %s'
                                         % sha(PROBE / 'report.json')),
        ledger=dict(old_file=str(PROBE / 'report.json'), old_sha256='6aef8a92b22ef9046f179c6ea83c690ac07d6b974d0e52fdef8f0526226b2bb2',
                    new_file=str(e / 'conclusion_two_paragraphs.json'), new_sha256=None,
                    reason='the blanket sentence contradicted the batch README and its own deltas; the current conclusion is stated in two '
                           'paragraphs, one per variable'),
    ), ensure_ascii=False, indent=1), encoding='utf-8')
    j = json.loads((e / 'conclusion_two_paragraphs.json').read_text(encoding='utf-8'))
    j['ledger']['new_sha256'] = sha(e / 'conclusion_two_paragraphs.json')
    (e / 'conclusion_two_paragraphs.json').write_text(json.dumps(j, ensure_ascii=False, indent=1), encoding='utf-8')
    rec(e / 'conclusion_two_paragraphs.json', None, sha(e / 'conclusion_two_paragraphs.json'),
        'two-paragraph conclusion published; the old blanket sentence is carried only as a withdrawn quotation', kind='new_file')

    # (1) the verification settlement
    (e / 'verification_settlement.json').write_text(json.dumps(dict(
        status='SETTLEMENT_OF_THE_PROBE_BATCH_VERIFICATION', at=now(),
        batch=str(PROBE),
        settled_verification=dict(agent='e736139b-f5a3-49b1-93ce-ddabb6c7beb6', artifact='early alert, then stopped',
                                  role='the single independent verification of this batch; no full report is produced, no second verifier is '
                                       'dispatched, and no search is re-run'),
        physics_stands=dict(a2_five_groups_first_nonfoot_steps=[2790, 2811, 2831, 2254, 3139],
                            legs_below_6=[544, 538, 532, 736, 304], zero_tarsus=[5075, 5082, 5072, 2515, 3119],
                            root_z_min_final_and_max_qacc='reproduced bit for bit including the max|qacc| steps',
                            a3='the reference-group qacc trace is bit-identical to the canonical time_series_25us.npz'),
        materials=[dict(id='V1', what='the probe report.json carried a blanket conclusion that contradicted the batch README and its own '
                                      'deltas; corrected, old text retained only as a withdrawn quotation'),
                   dict(id='V2', what='the verification window overlapped writes by this work line (17:55:30)'),
                   dict(id='V3', what='two verifiers ran in parallel at one point; both were stopped')],
    ), ensure_ascii=False, indent=1), encoding='utf-8')
    rec(e / 'verification_settlement.json', None, sha(e / 'verification_settlement.json'),
        'records that the early alert is the batch single verification and that its physics stands, without re-verifying', kind='new_file')

    # (3) the discipline event, with the writer's own account
    (e / 'discipline_event_1755.json').write_text(json.dumps(dict(
        status='DISCIPLINE_EVENT', id='IN-WINDOW-WRITE-1755', at=now(),
        when='2026-10-15 17:55:30 PDT (-0700)', continued='17:54:07 and 17:58:17',
        files=[dict(file=str(D / f), before_sha256=h, after_sha256=sha(D / f), mtime=mt(D / f))
               for b, D in BLOCKS.items() for f, h in (e3.get(b) or {}).items() if (D / f).exists()],
        writer_account=("this work line (the model) wrote these files while a verification window was open. The account is: the writes were "
                        "made on the owner's instruction to complete the freeze and the errata round before dispatching the verifier, and the "
                        "author did not check whether a window was already open. No independent writer process is involved and no separate "
                        "writer directory can be proven from disk."),
        rule_now='any further write to a frozen directory requires a NEW directory; in-place patching is forbidden from this point',
        tracked_in='GOAL_TRACKER.zh-CN.md discipline section',
    ), ensure_ascii=False, indent=1), encoding='utf-8')
    rec(e / 'discipline_event_1755.json', None, sha(e / 'discipline_event_1755.json'),
        'the 17:55:30 in-window write registered with time, files, before/after hashes and the writer own account', kind='new_file')

    (e / 'change_list.json').write_text(json.dumps(dict(
        status='ERRATA_9_CHANGE_LIST', manual_research_not_product=True, created_at=now(),
        rule='old file / old sha256 / new file / new sha256 / reason / time; delivered directories are not written',
        entries=changes, n=len(changes),
        settled_verification=str(e / 'verification_settlement.json'),
        asof=str(e / 'asof_anchored_records.json'), conclusion=str(e / 'conclusion_two_paragraphs.json'),
        discipline_event=str(e / 'discipline_event_1755.json'),
    ), ensure_ascii=False, indent=1), encoding='utf-8')
    (e / 'README.zh-CN.md').write_text('\n'.join([
        '# errata-9：探针批核验结案 ＋ as-of 历史锚定 ＋ 两段式结论 ＋ 窗口内写入纪律事件（**手动研究实施，不是成品**）', '',
        '## 一、结案（不再补报告、不再开第二个核验、不重跑搜索）',
        '- **e736139b 的早期警报 = 探针批唯一一次独立核验结论**：A2 五组 `2790/2811/2831/2254/3139`、`legs<6` `544/538/532/736/304`、'
        '零跗节 `5075/5082/5072/2515/3119`、`root_z_min/final` 与 `max|qacc|`（含其步号）**逐位复现**；**A3** 参考组 `qacc` 轨迹与 '
        '`canonical time_series_25us.npz` **逐位相同**。材料三条（旧全称句、窗口重叠、并行核验）见 `verification_settlement.json`。', '',
        '## 二、三条处置',
        '1. **自摘要**：**已删除自身条目**，`covers_complete` **语义限定为非自身文件**（条目数 = 其余文件数），**不允许"再算一次去对齐"的循环**；'
        '旧错值**只作历史**（`0b71990b…` / `72d34b3e…`→实测 `5417fabe…` / `7f805215…`），**不覆盖**。',
        '2. **结论**：旧全称句（`6aef8a92…` 那版）**只以"已撤回旧文"形式引用**，不出现在任何现行结论里；现行结论为**两段**：'
        '① **幅度不敏感** —— 整档 ±5% 只挪 **−21/+20 步**（2790/2831 对 2811）；② **时序敏感** —— **单帧（5 ms）错位挪 −557/+328 步**'
        '（2254/3139）。见 `conclusion_two_paragraphs.json`（含 old file / old sha256 / new file / new sha256 / reason）。',
        '3. **errata-3 的 9 处 current sha256**：改为**按时刻锚定的历史记录** —— **as-of 时间 ＋ 当时哈希 ＋ 现值 ＋ 差异原因**；'
        '**绝不把"现值"字段改写成今天哈希冒充当时值**。实测共 **%d 条**（其中 **%d 条**确实不同），见 `asof_anchored_records.json`。'
        % (len(asof_records), sum(1 for r in asof_records if r['differs'])), '',
        '## 三、纪律事件',
        '- **17:55:30 在核验窗口内的写入**已登记：**时间、被改文件、前后哈希、写入者自述**（`discipline_event_1755.json`）。'
        '**冻结目录此后任何再写必须开新目录，不许原地补写**，并已写入 tracker 纪律段。', '',
        '- **手动研究实施，不是成品**；不宣称站立/行走/完成度；绝对力值与单位 `NOT_ESTABLISHED`。']),
        encoding='utf-8')

    (e / 'FROZEN_DELIVERY_HASHES.json').write_text('{}', encoding='utf-8')
    m, listdir = freeze_by_template(e, dict(kind='this round errata record'))
    (e / 'FROZEN_DELIVERY_HASHES.json').write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding='utf-8')
    seal(e)
    print('ERRATA-9:', e, '| files', len(listdir), '| covered', len(m['files']), '| rule', len(m['files']) == len(listdir) - 1)
    print('asof records:', len(asof_records), 'differing:', sum(1 for r in asof_records if r['differs']))

    # ============ part 2: the phase-sensitivity curve (new computation) ============
    rep = json.loads((SCAN / 'report.json').read_text(encoding='utf-8'))
    res = {r['offset_frames']: r for r in rep['results']}
    ks = sorted(res)
    steps = [res[k]['first_nonfoot_contact_step'] for k in ks]
    ms = [res[k]['first_nonfoot_contact_ms'] for k in ks]
    adj = [dict(from_k=ks[i], to_k=ks[i + 1], steps_per_frame=(steps[i + 1] - steps[i]) / (ks[i + 1] - ks[i]),
                ms_per_frame=(ms[i + 1] - ms[i]) / (ks[i + 1] - ks[i])) for i in range(len(ks) - 1)]
    slope = round((steps[-1] - steps[0]) / (ks[-1] - ks[0]), 4)
    out = DESK / ('S2-phase-sensitivity-curve-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.array(ks, dtype=float)
    fig, ax = plt.subplots(2, 2, figsize=(15.5, 9.8))
    a = ax[0][0]
    a.plot(x, steps, 'o-', color='#1f77b4')
    for xi, s in zip(x, steps):
        a.annotate('%d' % s, (xi, s), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=8)
    a.set_xlabel('integer frame-index offset k [-] (k>0 = archive read later)')
    a.set_ylabel('first non-foot contact [executed steps]')
    a.set_title('phase-sensitivity curve: step vs alignment (endpoint slope %.1f steps/frame, NOT monotone)' % slope, fontsize=10)
    a.grid(alpha=0.3)
    a = ax[0][1]
    a.plot(x, ms, 'o-', color='#d62728')
    for xi, v in zip(x, ms):
        a.annotate('%.3f' % v, (xi, v), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=8)
    a.set_xlabel('integer frame-index offset k [-]')
    a.set_ylabel('first non-foot contact [ms] (from measured d.time, |d.time - step*dt| = 0.0)')
    a.set_title('same curve in milliseconds; adjacent slopes %s ms/frame'
                % ' / '.join('%+.4f' % ad['ms_per_frame'] for ad in adj), fontsize=9)
    a.grid(alpha=0.3)
    a = ax[1][0]
    a.plot(x, [res[k]['max_abs_qacc'] for k in ks], 'o-', color='#2ca02c')
    a.set_xlabel('integer frame-index offset k [-]'); a.set_ylabel('max|qacc| over the run [mm/s^2]')
    a.set_title('max|qacc| vs alignment (act never leaves [0,1]: %s)' % all(res[k]['act_out_of_domain_steps'] == 0 for k in ks), fontsize=10)
    a.grid(alpha=0.3)
    a = ax[1][1]
    a.plot(x, [res[k]['root_z_min'] for k in ks], 'o-', color='#9467bd', label='root z minimum')
    a.plot(x, [res[k]['root_z_final'] for k in ks], 's--', color='#8c564b', label='root z final')
    a.set_xlabel('integer frame-index offset k [-]'); a.set_ylabel('root z [mm]')
    a.set_title('root z minimum and final vs alignment', fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.suptitle('phase-sensitivity curve from the frozen frame-alignment scan: one archive, only the starting frame index differs\n'
                 'dt 2.5e-5 asserted, 20000 steps = 0.500 s, state written once, only ctrl in the loop, measured d.time (|d.time - step*dt| = 0.0)\n'
                 'manual research implementation, NOT a product; no standing/walking claim', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out / 'phase-sensitivity-curve.png', dpi=135)
    plt.close(fig)

    (out / 'curve.json').write_text(json.dumps(dict(
        status='PASS_PHASE_SENSITIVITY_CURVE', manual_research_not_product=True,
        source_of_numbers=str(SCAN), source_note='this directory re-plots and re-tabulates the already-delivered scan; it runs no new physics',
        protocol_inherited=rep['declared'],
        curve=[dict(k=k, first_nonfoot_step=steps[i], first_nonfoot_ms=ms[i],
                    legs_below_6=res[k]['first_step_legs_in_contact_below_6'],
                    first_zero_tarsus=res[k]['first_step_zero_tarsus_contacts'],
                    root_z_min=res[k]['root_z_min'], root_z_final=res[k]['root_z_final'],
                    max_abs_qacc=res[k]['max_abs_qacc'], max_abs_qacc_step=res[k]['max_abs_qacc_step'],
                    act_peak=res[k]['act_peak'], act_out_of_domain_steps=res[k]['act_out_of_domain_steps'],
                    max_u_diamond_tarsus=res[k]['max_u_diamond_tarsus'],
                    max_u_diamond_all=res[k]['max_u_diamond_all_contacts'],
                    first_dt_error=res[k]['first_dt_error']) for i, k in enumerate(ks)],
        adjacent_slopes=adj, endpoint_slope_steps_per_frame=slope,
        units_check=dict(ms_source='measured d.time', first_dt_error_per_group=[res[k]['first_dt_error'] for k in ks],
                         hand_written_ms_constants=False),
        answer='the collapse moment is governed by command timing (frame alignment): a one-frame shift moves it by 328-557 steps while a '
               '+/-5% amplitude rescale moves it by 20-21 steps',
        recorded_at=now()), ensure_ascii=False, indent=1), encoding='utf-8')
    (out / 'README.zh-CN.md').write_text('\n'.join([
        '# 相位-敏感度曲线（**手动研究实施，不是成品**）', '',
        '- **来源**：**重绘/重表**自已交付的 `/home/sensen/Desktop/engine修复验收/S2-frame-alignment-scan-20261015T180345/`'
        '（`report.json` ＋ `frame_alignment_scan.npz`），**本目录不跑任何新物理**。',
        '- **协议（继承，未改）**：`dt = 2.5e-5` 断言、20000 步 = 0.500 s、状态只写一次、循环内只写 `ctrl`、'
        '`idx = clip(step//200 + k, 0, 99)`、越界夹边。',
        '- **毫秒口径**：由**实测 `d.time`** 推出（各组 `|d.time − step×dt| = 0.0`），**无手写毫秒常量**。', '',
        '| k | 首次非足步 / ms | legs<6 | 零跗节 | root_z min / final | max\\|qacc\\| / 步 | act 峰 / 越域 | u_diamond 峰（两口径） |',
        '|---|---|---|---|---|---|---|---|',
        *['| %+d | %d / %s | %s | %s | %s / %s | %s / %s | %s / %s | %s / %s |'
          % (k, res[k]['first_nonfoot_contact_step'], res[k]['first_nonfoot_contact_ms'],
             res[k]['first_step_legs_in_contact_below_6'], res[k]['first_step_zero_tarsus_contacts'],
             res[k]['root_z_min'], res[k]['root_z_final'], res[k]['max_abs_qacc'], res[k]['max_abs_qacc_step'],
             res[k]['act_peak'], res[k]['act_out_of_domain_steps'], res[k]['max_u_diamond_tarsus'],
             res[k]['max_u_diamond_all_contacts']) for k in ks],
        '',
        '## 相邻斜率', '- %s' % '；'.join('%+d→%+d: %+.1f 步/帧（%+.4f ms/帧）' % (a['from_k'], a['to_k'], a['steps_per_frame'],
                                                                                a['ms_per_frame']) for a in adj),
        '- 端点斜率 **%.1f 步/帧**（−2..+2 全范围均值）；**不单调**（+2 处中断）。' % slope, '',
        '- **手动研究实施，不是成品**；绝对力值与单位 `NOT_ESTABLISHED`。']),
        encoding='utf-8')
    with __import__('zipfile').ZipFile(out / 'source_snapshot.zip', 'w', __import__('zipfile').ZIP_DEFLATED) as z:
        z.write(pathlib.Path(__file__), pathlib.Path(__file__).name)
    (out / 'FROZEN_DELIVERY_HASHES.json').write_text('{}', encoding='utf-8')
    m2, ld2 = freeze_by_template(out, dict(kind='phase-sensitivity curve, no new physics'))
    (out / 'FROZEN_DELIVERY_HASHES.json').write_text(json.dumps(m2, ensure_ascii=False, indent=1), encoding='utf-8')
    seal(out)
    print('CURVE:', out, '| files', len(ld2), '| covered', len(m2['files']), '| rule', len(m2['files']) == len(ld2) - 1)
    return e, out


if __name__ == '__main__':
    main()
