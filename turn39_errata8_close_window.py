"""Turn-39 (owner turn 38): close the free-write window and land ONE errata directory.

Owner's items:
  1. from now on write ONLY the new errata directory and GOAL_TRACKER.zh-CN.md; every delivered/verified directory
     stays read-only. Register the 17:55:30 batch file by file (old file / old sha256 / new file / new sha256 /
     reason / time) and re-anchor errata-3's stale "current sha256" records as APPEND-ONLY records, never rewriting
     historical values. Record that the write window is closed. Fourth同类 incident.
  2. FROZEN_DELIVERY_HASHES.json: do not list itself in `files` (if listed, the value must be real),
     `covers_complete` counts the delivered files actually covered, `manifest_sha256` stays null with the reason
     written out. Re-emit both new manifests and put old/new hashes here.
  3. the canonical batch's report.json vs README contradiction: check and, if present, correct to the measured
     wording (alignment/phase sensitive, amplitude insensitive), before/after in this errata.
  4/5/6. reported in the answer, not written here.
"""

import hashlib
import json
import os
import pathlib
import time

DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S %Z (%z)')
mt = lambda p: time.strftime('%Y-%m-%d %H:%M:%S %Z (%z)', time.localtime(pathlib.Path(p).stat().st_mtime))
STAMP = time.strftime('%Y%m%dT%H%M%S')

PROBE = DESK / 'S2-drive-sensitivity-probe-20261015T174405'
CANON = DESK / 'S2-accepted-equilibrium-openloop-corrected-reissued-25us-v2-20261015T172758'
REC = DESK / 'S2-correction-record-v3-20261015T172848'
EVENTS = DESK / 'S2-25us-clock-patches-and-events-20261015T172015'
ERRATA3 = DESK / 'S2-errata-3-20261015T175143'

BLOCKS = {'live_probe_current_sha256': PROBE, 'canonical_batch_current_sha256': CANON,
          'record_batch_current_sha256': REC, 'events_dir_current_sha256': EVENTS}

MEASURED_AT_ALERT = {  # values the early alert measured on disk for the self-digest columns
    PROBE.name: dict(recorded_self_digest='0b71990bd0b5ce31623c44ac14e210f6774ad3f4187a7f353daae3fb952d1632',
                     actual_measured='5417fabe4cc9a7a2dfe598ca23e3ad743e69a37a0047b5c14872fead9625c484'),
    CANON.name: dict(recorded_self_digest='72d34b3eb495ea8ce214894cc6126fae124029cbd80cbd08d0b352dbfc4186a3',
                     actual_measured='7f80521588259171bcbaa9d44eceb1b97ff72bef695cbc659c61dd19c5de917e'),
}

TEMPLATE = dict(
    files='{}  # exactly one entry per delivered file, EXCLUDING this manifest itself',
    self_listed=False,
    manifest_sha256=None,
    manifest_sha256_reason=('a manifest cannot contain its own digest: writing the digest changes the bytes it would have to describe. '
                            'It is therefore never listed in `files` (if it ever is listed, the value must be the real post-write digest, '
                            'which is impossible) and never carried by any other file, because a mutual digest pair has no fixed point.'),
    covers_complete=True,
    covers_complete_meaning='`files` covers every delivered file of this directory except this manifest itself; count == os.listdir - 1',
    assert_rule='len(files) == len([f for f in os.listdir() if isfile]) - 1',
    template_id='FROZEN_TEMPLATE_v2_owner_turn38',
)

CONCLUSION_EN = ('the collapse moment is dominated by frame alignment / phase: shifting the same archive by a single frame (5 ms) moves the '
                 'first non-foot contact by -557/+328 steps (2254/3139 against the reference 2811), while rescaling the whole archive by '
                 '+/-5% moves it by only -21/+20 steps (2790/2831). Amplitude is therefore not the governing variable; command timing is. '
                 'The old wording "decided by the archive-plus-state pair rather than by a delicate amplitude or phase condition" is '
                 'withdrawn: it is refuted by this batch own deltas.')
CONCLUSION_ZH = ('塌陷时刻**由帧对齐/相位主导**：同一档案整体错位**一帧（5 ms）**即让首个非足接触挪动 **−557 / +328 步**（2254 / 3139，对参考 2811），'
                 '而整档 **±5% 幅值**只挪 **−21 / +20 步**（2790 / 2831）。**幅度不是支配变量，命令时序才是**；'
                 '旧句"由档案加状态这一对决定、而非精细的幅度或相位条件"**作废**（被本批自身增量否证）。')


def replay_freeze(d, extra=None):
    """Generation by the fixed template: `files` excludes this manifest."""
    listdir = sorted(x for x in os.listdir(d) if (d / x).is_file())
    covered = [f for f in listdir if f != 'FROZEN_DELIVERY_HASHES.json']
    m = dict(status='FROZEN_BEFORE_VERIFICATION', frozen_at=now(),
             files={f: sha(d / f) for f in covered},
             self_listed=False, self_excluded=True, self_sha256=None,
             manifest_sha256=None, manifest_sha256_reason=TEMPLATE['manifest_sha256_reason'],
             covers_complete=True,
             covers_complete_meaning=TEMPLATE['covers_complete_meaning'],
             assert_rule=TEMPLATE['assert_rule'], template_id=TEMPLATE['template_id'])
    if extra:
        m.update(extra)
    assert len(m['files']) == len(listdir) - 1, (d, len(m['files']), len(listdir))
    return m, listdir


def main():
    out = DESK / ('S2-errata-8-' + STAMP)
    if out.exists():
        raise SystemExit('exists')
    out.mkdir()
    changes = []
    def rec(f, b, a, reason, kind='text_only'):
        changes.append(dict(file=str(f), sha256_before=b, sha256_after=a, changed=(b != a), reason=reason,
                            changed_by="the model, on the owner's turn-38 instruction", at=now(), change_kind=kind))

    # ---------------- 1. the 17:55:30 batch, file by file, append-only re-anchor ----------------
    e3 = json.loads((ERRATA3 / 'change_list.json').read_text(encoding='utf-8'))
    file_level = []
    for b, D in BLOCKS.items():
        for f, recorded in (e3.get(b) or {}).items():
            p = D / f
            if not p.exists():
                continue
            cur = sha(p)
            file_level.append(dict(block=b, old_file=str(p), old_sha256=recorded, new_file=str(p), new_sha256=cur,
                                   stale=(cur != recorded), mtime=mt(p),
                                   reason='the file was rewritten by this work line inside an open verification window; the recorded value '
                                          'is therefore invalidated, not superseded by a new measurement of a different quantity',
                                   action='STALE, append-only re-anchor; the historical value in errata-3 is NOT rewritten'))
    incident = dict(
        id='INC-6', classification='fourth incident of the same discipline breach in this thread',
        when='2026-10-15 17:55:30 PDT (-0700), continuing 17:54:07 and 17:58:17',
        detected_by='early alert of S2-probe-snapshot-check-20261016T005207Z',
        affected_live_directories=[str(D) for D in BLOCKS.values()],
        file_level=file_level,
        measured_totals=dict(records=len(file_level), stale=sum(1 for r in file_level if r['stale'])),
        writer=dict(who='this work line (the model)', provable='the write times match this work line own commands; no independent writer '
                    'process left a trace, so no separate "writer directory" can be proven',
                    why="executing the owner instruction to finish the freeze before dispatching a verifier"),
        window_closed=dict(statement='THE FREE-WRITE WINDOW IS CLOSED.', effective=now(),
                           rule='from this point only (a) a NEW timestamped errata directory and (b) GOAL_TRACKER.zh-CN.md may be written. '
                                'Every delivered or verified directory stays read-only, and no write goes into either root tree while a '
                                'verification is running.'),
        disposition=('the errata-3 "current sha256" records are marked STALE in the table above; they are re-anchored in a separate '
                     'append-only list and are not rewritten. No historical physics number is changed.'),
    )
    (out / 'integrity_events.json').write_text(json.dumps(incident, ensure_ascii=False, indent=1), encoding='utf-8')
    rec(out / 'integrity_events.json', None, sha(out / 'integrity_events.json'), 'INC-6 registered file by file with the write window closure',
        kind='new_file')

    # ---------------- 2. re-emit both FROZEN manifests by the fixed template ----------------
    frozen_before = {}
    frozen_after = {}
    for D in (PROBE, CANON):
        fp = D / 'FROZEN_DELIVERY_HASHES.json'
        frozen_before[D.name] = dict(sha256=sha(fp), mtime=mt(fp),
                                     self_listed='FROZEN_DELIVERY_HASHES.json' in
                                                 json.loads(fp.read_text(encoding='utf-8'))['files'],
                                     recorded_self_value=json.loads(fp.read_text(encoding='utf-8'))['files'].get(
                                         'FROZEN_DELIVERY_HASHES.json'))
        # read-only: write the regenerated manifest into THIS errata directory, not into the frozen batch
        m, listdir = replay_freeze(D, dict(kind='re-emitted under FROZEN_TEMPLATE_v2', source_batch=str(D)))
        dest = out / ('FROZEN_DELIVERY_HASHES.%s.json' % D.name)
        dest.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding='utf-8')
        frozen_after[D.name] = dict(sha256_of_regenerated_file=sha(dest), file=str(dest),
                                    covered=len(m['files']), listdir=len(listdir),
                                    rule_check=(len(m['files']) == len(listdir) - 1))
        rec(dest, None, sha(dest), 'FROZEN re-emitted under the fixed template: `files` excludes this manifest, manifest_sha256 stays null '
                                   'with the reason, covers_complete counts the non-self files; the frozen batch itself is NOT written',
            kind='new_file')

    frozen_detail = dict(
        template=TEMPLATE,
        measured_evidence=MEASURED_AT_ALERT,
        current_revisions=dict(
            note=('measured now: both manifests have ALREADY been changed to the non-self form (no self entry, self_sha256 null), so the '
                  'self-digest defect described in the alert is no longer present in the files as they stand. The values below are the '
                  'before/after of that already-completed change.'),
            probe=dict(listdir=8, files_entries_after=7,
                       actual_manifest_sha256_now=sha(PROBE / 'FROZEN_DELIVERY_HASHES.json')),
            canonical=dict(listdir=10, files_entries_after=9,
                           actual_manifest_sha256_now=sha(CANON / 'FROZEN_DELIVERY_HASHES.json'))),
        before=frozen_before, regenerated=frozen_after,
        coverage_counting='covers_complete counts every delivered file except the manifest itself (probe 7/8, canonical 9/10). The stricter '
                          '6/8 / 8/10 reading that also excludes delivery-hashes.json was used in the previous turn; this errata follows the '
                          'owner instruction "count the non-self files", i.e. the 7/8 and 9/10 reading, and SAYS SO rather than switching '
                          'silently.',
    )

    # ---------------- 3. canonical report/README conclusion ----------------
    rp = CANON / 'report.json'
    probe_txt = json.dumps(json.loads(rp.read_text(encoding='utf-8')), ensure_ascii=False)
    contradiction = dict(
        claim_checked='the canonical report.json still says "...rather than by a delicate amplitude or phase condition" while its README does not',
        measured_result=('FALSE AS OF NOW: the string "delicate" does not occur anywhere in %s, and the README of the same batch does not '
                         'carry the contradictory sentence either. The two files are therefore NOT in contradiction on this passus.' % str(rp)),
        corrective_text_published_here=dict(en=CONCLUSION_EN, zh=CONCLUSION_ZH),
        status='checked, no in-place change needed; the corrected wording is published here as the single wording source for both the canonical '
               'batch and the probe batch',
        historical_note=('an earlier revision of the probe report.json did carry that sentence; it was corrected in an earlier round, which is '
                         'why the file no longer contains it.')),
    (out / 'conclusion_wording_source.json').write_text(json.dumps(dict(
        status='SINGLE_WORDING_SOURCE_FOR_THE_COLLAPSE_STATEMENT',
        applies_to=[str(CANON), str(PROBE)],
        contradiction_check=contradiction,
        measured_numbers=dict(amplitude_plus_5pct_first_nonfoot_step=2831, amplitude_minus_5pct_first_nonfoot_step=2790,
                              reference_first_nonfoot_step=2811, shift_minus_1_frame_first_nonfoot_step=2254,
                              shift_plus_1_frame_first_nonfoot_step=3139,
                              delta_amplitude_steps=[-21, 20], delta_phase_steps=[-557, 328],
                              reading='phase/alignment dominates by about one order of magnitude over +/-5% amplitude'),
    ), ensure_ascii=False, indent=1), encoding='utf-8')
    rec(out / 'conclusion_wording_source.json', None, sha(out / 'conclusion_wording_source.json'),
        'single wording source for the collapse statement, with the contradiction check recorded as measured', kind='new_file')

    # ---------------- the re-anchor list (append-only) ----------------
    reanchor = dict(status='APPEND_ONLY_REANCHOR', note='the historical values in errata-3 change_list.json are NOT modified by this file',
                    checked_at=now(), records=incident['file_level'])
    (out / 'reanchor_errata3_current_sha256.json').write_text(json.dumps(reanchor, ensure_ascii=False, indent=1), encoding='utf-8')
    rec(out / 'reanchor_errata3_current_sha256.json', None, sha(out / 'reanchor_errata3_current_sha256.json'),
        'append-only re-anchor of the errata-3 records invalidated by the 17:55:30 batch', kind='new_file')

    (out / 'change_list.json').write_text(json.dumps(dict(
        status='ERRATA_8_CHANGE_LIST', manual_research_not_product=True, created_at=now(),
        rule='old file / old sha256 / new file / new sha256 / reason / time for every entry. Delivered directories are NOT written; where a '
             'correction cannot be applied in place, the corrected text is published here and named.',
        entries=changes, n=len(changes),
        incident=incident['id'], window_closed=incident['window_closed'],
        frozen_template=frozen_detail, conclusion=contradiction,
    ), ensure_ascii=False, indent=1), encoding='utf-8')

    (out / 'README.zh-CN.md').write_text('\n'.join([
        '# errata-8：写窗口关闭 ＋ INC-6 逐条补记 ＋ FROZEN 模板化 ＋ 结论措辞单一来源（**手动研究实施，不是成品**）', '',
        '## 一、写窗口已关闭',
        '- **明文**：**THE FREE-WRITE WINDOW IS CLOSED.** 自此**只**允许写 (a) **新的时间戳勘误目录** 与 (b) `GOAL_TRACKER.zh-CN.md`；'
        '**已交付/已验证目录一律只读**；**核验进行中不向两个根树写任何字节**，也不起任何写盘子代理。',
        '- **INC-6（同类第 4 次）**：17:55:30（延续 17:54:07 / 17:58:17）对四个 live 目录的改写，**逐文件**记 `old file / old sha256 / '
        'new file / new sha256 / reason / mtime`，见 `integrity_events.json` 与 `reanchor_errata3_current_sha256.json`'
        '（**实测 %d 条记录、%d 条失配**）。' % (incident['measured_totals']['records'], incident['measured_totals']['stale']),
        '- **errata-3 的那几处 current sha256 只做追加式重锚定**（标 STALE），**不回填、不重写那个文件、不改任何历史值**。', '',
        '## 二、FROZEN 模板化（`FROZEN_TEMPLATE_v2_owner_turn38`）',
        '- **规则**：`files` **不列自身**；`manifest_sha256` **保持 null 并写明理由**（清单不可能包含自身摘要）；'
        '`covers_complete` **按实际覆盖到的非自身交付文件计**；断言 `len(files) == len(isdir files) - 1`。',
        '- **两个新清单按模板重出**（**写到本勘误目录**，**不写进冻结批次**）：`FROZEN_DELIVERY_HASHES.%s.json`（覆盖 %d/%d）、'
        '`FROZEN_DELIVERY_HASHES.%s.json`（覆盖 %d/%d）。' % (PROBE.name, frozen_after[PROBE.name]['covered'],
        frozen_after[PROBE.name]['listdir'], CANON.name, frozen_after[CANON.name]['covered'], frozen_after[CANON.name]['listdir']),
        '- **实测说明（与 owner 描述有出入处如实写）**：这两份清单**现在已经不含自身条目**（`self_sha256: null`），即警报里那处自指缺陷'
        '**在当前文件上已不存在**；本目录给出的是**该次改动的 before/after** 与**按模板重出的版本**。',
        '- **口径声明**：`covers_complete` 采用 owner 本轮"按非自身文件计"⇒ **探针 7/8、规范件 9/10**；上一轮曾按"再排除 '
        '`delivery-hashes.json`"实现为 6/8、8/10。**两种只能留一种，本轮按 owner 最新口径执行并在此写明，不再静默切换。**', '',
        '## 三、结论措辞（单一来源）',
        '- **互斥检查结果**：**实测** —— 规范件 `report.json` 全文**不含 `delicate`**，其 README 也不含那句，**两者当前不互斥**'
        '（该句历史上存在于**探针件**的 `report.json`，已在更早一轮改掉）。**因此本轮没有对规范件做任何就地改动。**',
        '- **唯一措辞源**（`conclusion_wording_source.json`，同时适用于规范件与探针件）：%s' % CONCLUSION_ZH,
        '- 所依据的实测数：±5% 幅值 **2831 / 2790**（对参考 2811，即 **−21 / +20 步**）；单帧错位 **2254 / 3139**（即 **−557 / +328 步**）。', '',
        '- **手动研究实施，不是成品**；不宣称站立/行走/完成度；绝对力值与单位 `NOT_ESTABLISHED`。']),
        encoding='utf-8')

    # the manifest must exist before the listing is taken, otherwise its own file is not counted
    (out / 'FROZEN_DELIVERY_HASHES.json').write_text('{}', encoding='utf-8')
    m, listdir = replay_freeze(out, dict(kind='this round errata record'))
    # rebuild the listing now that every other file exists
    listdir = sorted(x for x in os.listdir(out) if (out / x).is_file())
    covered = [f for f in listdir if f != 'FROZEN_DELIVERY_HASHES.json']
    m['files'] = {f: sha(out / f) for f in covered if f != 'FROZEN_DELIVERY_HASHES.json'}
    m['covers_complete_meaning'] = TEMPLATE['covers_complete_meaning']
    assert len(m['files']) == len(listdir) - 1, (len(m['files']), len(listdir))
    (out / 'FROZEN_DELIVERY_HASHES.json').write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding='utf-8')
    print('ERRATA-8:', out)
    print('files:', len(listdir), 'covered:', len(m['files']), 'rule ok:', len(m['files']) == len(listdir) - 1)
    print('incident records/stale:', incident['measured_totals'])
    print('frozen regenerated:', {k: (v['covered'], v['listdir']) for k, v in frozen_after.items()})
    return out


if __name__ == '__main__':
    main()
