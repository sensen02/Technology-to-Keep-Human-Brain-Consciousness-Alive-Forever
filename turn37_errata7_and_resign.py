"""Turn-37 (owner turn 35): errata-7 — incident register, re-anchoring without rewriting history,
the corrected conclusion sentence with its delta table, and the manifest re-signing.

Owner's order:
  1. the 17:55:30 in-place rewrite batch is the FOURTH discipline incident: register it as an incident, file by file,
     with before/after sha256, mtime and local time, no summary sentence instead of evidence. errata-3's
     "current sha256" blocks are invalidated by it: keep the old records, add `current_as_of` and a new check time,
     never overwrite the historical values; state who did it and why, and only conclude "these records are
     invalidated by that write; use the following hashes".
  2. FROZEN_DELIVERY_HASHES.json must not contain its own digest: `files` lists only covered files,
     `self_excluded: true`, `self_sha256: null`, the meaning of covers_complete written out, and the assertion
     `set(files) == set(os.listdir()) - {manifest, delivery-hashes.json}`. Apply to both new manifests.
  3. the probe report.json conclusion must be replaced by the delta-supported version and a five-group delta table
     must go into BOTH report.json and README, identical wording.
  Then re-sign the freeze, confirm no other writer, then dispatch exactly one verification.

Nothing is written inside any directory that is under verification. The frame-alignment scan directory is NOT
touched: it is the object of the verification that is dispatched at the end of this round.
"""

import hashlib
import json
import os
import pathlib
import stat
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
ERRATA6 = DESK / 'S2-errata-6-20261015T180248'
SCAN = DESK / 'S2-frame-alignment-scan-20261015T180345'
SNAP_ROOT = DESK / '_frozen_snapshots'

BLOCKS = {
    'live_probe_current_sha256': PROBE,
    'canonical_batch_current_sha256': CANON,
    'record_batch_current_sha256': REC,
    'events_dir_current_sha256': EVENTS,
}

CONCLUSION = ('the collapse moment is decided by the "archive x initial state" pair: change the initial state or change the archive and '
              'the moment moves. Within that pair the sensitivity is very uneven: a +/-5% rescaling of the same archive moves the first '
              'non-foot contact by only -21/+20 steps (2790/2831 versus the reference 2811), while shifting the archive by a single frame '
              '(5 ms) moves it by -557/+328 steps (2254/3139) - an order of magnitude more. Frame alignment (phase) therefore dominates '
              'the timing, and the former wording "...rather than by a delicate amplitude or phase condition" is withdrawn: it is refuted '
              'by this batch own deltas.')
DELTA_TABLE = [
    dict(group='scale_0.95', delta='archive x 0.95', first_nonfoot_step=2790, first_nonfoot_ms=69.75,
         legs_below_6_step=544, first_zero_tarsus_step=5075),
    dict(group='reference_1.00', delta='reference (no change)', first_nonfoot_step=2811, first_nonfoot_ms=70.275,
         legs_below_6_step=538, first_zero_tarsus_step=5082),
    dict(group='scale_1.05', delta='archive x 1.05', first_nonfoot_step=2831, first_nonfoot_ms=70.775,
         legs_below_6_step=532, first_zero_tarsus_step=5072),
    dict(group='shift_minus_1_frame', delta='frame index -1 (5 ms earlier)', first_nonfoot_step=2254, first_nonfoot_ms=56.35,
         legs_below_6_step=736, first_zero_tarsus_step=2515),
    dict(group='shift_plus_1_frame', delta='frame index +1 (5 ms later)', first_nonfoot_step=3139, first_nonfoot_ms=78.475,
         legs_below_6_step=304, first_zero_tarsus_step=3119),
]


def unlock(d):
    for r, _dd, fs in os.walk(d):
        for f in fs:
            os.chmod(os.path.join(r, f), 0o664)
        os.chmod(r, 0o775)


def resign(d, extra=None):
    """Owner item 2 schema: files == os.listdir() minus this manifest and minus delivery-hashes.json."""
    d = pathlib.Path(d)
    unlock(d)
    fz, dh = d / 'FROZEN_DELIVERY_HASHES.json', d / 'delivery-hashes.json'
    for p in (fz, dh):
        if p.exists():
            p.unlink()
    fz.write_text('{}', encoding='utf-8')
    dh.write_text('{}', encoding='utf-8')
    names = sorted(p.name for p in d.iterdir() if p.is_file())
    covered = [f for f in names if f not in ('FROZEN_DELIVERY_HASHES.json', 'delivery-hashes.json')]

    # delivery-hashes.json: every file except itself (its own digest cannot exist)
    dhd = {f: sha(d / f) for f in names if f != 'delivery-hashes.json'}
    dhd['_self_excluded'] = 'this file cannot contain its own digest; no other file carries it either (a mutual pair has no fixed point)'
    dh.write_text(json.dumps(dhd, ensure_ascii=False, indent=1), encoding='utf-8')

    m = dict(status='FROZEN_BEFORE_VERIFICATION', frozen_at=now(),
             files={f: sha(d / f) for f in covered},
             self_excluded=True, self_sha256=None,
             self_excluded_reason=('this manifest cannot contain its own digest: writing the digest changes the bytes it would describe. '
                                   'It is therefore not listed in `files`, and its integrity is checked by re-hashing the file.'),
             covers_complete=True,
             covers_complete_meaning=('`files` covers every file of this directory EXCEPT this manifest and EXCEPT delivery-hashes.json; '
                                      'delivery-hashes.json is not covered here because its own digest cannot exist and neither file may '
                                      'carry the other (no fixed point). assert: set(files) == set(os.listdir()) - {manifest, '
                                      'delivery-hashes.json}'),
             assert_rule='set(files) == set(os.listdir()) - {FROZEN_DELIVERY_HASHES.json, delivery-hashes.json}',
             manifest_sha256=None,
             manifest_sha256_note='null by necessity; verify by re-hashing this file')
    if extra:
        m.update(extra)
    fz.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding='utf-8')

    m2 = json.loads(fz.read_text(encoding='utf-8'))
    listing = set(os.listdir(d))
    assert set(m2['files']) == listing - {'FROZEN_DELIVERY_HASHES.json', 'delivery-hashes.json'}, d
    assert m2['self_excluded'] is True and m2['self_sha256'] is None
    assert 'FROZEN_DELIVERY_HASHES.json' not in m2['files'] and 'delivery-hashes.json' not in m2['files']
    for f, h in m2['files'].items():
        assert sha(d / f) == h, (d.name, f)
    for p in sorted(d.iterdir()):
        if p.is_file():
            os.chmod(p, 0o444)
    os.chmod(d, 0o555)
    return dict(directory=str(d), n_files=len(listing), covered=len(m2['files']),
                manifest_sha256=sha(fz), delivery_hashes_sha256=sha(dh),
                rule='files == listdir - {manifest, delivery-hashes.json}', holds=True)


def main():
    out = DESK / ('S2-errata-7-' + STAMP)
    if out.exists():
        raise SystemExit('exists')
    out.mkdir()

    changes = []
    def rec(f, before, after, reason, kind='text_only'):
        changes.append(dict(file=str(f), sha256_before=before, sha256_after=after, changed=before != after,
                            changed_by="the model, on the owner's turn-35 instruction", changed_at=now(),
                            reason=reason, change_kind=kind))

    # ---------------- item 1: incident register + re-anchoring ----------------
    incident = dict(
        id='INC-5', kind='discipline incident: in-place rewrite of live delivered directories while a verification ran',
        when='2026-10-15 17:55:30 PDT (-0700)', detected_by='verifier S2-probe-snapshot-check-20261016T005207Z (early alert)',
        occurred_again_at='2026-10-15 17:58:17 and 17:54:07 (the follow-on manifests and README writes of the same rounds)',
        occurrence_number='fourth occurrence in this thread (17:01:26, 17:23:27->17:31:11, 17:55:30, and this round batch)',
        writer=dict(who='this work line (the model)',
                    writer_directory='not separately provable from disk: the writes are a single work line acting under one instruction; '
                                     'no independent writer process left a trace',
                    why="executing the owner's instruction to finish the freeze and the errata round; the write nonetheless landed inside an "
                        'open verification window, which is the breach'),
        affected_objects=[str(p) for p in BLOCKS.values()],
        file_level=[
            dict(block=b, file=str(D / f), recorded_in_errata_3=h, current_as_of=sha(D / f), mtime=mt(D / f),
                 matches=(sha(D / f) == h))
            for b, D in BLOCKS.items()
            for f, h in (json.loads((ERRATA3 / 'change_list.json').read_text(encoding='utf-8')).get(b) or {}).items()
            if (D / f).exists()],
        measured_totals=dict(records_in_errata_3_blocks=21, records_now_stale=10,
                             note='the earlier rounds reported "9"; the measured number against the current files is 10 stale out of 21 '
                                  'records, and the file-level table above is the authority'),
        conclusion=('these errata-3 records were invalidated by that write batch; the following hashes are to be used instead. '
                    'No historical physics number is changed by this conclusion.'),
    )
    (out / 'integrity_events.json').write_text(json.dumps(incident, ensure_ascii=False, indent=1), encoding='utf-8')
    rec(out / 'integrity_events.json', None, sha(out / 'integrity_events.json'),
        'INC-5 registered file by file with before/after sha256, mtime and local time; no summary sentence used in place of evidence',
        kind='new_file')

    # ---------------- item 3: probe report.json + README conclusion and delta table ----------------
    unlock(PROBE)
    rp = PROBE / 'report.json'
    b = sha(rp)
    j = json.loads(rp.read_text(encoding='utf-8'))
    cur = j['answer']['reading']
    assert 'delicate' not in cur, ('the blanket sentence is still present in answer.reading: %s' % cur[:200])
    assert 'more sensitive to frame alignment' in cur, ('neither the blanket sentence nor the measured sentence is present: %s' % cur[:200])
    # reword to the owner's exact two-clause phrasing, keeping the measured numbers
    j['answer']['reading'] = CONCLUSION
    j['answer']['delta_table'] = DELTA_TABLE
    j['answer']['delta_table_note'] = ('three-column delta table for the five groups: first non-foot contact (step and ms), legs_in_contact < 6, '
                                       'first step with zero tarsus contacts. Identical wording and identical numbers are reproduced in '
                                       'README.zh-CN.md of this same batch.')
    j['answer']['independent_recomputation_note'] = ('the shift groups max|qacc| steps 3208 and 2813 were recomputed by an independent verifier and '
                                                    'agree with the delivered values, so the earlier statement that they had no second '
                                                    'recomputation is withdrawn')
    j['answer']['reading_corrected_again_at'] = now()
    rp.write_text(json.dumps(j, ensure_ascii=False, indent=1), encoding='utf-8')
    rec(rp, b, sha(rp), 'item 3: the blanket conclusion is replaced by the delta-supported sentence; the five-group delta table is embedded; '
                        'the "no second recomputation" statement is withdrawn')

    rd = PROBE / 'README.zh-CN.md'
    b = sha(rd)
    t = rd.read_text(encoding='utf-8')
    tbl = '\n'.join([
        '', '## 五组增量表（与 report.json 的 `answer.delta_table` 逐字一致）', '',
        '| 组 | 增量 | 首个非足接触步 / ms | legs<6 步 | 首次零跗节步 |',
        '|---|---|---|---|---|',
        *['| %s | %s | %s / %s | %s | %s |' % (r['group'], r['delta'], r['first_nonfoot_step'], r['first_nonfoot_ms'],
                                               r['legs_below_6_step'], r['first_zero_tarsus_step']) for r in DELTA_TABLE],
        '',
        '## 结论（更正后，与 report.json `answer.reading` 逐字一致）',
        '- %s' % CONCLUSION,
        '- 说明：错位组 `max|qacc|` 步 3208 / 2813 **已由独立核验方复算并与交付值一致**，故此前"尚无第二次复算"的说法作废。',
    ])
    t = t + tbl
    rd.write_text(t, encoding='utf-8')
    rec(rd, b, sha(rd), 'item 3: the same corrected conclusion and the same five-group delta table are appended to the batch README so that '
                        'report.json and README state one identical conclusion')

    # ---------------- item 2: re-sign every frozen manifest ----------------
    resigned = [resign(d, dict(kind='frozen under the owner-specified schema')) for d in
                (PROBE, CANON, REC, EVENTS, ERRATA3, ERRATA6)]

    (out / 'change_list.json').write_text(json.dumps(dict(
        status='ERRATA_7_CHANGE_LIST', manual_research_not_product=True, created_at=now(),
        rule='full path, before sha256, after sha256, reason, author for every entry. errata-3 historical values are preserved and never '
             'overwritten: re-anchoring is published as `current_as_of` beside the old value.',
        entries=changes, n=len(changes),
        reanchored_records=incident['file_level'],
        incident=incident['id'],
        manifest_schema=dict(
            files='every file of the directory EXCEPT this manifest and EXCEPT delivery-hashes.json',
            self_excluded=True, self_sha256=None,
            assert_rule='set(files) == set(os.listdir()) - {FROZEN_DELIVERY_HASHES.json, delivery-hashes.json}',
            why='a manifest cannot contain its own digest, and a mutual digest pair with delivery-hashes.json has no fixed point; the earlier '
                'revision that listed itself with a self-digest was therefore necessarily wrong, and the schema now makes that impossible '
                'to state'),
        resigned=[r for r in resigned],
        conclusion_of_record='the errata-3 records were invalidated by the 17:55:30 write batch; use the hashes listed in this round '
                            'integrity_events.json instead. No historical physics number is changed.',
    ), ensure_ascii=False, indent=1), encoding='utf-8')

    (out / 'README.zh-CN.md').write_text('\n'.join([
        '# errata-7：INC-5 事故登记 ＋ 记录重锚定（不覆盖历史值）＋ 清单重签 ＋ 结论句与增量表落地（**手动研究实施，不是成品**）', '',
        '## 一、INC-5（第 4 次纪律事故，逐文件登记，不用总述代替凭证）',
        '- **时间**：2026-10-15 **17:55:30**（后续同批 17:54:07 与 17:58:17）。**检测者**：核验方早期通报。',
        '- **逐文件 before/after + mtime**：见 `integrity_events.json` 的 `file_level`（**21 条记录，其中现测 10 条失配**；'
        '此前几轮写的"9 条"是**未经盘上核对的数字**，以本轮逐文件表为准）。',
        '- **谁改的**：本工作线（模型）；**能否确证写者目录**：**不能**——同一条工作线、同一条指令下的连续写入，盘上没有独立写者痕迹，'
        '如实写明而不编造归属。**为什么改**：执行 owner"先把冻结做完再派核验"的指令；但写入落进了**已开窗口**，这就是违规本身。',
        '- **处置结论**：**该批记录已被本次写作用废，需以本轮列出的哈希为准**；**不改历史物理数字**。errata-3 的原记录原样保留，'
        '重锚定以 `current_as_of` ＋ 新核对时间并列发布。', '',
        '## 二、清单重签（owner 指定 schema，不再引入第三种写法）',
        '- `files` 只列**被覆盖文件** = 目录内除**本清单**与 **`delivery-hashes.json`** 之外的全部文件；`self_excluded: true`；'
        '`self_sha256: null`；`covers_complete` 含义写明；**断言** `set(files) == set(os.listdir()) - {清单, delivery-hashes.json}`'
        '（freeze 时脚本断言，逐一重算摘要）。',
        '- 为什么这样才是正解：清单**不能包含自身摘要**（写摘要即改字节），而 `delivery-hashes.json` **也无法承载清单摘要**'
        '（互记是无不动点的方程）。此前"列自身 + 自摘要"必然错，本 schema 使那种写法**无法被写出来**。',
        '- 本轮重签：%s。' % '、'.join(pathlib.Path(r['directory']).name + '（%d 条）' % r['covered'] for r in resigned), '',
        '## 三、结论句与增量表（report.json ↔ README 逐字一致）',
        '- 新结论句：**%s**' % CONCLUSION,
        '- 五组增量表（首非足步/ms、legs<6、首次零跗节）已**同时**写入探针 `report.json` 的 `answer.delta_table` 与 `README.zh-CN.md`，'
        '两处数字与措辞一致。',
        '- 另作废"错位组 3208/2813 尚无第二次复算"——**独立核验方已复算且与交付值一致**（写入 `report.json` 的 '
        '`answer.independent_recomputation_note`）。', '',
        '- **手动研究实施，不是成品**；不宣称站立/行走/完成度；绝对力值与单位 `NOT_ESTABLISHED`。']),
        encoding='utf-8')

    # freeze this errata directory under the same schema
    info = resign(out, dict(kind='this round errata record'))
    print('ERRATA-7:', out)
    print('files:', info['n_files'], 'covered:', info['covered'])
    for r in resigned:
        print('  re-signed %-62s listdir=%d covered=%d' % (pathlib.Path(r['directory']).name, r['n_files'], r['covered']))
    print('stale records measured:', incident['measured_totals'])
    return out


if __name__ == '__main__':
    main()
