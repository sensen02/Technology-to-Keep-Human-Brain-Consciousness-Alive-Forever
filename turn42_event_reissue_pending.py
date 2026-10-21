"""Turn-42 (owner turn 41): three new directories.

  1. the 17:55:30 rewrite is filed as its OWN timestamped event directory (not folded into errata-3): per file
     before/after sha256, mtime, and the writer's own account; errata-3's 9 (measured: 10) current-sha256 records
     are labelled only "superseded by the 17:55:30 event" and the ledger is NOT rewritten to match the disk.
  2. the two manifests are re-issued in a new directory under the tightened design:
     files = all other delivered files, covers_all_other_files=true, covers_self=false, and the text
     "manifest_sha256: null by design; this manifest's own digest is recorded only by the ledger".
     No in-place overwrite; only that one thing is changed.
  3. the ~70 ms attribution is marked PENDING VERIFICATION (tentative "archive x state inseparable") until the
     verifier's complete report lands.
Plus the tracker's freeze rule: any further in-window write VOIDs the batch and it is redone.
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
utc = lambda: time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
mt = lambda p: time.strftime('%Y-%m-%d %H:%M:%S %Z (%z)', time.localtime(pathlib.Path(p).stat().st_mtime))
TS = time.strftime('%Y%m%dT%H%M%S')

PROBE = DESK / 'S2-drive-sensitivity-probe-20261015T174405'
CANON = DESK / 'S2-accepted-equilibrium-openloop-corrected-reissued-25us-v2-20261015T172758'
REC = DESK / 'S2-correction-record-v3-20261015T172848'
EVENTS = DESK / 'S2-25us-clock-patches-and-events-20261015T172015'
ERRATA3 = DESK / 'S2-errata-3-20261015T175143'
BLOCKS = {'live_probe_current_sha256': PROBE, 'canonical_batch_current_sha256': CANON,
          'record_batch_current_sha256': REC, 'events_dir_current_sha256': EVENTS}


def seal(d):
    for p in sorted(d.iterdir()):
        if p.is_file():
            os.chmod(p, 0o444)
    os.chmod(d, 0o555)


def template(d, extra=None):
    listdir = sorted(x for x in os.listdir(d) if (d / x).is_file())
    covered = [f for f in listdir if f != 'FROZEN_DELIVERY_HASHES.json']
    m = dict(status='FROZEN_BEFORE_VERIFICATION', frozen_at=now(),
             files={f: sha(d / f) for f in covered},
             covers_all_other_files=True, covers_self=False,
             manifest_sha256=None,
             manifest_sha256_note="manifest_sha256: null by design; this manifest's own digest is recorded only by the ledger",
             excluded='FROZEN_DELIVERY_HASHES.json (this manifest itself)',
             excluded_reason='a manifest cannot contain its own digest (writing it changes the bytes); no other file carries it either because '
                             'a mutual digest pair has no fixed point',
             design_rule='files covers every delivered file except this manifest; len(files) == len(os.listdir()) - 1',
             template_id='FROZEN_TEMPLATE_v4_owner_turn41')
    if extra:
        m.update(extra)
    assert len(m['files']) == len(listdir) - 1, (d, len(m['files']), len(listdir))
    return m, listdir


def main():
    # ================= 1. the 17:55:30 event, filed on its own =================
    ev = DESK / ('S2-integrity-event-175530-' + TS)
    if ev.exists():
        raise SystemExit('exists')
    ev.mkdir()
    e3 = json.loads((ERRATA3 / 'change_list.json').read_text(encoding='utf-8'))
    files = []
    for b, D in BLOCKS.items():
        for f, recorded in (e3.get(b) or {}).items():
            p = D / f
            if not p.exists():
                continue
            files.append(dict(
                file=str(p), mtime=mt(p), block=b,
                before_sha256=dict(errata_3_recorded=recorded,
                                   verifier_17_29_33=('cf5708c86b1cae97e3ee9084ab5f10bd91e77fc59fc87a4f06b9222411260c53'
                                                      if D is CANON and f == 'delivery-hashes.json' else None)),
                after_sha256=sha(p),
                superseded_by_this_event=(sha(p) != recorded),
            ))
    who = dict(
        who_wrote='this work line: the model, executing the owner instruction to finish the freeze and the errata round before dispatching a '
                  'verifier',
        motive=('the instruction was to complete the FROZEN manifests and the errata records first. The author did not check whether a '
                'verification window was already open, and it was: verifier 1ddb238f had already started. That is the breach.'),
        not_external='this was NOT an external or unknown writer. It was the same work line, in the same session, under one instruction. No '
                     'independent process left a trace, so a distinct "writer directory" cannot be proven from disk, and none is claimed.',
        provable='the write times (17:54:07, 17:55:30, 17:58:17) match this work line own commands in the same session',
    )
    (ev / 'event.json').write_text(json.dumps(dict(
        status='INTEGRITY_EVENT_STANDALONE', id='EV-175530', at=now(),
        when='2026-10-15 17:55:30 PDT (-0700)', continued='17:54:07 and 17:58:17',
        files=files, n=len(files), n_superseded=sum(1 for r in files if r['superseded_by_this_event']),
        writer=who,
        errata_3_handling=dict(
            file=str(ERRATA3 / 'change_list.json'), current_sha256=sha(ERRATA3 / 'change_list.json'),
            mtime=mt(ERRATA3 / 'change_list.json'),
            label_only='the records it holds are labelled "superseded by the 17:55:30 event" and NOTHING in that file is rewritten. '
                       'The measured number of affected records is %d, not the 9 quoted earlier; the measured figure is authoritative.'
                       % sum(1 for r in files if r['superseded_by_this_event'])),
        consequence='every affected directory stays read-only; all later corrections live in new timestamped directories with a ledger',
    ), ensure_ascii=False, indent=1), encoding='utf-8')
    (ev / 'README.zh-CN.md').write_text('\n'.join([
        '# 完整性事件（独立立案）：2026-10-15 17:55:30 核验窗口内写入（**手动研究实施，不是成品**）', '',
        '- **逐文件**：`event.json` 给出**全路径 / mtime / errata-3 recorded 值 / 核验方 17:29:33 实测值 / 现值 / 是否被本事件超越**，'
        '实测 **%d 条**、**%d 条被超越**。' % (len(files), sum(1 for r in files if r['superseded_by_this_event'])),
        '- **谁在写、什么动机**：**本工作线（模型）**，动因是执行"先完成冻结与 errata 再派核验"的指令；**当时未检查窗口是否已开**，'
        '而窗口已开（核验者 `1ddb238f` 已起手）—— 这就是违规本身。**明确不是"外部"写者**：同一会话、同一条指令下的连续写入；'
        '盘上无独立进程痕迹，**不主张存在"另一个写者目录"**。',
        '- **errata-3 的 ledger 未被改写**（现值 `%s`）：其中记录**只标注**"已被 17:55:30 事件超越"，'
        '**绝不为了对上磁盘去原地改写 ledger**。' % sha(ERRATA3 / 'change_list.json'), '',
        '- **手动研究实施，不是成品**；绝对力值与单位 `NOT_ESTABLISHED`。']),
        encoding='utf-8')
    (ev / 'FROZEN_DELIVERY_HASHES.json').write_text('{}', encoding='utf-8')
    m, ld = template(ev, dict(kind='this event record'))
    (ev / 'FROZEN_DELIVERY_HASHES.json').write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding='utf-8')
    seal(ev)

    # ================= 2. the two manifests re-issued under the tightened design =================
    re = DESK / ('S2-frozen-manifest-reissue-' + TS)
    if re.exists():
        raise SystemExit('exists')
    re.mkdir()
    led = []
    for D in (PROBE, CANON):
        m2, ld2 = template(D, dict(kind='re-issued manifest; the live file is NOT overwritten'))
        dest = re / ('FROZEN_DELIVERY_HASHES.%s.json' % D.name)
        dest.write_text(json.dumps(m2, ensure_ascii=False, indent=1), encoding='utf-8')
        led.append(dict(old_file=str(D / 'FROZEN_DELIVERY_HASHES.json'),
                        old_sha256=sha(D / 'FROZEN_DELIVERY_HASHES.json'),
                        new_file=str(dest), new_sha256=sha(dest),
                        live_listdir=len(ld2), covered=len(m2['files']),
                        rule_check=(len(m2['files']) == len(ld2) - 1),
                        reason='the design error: the manifest must not list itself with a digest that cannot be valid. Re-issued with '
                               'covers_all_other_files=true, covers_self=false and the manifest_sha256: null by design note. Only this one '
                               'thing changes; the non-self entries were already correct and are untouched.',
                        applied=False,
                        applied_note='NOT overwritten in place: the live file is left as it is and this re-issue is the ledger entry to be '
                                     'used when the live file is next legitimately replaced'))
    (re / 'reissue_ledger.json').write_text(json.dumps(dict(
        status='FROZEN_MANIFEST_REISSUE_LEDGER', at=now(), template_id='FROZEN_TEMPLATE_v4_owner_turn41',
        design_rule=template(PROBE)[0]['design_rule'], ledger=led, n=len(led),
    ), ensure_ascii=False, indent=1), encoding='utf-8')
    (re / 'README.zh-CN.md').write_text('\n'.join([
        '# 冻结清单重出（设计错误更正，**不原地覆盖**）（**手动研究实施，不是成品**）', '',
        '- **设计规则**：`files` **只覆盖其它交付文件**；`covers_all_other_files = true`；`covers_self = false`；'
        '**明写** `manifest_sha256: null by design；本清单自身哈希只由 ledger 记录`；`len(files) == os.listdir − 1`。',
        '- **只改这一处**：非自身条目本来就是对（探针 `7/8`、规范件 `9/10`），**未顺手动其它字段**。',
        '- **未原地覆盖**：两份 live 清单保持原样；重出件写在本目录，`reissue_ledger.json` 逐条记 **old file / old sha256 / new file / '
        'new sha256 / 理由**，供下次合法替换时使用。', '',
        '- **手动研究实施，不是成品**；绝对力值与单位 `NOT_ESTABLISHED`。']),
        encoding='utf-8')
    (re / 'FROZEN_DELIVERY_HASHES.json').write_text('{}', encoding='utf-8')
    m3, ld3 = template(re, dict(kind='this re-issue record'))
    (re / 'FROZEN_DELIVERY_HASHES.json').write_text(json.dumps(m3, ensure_ascii=False, indent=1), encoding='utf-8')
    seal(re)

    # ================= 3. pending-verification status =================
    pv = DESK / ('S2-pending-verification-70ms-' + TS)
    if pv.exists():
        raise SystemExit('exists')
    pv.mkdir()
    (pv / 'status.json').write_text(json.dumps(dict(
        status='PENDING_VERIFICATION', at=now(), at_utc=utc(),
        question='is the ~70 ms collapse decided by the command archive itself (including frame alignment), or is it a property of the accepted '
                 'state?',
        current_formulation=dict(
            text='TENTATIVE: the ~70 ms first non-foot contact is a joint property of the state and of the archive frame alignment; the two are '
                 'NOT separable from the measurements available so far, because every group shares one state and one archive.',
            tier='tentative / to be confirmed only by the verifier complete report',
            cannot_be_separated_so_far='no group varies the state while holding the archive fixed, and no group varies the archive while holding '
                                       'the state fixed in a way that isolates the two factors, so "archive x state inseparable" is a statement '
                                       'about this design, not a measured decomposition'),
        what_is_already_measured=dict(amplitude_steps=[-21, 20], phase_steps=[-557, 328],
                                      amplitude_ms=[-0.525, 0.525], phase_ms=[-13.925, 8.2],
                                      reading='phase/alignment sensitivity exceeds amplitude sensitivity by about one order of magnitude '
                                              '(measured, not extrapolated)'),
        what_is_pending='the verifier complete report on the errata-4 object; until it lands, no final sentence on archive-vs-state is written',
        rule='final wording only after the report lands',
    ), ensure_ascii=False, indent=1), encoding='utf-8')
    (pv / 'README.zh-CN.md').write_text('\n'.join([
        '# ~70 ms 塌陷归因：**待验状态**（**手动研究实施，不是成品**）', '',
        '- **在核验方完整报告落盘前，一律写"待验"**；"archive × state 不可分"**只能是暂定表述**（它说的是本设计的事实：所有组共用同一状态'
        '与同一档案，**没有任何一组单独变动其中一个因子**，因此无法据现有测量做分解）。',
        '- **已经实测的**：幅度 ±5% → **−21/+20 步（−0.525/+0.525 ms）**；单帧错位 → **−557/+328 步（−13.925/+8.2 ms）**；'
        '**相位/对齐敏感度比幅度高约一个数量级（实测结论，非外推）**。',
        '- **待报告到达后定稿**。', '',
        '- 绝对力值与单位 `NOT_ESTABLISHED`。']),
        encoding='utf-8')
    (pv / 'FROZEN_DELIVERY_HASHES.json').write_text('{}', encoding='utf-8')
    m4, ld4 = template(pv, dict(kind='pending-verification status record'))
    (pv / 'FROZEN_DELIVERY_HASHES.json').write_text(json.dumps(m4, ensure_ascii=False, indent=1), encoding='utf-8')
    seal(pv)

    # ================= tracker: freeze rule tightened =================
    t = ROOT / 'GOAL_TRACKER.zh-CN.md'
    txt = t.read_text(encoding='utf-8')
    rule = ('\n## 冻结期规则（owner 第 41 轮从紧）\n\n'
            '- **核验窗口 = verifier 起手 → 它把报告落盘**。窗口期间**整棵交付树一个字节都不许写**（不只是对象目录）。'
            '窗口结束后要改，**一律新目录 ＋ ledger**（旧文件、旧 sha256、新文件、新 sha256、理由）。\n'
            '- **再在窗口内写一次 ⇒ 该批次直接 VOID 重做**，**不再补记、不修补**。已发生的同类事件：17:01:26、17:23:27→17:31:11、'
            '17:55:30（本次已按独立事件立案 `S2-integrity-event-175530-*`）。\n'
            '- **窗口内不派第二/并行核验**；一次一个，顺序执行。\n')
    if '冻结期规则（owner 第 41 轮从紧）' not in txt:
        txt = txt.rstrip('\n') + '\n' + rule
        t.write_text(txt, encoding='utf-8')
    print('EVENT   :', ev)
    print('REISSUE :', re, '| ledger rows', len(led))
    print('PENDING :', pv)
    print('tracker sha256:', sha(t))
    return ev, re, pv


if __name__ == '__main__':
    main()
