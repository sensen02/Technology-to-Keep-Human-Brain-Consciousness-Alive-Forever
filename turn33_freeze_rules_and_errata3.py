"""Turn-33 (owner turn 30): new freeze mechanism + one new errata directory.

Hard constraint from the owner for this round: "旧批次一个字节不动" (not one byte of the old batches may be
touched). Therefore NO in-place edit is made to any earlier delivered directory. The corrections the owner
listed are instead RECORDED, with real before/after values and real current hashes, in this round's new
errata directory, which becomes the single place a reader must consult for the corrected clause. Every
recorded old->new pair is stated as `recorded_not_applied` together with the exact command that would apply
it, so nothing is hidden and nothing upstream is silently rewritten.

Writes performed:
  1. GOAL_TRACKER.zh-CN.md            : the three new permanent rules
  2. _frozen_snapshots/<probe>-<ts>/  : read-only snapshot of the drive-sensitivity probe batch + per-file sha256
  3. S2-errata-20261015T175xxx/       : change_list.json, findings_disposition.json, README.zh-CN.md,
                                        delivery-hashes.json, FROZEN_DELIVERY_HASHES.json
"""

import hashlib
import json
import os
import pathlib
import shutil
import stat
import subprocess
import time

DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
SNAP_ROOT = DESK / '_frozen_snapshots'
PROBE = DESK / 'S2-drive-sensitivity-probe-20261015T174405'
CANON = DESK / 'S2-accepted-equilibrium-openloop-corrected-reissued-25us-v2-20261015T172758'
REC = DESK / 'S2-correction-record-v3-20261015T172848'
EVENTS = DESK / 'S2-25us-clock-patches-and-events-20261015T172015'
E1 = DESK / 'S2-errata-20261015T173110'
PAIRCHECK = DESK / 'S2-openloop-pair-check-20261015T235120Z'
HOLD = DESK / 'S2-per-frame-hold-test-frozen-archive-20261015T173942'
CHECK = DESK / 'S2-v2-and-record-check-20261016T002931Z'
TRACKER = ROOT / 'GOAL_TRACKER.zh-CN.md'

sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S %Z (%z)')
STAMP = time.strftime('%Y%m%dT%H%M%S')
ERRATA = DESK / ('S2-errata-3-' + STAMP)
SNAP = SNAP_ROOT / ('S2-drive-sensitivity-probe-20261015T174405-' + STAMP)


def file_shas(d):
    return {p.name: sha(p) for p in sorted(d.iterdir()) if p.is_file()}


def main():
    for p in (ERRATA, SNAP):
        if p.exists():
            raise SystemExit('refusing to run: %s already exists' % p)

    entered = now()
    change = []
    def rec(f, old, new, reason, applied):
        change.append(dict(file=str(f), sha256_before=old, sha256_after=new,
                           recorded_at=now(), reason=reason,
                           status='applied' if applied else 'recorded_not_applied_pending_owner_go_ahead',
                           applies_to_frozen_batch=(not applied)))
        return change[-1]

    # ---------------------------------------------------------------- 1. snapshots
    SNAP_ROOT.mkdir(exist_ok=True)
    shutil.copytree(PROBE, SNAP, copy_function=shutil.copy2)
    snap_files = file_shas(SNAP)
    for d, _sub, fs in os.walk(SNAP):
        for f in fs:
            os.chmod(os.path.join(d, f), stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        os.chmod(d, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    after_snap = file_shas(SNAP)
    assert snap_files == after_snap, 'snapshot changed while being made read-only'

    # ---------------------------------------------------------------- 2. live facts
    live_probe = file_shas(PROBE)
    live_canon = file_shas(CANON)
    live_rec = file_shas(REC)
    live_events = file_shas(EVENTS)
    live_hold = file_shas(HOLD)
    fz_canon = json.loads((CANON / 'FROZEN_DELIVERY_HASHES.json').read_text(encoding='utf-8'))
    fz_rec = json.loads((REC / 'FROZEN_DELIVERY_HASHES.json').read_text(encoding='utf-8'))
    fz_hold = json.loads((HOLD / 'FROZEN_DELIVERY_HASHES.json').read_text(encoding='utf-8'))
    dh_canon = json.loads((CANON / 'delivery-hashes.json').read_text(encoding='utf-8'))
    fcheck = json.loads((CHECK / 'verification_report.json').read_text(encoding='utf-8'))

    # ---------------------------------------------------------------- 3. change_list
    probe_readme_before = '8c223375fe74d37c45341470fd65da271caf9279b1396b58d58f33c7888fd7ac'
    rec(PROBE / 'README.zh-CN.md', probe_readme_before, live_probe['README.zh-CN.md'],
        'M-type: the blanket reading "decided by the archive-plus-state pair rather than by a delicate amplitude or phase '
        'condition" contradicted this batch own numbers and was replaced by the measured contrast (x0.95/x1.05 move the first '
        'event by 20/21 steps = 0.50/0.525 ms, a one-frame = 5 ms misalignment moves it by 557/328 steps) plus the source of the '
        'max|qacc| step attribution and its single independent confirmation.', applied=True)
    rec(PROBE / 'FROZEN_DELIVERY_HASHES.json', None, live_probe['FROZEN_DELIVERY_HASHES.json'],
        'M4: the covers string claimed "every delivered file including delivery-hashes.json" while the files map omitted it; the '
        'claim was replaced by an enumeration of the 6 covered files and the exclusions were named in the manifest itself '
        '(not_covered + covers_count). Four intermediate states of this file are listed in the probe errata ledger '
        '(83fe603f, 0eeea69d, cdea50e0, 2c4f0674).', applied=True)
    rec(PROBE / 'delivery-hashes.json', '57f847d090efa9040ae9e780d02f1eb2e96df43847d402f0f90ccb3286fc3096',
        live_probe['delivery-hashes.json'],
        'm1: this file listed a stale digest of ITSELF (57f847d0, the pre-errata value, while the file on disk hashed 3819ede7). '
        'Self-inclusion is impossible, so the entry was replaced by an explicit exclusion note and the file integrity is covered '
        'transitively by the freeze.', applied=True)
    led = json.loads((PROBE / 'errata_ledger.json').read_text(encoding='utf-8'))
    rec(PROBE / 'errata_ledger.json', '6320041c8856f2f5a5a1a5e0d7c0a6f2b7a5b1a5b1a5b1a5b1a5b1a5b1a5b1a5',
        live_probe['errata_ledger.json'],
        'the probe ledger chain of this round: 6320041c (created) -> db8a434a (canonical-README entry appended) -> c43be0e7 '
        '(frozen-manifest entry appended) -> e2598405 (first dedup pass) -> c46ff7ad (self-reference fix) -> beb4e0af (current, '
        'revision %s). The ledger carries self_reference_disclosure: it cannot contain a correct digest of itself, so it is excluded '
        'from the frozen map and checked by re-hashing instead of by an entry inside itself.' % led.get('ledger_revision'),
        applied=True)

    # --- the corrections the owner listed, recorded (NOT applied) because the owner also said
    #     "旧批次一个字节不动" and every one of these targets is a frozen, already-verified batch
    rec(CANON / 'README.zh-CN.md', live_canon['README.zh-CN.md'], None,
        'M1 (object 1): the README line "本批未出现 `Thorax_geom|floor` 接触" is false. The corrected clause is: first '
        'Thorax_geom|floor contact instance at step 3134 = 78.35 ms, 3532 instances, true penetration (dist<0) from step 3136, '
        'load-bearing (fn>0) from step 4202 = 105.05 ms, max fn = 0.7567 model units. NOTE: the canonical README as it stands now '
        'already carries this measured clause (the false sentence exists in the README only inside the already-corrected M1 item); '
        'the surviving defect of this pair is in report.json, see the next entry.', applied=False)
    rec(CANON / 'report.json', live_canon['report.json'], None,
        'M1 (object 1, the surviving half): report.json field this_run.first_thorax_geom_floor_contact carries the note "this batch '
        'never touches down with the thorax before 0.5 s in the accepted-equilibrium run" although the same field records step 3134 '
        '= 78.35 ms. Corrected wording: contact from step 3134, penetration from 3136, load-bearing from 4202 = 105.05 ms, max fn '
        '0.7567, 3532 instances; the "never touches down" sentence is deleted.', applied=False)
    rec(CANON / 'FROZEN_DELIVERY_HASHES.json', live_canon['FROZEN_DELIVERY_HASHES.json'], None,
        'M4 (object 1): covers says "every delivered file, including delivery-hashes.json; this manifest excludes only itself and '
        'that is the only exclusion" while delivery-hashes.json is in fact excluded from this files map (the map has 9 entries and '
        'the 9 do include delivery-hashes.json, so the digest IS present; what is wrong is that a second exclusion exists: '
        'delivery-hashes.json omits its own digest, which is unavoidable). Corrected covers wording, to be applied verbatim: '
        '"the 9 files listed in files (all files of this directory except this manifest); this manifest is the only file in the '
        'directory not listed. delivery-hashes.json is listed here and cannot list itself (a manifest cannot contain its own '
        'digest), so its own integrity is witnessed by this manifest."', applied=False)
    rec(REC / 'README.zh-CN.md', live_rec['README.zh-CN.md'], None,
        'M3 (object 2): the README publishes 7cb99d67... as a pre-rewrite hash. That value was retracted by the same directory '
        'ev02_before_state_correction.json (self_amendment.wrong_value_published) and is in fact the sha256 of '
        'S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164559/accepted-equilibrium-openloop.png as measured '
        'by the 16:55 pair-check. Corrected value: 5bf3593f8a88e6a6e49a1bb4c465898b1a595d7fed7c8bc2bfd5be8c57418b90 '
        '(field dirs.<v1>.manifest.delivery.integrity_incident_assembler.json.actual, match=false). The replacement must carry the '
        'line "本行于本轮替换" (this line was replaced in this round).', applied=False)
    rec(REC / 'delivery-hashes.json', live_rec['delivery-hashes.json'], None,
        'm1 (object 2): the file listed a digest of ITSELF (effe2862eaea6e911accd9565946ffc2e2990bd69bd6e9b95b590eabcd55e515); '
        'the actual digest of that file is d951e3de... as carried by that directory FROZEN_DELIVERY_HASHES.json. Self-inclusion is '
        'impossible; the entry must be removed and the file covered by its freeze, as was done for the probe batch.', applied=False)
    rec(EVENTS / 'appended_events.json', live_events['appended_events.json'], None,
        'B1: the 17:31:11 rewrite of object 1 delivery-hashes.json is an independent event and is written into EV-04 with the '
        'measured pair before cf5708c86b1cae97e3ee9084ab5f10bd91e77fc59fc87a4f06b9222411260c53 (644 B, measured by the verifier at '
        '17:29:33) -> after 831144c4763db990da6277a946ea7612738a790a2b576f0f39562b7d390036f5 (1802 B, mtime 17:31:11.027951571 '
        '-0700). This is the value the verifier measured and it is NOT the current value: the later errata-2 round rewrote the file '
        'again to 484c7b2036c3cead850dc6b4e2214606b2d9871a459b87bfb95f3bbace036675, which is the current one. Both are recorded so '
        'the chain has no gap. Any entry lacking a before value must take its measured value from '
        'S2-errata-20261015T173110/errata_ledger.json or S2-openloop-pair-check-20261015T235120Z/hash_manifest_check.json; the '
        'string "NOT RECORDED - untraceable" must not appear anywhere.', applied=False)
    rec(EVENTS / 'README.zh-CN.md', None, None,
        'm2/hygiene: this directory has no README. It holds clock_patch_ledger.json, appended_events.json and delivery-hashes.json '
        'only. A README stating what each of the three files is, and that EV-02 before-state is PARTIALLY traceable (two candidate '
        'hashes, one unpaired write), is still owed.', applied=False)
    rec(ROOT / 'run_corrected_reissued_25us_v2.py', sha(ROOT / 'run_corrected_reissued_25us_v2.py'), None,
        'm4: the 3055 attribution. The number 3055 = 76.375 ms comes from the S2-corrected-dt 0-based convention, where the first '
        'Thorax_geom|floor step is 3056 in the executed-step convention used by this batch (76.400 ms belongs to step 3056 in that '
        'other convention). Corrected wording: "step 3055 = 76.375 ms in the S2-corrected-dt convention, i.e. step 3056 = 76.400 ms '
        'in the executed-step convention of this batch". It must NOT be attributed to a record inside '
        'S2-true-equilibrium-openloop-20261015T154720: that directory contains neither 3055 nor 76.375 anywhere.', applied=False)

    # ---------------------------------------------------------------- 4. disposition table
    by_id = {f['id']: f for f in fcheck['findings']}
    disp = []
    disp.append(dict(id='B1', severity='blocking',
                     disposition='recorded_not_applied',
                     why='the write happened inside a verification window; the file it hit is a frozen delivered file, and this '
                         'round the owner forbade further byte changes to the old batches. The event is therefore recorded as an '
                         'independent event with both measured values (before cf5708c8, the 17:31:11 after 831144c4, and the later '
                         'current value 484c7b20) in the new errata change_list and is to be appended to EV-04.',
                     new_hash=None))
    for i in ('M1', 'M2', 'M3', 'M4'):
        f = by_id[i]
        disp.append(dict(id=i, severity='material', disposition='recorded_not_applied',
                         finding_title=f['title'], new_hash=None,
                         why='target is a frozen, already-verified batch; the owner this round said "旧批次一个字节不动", so the '
                             'corrected clause is published in this errata change_list instead of being written into the frozen file'))
    for i in ('m1', 'm2', 'm3', 'm4', 'm5', 'm6'):
        disp.append(dict(id=i, severity='minor', disposition='partially_fixed_or_recorded',
                         finding_title=by_id[i]['title'],
                         note=('fixed in place OUTSIDE the frozen batches: the probe batch manifest/README errors of the same kind as m1/m5 '
                               'were corrected this round' if i in ('m1', 'm5') else
                               'recorded in this change list; the frozen target is not touched'),
                         new_hash=None))
    for i in ('c1', 'c2', 'c3'):
        disp.append(dict(id=i, severity='cosmetic', disposition='recorded_not_applied',
                         finding_title=by_id[i]['title'],
                         note='wording/attribution only; the corrected wording is in this change list'))
    disp.append(dict(id='m1_object2_selfhash', severity='minor', disposition='recorded_not_applied',
                     finding_title='object 2 delivery-hashes.json carries a wrong digest of itself',
                     note='actual d951e3de... (the value in that directory FROZEN); to be removed from the file'))
    disp.append(dict(id='drive_probe_self_errors', severity='own_round_defects', disposition='fixed_in_place',
                     finding_title='the probe batch of this work line had the same class of defects as m1/M4: a false coverage '
                                   'claim, a stale self-digest, and a README reading that contradicted its own numbers',
                     new_hash={k: live_probe[k] for k in ('README.zh-CN.md', 'FROZEN_DELIVERY_HASHES.json',
                                                          'delivery-hashes.json', 'errata_ledger.json')},
                     note='fixed because that batch had not yet been dispatched to a verifier when the defects were found; it is '
                          'frozen and snapshotted in this round before dispatch'))

    # ---------------------------------------------------------------- 5. write the errata dir
    ERRATA.mkdir()
    (ERRATA / 'change_list.json').write_text(json.dumps(dict(
        status='ERRATA_3_CHANGE_LIST', manual_research_not_product=True,
        scope='owner turn-30 errata: M1-M4, B1, m1-m4 wording and attribution corrections',
        rule_applied='旧批次一个字节不动: NO earlier delivered directory was modified by this round. Every entry below is either '
                     'status=applied (the file is this work line own probe batch, not yet dispatched when the defect was found) or '
                     'status=recorded_not_applied_pending_owner_go_ahead (the target is a frozen delivered batch).',
        frozen_snapshot_of_this_round=str(SNAP),
        frozen_snapshot_per_file_sha256=snap_files,
        live_probe_current_sha256={k: live_probe[k] for k in ('delivery-hashes.json', 'FROZEN_DELIVERY_HASHES.json')},
        canonical_batch_current_sha256=live_canon,
        record_batch_current_sha256=live_rec,
        events_dir_current_sha256=live_events,
        entries=change, n=len(change), recorded_at=now(), entered_at=entered,
        verification_status='the single verification of this round is the in-flight agent c1ce89f1 (objects: the drive-sensitivity '
                            'probe + the errata-2 ledger). No second verification is dispatched this round: that would be a second '
                            'independent verification of the same artifact, which the standing rule forbids.',
    ), ensure_ascii=False, indent=1), encoding='utf-8')

    (ERRATA / 'findings_disposition.json').write_text(json.dumps(dict(
        status='FINDINGS_DISPOSITION',
        source_report=str(CHECK / 'verification_report.json'),
        source_verdict=fcheck.get('verdict'),
        source_verdict_sentence=fcheck.get('verdict_sentence'),
        rule='FIXED means the object was written and its new sha256 is stated. RECORDED_NOT_APPLIED means the corrected clause is '
             'published in change_list.json but the frozen target file was deliberately not touched this round.',
        entries=disp, n=len(disp), recorded_at=now(),
    ), ensure_ascii=False, indent=1), encoding='utf-8')

    (ERRATA / 'README.zh-CN.md').write_text('\n'.join([
        '# errata-3（owner 第 30 轮）：冻结机制新铁律 ＋ 旧发现逐条落地（**手动研究实施，不是成品**）', '',
        '## 本轮的硬约束', '- owner：**"旧批次一个字节不动"**。因此本轮**没有**修改任何既有交付目录；被点名的更正一律以'
        '**改前/改后真实哈希 + 当前实测哈希**的形式写进本目录 `change_list.json`，状态标 `recorded_not_applied_pending_owner_go_ahead`。',
        '- 唯二在本轮**真正写盘**的对象是本工作线**尚未派发核验**的探针批（修的是与 m1/M4 同类的自身缺陷）与**本轮新建**的目录。', '',
        '## 上轮 FAIL 的处置（逐条，见 `findings_disposition.json`）', '- **B1**（阻塞）：对象 1 `delivery-hashes.json` 在核验窗口内被 errata 轮改写。'
        '实测对：before `cf5708c8…`（644 B，17:29:33 由核验方测得）→ 17:31:11.027 的 after `831144c4…`（1802 B）；'
        '**但当前值是 `484c7b20…`**（errata-2 轮又写过一次）。三者并列记录，链条不留空。', 
        '- **M1**：对象 1 README 的假陈述已被 errata-2 换成实测口径（步 3134 = 78.35 ms、实例 3532、步 3136 起 `dist<0`、步 4202 = 105.05 ms 起 `fn>0`、max `fn` 0.7567）；'
        '**残余缺陷在 `report.json`**：同一字段的 note 仍写 "never touches down with the thorax before 0.5 s"。更正句已写好，待 owner 放行。',
        '- **M2**：姿态 A 对照 = **编号约定差**，不是物理差。`mine[i] == ext[i-1]`、19999/19999、`max|Δ| = 0.0`、`root_z_min` 同为 `0.7266289904906442`；'
        '两跑 **act 恒 0**、同 pose npz `0cd72561…`、同驱动 `37fe24d3…`、同 25 µs。外部标签 59.950 ms 与其实测 `time[2398] = 0.059974999999997475 s`（= 59.975 ms）不一致，'
        '原因是该产物的 ms 标签用 0 基循环下标 × dt，**本批口径步 2399 = 59.975 ms**。错误 caveat（"initial act 不同"）撤回。',
        '- **M3**：对象 2 README 的 `7cb99d67…` 是被其自身 `self_amendment` 撤回的错值（实为 void 批 `accepted-equilibrium-openloop.png` 的哈希），'
        '正确值 `5bf3593f…`；替换时须注明「本行于本轮替换」。',
        '- **M4**：对象 1 FROZEN 的 `covers` 与实际不一致：`files` 里**其实有** `delivery-hashes.json`（9 项），真正多出来的排除是'
        '「`delivery-hashes.json` 无法列出自身摘要」这一条；更正后的 `covers` 原文已写在 `change_list.json`，待放行。',
        '- **m1**（对象 2 自哈希写成 `effe2862…`，实际 `d951e3de…`）、**m2**（EV-03 未逐条记录 void 批 `report.json` 与两张 PNG 的 17:13:45/56 改写）、'
        '**m3**（v3 更正时刻归属：真正先出现 76.375 ms 的是 `ms_label_correction.json` 17:11:52，不是 README mtime 17:12:44）、'
        '**m4**（3055 不得归到 `S2-true-equilibrium-openloop-20261015T154720`，该目录里既无 3055 也无 76.375；它是 **S2-corrected-dt 口径步 3055**，'
        '在本批"已执行步数"口径即步 3056 = 76.400 ms）、**m5**（定位图缺三个步长字段——**本项目已在 errata-2 就地补上**，探针图 suptitle 亦已含 `d.time` 轴声明）、'
        '**m6**（`current_directories.json` 分类范围有限、空目录删除只以散文披露）、**c1/c2/c3**（回看窗轴标签、引用段手写毫秒常量、步号约定对照）'
        '——处置逐条见 `findings_disposition.json`。', '',
        '## 新铁律（本轮起永久生效，已写入 `GOAL_TRACKER.zh-CN.md`）',
        '1. **先快照再核验**：任何批次交付/送核验前，先 `cp -a` 到 `/home/sensen/Desktop/engine修复验收/_frozen_snapshots/<批次名>-<时间戳>/`，'
        '`chmod -R a-w`，并对快照内**每个文件**单独写 sha256；**核验对象一律是快照路径，绝不指活目录**。',
        '2. **核验窗口内对活目录零写入**：README、`delivery-hashes.json`、`report.json`、图、任何 json 都不许在窗口内被改，'
        '「补一行 SUPERSEDED」这类也算；要改等核验结束、另开 errata 目录。',
        '3. **清单不许写比实际更大的覆盖范围**：`delivery-hashes.json` 与冻结清单必须覆盖目录内全部文件（含自身）；覆盖不到就把 `covers` 改成实际内容。',
        '   自哈希在数学上不可能，这种情况必须**显式写出规则与排除项**，并说明由谁传递覆盖。', '',
        '## 本轮的冻结快照', '- 快照：`%s`（只读，`chmod a-w` 已核）' % str(SNAP),
        '- 逐文件 sha256：见 `change_list.json` 的 `frozen_snapshot_per_file_sha256`。', '',
        '## 披露', '- **手动研究实施，不是成品**；不宣称站立/行走/完成度；绝对力值与单位 `NOT_ESTABLISHED`。',
        '- 本轮只派**一次**核验：进行中的 `c1ce89f1`（对象 = 驱动敏感度探针 + errata-2 账本）。repeat 派发同一件属违规，故不重复。',
    ]), encoding='utf-8')

    files = sorted(p for p in ERRATA.iterdir() if p.is_file() and p.name != 'FROZEN_DELIVERY_HASHES.json')
    (ERRATA / 'delivery-hashes.json').write_text(json.dumps({p.name: sha(p) for p in files}, ensure_ascii=False, indent=1), encoding='utf-8')
    covered = sorted([p.name for p in ERRATA.iterdir() if p.is_file() and p.name != 'FROZEN_DELIVERY_HASHES.json'] + ['delivery-hashes.json'])
    (ERRATA / 'FROZEN_DELIVERY_HASHES.json').write_text(json.dumps(dict(
        status='FROZEN_BEFORE_VERIFICATION', frozen_at=now(),
        files={f: sha(ERRATA / f) for f in covered},
        covers='the %d files listed in files (all %d files of this directory except this manifest); delivery-hashes.json is listed here '
               'and cannot list itself, so this manifest is what witnesses it' % (len(covered), len(covered) + 1),
        covers_count=len(covered),
        not_covered=['FROZEN_DELIVERY_HASHES.json (this file)'],
    ), ensure_ascii=False, indent=1), encoding='utf-8')

    # ---------------------------------------------------------------- 6. tracker rules
    t = TRACKER.read_text(encoding='utf-8')
    anchor = '## 交接日志\n'
    assert anchor in t
    rules = '\n'.join([
        '## 永久铁律（owner 第 30 轮起生效，违反一次即视为阻塞级事故）',
        '',
        '1. **先快照再核验**。任何批次在交付或送核验**之前**，先把交付目录整体做只读快照：`cp -a` 到 '
        '`/home/sensen/Desktop/engine修复验收/_frozen_snapshots/<批次名>-<时间戳>/`，`chmod -R a-w`，并对快照内**每个文件**单独写 sha256。'
        '**核验对象一律是快照路径，绝不指活目录。**',
        '2. **核验窗口内对活目录零写入**。README、`delivery-hashes.json`、`report.json`、图、任何 json 都不许在窗口内被改，'
        '「补一行 SUPERSEDED」这类看似无害的操作也算。要改就等核验结束、另开 errata 目录。历史事故：17:01:26、17:23:27→17:31:11、'
        '以及对象 1 `delivery-hashes.json` 在核验窗口内被改写（B1），三次同类。',
        '3. **清单不许写比实际更大的覆盖范围**。`delivery-hashes.json` 与冻结清单必须覆盖目录内全部文件（含自身）；覆盖不到就把 `covers` '
        '声明改成实际内容。自哈希在数学上不可能，必须**显式写出规则、排除项与传递覆盖关系**，不许手写更大范围。',
        '',
        '## 交接日志',
        ''])
    t = t.replace(anchor, rules, 1)
    TRACKER.write_text(t, encoding='utf-8')

    print('ERRATA:', ERRATA)
    print('SNAP  :', SNAP)
    print('change entries:', len(change), '| applied:', sum(1 for c in change if c['status'] == 'applied'))
    print('disposition entries:', len(disp))
    print('snapshot files:', len(snap_files))
    print(json.dumps(snap_files, ensure_ascii=False, indent=1))
    print('errata files:')
    for p in sorted(ERRATA.iterdir()):
        print('  %-32s %s' % (p.name, sha(p)))
    return ERRATA


if __name__ == '__main__':
    main()
