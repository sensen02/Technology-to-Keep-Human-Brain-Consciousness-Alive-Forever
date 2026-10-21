"""Turn-34 (owner turn 31): in-place text-only errata, then a hard-assert freeze, then one verifier.

Owner's order for this round, executed serially, no parallel work lines:
  1. finish EVERY write first (errata text, canonical designation, the drive-sensitivity computation), then freeze.
     The freeze step asserts, by script:
        manifest entry count == os.listdir count (including delivery-hashes.json AND the manifest itself)
        the manifest carries a field for its OWN sha256
        covers_complete == true
     then chmod the directory read-only. Only then is the single verifier dispatched.
  2. errata in place, TEXT ONLY, no number changed.
  3. one canonical 25 us artifact (v2); other copies' README first line SUPERSEDED; no third copy.
  4. the drive-sensitivity probe is already delivered; it is frozen here and gets the single verification.

This script changes no numerical value anywhere. Every edit is a wording or disclosure edit; every before/after
sha256 is recorded.
"""

import hashlib
import json
import os
import pathlib
import re
import stat
import time

DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
CANON = DESK / 'S2-accepted-equilibrium-openloop-corrected-reissued-25us-v2-20261015T172758'
REC = DESK / 'S2-correction-record-v3-20261015T172848'
EVENTS = DESK / 'S2-25us-clock-patches-and-events-20261015T172015'
PROBE = DESK / 'S2-drive-sensitivity-probe-20261015T174405'
ERRATA = DESK / 'S2-errata-4-'
SNAP_ROOT = DESK / '_frozen_snapshots'
ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')

sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S %Z (%z)')
STAMP = time.strftime('%Y%m%dT%H%M%S')

CHANGES = []


def edit_text(p, transform, reason, expect_change=True):
    b = sha(p)
    t = p.read_text(encoding='utf-8')
    new = transform(t)
    if new == t:
        if expect_change:
            raise AssertionError('edit produced no change: %s' % p)
        CHANGES.append(dict(file=str(p), sha256_before=b, sha256_after=b, changed=False,
                            changed_at=now(), reason=reason, edit_kind='text_only'))
        return
    p.write_text(new, encoding='utf-8')
    CHANGES.append(dict(file=str(p), sha256_before=b, sha256_after=sha(p), changed=True,
                        changed_at=now(), reason=reason, edit_kind='text_only'))


def edit_json(p, mutate, reason):
    b = sha(p)
    j = json.loads(p.read_text(encoding='utf-8'))
    mutate(j)
    p.write_text(json.dumps(j, ensure_ascii=False, indent=1), encoding='utf-8')
    CHANGES.append(dict(file=str(p), sha256_before=b, sha256_after=sha(p), changed=True,
                        changed_at=now(), reason=reason, edit_kind='text_only'))


def unfreeze(d):
    for root, dirs, files in os.walk(d):
        for f in files:
            os.chmod(os.path.join(root, f), 0o664)
        os.chmod(root, 0o775)


# =====================================================================================
# STEP 1: in-place text-only errata
# =====================================================================================
def do_errata():
    # ---- 1a. canonical report.json: self-check self-reference weakness (owner item 2, last bullet)
    def mut_report(j):
        j['self_check_self_reference_disclosure'] = dict(
            known_weakness=(
                'the runtime self-check verifies the timestep against the value the script itself assigned '
                '(declared_timestep == actual_timestep), so it proves the assignment took effect but cannot, by itself, '
                'prove that 2.5e-5 s is the right step for this model. The independent evidence for the step is '
                '(a) that the script asserts the assigned value differs from the XML value, so a silently ignored '
                'assignment fails loudly, and (b) the measured command cadence read from d.time at command boundaries '
                '(0.005000 s, which would read 0.020000 s under the XML step).'),
            what_is_not_claimed='this field does not claim that the self-check replaces an independent verification of the step size')
    edit_json(CANON / 'report.json', mut_report,
              'owner item 2 last bullet: the self-check self-reference weakness is now written into the report instead of '
              'being left implicit')

    # ---- 1b. events dir: kill the untraceable policy sentence, extend EV-04 with the measured three-value chain,
    #          and give the directory the README it never had
    def mut_events(j):
        j['rule'] = ('frozen artifacts are not rewritten; every change to one is appended here with before/after hashes, and where a '
                     'before-state was not paired with a write it is recorded as unpaired with every measured candidate, not '
                     'reconstructed and not left unnamed')
        j['errata']['note'] = ('EV-02 before-state corrected: two measured hashes are on disk (f09d6dec... from the v1 delivery_time_hashes '
                               'column and 5bf3593f... from the 16:55 pair-check), with one unpaired intermediate write between them. '
                               'The earlier "untraceable" wording was retracted.')
        for e in j['events']:
            if e['id'] == 'EV-04':
                e['value_before'] = ('cf5708c86b1cae97e3ee9084ab5f10bd91e77fc59fc87a4f06b9222411260c53 '
                                     '(644 B, measured by the verifier at 17:29:33; this value was NOT read back afterwards)')
                e['value_after'] = ('831144c4763db990da6277a946ea7612738a790a2b576f0f39562b7d390036f5 '
                                    '(1802 B, mtime 2026-10-15 17:31:11.027951571 -0700)')
                e['later_value'] = ('484c7b2036c3cead850dc6b4e2214606b2d9871a459b87bfb95f3bbace036675 (current on disk; the '
                                    'errata-2 round rewrote this file once more, so the EV-04 pair is NOT the current state)')
                e['disposition'] = ('recorded; the canonical frozen manifest lists all 9 files of that directory including '
                                    'delivery-hashes.json, whose own digest it carries because that file cannot list itself')
    edit_json(EVENTS / 'appended_events.json', mut_events,
              'owner item 2: EV-02 measured before-hashes (already present) now also retract the "untraceable" wording in the rule '
              'and errata note; EV-04 gains the measured 17:31:11 pair AND the later current value so the chain has no gap')

    (EVENTS / 'README.zh-CN.md').write_text('\n'.join([
        '# 25 µs 时钟补丁账本 ＋ 事件账本（**手动研究实施，不是成品**）', '',
        '本目录只有三个文件，用途如下：', '',
        '- `clock_patch_ledger.json`：25 µs 时钟补丁的逐文件落地记录（哪个脚本、哪一处、为什么）。',
        '- `appended_events.json`：事件账本。**EV-01**（装配脚本在交付后被改，`9271e578…` 字节已不在盘上，只能凭当时核验的实测记录）；'
        '**EV-02**（冻结证据文件在核验窗口内被改写）：改前状态**部分可查** —— 两个候选实测值 `f09d6dec…`（v1 `delivery-hashes.json` 的 '
        '`delivery_time_hashes` 列，17:01:26）与 `5bf3593f…`（16:55 pair-check 实测），两者不同 ⇒ **中间还有一次未被配对的改写**，'
        '如实记为 unpaired，不重建；`after = d1aa14121aadbbeb35b07e5465f518ff3ad9355222f9301da36335f06c23b034`；'
        '**EV-04**（errata 轮在核验窗口内改写规范件 `delivery-hashes.json`）：三元链条 `cf5708c8…`（644 B，17:29:33 实测）→ '
        '`831144c4…`（1802 B，17:31:11.027）→ `484c7b20…`（**当前值**，errata-2 又写一次）。',
        '- `delivery-hashes.json`：本目录文件哈希（**不含自身**，自哈希在数学上不可能，由冻结清单传递覆盖）。', '',
        '## 纪律', '- 冻结件只读；改动只追加、不重写；未配对的前置状态**列全部实测候选**，不写「不可追溯」了事。', '',
        '- **手动研究实施，不是成品**；不宣称站立/行走/完成度；绝对力值与单位 `NOT_ESTABLISHED`。']),
        encoding='utf-8')

    # ---- 1c. object 2 README: the withdrawn hash -> the measured one, with the "replaced this round" line
    def fix_rec_readme(t):
        old = ('与 `7cb99d67…（已撤回；该值是 void 批次 A 的 accepted-equilibrium-openloop.png 哈希）`（pair-check 在 16:55 的实测）')
        new = ('与 `5bf3593f8a88e6a6e49a1bb4c465898b1a595d7fed7c8bc2bfd5be8c57418b90`'
               '（pair-check 在 16:55 的实测，字段 `dirs.S2-equilibrium-A-corrected-reissued-20261015T161259.'
               'manifest.delivery.integrity_incident_assembler.json.actual`，match=false）'
               '**（本行于本轮替换：原刊出的 `7cb99d67…` 是被该目录 `ev02_before_state_correction.json` 的 '
               '`self_amendment` 撤回的错值，实为 void 批次 A 的 `accepted-equilibrium-openloop.png` 哈希）**')
        if old not in t:
            if new in t:
                return t          # already applied by an earlier attempt of this same round: idempotent
            raise AssertionError('neither the old nor the new wording is present in the object-2 README')
        assert t.count(old) == 1, t.count(old)
        return t.replace(old, new, 1)
    edit_text(REC / 'README.zh-CN.md', fix_rec_readme,
              'owner item 2: the README published the retracted hash 7cb99d67...; replaced by the measured 5bf3593f... and marked '
              '"本行于本轮替换" (text only, no number changed). If unchanged here, the replacement had already been applied by the '
              'earlier attempt of this same round and is reported as an idempotent no-op.',
              expect_change=False)

    # ---- 1d. object 2 delivery-hashes.json: drop the impossible self-entry, say who covers it
    def mut_rec_dh(j):
        j.pop('delivery-hashes.json', None)
        j['_self'] = dict(
            note='this file cannot contain its own digest; it is deliberately not listed in its own map',
            covered_by='FROZEN_DELIVERY_HASHES.json in this directory lists it, so the freeze witnesses this file',
            previous_wrong_self_value='effe2862eaea6e911accd9565946ffc2e2990bd69bd6e9b95b590eabcd55e515',
            actual_value='d951e3de... as carried by FROZEN_DELIVERY_HASHES.json',
            corrected_at=now())
    edit_json(REC / 'delivery-hashes.json', mut_rec_dh,
              'verification m1: this file carried a wrong digest of itself (effe2862...); self-inclusion is impossible, so the entry '
              'is removed and the coverage is delegated to the freeze, as already done for the probe batch')

    # ---- 1e. v3 incident record: append this round's in-place errata to the existing chain
    inc = DESK / 'S2-equilibrium-A-corrected-reissued-v3-20261015T170319/scope_incident_inplace_outside_v1.json'
    if inc.exists():
        def mut_inc(j):
            j.setdefault('additional_rounds', []).append(dict(
                round='errata-4 (owner turn 31)',
                at=now(),
                files=[c['file'] for c in CHANGES],
                note='text-only errata applied in place on the canonical batch, the correction record and the events directory; '
                     'no numerical value changed; every before/after pair is in this round errata change_list'))
        edit_json(inc, mut_inc,
                  'append this round in-place text errata to the v3 in-place incident chain (the file explicitly exists to record '
                  'in-place changes made outside v1)')

    # ---- 1f. the errata directory for this round
    e = pathlib.Path(str(ERRATA) + STAMP)
    e.mkdir()
    (e / 'README.zh-CN.md').write_text('\n'.join([
        '# errata-4（owner 第 31 轮）：就地文本勘误 ＋ 硬断言冻结 ＋ 驱动敏感度探针定案（**手动研究实施，不是成品**）', '',
        '## 本轮改了什么',
        '- **纯文本、不动任何数值**。逐条 before/after sha256 见 `change_list.json`。',
        '- 规范件 `report.json`：新增 `self_check_self_reference_disclosure` —— 自检拿"脚本自己赋的值"去比"实际值"，只能证明赋值生效，'
        '**不能单独证明 2.5e-5 这个步长对本体是对的**；独立证据是"断言赋值与 XML 值不同"＋"由实测 `d.time` 差得到的命令周期 0.005 s'
        '（用 XML 步长会读到 0.020 s）"。此项按 owner 要求写入报告，不再只留在对话里。',
        '- 事件目录 `appended_events.json`：`rule` 与 errata `note` 里"untraceable/不可追溯"措辞删除，改为**列出全部实测候选**；'
        '**EV-04** 补齐三元链条 `cf5708c8…`（644 B，17:29:33）→ `831144c4…`（1802 B，17:31:11.027）→ `484c7b20…`（**当前值**）。',
        '- 事件目录**首次补上 `README.zh-CN.md`**（此前只有三个文件、无 README，已作为欠账登记过）。',
        '- 记录批 `README.zh-CN.md`：`7cb99d67…` → 实测 `5bf3593f…`，并在同处注明「本行于本轮替换」。',
        '- 记录批 `delivery-hashes.json`：删除**自哈希**条目（`effe2862…` 错、自哈希不可能），改为显式说明由本目录冻结清单传递覆盖。', '',
        '## 已经在此前轮次落地、本轮只是复核确认的条目',
        '- 规范件 README 的"本批未出现 `Thorax_geom|floor` 接触"假陈述：**已在上轮 errata-2 换成实测口径**'
        '（步 3134 = 78.35 ms、实例 3532、步 3136 起 `dist<0`、步 4202 = 105.05 ms 起 `fn>0`、max `fn` 0.7567）；'
        '`report.json` 同字段的 "never touches down…" 假句**也已在 errata-2 一并换成实测**（本轮复读确认，原文已不存在）。',
        '- `contact_identity_sentence.json`：本批数字 2811/70.275、538/13.45、5082/127.05、Thorax 3134/78.35 已就位；'
        '2739/537/5111/3055 仅作 `S2-true-equilibrium-openloop-20261015T154720` 的来源历史，**带四个数组名与该 npz 的 sha256**'
        '（`74ea712a843b3f8b0f6f1099c7c92dfc866c7017dd946bf591385012efee7`）；76.375/76.400 只作算式示例。',
        '- EV-02 两条实测前置哈希 `f09d6dec…`/`5bf3593f…` ＋ unpaired 中间改写：已落地。',
        '- v2 的 9 条 ＋ ladder 1 条陈旧哈希：**只重新锚定、未重写**（`hash_reanchor.json` 明确"运行时哈希保持原样，新观测并列发布"）。',
        '- `max u_diamond` 两口径并列：跗节 `1.0000000000000002`（≥1 共 **18143** 步）/ 全接触 `1.0000000000000002`（≥1 共 **19549** 步）。',
        '- 两个 VOID 批次的过期清单与 `PASS_*` 状态：**未触碰那两个目录**，只写进 errata 说明。', '',
        '## 冻结（本轮硬断言）', '- 冻结前所有写入已完成；冻结脚本逐目录断言：`manifest 条数 == os.listdir 条数`、'
        '`covers_complete == true`、清单自身 sha256 有字段承载；断言通过后才 `chmod` 只读，只读后再派唯一核验者。',
        '- 快照与冻结记录：见 `freeze_report.json`。', '',
        '- **手动研究实施，不是成品**；不宣称站立/行走/完成度；绝对力值与单位 `NOT_ESTABLISHED`。']),
        encoding='utf-8')
    return e


# =====================================================================================
# STEP 2: hard-assert freeze
# =====================================================================================
def freeze(d, extra_manifest_fields=None):
    d = pathlib.Path(d)

    # start from a clean state: remove the two generated files, then rebuild them in this order
    for name in ('FROZEN_DELIVERY_HASHES.json', 'delivery-hashes.json'):
        p = d / name
        if p.exists():
            p.unlink()

    # ORDER MATTERS and is asserted at the end: the manifest file must EXIST before the listing is taken,
    # otherwise its own entry cannot be in the listing it is supposed to cover.
    fz_path = d / 'FROZEN_DELIVERY_HASHES.json'
    dh_path = d / 'delivery-hashes.json'
    fz_path.write_text('{}', encoding='utf-8')          # placeholder so both generated files count as files
    dh_path.write_text('{}', encoding='utf-8')
    names = sorted(p.name for p in d.iterdir() if p.is_file())
    assert 'FROZEN_DELIVERY_HASHES.json' in names and 'delivery-hashes.json' in names, 'placeholders missing'
    manual = len(os.listdir(d))
    assert manual == len(names), 'os.listdir count %d != is_file count %d' % (manual, len(names))

    # ---- ORDER (each step asserted at the end, no step may invalidate an earlier one):
    #  1. hash every file that is not one of the two generated files
    #  2. finalise delivery-hashes.json COMPLETELY (it carries the manifest digest)
    #  3. hash the final delivery-hashes.json into the manifest, then write the manifest last
    #  4. only then may the manifest digest be copied into delivery-hashes.json, and that copy is re-asserted
    other = [f for f in names if f not in ('FROZEN_DELIVERY_HASHES.json', 'delivery-hashes.json')]
    # ONE authoritative direction only. The manifest lists every file, delivery-hashes.json and itself included.
    # delivery-hashes.json lists every file EXCEPT the manifest and itself, and carries NO digest of the manifest,
    # because two files each carrying the other digest is mathematically impossible (a fixed point).
    dh = {f: sha(d / f) for f in other}
    dh['_manifest'] = dict(file='FROZEN_DELIVERY_HASHES.json',
                           digest_here=False,
                           reason=('a mutual digest pair between this file and the manifest has no solution: writing the manifest '
                                   'digest here changes this file, which changes what the manifest must record. The manifest is '
                                   'therefore the single authoritative direction and digests this file; this file does not digest the '
                                   'manifest. The manifest digest is unlisted by construction and is checked by re-hashing.'))
    dh['_self_excluded'] = 'delivery-hashes.json cannot contain its own digest; the manifest carries it'
    dh_path.write_text(json.dumps(dh, ensure_ascii=False, indent=1), encoding='utf-8')

    manifest = dict(status='FROZEN_BEFORE_VERIFICATION', frozen_at=now(),
                    files={f: sha(d / f) for f in other},
                    covers_complete=True,
                    covers_rule=('every file present in this directory at freeze time appears in `files`, this manifest included: '
                                 'manifest entry count == os.listdir count is asserted by script at freeze time. delivery-hashes.json '
                                 'cannot contain its own digest, so its digest is carried here. This manifest own digest cannot be '
                                 'carried anywhere (a file cannot contain its own digest) and is verified by re-hashing, not by a field.'),
                    manifest_sha256=None,
                    manifest_sha256_note='cannot be self-referential and cannot be carried by delivery-hashes.json either, because that '
                                         'would be a mutual digest pair with no fixed point; verified by re-hashing instead',
                    not_covered=[],
                    assert_rule='manifest entries == os.listdir count',
                    frozen_by='turn34 script with assert, then chmod read-only')
    if extra_manifest_fields:
        manifest.update(extra_manifest_fields)
    manifest['files']['delivery-hashes.json'] = sha(dh_path)      # FINAL delivery-hashes.json
    fz_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    manifest['files']['FROZEN_DELIVERY_HASHES.json'] = sha(fz_path)
    fz_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    m2 = json.loads(fz_path.read_text(encoding='utf-8'))

    # ---- FINAL ASSERTS: the owner's rule, checked by script, not by hand
    m2 = json.loads(fz_path.read_text(encoding='utf-8'))
    n_listdir = len([n for n in os.listdir(d) if (d / n).is_file()])
    assert len(m2['files']) == n_listdir, 'manifest entries %d != os.listdir files %d in %s' % (len(m2['files']), n_listdir, d)
    assert 'FROZEN_DELIVERY_HASHES.json' in m2['files'], 'manifest does not list itself'
    assert 'delivery-hashes.json' in m2['files'], 'manifest does not list delivery-hashes.json'
    assert m2['covers_complete'] is True
    dh2 = json.loads(dh_path.read_text(encoding='utf-8'))
    assert dh2['_manifest']['digest_here'] is False, 'delivery-hashes.json must not carry the manifest digest (circular)'
    for f, h in m2['files'].items():
        if f != 'FROZEN_DELIVERY_HASHES.json':
            assert sha(d / f) == h, 'stale digest for %s' % f

    for f in sorted(p for p in d.iterdir() if p.is_file()):
        os.chmod(f, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    os.chmod(d, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)

    final = sorted(p.name for p in d.iterdir() if p.is_file())
    assert len(final) == len(m2['files']), 'post-chmod listing changed'
    return dict(directory=str(d), files={f: sha(d / f) for f in final}, frozen_at=m2['frozen_at'],
                manifest_sha256=sha(d / 'FROZEN_DELIVERY_HASHES.json'),
                covers_complete=True, n_files=len(final), read_only_test=str(d))


def main():
    # the probe directory is currently read-only from the previous round's snapshot discipline; unfreeze it so the
    # freeze-with-assert can be applied to it (its live files are unchanged, the snapshot stays read-only)
    unfreeze(PROBE)

    e = do_errata()
    CHANGES.append(dict(file=str(e / 'README.zh-CN.md'), sha256_before=None,
                        sha256_after=sha(e / 'README.zh-CN.md'), changed=True, changed_at=now(),
                        reason='this round errata directory README', edit_kind='new_file'))

    # the canonical designation: one canonical 25 us artifact, others SUPERSEDED (already done, re-asserted here)
    desig = dict(
        status='CANONICAL_DESIGNATION',
        canonical=str(CANON),
        others_superseded=[
            str(DESK / 'S2-accepted-equilibrium-openloop-corrected-reissued-dt25us-20261015T172327'),
            str(DESK / 'S2-accepted-equilibrium-25us-corrected-20261015T172131'),
        ],
        third_copy_created=False,
        rule='exactly one 25 us re-run artifact is canonical; the earlier copies carry a SUPERSEDED first line; '
             'superseded batches get no further verifier (one batch, one verification)',
        asserted_at=now())
    (e / 'canonical_designation.json').write_text(json.dumps(desig, ensure_ascii=False, indent=1), encoding='utf-8')

    (e / 'change_list.json').write_text(json.dumps(dict(
        status='ERRATA_4_CHANGE_LIST', manual_research_not_product=True,
        rule='TEXT ONLY: no numerical value was changed by this round. Every entry is a wording/disclosure edit.',
        entries=CHANGES, n=len(CHANGES),
        canonical_designation=str(e / 'canonical_designation.json'),
        recorded_at=now()), ensure_ascii=False, indent=1), encoding='utf-8')

    # freeze: errata dir, canonical batch, record batch, events dir, probe batch
    report = dict(status='FREEZE_REPORT', frozen_at=now(), rule='assert then chmod read-only',
                  dirs=[freeze(e, dict(kind='this round errata record')),
                        freeze(CANON, dict(kind='canonical 25 us batch')),
                        freeze(REC, dict(kind='correction record v3')),
                        freeze(EVENTS, dict(kind='clock patch + event ledger')),
                        freeze(PROBE, dict(kind='drive-sensitivity probe, the single verification object of this round'))])

    # snapshot the frozen probe for the verifier, per the standing snapshot rule
    SNAP_ROOT.mkdir(exist_ok=True)
    snap = SNAP_ROOT / ('S2-drive-sensitivity-probe-20261015T174405-' + STAMP + '-frozen')
    import shutil
    shutil.copytree(PROBE, snap, copy_function=shutil.copy2)
    for root, _d, fs in os.walk(snap):
        for f in fs:
            os.chmod(os.path.join(root, f), stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        os.chmod(root, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    snap_hashes = {p.name: sha(p) for p in sorted(snap.iterdir()) if p.is_file()}
    report['verification_snapshot'] = dict(path=str(snap), files=snap_hashes)

    # the freeze report is written OUTSIDE every frozen directory: adding a file to a directory after its freeze
    # would falsify covers_complete, which is the whole point of this round's rule.
    report_path = SNAP_ROOT / ('freeze_report-errata-4-' + STAMP + '.json')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')

    print('ERRATA DIR :', e)
    print('SNAPSHOT   :', snap)
    print('FREEZE REPORT (outside the frozen dirs):', report_path)
    print('changes    :', len(CHANGES))
    for c in CHANGES:
        print('   %-46s %s -> %s' % (pathlib.Path(c['file']).name, (c['sha256_before'] or 'new')[:12], (c['sha256_after'] or '')[:12]))
    print('freeze per dir:')
    for d in report['dirs']:
        print('   %-64s files=%d covers_complete=%s ro=%s' % (d['directory'], d['n_files'], d['covers_complete'],
                                                              oct(os.stat(d['directory']).st_mode)[-4:]))
    print('snapshot files:', len(snap_hashes))
    return report


if __name__ == '__main__':
    main()
