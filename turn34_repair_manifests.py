"""Turn-34 repair (verifier alert from S2-probe-snapshot-check-20261016T005207Z).

The verifier reported two defects, both mine:

  D1  the new manifests list THEMSELVES in `files` with a digest that cannot be valid (writing the manifest
      changes it). The count assertion passed, so my own assert was satisfied by a lie. This is exactly the
      m1/M4 defect class I was supposed to close.
      FIX: `files` lists every file EXCEPT the manifest. The manifest records a machine-checkable invariant
      (len(files) + 1 == the directory file count, the manifest being the +1), and states plainly that no
      manifest can contain its own digest and that its digest is verified by re-hashing.
  D2  `report.json` of the canonical batch still carried the old blanket reading (contradicting its own README,
      which the earlier errata had corrected). The verifier also showed my errata-3 README had FALSELY claimed
      that report.json's wording was already fixed.
      FIX: report.json's answer.reading is replaced by the measured reading; the false claim is corrected in
      errata-4's README and recorded here.

Also recorded: errata-3's change_list "current sha256" blocks were stale (the verifier measured 9 mismatches),
because those directories were rewritten after errata-3 froze. Those blocks are superseded here by this
round's post-repair digests.
"""

import hashlib
import json
import os
import pathlib
import stat
import time

DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
CANON = DESK / 'S2-accepted-equilibrium-openloop-corrected-reissued-25us-v2-20261015T172758'
REC = DESK / 'S2-correction-record-v3-20261015T172848'
EVENTS = DESK / 'S2-25us-clock-patches-and-events-20261015T172015'
PROBE = DESK / 'S2-drive-sensitivity-probe-20261015T174405'
ERRATA3 = DESK / 'S2-errata-3-20261015T175143'
ERRATA4 = sorted(DESK.glob('S2-errata-4-*'))[-1]
SNAP_ROOT = DESK / '_frozen_snapshots'
EXTRA = [DESK / 'S2-accepted-equilibrium-25us-corrected-20261015T172131',
         DESK / 'S2-accepted-equilibrium-openloop-corrected-reissued-25us-v2-20261015T172533']

sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S %Z (%z)')

READING_OLD = ('the first non-foot contact happens in every group inside a narrow band around the reference step, so the near-term '
               'collapse is decided by the archive-plus-state pair rather than by a delicate amplitude or phase condition; the numbers '
               'above state the band explicitly. The root z minimum and the u_diamond peak are reported per group as well.')
READING_NEW = ('scaling the whole archive by 0.95 / 1.05 moves the first non-foot contact by only 20 / 21 steps (0.50 / 0.525 ms), while '
               'misaligning it by a single frame (5 ms, the archive cadence) moves it by 557 / 328 steps: the ~70 ms collapse is fixed by '
               'the archive-plus-state pair, and within that pair it is far more sensitive to frame alignment than to fine drive amplitude. '
               'max|qacc| does not track the first-event step in the same direction (the -1 frame group has the smallest max|qacc| and the '
               'earliest event), so amplitude is not read as the cause of the timing. The root z minimum and the u_diamond peak are '
               'reported per group as well.')


def unfreeze(d):
    for root, _dirs, files in os.walk(d):
        for f in files:
            os.chmod(os.path.join(root, f), 0o664)
        os.chmod(root, 0o775)


def freeze(d, extra=None):
    """The repaired freeze: the manifest lists everything except itself, and records the invariant instead of a lie."""
    d = pathlib.Path(d)
    for name in ('FROZEN_DELIVERY_HASHES.json', 'delivery-hashes.json'):
        p = d / name
        if p.exists():
            p.unlink()
    fz_path = d / 'FROZEN_DELIVERY_HASHES.json'
    dh_path = d / 'delivery-hashes.json'
    fz_path.write_text('{}', encoding='utf-8')
    dh_path.write_text('{}', encoding='utf-8')
    names = sorted(p.name for p in d.iterdir() if p.is_file())
    total = len(os.listdir(d))
    assert total == len(names) and 'FROZEN_DELIVERY_HASHES.json' in names and 'delivery-hashes.json' in names, d

    other = [f for f in names if f not in ('FROZEN_DELIVERY_HASHES.json', 'delivery-hashes.json')]

    # delivery-hashes.json: every file except itself and except the manifest; no digest of the manifest (no fixed point exists)
    dh = {f: sha(d / f) for f in other}
    dh['_self_excluded'] = ('delivery-hashes.json cannot contain its own digest: writing it would change the file being digested. Its '
                            'digest is carried by FROZEN_DELIVERY_HASHES.json in this directory.')
    dh['_manifest'] = dict(file='FROZEN_DELIVERY_HASHES.json', digest_here=False,
                           reason='a mutual digest pair between these two files has no fixed point: writing the manifest digest here '
                                  'changes this file, which changes the digest the manifest must carry. One direction only.')
    dh_path.write_text(json.dumps(dh, ensure_ascii=False, indent=1), encoding='utf-8')

    manifest = dict(
        status='FROZEN_BEFORE_VERIFICATION', frozen_at=now(),
        files={f: sha(d / f) for f in other},
        covers_complete=True,
        covers_rule=('`files` lists every file in this directory EXCEPT this manifest itself, and includes delivery-hashes.json. '
                     'The invariant below is asserted by script at freeze time. This manifest cannot list itself: a file cannot contain '
                     'a correct digest of itself, because writing the digest changes it. This manifest own digest is therefore unlisted '
                     'by construction and is checked by re-hashing the file.'),
        invariant=dict(directory_file_count=total, listed_in_files=len(other) + 2,  # + delivery-hashes.json + this manifest
                       rule='len(files) + 1 (this manifest) == directory file count', holds=True,
                       note='len(files) below counts delivery-hashes.json but not this manifest, so len(files) + 1 == count'),
        manifest_sha256=None,
        manifest_sha256_note=('null by necessity, not by omission: carrying its own digest is impossible, and carrying it in '
                              'delivery-hashes.json would create a mutual pair with no fixed point. Verify by re-hashing.'),
        not_covered=['FROZEN_DELIVERY_HASHES.json (this file) only'],
        assert_rule='len(files) + 1 == os.listdir count; every listed digest == bytes on disk',
        frozen_by='turn34 repair script with assert, then chmod read-only',
        repair_note=('this manifest was rebuilt after a verification showed the previous revision listed itself with a digest that could '
                     'not be valid. The previous revision and its wrong self-entry are recorded in this round errata change_list.'),
    )
    if extra:
        manifest.update(extra)
    manifest['files']['delivery-hashes.json'] = sha(dh_path)
    fz_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')

    # ---- asserts: the invariant, not a self-digest
    m = json.loads(fz_path.read_text(encoding='utf-8'))
    n = len([x for x in os.listdir(d) if (d / x).is_file()])
    assert 'FROZEN_DELIVERY_HASHES.json' not in m['files'], 'manifest must not list itself'
    assert 'delivery-hashes.json' in m['files'], 'manifest must list delivery-hashes.json'
    assert len(m['files']) + 1 == n, 'len(files)+1 = %d != directory count %d in %s' % (len(m['files']) + 1, n, d)
    assert m['invariant']['directory_file_count'] == n
    assert m['covers_complete'] is True
    for f, h in m['files'].items():
        assert sha(d / f) == h, 'stale digest for %s in %s' % (f, d)
    dh2 = json.loads(dh_path.read_text(encoding='utf-8'))
    assert dh2['_manifest']['digest_here'] is False
    assert 'FROZEN_DELIVERY_HASHES.json' not in dh2

    for f in sorted(p for p in d.iterdir() if p.is_file()):
        os.chmod(f, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    os.chmod(d, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    return dict(directory=str(d), n_files=len([x for x in os.listdir(d) if (d / x).is_file()]),
                covers_complete=True, invariant_holds=True, manifest_sha256=sha(fz_path),
                files={f: sha(d / f) for f in sorted(x for x in os.listdir(d) if (d / x).is_file())})


def main():
    changes = []
    def rec(f, b, a, reason, kind='text_only'):
        changes.append(dict(file=str(f), sha256_before=b, sha256_after=a, changed=b != a,
                            changed_at=now(), reason=reason, edit_kind=kind))

    # capture the WRONG self-entries before rebuilding
    wrong = {}
    for d in [PROBE, CANON] + EXTRA:
        p = d / 'FROZEN_DELIVERY_HASHES.json'
        if p.exists():
            m = json.loads(p.read_text(encoding='utf-8'))
            wrong[d.name] = dict(recorded_self_digest=m.get('files', {}).get('FROZEN_DELIVERY_HASHES.json'),
                                 actual_before_rebuild=sha(p))

    # ---- D2: THE PROBE BATCH's report.json still carries the blanket reading.
    # The verifier's wording was "report.json of the frozen batch"; I first looked in the canonical batch, then located it
    # precisely: it is /S2-drive-sensitivity-probe-.../report.json answer.reading, fixed in that batch's README by an earlier
    # errata while this field was missed, so the two files of the same frozen batch stated opposite conclusions.
    unfreeze(PROBE)
    rp = PROBE / 'report.json'
    b = sha(rp)
    j = json.loads(rp.read_text(encoding='utf-8'))
    assert j['answer']['reading'] == READING_OLD, 'the old blanket reading is not where it was located'
    j['answer']['reading'] = READING_NEW
    j['answer']['reading_corrected_at'] = now()
    j['answer']['reading_correction_note'] = ('this field previously carried the blanket "rather than by a delicate amplitude or phase '
                                             'condition" wording, which contradicted this batch own deltas (-557/+328 steps for a one-frame '
                                             'shift versus -21/+20 for +/-5% amplitude). Corrected here; README.zh-CN.md of this same batch '
                                             'had been corrected by an earlier errata while this field was missed, so the two files of one '
                                             'frozen batch stated opposite conclusions.')
    rp.write_text(json.dumps(j, ensure_ascii=False, indent=1), encoding='utf-8')
    rec(rp, b, sha(rp), 'D2: the probe report.json still carried the blanket reading contradicted by its own deltas; replaced by the measured '
                        'reading so that report.json and README.zh-CN.md of the same frozen batch agree')

    # ---- D1: rebuild every frozen manifest with the honest invariant
    rebuilt = []
    for d in [PROBE, CANON, REC, EVENTS, ERRATA4, ERRATA3] + EXTRA:
        if not d.exists():
            continue
        unfreeze(d)
        info = freeze(d, dict(kind='frozen under the repaired rule: no self-digest, invariant recorded'))
        rebuilt.append(info)
        rec(d / 'FROZEN_DELIVERY_HASHES.json', wrong.get(d.name, {}).get('actual_before_rebuild'), info['manifest_sha256'],
            'D1: rebuilt. The previous revision listed this manifest itself in `files` with the digest %s, which cannot be valid because '
            'writing the manifest changes it (the verifier measured the on-disk value and they differed). The repaired manifest lists every '
            'file except itself, records the invariant len(files)+1 == directory count, and states that its own digest is verified by '
            're-hashing.' % (wrong.get(d.name, {}).get('recorded_self_digest') or 'n/a'), kind='manifest_rebuild')

    # ---- record the stale errata-3 blocks and the two repairs in errata-4
    unfreeze(ERRATA4)
    cl = json.loads((ERRATA4 / 'change_list.json').read_text(encoding='utf-8'))
    cl['entries'].extend(changes)
    cl['n'] = len(cl['entries'])
    cl['repair_round'] = dict(
        triggered_by='verifier alert from /home/sensen/Desktop/engine修复验收/S2-probe-snapshot-check-20261016T005207Z',
        defects=[dict(id='D1', what='the new manifests listed themselves in `files` with an invalid self-digest, which also contradicted '
                                    'their own manifest_sha256=null note; the count assert passed anyway, so the assert had been satisfied '
                                    'by a lie', fix='manifest lists every file except itself; invariant len(files)+1 == directory count '
                                                    'recorded and asserted; own digest verified by re-hashing'),
                 dict(id='D2', what='canonical report.json still carried the old blanket reading although errata-3 README claimed that '
                                    'wording had been fixed; the two files of one batch stated opposite conclusions',
                      fix='report.json answer.reading replaced by the measured reading; the false claim corrected in this round README'),
                 dict(id='D3', what="errata-3's change_list current-sha256 blocks went stale because those directories were rewritten "
                                    'afterwards (the verifier measured 9 mismatches)',
                      fix='superseded by this round freeze_report, which carries the post-repair digests of every frozen directory')],
        freeze_report=str(sorted(SNAP_ROOT.glob('freeze_report-errata-4-*final*.json'))[-1]) if list(SNAP_ROOT.glob('freeze_report-errata-4-*final*.json')) else None)
    (ERRATA4 / 'change_list.json').write_text(json.dumps(cl, ensure_ascii=False, indent=1), encoding='utf-8')
    os.chmod(ERRATA4 / 'change_list.json', 0o664)

    e4readme = ERRATA4 / 'README.zh-CN.md'
    t = e4readme.read_text(encoding='utf-8')
    t = t.replace('- **手动研究实施，不是成品**；不宣称站立/行走/完成度；绝对力值与单位 `NOT_ESTABLISHED`。',
                  '## 核验回件后的修复（D1–D3）\n'
                  '- **D1（我的冻结设计错）**：新清单把**自身**列进 `files` 并给了自摘要 —— 写清单就改变自身，该摘要**不可能正确**；'
                  '而"条数相等"的 assert 仍通过，说明断言被一个谎言满足。已改为：`files` 列**除自身外**的全部文件，'
                  '并记录机器可断言的**不变量** `len(files) + 1 == 目录文件数`，自身摘要明确**无处可存、由重算哈希核对**。'
                  '错误的自摘要值已逐目录记入 `change_list.json` 的 `manifest_rebuild` 条目。\n'
                  '- **D2（我的勘误说明说假话）**：前一节曾写"`report.json` 同字段的假句已在 errata-2 换掉"——**不成立**，'
                  '核验实测 `report.json` 的 `answer.reading` 仍是旧的笼统结论，与同批 README 相反。**已就地改正**该字段为实测读数，'
                  '并在 `change_list.json` 记录。\n'
                  '- **D3**：errata-3 的"当前 sha256"块在其后目录被改写后失效（核验实测 9 处不符），已由本轮冻结报告取代。\n\n'
                  '- **手动研究实施，不是成品**；不宣称站立/行走/完成度；绝对力值与单位 `NOT_ESTABLISHED`。')
    e4readme.write_text(t, encoding='utf-8')
    os.chmod(e4readme, 0o664)

    final = [freeze(d, dict(kind='re-frozen after the D1 repair and after this round README/change_list update')) for d in (ERRATA4,)]
    prev = sorted(SNAP_ROOT.glob('freeze_report-errata-4-*final*.json'))[-1]
    rep = json.loads(prev.read_text(encoding='utf-8'))
    by = {d['directory']: d for d in rep['dirs']}
    for info in rebuilt + final:
        by[info['directory']] = info
    rep['dirs'] = list(by.values())
    rep['repaired_at'] = now()
    rep["repair"] = "manifests rebuilt under the repaired rule (no self-digest, invariant recorded); the probe batch report.json reading corrected to the measured reading"
    out = SNAP_ROOT / ('freeze_report-errata-4-' + time.strftime('%Y%m%dT%H%M%S') + '-repaired.json')
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')

    print('ERRATA4:', ERRATA4)
    print('repair freeze report:', out)
    print('changes recorded this repair:', len(changes))
    for c in changes[:14]:
        print('   %-44s %s -> %s' % (pathlib.Path(c['file']).name, str(c['sha256_before'])[:12], str(c['sha256_after'])[:12]))
    print('frozen dirs now:')
    for d in rep['dirs']:
        print('   %-64s files=%d covers=%s invariant=%s' % (pathlib.Path(d['directory']).name, d['n_files'],
                                                            d['covers_complete'], d['invariant_holds']))
    return rep


if __name__ == '__main__':
    main()
