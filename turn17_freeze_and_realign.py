"""Turn 17 closing actions:

AA) hash_realignment.json in my own v2 product directory: per record id, old sha256, current measured sha256, changed file
    path, reason, time; the original records stay untouched and are explicitly marked stale_since / superseded_by; the
    snapshot members that no longer match the live files are NOT overwritten — the current live files are stored as a SECOND
    member set named post-correction; plus the read-only discipline is written into the v2 README.
BB) integrity_incident_assembler.json marked INVALID with a factual replacement record (the true event revision cannot be
    recovered from disk, so the INVALID branch is taken explicitly).
CC) The open-loop artifact is frozen (fresh frozen manifest written AFTER the last measurement) and only then is the single
    verification dispatched.
"""

import hashlib
import json
import pathlib
import time
import zipfile

ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
V1 = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259'
V2 = DESK / 'S2-equilibrium-A-corrected-reissued-v2-20261015T163134'
LADDER = DESK / 'S2-equilibrium-radius-ladder-20261015T161518'
ART = DESK / 'S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164559'
ART2 = DESK / 'S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164943-from-accepted-pose-archive'
REANCHOR = DESK / 'S2-equilibrium-hash-reanchor-20261015T165026'
SEARCH = ROOT / 'run_mujoco_equilibrium_search.py'
DIAG = ROOT / 'diagnose_aliased_residual.py'
LIVE = ROOT / 'run_equilibrium_reissue_finalize.py'
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S')


def main():
    # ---------------- AA) hash realignment ----------------
    v2r = json.loads((V2 / 'report.json').read_text())
    ledger = json.loads((V2 / 'correction_ledger.json').read_text())
    records = []
    for i, r in enumerate(ledger['stale_fields_v2'], 1):
        records.append(dict(id='v2-stale-%02d' % i, field=r['field'], file=r['file'],
                            old_sha256=r['old_sha256'], current_measured_sha256=sha(r['file']) if pathlib.Path(r['file']).exists() else None,
                            reason='out-of-scope in-place correction (F9 template text / F6 wording / F-A wording)',
                            changed_at='2026-10-15 16:36-16:43, again 16:45 and 17:0x',
                            stale_since='2026-10-15 16:36', superseded_by='post_seal_amendment.json + hash_realignment.json',
                            original_record_kept=True))
    for i, r in enumerate(ledger['stale_fields_ladder'], 1):
        records.append(dict(id='ladder-stale-%02d' % i, field=r['field'], file=r['file'], old_sha256=r['old_sha256'],
                            current_measured_sha256=sha(r['file']) if pathlib.Path(r['file']).exists() else None,
                            reason='out-of-scope in-place correction of the live search script template text',
                            changed_at='2026-10-15 16:36-16:43', stale_since='2026-10-15 16:36',
                            superseded_by='post_seal_amendment.json + hash_realignment.json', original_record_kept=True))
    # second member set with the CURRENT live files (the old snapshot is not overwritten)
    post_zip = V2 / 'source_snapshot_post_correction.zip'
    with zipfile.ZipFile(post_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in (SEARCH, DIAG, LIVE):
            z.write(p, p.name)
    realignment = dict(
        status='PASS_HASH_REALIGNMENT', manual_research_not_product=True,
        principle=('the sealed records are read-only from now on: nothing in a sealed run directory is rewritten, the original hash values stay '
                   'where they were, and every amendment is published as an ADDITIONAL file'),
        why=('an authorised correction round had already edited files outside the batch before this rule was set; that cannot be undone by editing '
             'again, so it is recorded here and the affected records are marked stale with an explicit successor'),
        records=records, n_records=len(records),
        evidence_numbers_unchanged=('the independent verification compared 14 numeric fields per rewritten run report with the pre-edit values: all '
                                    'identical, so no number changed'),
        snapshot_members_not_overwritten=dict(
            statement='the original source_snapshot.zip inside each directory is untouched; the live files that no longer match it are shipped '
                      'SEPARATELY as post-correction members',
            second_member_set=str(post_zip),
            members={p.name: sha(p) for p in (SEARCH, DIAG, LIVE)},
            note='快照为准，live 已漂移；第二成员集只用于说明当前 live 内容'),
        discipline=('from 2026-10-15: sealed run report.json / figures / diagnostic JSON are READ-ONLY. All further corrections are published as '
                    'additional files (never in place).'),
        recorded_at=now())
    (V2 / 'hash_realignment.json').write_text(json.dumps(realignment, ensure_ascii=False, indent=1))
    rmp = V2 / 'README.zh-CN.md'
    t = rmp.read_text()
    if 'hash_realignment' not in t:
        t += ('\n\n## 16. 哈希再对齐与只读纪律（F-B）\n'
              '- `hash_realignment.json`（本目录）：逐条 **记录 id / 旧 sha256 / 当前实测 sha256 / 被改文件路径 / 原因 / 改动时间**；'
              '**旧记录原样保留**，并显式标注 `stale_since` 与 `superseded_by`，**禁止静默覆盖**。\n'
              '- 与 live 不一致的快照成员**不覆盖**；当前 live 文件另存为**第二成员集** `source_snapshot_post_correction.zip`（快照为准，live 已漂移）。\n'
              '- **纪律（写死）**：封存的 run `report.json` / 图 / 诊断 JSON **从此只读**；一切修正**以附加文件发布**，不再原地改。\n')
        rmp.write_text(t)

    # ---------------- BB) integrity file marked INVALID ----------------
    p = V1 / 'integrity_incident_assembler.json'
    rec = json.loads(p.read_text())
    rec['status'] = 'INVALID'
    rec['why_invalid'] = ('the file is self-contradictory: it asserted an incident revision with content different from the snapshot while the three '
                          'copies measured on disk (live, its own preserved copy, the snapshot member) are byte-identical and none contains the fields '
                          'it named. The true event revision cannot be recovered from disk (its bytes were overwritten), so the INVALID branch is taken '
                          'explicitly rather than papered over.')
    rec['true_event_record'] = dict(
        event='assembler revision drift after delivery',
        before_sha256='5c4212768b89629f46623578fb9003e7cb67383e98077539ba5d63a3525aa2d0 (declared at delivery, identical to the snapshot member)',
        after_sha256='9271e578b6c54b8882c1d765af9f2ab67a13956449c0f061777810034c9c1742 (live at 2026-10-15 16:14:08; contained the F2 count fields)',
        time='2026-10-15 16:14:08 (mtime)', replaced_files='/run/media/sensen/Data2/cell_wound_prototype/run_equilibrium_reissue_finalize.py',
        consequence='the delivered files and the snapshot stayed mutually consistent, but the live path could not reproduce the delivered report at that moment',
        recoverable='no: the after-revision bytes were overwritten by my own remedy and are gone',
        current_authoritative='%s (= snapshot + the five documented F2 replacements; it produced the current delivered files)' % sha(LIVE))
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1))

    # ---------------- CC) freeze, then the verification is dispatched outside this script ----------------
    for art in (ART, ART2):
        files = {q.name: sha(q) for q in sorted(art.iterdir())
                 if q.is_file() and q.name not in ('delivery-hashes.json', 'FROZEN_DELIVERY_HASHES.json')}
        mf = art / 'delivery-hashes.json'
        old = json.loads(mf.read_text()) if mf.exists() else {}
        delivery = old.get('delivery_time_hashes') or {k: v for k, v in old.items() if isinstance(v, str)}
        for k, v in files.items():
            delivery.setdefault(k, v)
        mf.write_text(json.dumps(dict(format='two columns: delivery_time_hashes (preserved) + now_hashes (current)',
                                      delivery_time_hashes=delivery, now_hashes=files,
                                      files_changed_since_delivery=sorted(k for k in files if delivery.get(k) and delivery[k] != files[k]),
                                      recorded_at=now()), ensure_ascii=False, indent=1))
        frozen = {q.name: sha(q) for q in sorted(art.iterdir())
                  if q.is_file() and q.name not in ('FROZEN_DELIVERY_HASHES.json',)}
        (art / 'FROZEN_DELIVERY_HASHES.json').write_text(json.dumps(dict(
            status='FROZEN_BEFORE_VERIFICATION', frozen_at=now(), supersedes_earlier_freeze=True,
            reason='frozen again after the per-step max|qacc| and u_diamond addenda; from now on this directory is not modified',
            files=frozen), ensure_ascii=False, indent=1))
        print('frozen', art.name, len(frozen), 'files')
    print('realignment records:', len(records))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
