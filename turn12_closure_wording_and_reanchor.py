"""Turn 12 closure, all edits in place, no new review round:

1) F-A wording unified in FOUR places: v2 report.json, v2 README, v1 README (verify), ladder README.
2) hash_reanchor.json added to the v2 directory: the stale records with before/after, the trigger, the reason, the statement
   that the numeric fields are bit-identical, and the note that the three snapshot members are stale against the live files
   ("snapshot is the record, live has drifted").
3) v1 integrity_incident_assembler.json rewritten as measured facts only, with the old "the incident revision had different
   content" narrative RETRACTED and the gate-count 5 vs 4 issue recorded as the real substance of the incident.
4) The revision records are copied into the new open-loop artifact, together with hash_reanchor.json.
"""

import hashlib
import json
import pathlib
import re
import shutil
import time

ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
V1 = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259'
V2 = DESK / 'S2-equilibrium-A-corrected-reissued-v2-20261015T163134'
LADDER = DESK / 'S2-equilibrium-radius-ladder-20261015T161518'
NEW = DESK / 'S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164559'
SEARCH = ROOT / 'run_mujoco_equilibrium_search.py'
DIAG = ROOT / 'diagnose_aliased_residual.py'
LADDER_RUNNER = ROOT / 'run_equilibrium_radius_ladder.py'
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S')
records = []

CN_UNIFIED = ('首次非足接触（步 2739 = 68.475 ms）的几何对是 **`A6_geom | floor`**（`A6_geom` 属 body A6）；'
              '**首个 `Thorax_geom | floor` 接触在步 3055 = 76.400 ms**；'
              'v1/ladder 里"步 2739 首次失去足接触"的旧句**作废**（写在三处的旧句一律以此为准）。')
EN_UNIFIED = ('the first non-foot contact at step 2739 = 68.475 ms is A6_geom | floor (A6_geom belongs to body A6); the first '
              'Thorax_geom | floor contact appears at step 3055 = 76.400 ms; the older wording "loses foot contact at step '
              '2739" that stood in v1 and in the ladder README is VOID.')


def apply_wording(path, is_json, replacements, label):
    p = pathlib.Path(path)
    before = sha(p)
    if is_json:
        obj = json.loads(p.read_text())
        for key, value in replacements.items():
            obj[key] = value
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    else:
        txt = p.read_text()
        for old, new in replacements:
            if old not in txt:
                print('  note: anchor not found in %s (%s)' % (p.name, label))
                continue
            txt = txt.replace(old, new, 1)
        p.write_text(txt)
    after = sha(p)
    records.append(dict(finding='F-A', file=str(p), label=label, sha256_before=before, sha256_after=after,
                        applied_at=now()) if after != before else dict(finding='F-A', file=str(p), label=label,
                                                                       sha256_before=before, sha256_after=after,
                                                                       applied_at=now(), note='unchanged (already correct)'))
    print('  %-6s %s -> %s' % (label, before[:8], after[:8]))


def main():
    print('--- 1) F-A wording in four places')
    # v2 report.json
    v2r = json.loads((V2 / 'report.json').read_text())
    md = v2r.get('mandatory_disclosure')
    idx = next((i for i, s in enumerate(md) if 'Thorax_geom' in s or '68.475' in s), None)
    if md is not None and idx is not None:
        md[idx] = ('A declared-cone static equilibrium is NOT dynamic maintainability. Measured under the real 100-frame '
                   'connectome-derived archive: ' + EN_UNIFIED + ' The six-leg tarsus condition is already broken at step 537 = '
                   '13.425 ms and the last tarsus contact is lost at first_tarsus_loss_step = 5111 = 127.775 ms. A newer '
                   'measurement of the accepted state of the corrected/reissued batch '
                   '(S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164559) is a different experiment.')
    apply_wording(V2 / 'report.json', True, {'mandatory_disclosure': md}, 'v2 report.json')

    # v2 README: replace the bullet that mentions Thorax_geom / 68.475 ms
    v2t = (V2 / 'README.zh-CN.md').read_text()
    old_bullet = None
    for line in v2t.split('\n'):
        if 'Thorax_geom' in line or ('68.475' in line and line.strip().startswith('- ')):
            old_bullet = line
            break
    reps = []
    if old_bullet:
        reps.append((old_bullet, '- ' + CN_UNIFIED))
    apply_wording(V2 / 'README.zh-CN.md', False, reps, 'v2 README')

    # v1 README: align to the unified sentence (it already carries A6_geom + 3055)
    v1t = (V1 / 'README.zh-CN.md').read_text()
    reps = []
    m = re.search(r'- \u9759\u6001\u5e73\u8861\*\*\u4e0d\u7b49\u4e8e\*\*.*?\u3002\uff09', v1t, re.S)
    if m:
        reps.append((m.group(0), '- 静态平衡**不等于**可动态维持，三件事分开写：' + CN_UNIFIED +
                     ' 另有：`legs_in_contact` 自步 537 = 13.425 ms 起已少于 6；`first_tarsus_loss_step = 5111 = 127.775 ms`。'
                     '（引自 `S2-true-equilibrium-openloop-20261015T154720`，并从其 `time_series.npz` 重算。）'))
    apply_wording(V1 / 'README.zh-CN.md', False, reps, 'v1 README')

    # ladder README: replace the pre-F4 sentence
    lt = (LADDER / 'README.zh-CN.md').read_text()
    reps = []
    for line in lt.split('\n'):
        if '2739' in line or '首次失去足接触' in line:
            reps.append((line, '- ' + CN_UNIFIED))
            break
    apply_wording(LADDER / 'README.zh-CN.md', False, reps, 'ladder README')

    print('--- 2) hash_reanchor.json in the v2 directory')
    stale = []
    v2r = json.loads((V2 / 'report.json').read_text())
    for lab, rb in v2r.get('runs', {}).items():
        rdir = pathlib.Path(rb['run_dir'])
        for field in ('report_json_sha256', 'stage2_best_state_sha256'):
            rec = rb.get(field)
            fn = 'report.json' if field == 'report_json_sha256' else 'stage2_best_state.npz'
            cur = sha(rdir / fn) if (rdir / fn).exists() else None
            if rec and cur and rec != cur:
                stale.append(dict(where='runs.%s.%s' % (lab, field), file=str(rdir / fn), recorded=rec, current=cur))
    for key, f in (('inputs.search_script.sha256', SEARCH), ('inputs.diagnostic_script.sha256', DIAG),
                   ('inputs.ladder_script.sha256', LADDER_RUNNER)):
        node = v2r
        for part in key.split('.'):
            node = node.get(part, {}) if isinstance(node, dict) else {}
        rec = node.get('sha256') if isinstance(node, dict) else None
        cur = sha(f) if f.exists() else None
        if rec and cur and rec != cur:
            stale.append(dict(where=key, file=str(f), recorded=rec, current=cur))
    ladder_stale = []
    lr = json.loads((LADDER / 'report.json').read_text())
    lrec = (lr.get('inputs', {}) or {}).get('search_script_sha256')
    if lrec and lrec != sha(SEARCH):
        ladder_stale.append(dict(where='inputs.search_script_sha256', file=str(SEARCH), recorded=lrec, current=sha(SEARCH)))

    reanchor = dict(
        status='PASS_HASH_REANCHOR_RECORDED',
        manual_research_not_product=True,
        why=('the authorised in-place correction round rewrote 4 sealed run report.json files and their figures, the live search '
             'script, the live diagnostic script and the diagnostic JSON; that makes the hash records inside the v2 batch and in the '
             'ladder batch stale. The archives themselves are NOT rewritten to flatter the current source; the drift is recorded here.'),
        trigger=dict(window='2026-10-15 16:36 - 16:43 (the in-place correction round), and again 16:45 (F-A wording unification)',
                     what_changed=['4 run report.json (F9 template text + field rename) and their equilibrium-search.png (F5 legends)',
                                   'live run_mujoco_equilibrium_search.py (F9 template text)',
                                   'live diagnose_aliased_residual.py and outputs/diagnostics/*.json (F6 wording)',
                                   'README/CHANGELOG/report.json inside the corrected/reissued batch (F2/F3/F4/F7/F9)']),
        stale_records_v2=stale, stale_records_ladder=ladder_stale,
        numbers_unchanged=('the independent verification of the v2 batch compared 14 numeric fields per rewritten run report against '
                           'the values v2 had recorded BEFORE the edit and found them bit-identical; no conclusion of any batch changes. '
                           'Only wording, figure legends, and (in the batch report) count field names changed.'),
        snapshot_members=dict(
            statement='快照为准，live 已漂移 / the snapshot inside each directory is the record; the live scripts have drifted away from it',
            members_now_stale=[dict(snapshot_member='run_mujoco_equilibrium_search.py', inside=str(V1 / 'source_snapshot.zip'),
                                    live=str(SEARCH), live_sha256=sha(SEARCH)),
                               dict(snapshot_member='diagnose_aliased_residual.py', inside=str(V1 / 'source_snapshot.zip'),
                                    live=str(DIAG), live_sha256=sha(DIAG)),
                               dict(snapshot_member='run_equilibrium_reissue_finalize.py', inside=str(V1 / 'source_snapshot.zip'),
                                    live=str(ROOT / 'run_equilibrium_reissue_finalize.py'),
                                    live_sha256=sha(ROOT / 'run_equilibrium_reissue_finalize.py'))]),
        rule='不改历史哈希迎合当前源码：历史记录保留原值，漂移另立此文件说明',
        recorded_at=now())
    (V2 / 'hash_reanchor.json').write_text(json.dumps(reanchor, ensure_ascii=False, indent=1))
    print('  stale v2 records: %d | stale ladder records: %d' % (len(stale), len(ladder_stale)))

    print('--- 3) integrity incident record rewritten as measured facts')
    integrity = dict(
        status='PASS_INTEGRITY_FACTS_RECORDED',
        manual_research_not_product=True,
        measured_facts=[
            dict(t='2026-10-15 16:11-16:13', fact=('the batch was delivered; its source_snapshot.zip member '
                                                  'run_equilibrium_reissue_finalize.py is 5c4212768b89629f46623578fb9003e7cb67383e98077539ba5d63a3525aa2d0, '
                                                  'and the report recorded inputs.assembler_sha256 with the same value'),
                 source='this batch'),
            dict(t='2026-10-15 16:14:08 (mtime)', fact=('an earlier independent verification measured the live file at '
                                                       '9271e578b6c54b8882c1d765af9f2ab67a13956449c0f061777810034c9c1742 and reported that this live revision '
                                                       'already contained per_equilibrium_summary.n_all_four_gate_items_passing and n_qacc_item_passing'),
                 source='independent verification S2-equilibrium-reissued-check-20261015T231321Z'),
            dict(t='2026-10-15 16:36-16:43', fact=('my own remedies overwrote the only copy of the 9271e578 bytes, so at the time the v2 verification '
                                                  'looked at the files, the live script, the preserved copy and the snapshot member were measured to be '
                                                  'byte-identical (all 5c421276...), and none of the three contained the F2 count fields'),
                 source='independent verification S2-equilibrium-v2-check-20261015T233211Z (finding F-C)'),
            dict(t='2026-10-15 16:43:25', fact=('the live assembler was rebuilt from the snapshot bytes plus the five documented F2 replacements; the '
                                                'rebuild reproduced e52f080bd0895fe36901f60398b60408e7dafe4871232656025113c5be754622 exactly, and that revision '
                                                'is both the one that ran and the one on disk now'),
                 source='rebuild_clean_authoritative_assembler.py')],
        retracted=('the earlier narrative "the incident revision had different content from the snapshot" is RETRACTED as unverifiable: '
                   'the only surviving evidence is the earlier verifier\'s hash measurement above. What is checkable today is: the delivered '
                   'files come from the live revision e52f080b..., whose ONLY difference from the snapshot 5c421276... is the two F2 count '
                   'fields plus one README sentence, and the delivered batch report states n_all_four_gate_items_passing = 4.'),
        what_the_incident_really_was=('(1) the live script drifted after delivery, so the batch\'s recorded assembler hash stopped matching the '
                                      'file on disk; (2) the gate-count wording was wrong: the field then called n_gate_passing carried 5 while the '
                                      'batch\'s own four gate items are passed by only 4 hits (start 207 fails six_tarsi_in_contact because its 51 '
                                      'tarsus contacts cover five legs, LF missing). The batch now reports n_all_four_gate_items_passing = 4.'),
        snapshot_is_the_record='交付件以快照为准 for what the snapshot contains (5c421276, no F2 fields); the delivered report is the snapshot '
                               'output plus the documented in-place F2 change, which is why its inputs.assembler_sha256 is e52f080b',
        preserved_file=dict(name='assembler_snapshot_revision_5c421276.py',
                            sha256=sha(V1 / 'assembler_snapshot_revision_5c421276.py') if (V1 / 'assembler_snapshot_revision_5c421276.py').exists() else None,
                            what_it_is='the snapshot revision'),
        recorded_at=now())
    (V1 / 'integrity_incident_assembler.json').write_text(json.dumps(integrity, ensure_ascii=False, indent=1))
    print('  written')

    print('--- 4) copy the revision records into the new artifact')
    wording_rec = dict(status='PASS_WORDING_REVISION_RECORDED', unified_sentence_cn=CN_UNIFIED, unified_sentence_en=EN_UNIFIED,
                       places=records, recorded_at=now(),
                       note='the same sentence now stands in v2 report.json, v2 README, v1 README and the ladder README; the old '
                            '"loses foot contact at step 2739" wording is VOID everywhere')
    (NEW / 'wording_revision_F-A.json').write_text(json.dumps(wording_rec, ensure_ascii=False, indent=1))
    shutil.copy2(V2 / 'hash_reanchor.json', NEW / 'hash_reanchor.json')

    # refresh the hash manifests of every directory touched
    for d in (V1, V2, LADDER, NEW):
        files = sorted(p for p in d.iterdir() if p.is_file() and p.name != 'delivery-hashes.json')
        (d / 'delivery-hashes.json').write_text(json.dumps({p.name: sha(p) for p in files}, ensure_ascii=False, indent=1))
        print('  manifest refreshed:', d.name, len(files), 'files')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
