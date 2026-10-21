"""Turn 15 closure, all in place:

1) F-A: each of the four places gets a note naming the correction source (the independent verification that measured the
   identities) and the measured step numbers, so only one wording exists in the delivered files.
2) F-B: a post_seal_amendment record with path / delivery-time sha256 / current sha256 / change time / reason / the
   "14 numeric fields identical" statement, plus the freeze rule. The old hash values in the batches are NOT overwritten.
3) F-C: integrity_incident_assembler.json rewritten as a factual record, with the one clause that cannot hold as stated
   flagged explicitly instead of being asserted.
4) The delivered batch states the all-four-gate count (4) and lists ALL THREE assembler revisions with their roles and
   which one produced the current files.
5) F3 checked in both directions.
"""

import hashlib
import json
import pathlib
import time

ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
V1 = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259'
V2 = DESK / 'S2-equilibrium-A-corrected-reissued-v2-20261015T163134'
LADDER = DESK / 'S2-equilibrium-radius-ladder-20261015T161518'
REANCHOR = DESK / 'S2-equilibrium-hash-reanchor-20261015T165026'
BATCH = DESK / 'S2-openloop-pair-batch-20261015T165026'
ART = DESK / 'S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164559'
SEARCH = ROOT / 'run_mujoco_equilibrium_search.py'
DIAG = ROOT / 'diagnose_aliased_residual.py'
LIVE = ROOT / 'run_equilibrium_reissue_finalize.py'
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S')
SOURCE_NOTE = ('（更正来源：独立核验 `S2-equilibrium-v2-check-20261015T233211Z` 用确定性重放实测步 2739 的接触对为 '
               '`(floor, A6_geom)`、首个 `Thorax_geom | floor` 在步 3055 = 76.400 ms）')
SOURCE_NOTE_EN = (' (correction source: the independent verification S2-equilibrium-v2-check-20261015T233211Z replayed the '
                  'deterministic open loop and measured the pair at step 2739 to be (floor, A6_geom), with the first '
                  'Thorax_geom | floor contact at step 3055 = 76.400 ms)')
changed = []


def edit(path, fn, reason):
    p = pathlib.Path(path)
    b = sha(p)
    t = p.read_text()
    nt = fn(t)
    if nt != t:
        p.write_text(nt)
    a = sha(p)
    if a != b:
        changed.append(dict(file=str(p), before=b, after=a, reason=reason, at=now()))
    return p


def two_col(d):
    d = pathlib.Path(d)
    old = json.loads((d / 'delivery-hashes.json').read_text()) if (d / 'delivery-hashes.json').exists() else {}
    delivery = old.get('delivery_time_hashes') or {k: v for k, v in old.items() if isinstance(v, str)}
    nowh = {p.name: sha(p) for p in sorted(d.iterdir()) if p.is_file() and p.name != 'delivery-hashes.json'}
    for k, v in nowh.items():
        delivery.setdefault(k, v)
    (d / 'delivery-hashes.json').write_text(json.dumps(dict(
        format='two columns: delivery_time_hashes (preserved, never overwritten) + now_hashes (current)',
        delivery_time_hashes=delivery, now_hashes=nowh,
        files_changed_since_delivery=sorted(k for k in nowh if delivery.get(k) and delivery[k] != nowh[k]),
        note='post-seal amendments are indexed in post_seal_amendment.json; the original values stay in the left column',
        recorded_at=now()), ensure_ascii=False, indent=1))


def main():
    # ---------------- 1) F-A with the correction-source note ----------------
    def add_note_en(t):
        return t.replace('first non-foot contact = (floor, A6_geom), body A6, step 2739 = 68.475 ms',
                         'first non-foot contact = (floor, A6_geom), body A6, step 2739 = 68.475 ms' + SOURCE_NOTE_EN) \
            if 'correction source' not in t else t

    def add_note_cn(t):
        if '更正来源' in t:
            return t
        return t.replace('**首个 `Thorax_geom | floor` 接触在步 3055 = 76.400 ms**',
                         '**首个 `Thorax_geom | floor` 接触在步 3055 = 76.400 ms**' + SOURCE_NOTE)

    edit(V2 / 'report.json', lambda t: json.dumps({**json.loads(t), 'mandatory_disclosure': [
        (s + SOURCE_NOTE_EN) if ('A6_geom' in s and 'correction source' not in s) else s
        for s in json.loads(t).get('mandatory_disclosure', [])]}, ensure_ascii=False, indent=1),
        'F-A: correction-source note added to the disclosure')
    edit(V2 / 'README.zh-CN.md', add_note_cn, 'F-A: correction-source note added (v2 README)')
    edit(V1 / 'README.zh-CN.md', add_note_cn, 'F-A: correction-source note added (v1 README)')
    edit(LADDER / 'README.zh-CN.md', add_note_cn, 'F-A: correction-source note added (ladder README)')

    # ---------------- 2) F-B post-seal amendment ----------------
    seal = [('S2-mujoco-equilibrium-A_ms240-20261015T160147', 'run report.json + equilibrium-search.png'),
            ('S2-mujoco-equilibrium-A_ms_r0p20-20261015T160147', 'run report.json + equilibrium-search.png'),
            ('S2-mujoco-equilibrium-A_ms_r0p10-20261015T160147', 'run report.json + equilibrium-search.png'),
            ('S2-mujoco-equilibrium-A_ms_r0p05-20261015T160147', 'run report.json + equilibrium-search.png')]
    v2r = json.loads((V2 / 'report.json').read_text())
    amendments = []
    for name, what in seal:
        d = DESK / name
        rec = next((rb for rb in v2r.get('runs', {}).values() if pathlib.Path(rb['run_dir']).name == name), None)
        delivery_hash = rec.get('report_json_sha256') if rec else None
        amendments.append(dict(path=str(d / 'report.json'), delivery_time_sha256=delivery_hash,
                               now_sha256=sha(d / 'report.json'),
                               changed_at='2026-10-15 16:36-16:43 (F9 template text), again 16:45 (F-A wording)',
                               reason='out-of-scope in-place correction of the run template text and figure legends, ordered by the owner',
                               also_changed=[str(d / 'equilibrium-search.png')], what=what))
    for name in ('S2-mujoco-equilibrium-ladderR045-20261015T161518', 'S2-mujoco-equilibrium-ladderR040-20261015T161518',
                 'S2-mujoco-equilibrium-ladderR035-20261015T161518', 'S2-mujoco-equilibrium-ladderR030-20261015T161518',
                 'S2-mujoco-equilibrium-ladderR025-20261015T161518'):
        pass
    for f, field_before, what in ((SEARCH, (v2r['inputs'].get('search_script') or {}).get('sha256'), 'live search script (F9 template text)'),
                                  (DIAG, (v2r['inputs'].get('diagnostic_script') or {}).get('sha256'), 'live diagnostic script (F6 wording)')):
        amendments.append(dict(path=str(f), delivery_time_sha256=field_before, now_sha256=sha(f),
                               changed_at='2026-10-15 16:36-16:43', reason='out-of-scope in-place correction ordered by the owner',
                               also_changed=[], what=what))
    amendment = dict(
        status='PASS_POST_SEAL_AMENDMENT_INDEX', manual_research_not_product=True,
        freeze_rule=('from 2026-10-15 the four sealed run directories (report.json and their figures) are FROZEN: no further in-place '
                     'edits. Later corrections go to a new timestamped directory. This file records what already happened before the freeze.'),
        numbers_unchanged='the independent verification of the v2 batch compared 14 numeric fields per rewritten run report with the values '
                          'recorded before the edit: every value identical, so no number changed',
        never_overwrite='the old hash values stay in the batches and in the delivery-time column of every delivery-hashes.json; the amended '
                        'state is recorded here and in correction_ledger.json, never written back into the original fields',
        amendments=amendments,
        stale_records_index=dict(v2=str(V2 / 'correction_ledger.json'), table=str(REANCHOR / 'hash_reanchor_table.json')),
        recorded_at=now())
    (REANCHOR / 'post_seal_amendment.json').write_text(json.dumps(amendment, ensure_ascii=False, indent=1))
    (V2 / 'post_seal_amendment.json').write_text(json.dumps(amendment, ensure_ascii=False, indent=1))
    (BATCH / 'post_seal_amendment.json').write_text(json.dumps(amendment, ensure_ascii=False, indent=1))

    # ---------------- 3) F-C factual record ----------------
    record = dict(
        status='FACTS_ONLY_ONE_CLAUSE_FLAGGED', manual_research_not_product=True,
        measurements=[
            dict(at='2026-10-15 16:11-16:13', fact='the batch was delivered; inputs.assembler_sha256 = '
                                                   '5c4212768b89629f46623578fb9003e7cb67383e98077539ba5d63a3525aa2d0 and the '
                                                   'source_snapshot.zip member of that path has the same hash',
                 source='the delivered batch'),
            dict(at='2026-10-15 16:14:08 (mtime)', fact='the first independent verification measured the LIVE file at '
                                                        '9271e578b6c54b8882c1d765af9f2ab67a13956449c0f061777810034c9c1742 and reported that this live '
                                                        'revision contained per_equilibrium_summary.n_all_four_gate_items_passing and '
                                                        'n_qacc_item_passing, i.e. the F2 fix lived in the LIVE revision',
                 source='independent verification S2-equilibrium-reissued-check-20261015T231321Z (finding F1)'),
            dict(at='2026-10-15 16:36-16:43', fact='my remedies overwrote the only copy of those bytes; the second independent verification then '
                                                   'measured the live file, the preserved copy and the snapshot member to be BYTE-IDENTICAL (all '
                                                   '5c421276...) and none of them to contain the F2 fields',
                 source='independent verification S2-equilibrium-v2-check-20261015T233211Z (finding F-C)'),
            dict(at='2026-10-15 16:43:25', fact='the live file was rebuilt from the snapshot bytes plus the five documented F2 replacements and the '
                                                'rebuild reproduced e52f080bd0895fe36901f60398b60408e7dafe4871232656025113c5be754622 exactly; that '
                                                'revision is both the one that ran and the file on disk now',
                 source='rebuild_clean_authoritative_assembler.py')],
        clause_that_cannot_hold=( 'the instruction "live, incident copy and snapshot member are byte-identical AND none contains the F2 fields, AND the '
                                   'F2 fix is in the live 9271e578" is self-contradictory and is therefore NOT asserted: the byte-identity and the '
                                   'F2-free state were measured AFTER my restore overwrote the incident bytes, while the F2 fix was measured IN the '
                                   'live 9271e578 BEFORE that restore. Both measurements are recorded above with their timestamps.'),
        current_roles=dict(
            delivery_time_declared='5c4212768b89629f46623578fb9003e7cb67383e98077539ba5d63a3525aa2d0 (snapshot member, no F2 fields)',
            post_delivery_live_16_14_08='9271e578b6c54b8882c1d765af9f2ab67a13956449c0f061777810034c9c1742 (contained the F2 fields; bytes LOST)',
            current_live_and_producer='%s (byte-identical to the revision that produced the current files; = snapshot + the five documented F2 '
                                      'replacements)' % sha(LIVE)),
        gate_count='n_all_four_gate_items_passing = 4 (start 207 fails six_tarsi_in_contact: 51 tarsus contacts covering five legs, LF missing)',
        reference_pose='the full box (R = 0.40 rad) is centred on the SOURCE pose; only the nested boxes are centred on pose A '
                       '(accepted state: 0.42313245798143917 rad from A = outside, 0.38823562709632525 rad from the source pose = inside)',
        recorded_at=now())
    (V1 / 'integrity_incident_assembler.json').write_text(json.dumps(record, ensure_ascii=False, indent=1))

    # ---------------- 4) counts + three assembler hashes in the delivered batch ----------------
    def f_report(t):
        obj = json.loads(t)
        obj['assembler_provenance']['all_revisions'] = [
            dict(role='declared at delivery time / snapshot member', sha256='5c4212768b89629f46623578fb9003e7cb67383e98077539ba5d63a3525aa2d0',
                 contains_F2_fields=False),
            dict(role='live revision at 2026-10-15 16:14:08 (post-delivery drift)',
                 sha256='9271e578b6c54b8882c1d765af9f2ab67a13956449c0f061777810034c9c1742', contains_F2_fields=True,
                 bytes_available=False),
            dict(role='CURRENT live revision and the one that produced the current report.json / README.zh-CN.md',
                 sha256=sha(LIVE), contains_F2_fields=True, bytes_available=True)]
        obj['assembler_provenance']['which_produced_the_current_files'] = sha(LIVE)
        obj['per_equilibrium_summary']['gate_count_statement'] = ('the four declared gate items are passed by 4 hits, not 5: start 207 has '
                                                                  'six_tarsi_in_contact = false because its 51 tarsus contacts cover only five '
                                                                  'legs (LF missing)')
        return json.dumps(obj, ensure_ascii=False, indent=1)

    edit(V1 / 'report.json', f_report, 'F1/F2: all three assembler revisions listed with roles, and the count 4 restated')
    edit(V1 / 'README.zh-CN.md', lambda t: t + (
        '\n\n## 13. 装配脚本三个版本与计数口径（F1/F2）\n'
        '- 交付时声明/快照成员：`5c4212768b89629f46623578fb9003e7cb67383e98077539ba5d63a3525aa2d0`（**不含** F2 字段）。\n'
        '- 交付后漂移的实盘版本（mtime 2026-10-15 16:14:08）：`9271e578b6c54b8882c1d765af9f2ab67a13956449c0f061777810034c9c1742`'
        '（**含** F2 字段；字节已丢失，无法再生）。\n'
        '- **当前实盘版本，也是产出本目录 report.json / README 的那一版**：`%s`（= 快照 + 五个已记录替换；与运行时字节相同）。\n'
        '- **四项门限全过的命中数 = 4，不是 5**：起点 207 的 `six_tarsi_in_contact = false`（51 个跗节接触只覆盖 5 条腿，缺 LF）。\n'
        '- 参考姿态：**全盒以源姿态为心**（被接受点 vs A = 0.42313245798143917 rad，超出；vs 源姿态 = 0.38823562709632525 rad，在盒内）；'
        '**三个嵌套盒才是以姿态 A 为心**。\n'
        '- post-seal 修订索引：`post_seal_amendment.json`（本目录与 reanchor 目录各一份）——旧哈希值一律保留，不写回原字段。\n'
        % sha(LIVE)), 'F1/F2/F3: README section with the three revisions, the count 4 and both pose references')

    # ---------------- 5) F3 check ----------------
    v1r = json.loads((V1 / 'report.json').read_text())
    f3 = v1r.get('reference_pose_of_each_box', {})
    f3_ok = (f3.get('full_box', {}).get('joint_reference', '').startswith('SOURCE')
             and f3.get('nested_probes', {}).get('joint_reference', '').startswith('DELIVERED'))
    print('F3 both directions present in report.json:', f3_ok)
    print('README has both distances:', '0.42313245798143917' in (V1 / 'README.zh-CN.md').read_text()
          and '0.38823562709632525' in (V1 / 'README.zh-CN.md').read_text())

    for d in (V1, V2, LADDER, REANCHOR, BATCH, ART):
        two_col(d)
    print('edits:', len(changed))
    for c in changed:
        print('  %-42s %s -> %s' % (pathlib.Path(c['file']).parent.name + '/' + pathlib.Path(c['file']).name,
                                    c['before'][:8], c['after'][:8]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
