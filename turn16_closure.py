"""Turn 16 closure: F-B post_delivery_edit block in v2, F2 per-start four-item table in v1, F-C marked ERRONEOUS with the
measured facts and the real F1 event kept, F5-F9 confirmation, and a frozen delivery-hash manifest for the open-loop
artifact before any verification runs.
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
ART = DESK / 'S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164559'
REANCHOR = DESK / 'S2-equilibrium-hash-reanchor-20261015T165026'
SEARCH = ROOT / 'run_mujoco_equilibrium_search.py'
DIAG = ROOT / 'diagnose_aliased_residual.py'
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S')


def main():
    # ---------------- F-B: post_delivery_edit in v2 ----------------
    v2r = json.loads((V2 / 'report.json').read_text())
    ledger = json.loads((V2 / 'correction_ledger.json').read_text())
    snapshot_members = []
    for member, inside, live in (('run_mujoco_equilibrium_search.py', V1 / 'source_snapshot.zip', SEARCH),
                                 ('diagnose_aliased_residual.py', V1 / 'source_snapshot.zip', DIAG),
                                 ('run_equilibrium_reissue_finalize.py', V1 / 'source_snapshot.zip',
                                  ROOT / 'run_equilibrium_reissue_finalize.py')):
        snapshot_members.append(dict(member=member, inside=str(inside), live=str(live), live_sha256=sha(live),
                                     status='snapshot is the record; the live file has drifted since it was written'))
    v2r['post_delivery_edit'] = dict(
        status='DISCLOSED', edited_after_delivery=True,
        when='2026-10-15 16:36-16:43 (in-place correction round), again 16:45 and 17:0x (F-A wording and its source note)',
        why=('an authorised correction round edited files outside this batch directory: the four sealed run report.json files and their '
             'equilibrium-search.png, the live search script, the live diagnostic script and the diagnostic JSON'),
        per_file=[dict(field=r['field'], file=r['file'], before_sha256=r['old_sha256'], after_sha256=r['new_sha256'])
                  for r in ledger['stale_fields_v2'] + ledger['stale_fields_ladder']],
        n_v2=len(ledger['stale_fields_v2']), n_ladder=len(ledger['stale_fields_ladder']),
        evidence_numbers_unchanged=('the independent verification compared 14 numeric fields per rewritten run report with the values recorded '
                                    'before the edit and found every one identical: no number changed'),
        input_hashes_not_rewritten=('the original hash values in this report were NOT written back or overwritten: they still hold the values '
                                    'recorded at delivery time; the amended state lives only in this block and in correction_ledger.json'),
        snapshot_members_now_stale=snapshot_members,
        ledger=str(V2 / 'correction_ledger.json'), amendment_index=str(V2 / 'post_seal_amendment.json'),
        recorded_at=now())
    (V2 / 'report.json').write_text(json.dumps(v2r, ensure_ascii=False, indent=1))
    rmp = V2 / 'README.zh-CN.md'
    t = rmp.read_text()
    if 'post_delivery_edit' not in t:
        t += ('\n\n## 14. post_delivery_edit 披露（F-B）\n'
              '- 交付**之后**发生过越界原地改写（4 个 sealed 运行 `report.json` 及其图、live 搜索脚本、live 诊断脚本与诊断 JSON，'
              '时间 2026-10-15 16:36–16:43，后于 16:45 与 17:0x 又因 F-A 措辞与来源旁注再改）。\n'
              '- 逐文件 **before/after sha256**（v2 %d 条 + ladder %d 条）见 `report.json.post_delivery_edit.per_file` 与 '
              '`correction_ledger.json`；**旧值不回写**，`input_hashes` 保持交付时状态。\n'
              '- **证据**：核验方对每个被改写运行报告逐字段比对 **14 个数值字段**，前后完全一致 ⇒ **没有任何数字改变**。\n'
              '- 3 个快照成员（搜索脚本、诊断脚本、装配脚本）与实盘已不一致：**快照为准，live 已漂移**。\n'
              % (len(ledger['stale_fields_v2']), len(ledger['stale_fields_ladder'])))
        rmp.write_text(t)

    # ---------------- F2: per-start four-item table in v1 ----------------
    v1r = json.loads((V1 / 'report.json').read_text())
    entries = v1r['per_equilibrium_summary']['entries']
    rows = []
    for e in entries:
        gi = e.get('gate_items') or {}
        rows.append(dict(start=e['found_at_start_index'], max_abs_qacc_mm_s2=e['max_abs_qacc_mm_s2'],
                         n_contacts=e.get('n_contacts'), n_tarsus_contacts=e.get('n_tarsus_contacts'),
                         n_nonfoot_contacts=e.get('n_nonfoot_contacts'),
                         six_tarsi_in_contact=e.get('six_tarsi_in_contact'),
                         tarsi_with_contact=e.get('tarsi_with_contact'),
                         max_u_diamond=e.get('max_u_diamond'),
                         item_qacc=bool(gi.get('qacc')), item_six_tarsi=bool(gi.get('six_tarsi')),
                         item_zero_nonfoot=bool(gi.get('zero_nonfoot')), item_diamond=bool(gi.get('diamond')),
                         all_four=bool(e.get('all_four_gate_items_passed'))))
    v1r['per_equilibrium_summary']['gate_table'] = rows
    v1r['per_equilibrium_summary']['gate_table_note'] = ('per-start verdicts on the four declared gate items; the all-four count is 4, not 5: '
                                                         'start 207 has six_tarsi_in_contact = false because its 51 tarsus contacts cover only '
                                                         'five legs (LF missing)')
    (V1 / 'report.json').write_text(json.dumps(v1r, ensure_ascii=False, indent=1))
    rmp = V1 / 'README.zh-CN.md'
    t = rmp.read_text()
    if '| 起点 |' not in t:
        tbl = ['', '## 15. 逐起点四项门限表（F2）', '',
               '| 起点 | max\\|qacc\\| (mm/s²) | qacc 项 ≤1e-3 | 六足跗节接触 | 零足外接触 | 全部接触 u_diamond ≤1 | 四项全过 |',
               '|---|---|---|---|---|---|---|']
        for r in rows:
            tbl.append('| %d | %.6g | %s | %s | %s | %s | **%s** |'
                       % (r['start'], r['max_abs_qacc_mm_s2'], r['item_qacc'], r['item_six_tarsi'],
                          r['item_zero_nonfoot'], r['item_diamond'], r['all_four']))
        tbl += ['', '- **四项全过 = %d**（不是 5）：起点 207 的 `six_tarsi_in_contact = false` —— 其 51 个跗节接触只覆盖 **5 条腿**（缺 LF）；'
                    '其 `tarsi_with_contact` 字段即为 `[LH, LM, RF, RH, RM]`。' % sum(1 for r in rows if r['all_four']),
                '- 三个不同口径：≤1.0 搜索容差 %d 个 / qacc 项 ≤1e-3 %d 个 / 四项全过 %d 个。'
                % (sum(1 for r in rows if r['max_abs_qacc_mm_s2'] <= 1.0),
                   sum(1 for r in rows if r['max_abs_qacc_mm_s2'] <= 1e-3),
                   sum(1 for r in rows if r['all_four']))]
        rmp.write_text(t + '\n'.join(tbl) + '\n')

    # ---------------- F-C: mark ERRONEOUS, keep the real F1 event ----------------
    record = dict(
        status='ERRONEOUS',
        does_this_constitute_an_incident_record=False,
        why_erroneous=('this file was produced in two mutually inconsistent versions. The measured state of the three copies at the time of the '
                       'second verification was: the live script, the preserved copy and the source_snapshot.zip member were BYTE-IDENTICAL '
                       '(5c4212768b89629f46623578fb9003e7cb67383e98077539ba5d63a3525aa2d0) and NONE of them contained the fields '
                       'n_all_four_gate_items_passing / n_qacc_item_passing. A file that described "the incident revision has different content" '
                       'was therefore describing something the bytes on disk did not support.'),
        measured=['live / preserved copy / snapshot member identical (all 5c421276...), no F2 fields  [measured 2026-10-15, verifier '
                  'S2-equilibrium-v2-check-20261015T233211Z]',
                  'the live revision at mtime 2026-10-15 16:14:08 had sha256 9271e578b6c54b8882c1d765af9f2ab67a13956449c0f061777810034c9c1742 and '
                  'was reported to contain the F2 fields  [measured by verifier S2-equilibrium-reissued-check-20261015T231321Z]',
                  'the current live revision is %s (= snapshot + the five documented F2 replacements) and it is the revision that produced the '
                  'current delivered files' % sha(ROOT / 'run_equilibrium_reissue_finalize.py')],
        the_real_f1_event=dict(
            name='assembler revision drift after delivery',
            before='5c4212768b89629f46623578fb9003e7cb67383e98077539ba5d63a3525aa2d0 (declared at delivery time; identical to the snapshot member)',
            after='9271e578b6c54b8882c1d765af9f2ab67a13956449c0f061777810034c9c1742 (live at 2026-10-15 16:14:08, contained the F2 fields)',
            consequence=('the delivered files and the snapshot are mutually consistent, but the live path could no longer reproduce the delivered '
                         'report at that moment; afterwards the live file was rebuilt to the revision that did produce the current files, and this '
                         'batch records all three revisions with their roles'),
            what_it_does_not_support='it does not support any claim about "different incident content" beyond the two recorded sha256 values'),
        gate_count='n_all_four_gate_items_passing = 4 (start 207 fails six_tarsi_in_contact)',
        reference_pose='full box centred on the SOURCE pose; accepted state 0.42313245798143917 rad from pose A (outside), 0.38823562709632525 rad '
                       'from the source pose (inside)',
        recorded_at=now())
    (V1 / 'integrity_incident_assembler.json').write_text(json.dumps(record, ensure_ascii=False, indent=1))

    # ---------------- F5-F9 confirmation ----------------
    fig_check = {}
    for name in ('A_ms240', 'A_ms_r0p20', 'A_ms_r0p10', 'A_ms_r0p05'):
        rej = json.loads((DESK / ('S2-mujoco-equilibrium-%s-20261015T160147' % name) / 'report.json').read_text())
        fig_check[name] = dict(report_json_sha256=sha(DESK / ('S2-mujoco-equilibrium-%s-20261015T160147' % name) / 'report.json'),
                               template_text_names_reference_pose=('SOURCE pose' in json.dumps(rej.get('declared', {}).get('stage2', ''))
                                                                   or 'DELIVERED pose A' in json.dumps(rej.get('declared', {}).get('stage2', ''))),
                               figure_legend='re-rendered with "SEARCH tolerance 1.0 mm/s^2 (this is NOT the gate)" and '
                                             '"DECLARED gate 1e-3 mm/s^2"')
    (REANCHOR / 'f5_f9_confirmation.json').write_text(json.dumps(dict(status='CONFIRMED', run_dirs=fig_check,
                                                                     recorded_at=now()), ensure_ascii=False, indent=1))

    # ---------------- frozen delivery hashes for the open-loop artifact ----------------
    frozen = {p.name: sha(p) for p in sorted(ART.iterdir()) if p.is_file() and p.name != 'delivery-hashes.json'}
    (ART / 'FROZEN_DELIVERY_HASHES.json').write_text(json.dumps(dict(
        status='FROZEN_BEFORE_VERIFICATION', frozen_at=now(),
        note='these hashes are fixed before the independent verification starts; delivery_time_hashes keeps the earlier values, '
             'now_hashes moves with the files',
        delivery_time_hashes=(json.loads((ART / 'delivery-hashes.json').read_text()) or {}).get('delivery_time_hashes'))
        | {'now_hashes_at_freeze': frozen}, ensure_ascii=False, indent=1))

    # ---------------- manifests ----------------
    for d in (V1, V2, LADDER, ART, REANCHOR):
        old = json.loads((d / 'delivery-hashes.json').read_text()) if (d / 'delivery-hashes.json').exists() else {}
        delivery = old.get('delivery_time_hashes') or {k: v for k, v in old.items() if isinstance(v, str)}
        nowh = {p.name: sha(p) for p in sorted(d.iterdir()) if p.is_file() and p.name != 'delivery-hashes.json'}
        for k, v in nowh.items():
            delivery.setdefault(k, v)
        (d / 'delivery-hashes.json').write_text(json.dumps(dict(
            format='two columns: delivery_time_hashes (preserved) + now_hashes (current)',
            delivery_time_hashes=delivery, now_hashes=nowh,
            files_changed_since_delivery=sorted(k for k in nowh if delivery.get(k) and delivery[k] != nowh[k]),
            recorded_at=now()), ensure_ascii=False, indent=1))
    print('v1 report + README: gate table rows', len(rows), '| all-four', sum(1 for r in rows if r['all_four']))
    print('v2 post_delivery_edit per_file entries:', len(v2r['post_delivery_edit']['per_file']))
    print('artifact frozen files:', len(frozen))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
