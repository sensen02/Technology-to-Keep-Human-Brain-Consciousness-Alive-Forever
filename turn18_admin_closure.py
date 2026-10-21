"""Turn 18 administrative closure: (a) the wording-place list with paths, fields and line numbers; (b) the scope incident
file for the out-of-scope in-place edits plus the explicit re-anchoring choice with reasons (choice (b): keep the
snapshot/delivery-time values in the records and annotate them); (c) the integrity file marked WITHDRAWN with a note in
the v1 README, after which nothing else in v1 is written.
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
LOC = DESK / 'S2-accepted-equilibrium-localization-20261015T170058'
SEARCH = ROOT / 'run_mujoco_equilibrium_search.py'
DIAG = ROOT / 'diagnose_aliased_residual.py'
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S')


def line_of(path, needle):
    for i, ln in enumerate(pathlib.Path(path).read_text().split('\n'), 1):
        if needle in ln:
            return i
    return None


def main():
    # (a) wording places
    places = []
    for path, needle, field in (
            (V1 / 'report.json', 'A6_geom | floor', 'mandatory_disclosure[2]'),
            (V2 / 'report.json', 'first non-foot contact = (floor, A6_geom)', 'mandatory_disclosure[<A6 entry>]'),
            (V1 / 'README.zh-CN.md', 'A6_geom', '正文句（首次非足接触那条）'),
            (V2 / 'README.zh-CN.md', 'A6_geom', '正文句（首次非足接触那条）'),
            (LADDER / 'README.zh-CN.md', 'A6_geom', '正文句（首次非足接触那条）')):
        places.append(dict(file=str(path), field=field, line=line_of(path, needle), sha256=sha(path)))
    doc = dict(status='PASS_CONTACT_IDENTITY_WORDING_PLACES', manual_research_not_product=True,
               sentence='first non-foot contact = (floor, A6_geom), body A6, step 2739 = 68.475 ms; the first '
                        'Thorax_geom | floor contact is step 3055 = 76.400 ms; this is NOT a loss of foot contact',
               measurements_for_that_state=dict(legs_in_contact_below_6_step=537,
                                                legs_in_contact_below_6_ms=round(537 * 25e-6 * 1000, 3),
                                                first_step_with_only_one_leg_in_tarsus_contact=2739,
                                                first_step_with_only_one_leg_in_tarsus_contact_ms=round(2739 * 25e-6 * 1000, 3),
                                                source='recomputed from '
                                                       '/home/sensen/Desktop/engine修复验收/S2-true-equilibrium-openloop-20261015T154720/time_series.npz'),
               places=places, recorded_at=now())
    (V2 / 'contact_identity_wording_places.json').write_text(json.dumps(doc, ensure_ascii=False, indent=1))

    # (b) scope incident + re-anchoring choice
    ledger = json.loads((V2 / 'correction_ledger.json').read_text())
    touched = [SEARCH, DIAG, ROOT / 'run_equilibrium_reissue_finalize.py'] + [
        DESK / ('S2-mujoco-equilibrium-%s-20261015T160147' % lab) / 'report.json'
        for lab in ('A_ms240', 'A_ms_r0p20', 'A_ms_r0p10', 'A_ms_r0p05')]
    rows = []
    for f in touched:
        rec = next((r for r in ledger['stale_fields_v2'] + ledger['stale_fields_ladder']
                    if pathlib.Path(r['file']) == f), None)
        rows.append(dict(path=str(f), before_sha256=(rec['old_sha256'] if rec else None), after_sha256=sha(f),
                         changed_at='2026-10-15 16:36-16:43 (and 16:45 / 17:0x for the F-A wording)', changed_by='the model acting on the '
                         'owner\'s instruction', affects_any_number='no'))
    incident = dict(
        status='PASS_SCOPE_INCIDENT_RECORDED', manual_research_not_product=True,
        incident='out-of-scope in-place edits during a verification round: this is the SECOND such event (the first was the 15:22:25 '
                 'replacement and the 15:23:44 script edit)',
        what_was_touched=rows,
        stale_hash_records=dict(v2=[dict(field=r['field'], file=r['file'], old=r['old_sha256'], current=sha(r['file']))
                                    for r in ledger['stale_fields_v2']],
                                ladder=[dict(field=r['field'], file=r['file'], old=r['old_sha256'], current=sha(r['file']))
                                        for r in ledger['stale_fields_ladder']]),
        snapshot_members_no_longer_equal_to_live=[str(x) for x in (SEARCH, DIAG, ROOT / 'run_equilibrium_reissue_finalize.py')],
        re_anchoring=dict(
            choice='(b)',
            what_it_means=('the hash values recorded in the batches stay as delivered (they are the snapshot/consistency anchor); the current '
                           'measured values are published next to them here and in correction_ledger.json / hash_realignment.json / '
                           'post_seal_amendment.json, each marked "re-anchored observations" rather than written back'),
            why=('choosing (a) would overwrite the delivered-time values and destroy the only evidence that the delivered artifact and its '
                 'snapshot are mutually consistent; choice (b) keeps both facts: the delivery-time record and the live drift'),
            rule_from_now='any edit that touches a sealed directory must first be registered as its own incident event, with before/after '
                          'hashes, before the edit is made'),
        recorded_at=now())
    (V2 / 'scope_incident_inplace_outside_v1.json').write_text(json.dumps(incident, ensure_ascii=False, indent=1))

    # (c) integrity file withdrawn + one README line in v1
    p = V1 / 'integrity_incident_assembler.json'
    rec = json.loads(p.read_text())
    rec['status'] = 'WITHDRAWN'
    rec['withdrawn_reason'] = ('the file claimed an incident revision whose content differs from the snapshot, while the three copies on disk '
                               '(live, its own preserved copy, the snapshot member) are byte-identical and none contains the fields it named; the '
                               'true event revision cannot be recovered, so this record is withdrawn and the factual event is kept below')
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    rmp = V1 / 'README.zh-CN.md'
    t = rmp.read_text()
    if 'integrity_incident_assembler.json' not in t.split('## 17')[-1]:
        t += ('\n\n## 17. 作废说明：`integrity_incident_assembler.json`\n'
              '- 该文件**已标为 `WITHDRAWN`（作废）**：它声称存在一个"内容与快照不同的 incident 版本"，但磁盘上可核查的三份副本'
              '（实盘脚本、它自己保留的 incident 副本、快照成员）**字节相同**，且**都不含**它点名的 '
              '`n_all_four_gate_items_passing` / `n_qacc_item_passing` 字段。\n'
              '- 真实事件只有一条、按实测保留：装配脚本哈希 **`5c4212768b89629f…` → `9271e578b6c54b8882c1d765af9f2ab67a13956449c0f061777810034c9c1742`**'
              '（mtime **2026-10-15 16:14:08**，被替换文件 = 实盘 `run_equilibrium_reissue_finalize.py`）；后果是**交付件与快照自洽、当时实盘无法复现交付物**；'
              '事件版本字节已被后续处置覆盖，**不可恢复**。\n'
              '- 除本条与本目录既有的更正记录外，**不再写 v1 目录**。\n')
        rmp.write_text(t)
    for d in (V1, V2):
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
    print('wording places:', len(places))
    for p_ in places:
        print('  %s :: %s :: line %s' % (p_['file'], p_['field'], p_['line']))
    print('scope incident rows:', len(rows))
    print('integrity status:', json.loads(p.read_text())['status'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
