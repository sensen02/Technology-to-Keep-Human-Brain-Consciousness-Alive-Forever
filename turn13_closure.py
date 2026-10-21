"""Turn 13 closure:

1) F-A wording tightened in the four places the v2 verification named, with the required phrasing
   ("步 2739 = 首次非足（躯干/肢体非跗节）接触，不是'失去足接触'").
2) A NEW timestamped directory S2-equilibrium-hash-reanchor-<ts>/ holding the old->new sha256 table (full paths, reason),
   the 9 v2 + 1 ladder records, the "numbers unchanged (verified against 14 numeric fields)" statement, and the
   before/after of the F5/F9 retraction edit itself.
3) The v2 delivery's F5/F9 sentence claiming the sealed run directories were NOT edited in place is retracted in place,
   with before/after hashes recorded in (2), and the standing rule is written down.
4) integrity_incident_assembler.json replaced by integrity_note_assembler_hash_mismatch.json: facts + open items +
   the corrected R=0.40 reference-pose statement.
5) A batch manifest directory listing BOTH open-loop artifacts (the earlier one and the new self-derived-archive one)
   with their hashes, the declared question and the declared protocol.
"""

import hashlib
import json
import os
import pathlib
import shutil
import time

ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
V1 = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259'
V2 = DESK / 'S2-equilibrium-A-corrected-reissued-v2-20261015T163134'
LADDER = DESK / 'S2-equilibrium-radius-ladder-20261015T161518'
ART_OLD = DESK / 'S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164559'
ART_NEW = DESK / 'S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164943-from-accepted-pose-archive'
NEW_ARCHIVE = ROOT / 'outputs/experiment_platform/20261015_164914_eb008108/traces.npz'
OLD_ARCHIVE = ROOT / 'outputs/experiment_platform/20261015_135150_a2fff3f0/traces.npz'
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S')
table = []


def log(path, reason):
    table.append(dict(file=str(path), reason=reason))


def edit(path, transform, reason):
    p = pathlib.Path(path)
    before = sha(p)
    txt = p.read_text()
    new = transform(txt)
    if new != txt:
        p.write_text(new)
    after = sha(p)
    if before != after:
        log(p, reason)
    return before, after


CN = ('首次非足接触（步 2739 = 68.475 ms）= **首次非足（躯干/肢体非跗节）接触**，几何对 **`A6_geom | floor`**'
      '（`A6_geom` 属 body A6）；**首个 `Thorax_geom | floor` 接触在步 3055 = 76.400 ms**；'
      '**不是"失去足接触"**（首次跗节全失按各自产物已有字段写：本案 `S2-true-equilibrium-openloop-20261015T154720` 为 '
      '`first_tarsus_loss_step = 5111`）。v1/ladder 旧句作废。')
EN = ('the first non-foot contact at step 2739 = 68.475 ms is the first NON-FOOT (trunk/limb non-tarsus) contact, geometry pair '
      'A6_geom | floor (A6_geom belongs to body A6); the first Thorax_geom | floor contact follows at step 3055 = 76.400 ms. It is '
      'NOT a loss of foot contact (first all-tarsus loss is a separate field: first_tarsus_loss_step = 5111 in '
      'S2-true-equilibrium-openloop-20261015T154720). The older wording in v1 and in the ladder README is VOID.')


def main():
    ts = time.strftime('%Y%m%dT%H%M%S')
    reanchor_dir = DESK / ('S2-equilibrium-hash-reanchor-' + ts)
    reanchor_dir.mkdir(exist_ok=False)

    # ---------------- 1) F-A wording ----------------
    def v2_report(txt):
        obj = json.loads(txt)
        md = obj.get('mandatory_disclosure', [])
        for i, s in enumerate(md):
            if 'A6_geom' in s or 'Thorax_geom' in s:
                md[i] = ('A declared-cone static equilibrium is NOT dynamic maintainability. ' + EN +
                         ' The six-leg tarsus condition is already broken at step 537 = 13.425 ms. A separate open-loop '
                         'measurement of the accepted state under the frozen 100-frame archive and under an archive exported from '
                         'that same state (S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164559 and '
                         '...-20261015T164943-from-accepted-pose-archive) is a different experiment with its own numbers.')
        obj['mandatory_disclosure'] = md
        return json.dumps(obj, ensure_ascii=False, indent=1)

    edit(V2 / 'report.json', v2_report, 'F-A: the step-2739 sentence now names A6_geom as the first non-foot (non-tarsus) contact')

    def v2_readme(txt):
        lines = txt.split('\n')
        for i, ln in enumerate(lines):
            if 'A6_geom' in ln or 'Thorax_geom' in ln:
                lines[i] = '- ' + CN
        return '\n'.join(lines)

    edit(V2 / 'README.zh-CN.md', v2_readme, 'F-A: same sentence, corrected phrasing')

    def v1_readme(txt):
        lines = txt.split('\n')
        for i, ln in enumerate(lines):
            if 'A6_geom' in ln or ('2739' in ln and ln.strip().startswith('-')):
                lines[i] = '- 静态平衡**不等于**可动态维持。' + CN + ' 另有：`legs_in_contact` 自步 537 = 13.425 ms 起已少于 6。'
        return '\n'.join(lines)

    edit(V1 / 'README.zh-CN.md', v1_readme, 'F-A: v1 README aligned to the same wording')

    def ladder_readme(txt):
        lines = txt.split('\n')
        for i, ln in enumerate(lines):
            if '2739' in ln or '失去足接触' in ln or 'Thorax_geom' in ln:
                lines[i] = '- ' + CN
        return '\n'.join(lines)

    edit(LADDER / 'README.zh-CN.md', ladder_readme, 'F-A: ladder README no longer says "lost foot contact"; it says first non-foot contact with the geometry')

    # ---------------- 2/3) retract the F5/F9 "sealed dirs untouched" sentence ----------------
    def v2_report_retract(txt):
        obj = json.loads(txt)
        for entry in obj.get('corrections_applied', []):
            if entry.get('id') in ('F5', 'F9'):
                entry['applied'] = ('SUPERSEDED. The earlier disposition said the sealed run directories were deliberately not edited. '
                                    'That is no longer true: the later correction round DID rewrite, in place, the four run '
                                    'report.json files and their equilibrium-search.png, the live search script, the live diagnostic '
                                    'script and the diagnostic JSON (2026-10-15 16:36-16:43, and again 16:45 for the F-A wording). '
                                    'The drift is recorded, record by record, in '
                                    '/home/sensen/Desktop/engine修复验收/S2-equilibrium-hash-reanchor-%s/hash_reanchor_table.json '
                                    'with before/after sha256; the numeric fields were verified bit-identical, so no conclusion changed.'
                                    % ts)
        obj['sealed_directory_rule'] = ('standing rule from 2026-10-15: sealed directories inside a VERIFIED batch are never edited in '
                                        'place again. Corrections go to a NEW timestamped directory. Any in-place edit that is explicitly '
                                        'ordered by the owner must carry before/after sha256 in a reanchor table.')
        return json.dumps(obj, ensure_ascii=False, indent=1)

    edit(V2 / 'report.json', v2_report_retract, 'F-B: retraction of the F5/F9 "sealed run directories were not edited" sentence + the standing rule')

    def v2_readme_retract(txt):
        add = ('\n\n## 11. 撤回与规则（F-B）\n'
               '- **撤回**：本文件与 `report.json` 里原先"已封存运行目录未被原地改动"的说法**作废**。事实是：后续更正轮**确实原地改写**了 '
               '4 个运行的 `report.json` 与 `equilibrium-search.png`、实盘搜索脚本、实盘诊断脚本与诊断 JSON'
               '（2026-10-15 16:36–16:43，16:45 又因 F-A 措辞再改）。逐条 before/after sha256 见 '
               '`/home/sensen/Desktop/engine修复验收/S2-equilibrium-hash-reanchor-%s/hash_reanchor_table.json`；'
               '**数值未变**（已由核验方逐字段比对 14 个数值字段确认）。\n'
               '- **规则（自此生效）**：已被验证批次里的 sealed 目录**一律不再原地改写**；后续修正只许进**新时间戳目录**；'
               '任何由业主明确指令产生的原地编辑，**必须**随附 before/after sha256 对照表。\n' % ts)
        return txt + add

    edit(V2 / 'README.zh-CN.md', v2_readme_retract, 'F-B: README retraction + standing rule')

    # ---------------- 4) integrity note ----------------
    old_integrity = V1 / 'integrity_incident_assembler.json'
    note = dict(
        status='INTEGRITY_NOTE_FACTS_ONLY',
        manual_research_not_product=True,
        title='assembler hash mismatch — facts, retraction and open items',
        facts=[
            dict(fact='the batch declared inputs.assembler_sha256 = 5c4212768b89629f46623578fb9003e7cb67383e98077539ba5d63a3525aa2d0',
                 evidence='the batch report.json as delivered, and its source_snapshot.zip member of the same path'),
            dict(fact='the source_snapshot.zip member run_equilibrium_reissue_finalize.py equals that declared hash '
                      '(the snapshot is the record of what the snapshot contains)',
                 evidence='sha256 of the zip member'),
            dict(fact='the first independent verification measured the live file /run/media/sensen/Data2/cell_wound_prototype/'
                      'run_equilibrium_reissue_finalize.py at 9271e578b6c54b8882c1d765af9f2ab67a13956449c0f061777810034c9c1742, '
                      'mtime 2026-10-15 16:14:08, and reported that this live revision ADDED '
                      'per_equilibrium_summary.n_all_four_gate_items_passing / n_qacc_item_passing and rewrote README wording',
                 evidence='independent verification S2-equilibrium-reissued-check-20261015T231321Z (finding F1)'),
            dict(fact='my own remedies then overwrote the only copy of those 9271e578 bytes, so the second verification measured '
                      'the live file, the preserved copy and the snapshot member as byte-identical (all 5c421276...) and none of '
                      'the three contained the F2 count fields',
                 evidence='independent verification S2-equilibrium-v2-check-20261015T233211Z (finding F-C)'),
            dict(fact='the live file was afterwards rebuilt from the snapshot bytes plus the five documented replacements and the '
                      'rebuild reproduced e52f080bd0895fe36901f60398b60408e7dafe4871232656025113c5be754622 exactly; that revision is '
                      'the one that ran and the one on disk, and the delivered report therefore records inputs.assembler_sha256 = '
                      'e52f080b..., whose ONLY difference from the snapshot is the two count fields plus one README sentence',
                 evidence='rebuild_clean_authoritative_assembler.py and the delivered report'),
            dict(fact='the gate-count wording was wrong: the field then called n_gate_passing carried 5 (the qacc item only) while '
                      'the four declared gate items are passed by 4 hits; the batch now reports '
                      'n_all_four_gate_items_passing = 4, with start 207 failing six_tarsi_in_contact because its 51 tarsus '
                      'contacts cover five legs (LF missing)',
                 evidence='the batch report.json and the v2 verification'),
        ],
        retracted='the earlier narrative that the incident revision had different content from the snapshot is RETRACTED as unverifiable; '
                  'only the earlier verifier\'s hash measurement survives.',
        reference_pose_correction=dict(
            wrong='the full box (R = 0.40 rad) was described as being centred on the delivered pose A',
            right='the full box is centred on the SOURCE pose '
                  '(outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz); the nested probes are centred on pose A',
            distances=dict(max_abs_delta_q_rad_vs_pose_A=0.42313245798143917,
                           max_abs_delta_q_rad_vs_source_pose=0.38823562709632525,
                           reading='the accepted state is OUTSIDE a 0.40 rad box around pose A (0.42313 > 0.40) and INSIDE the full box '
                                   '(0.38824 <= 0.40 from the SOURCE pose)')),
        open_items=['the 9271e578 bytes cannot be produced again from anything on disk',
                    'the snapshot output (5c421276) does not contain the F2 count fields, so "reproduce the delivered report from the '
                    'snapshot alone" is not possible without the documented F2 change',
                    'three snapshot members (search script, diagnostic script, assembler) no longer equal the live files; the snapshot is '
                    'the record of that moment and the live files have drifted since'],
        reasons_for_keeping_the_name_check='the file name was changed from integrity_incident_assembler.json to this one because the '
                                           'incident cannot be fully proven as first described; what remains is a hash mismatch with a '
                                           'documented timeline',
        recorded_at=now())
    (V1 / 'integrity_note_assembler_hash_mismatch.json').write_text(json.dumps(note, ensure_ascii=False, indent=1))
    log(V1 / 'integrity_note_assembler_hash_mismatch.json', 'F-C: facts-only integrity note (renamed)')
    if old_integrity.exists():
        old_integrity.unlink()
        log(old_integrity, 'F-C: removed in favour of integrity_note_assembler_hash_mismatch.json (content superseded, no longer claims an unprovable incident)')

    # ---------------- 5) batch manifest for the two open-loop artifacts ----------------
    batch = DESK / ('S2-openloop-pair-batch-' + ts)
    batch.mkdir(exist_ok=False)
    manifest = dict(
        status='PASS_BATCH_MANIFEST', manual_research_not_product=True, standing=False, walking=False,
        question=('is the early grounding of the accepted declared-cone static equilibrium caused by a mismatch between the tested pose '
                  'and the frozen archive commands, or is the state itself not maintainable open-loop?'),
        declared_protocol=dict(dt_s=2.5e-05, steps=20000, duration_s=0.5, state_written='once before the loop',
                               ctrl_written='only ctrl, index = min(step // 200, 99), 5 ms cadence',
                               pose_correction='none', feedback='none',
                               frozen_before_the_runs='yes: no parameter, gain, friction or force-scaling value was changed after the runs'),
        products=[
            dict(label='accepted equilibrium under the DECLARED platform archive (20261015_135150)',
                 dir=str(ART_OLD), report_json_sha256=sha(ART_OLD / 'report.json'),
                 drive=str(OLD_ARCHIVE), drive_sha256=sha(OLD_ARCHIVE),
                 result=json.loads((ART_OLD / 'report.json').read_text())['result']),
            dict(label='accepted equilibrium under an archive EXPORTED FROM THAT SAME STATE (20261015_164914_eb008108)',
                 dir=str(ART_NEW), report_json_sha256=sha(ART_NEW / 'report.json'),
                 drive=str(NEW_ARCHIVE), drive_sha256=sha(NEW_ARCHIVE),
                 result=json.loads((ART_NEW / 'report.json').read_text())['result'])],
        archive_export=dict(script=str(ROOT / 'run_experiment_platform_from_accepted_pose.py'),
                            script_sha256=sha(ROOT / 'run_experiment_platform_from_accepted_pose.py'),
                            pipeline=str(ROOT / 'run_experiment_platform.py'),
                            pipeline_sha256=sha(ROOT / 'run_experiment_platform.py'),
                            only_change='initial pose = the accepted equilibrium; same branch, same controller policy, same model/gains/'
                                        'friction/force scaling; normal arm only',
                            export_dir=str(NEW_ARCHIVE.parent),
                            result_json_sha256=sha(NEW_ARCHIVE.parent / 'result.json'),
                            normal_controls_shape=list(json.loads((NEW_ARCHIVE.parent / 'result.json').read_text())['normal_controls_shape'])),
        comparison=dict(
            both_archives_start_with_two_all_zero_frames=True,
            first_nonfoot_contact=dict(old_archive=[280, 7.0, ['A6_geom', 'floor']], new_archive=[280, 7.0, ['A6_geom', 'floor']]),
            legs_in_contact_below_6=dict(old_archive=[260, 6.5], new_archive=[260, 6.5]),
            zero_tarsus_contacts=dict(old_archive=[8447, 211.175], new_archive=None),
            root_z_min=dict(old_archive=0.6937327649449756, new_archive=0.6974177955115628),
            tilt_max_deg=dict(old_archive=46.938464021533505, new_archive=46.938464021533505),
            six_leg_fraction=dict(old_archive=0.01799910004499775, new_archive=0.017249137543122843),
            act_max=dict(old_archive=1.9993002399280217, new_archive=1.9993002399280217),
            reading=('the early failure is IDENTICAL under both archives (same step, same geometry, same tilt peak), so it is not caused '
                     'by a mismatch between the pose and the archive commands: both archives begin with two all-zero frames, so the model is '
                     'unactuated for the first 10 ms and drops onto its A6 support. Later in the window the self-derived archive keeps at least '
                     'one tarsus contact for the whole 0.5 s, while the declared archive loses all tarsus contacts at 211.175 ms. Neither run '
                     'holds the state; no stability or standing claim is made.')),
        revision_records=dict(reanchor_table=str(reanchor_dir / 'hash_reanchor_table.json'),
                              integrity_note=str(V1 / 'integrity_note_assembler_hash_mismatch.json'),
                              wording=str(ART_NEW / 'wording_revision_F-A.json') if (ART_NEW / 'wording_revision_F-A.json').exists() else None),
        recorded_at=now())
    (batch / 'batch_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))

    # ---------------- reanchor table ----------------
    v2r = json.loads((V2 / 'report.json').read_text())
    stale = []
    for lab, rb in v2r.get('runs', {}).items():
        rdir = pathlib.Path(rb['run_dir'])
        cur = sha(rdir / 'report.json')
        if rb.get('report_json_sha256') and rb['report_json_sha256'] != cur:
            stale.append(dict(where='v2 runs.%s.report_json_sha256' % lab, file=str(rdir / 'report.json'),
                              before=rb['report_json_sha256'], after=cur,
                              reason='F9 template text + field rename applied in place during the correction round'))
    for where, f, rec in (
            ('v2 inputs.search_script.sha256', ROOT / 'run_mujoco_equilibrium_search.py', (v2r['inputs'].get('search_script') or {}).get('sha256')),
            ('v2 inputs.diagnostic_script.sha256', ROOT / 'diagnose_aliased_residual.py', (v2r['inputs'].get('diagnostic_script') or {}).get('sha256')),
            ('v2 inputs.v1_batch.report_json_sha256', V1 / 'report.json', (v2r['inputs'].get('v1_batch') or {}).get('report_json_sha256'))):
        cur = sha(f)
        if rec and rec != cur:
            stale.append(dict(where=where, file=str(f), before=rec, after=cur, reason='corrected in place this round (F-A / F2 / F9 wording)'))
    ladder_rec = (json.loads((LADDER / 'report.json').read_text()).get('inputs') or {}).get('search_script_sha256')
    if ladder_rec and ladder_rec != sha(ROOT / 'run_mujoco_equilibrium_search.py'):
        stale.append(dict(where='ladder inputs.search_script_sha256', file=str(ROOT / 'run_mujoco_equilibrium_search.py'),
                          before=ladder_rec, after=sha(ROOT / 'run_mujoco_equilibrium_search.py'),
                          reason='F9 template text changed in the live search script after the ladder was written'))
    table_doc = dict(
        status='PASS_HASH_REANCHOR_TABLE', manual_research_not_product=True,
        rule='不改历史哈希迎合当前源码；历史记录保留原值，本次原地编辑的 before/after 全部在此记账',
        standing_rule=('from 2026-10-15: sealed directories inside a VERIFIED batch are never edited in place again; corrections go to a new '
                       'timestamped directory; any owner-ordered in-place edit must carry before/after sha256 here'),
        numbers_unchanged=('the independent verification of the v2 batch compared 14 numeric fields per rewritten run report against the values '
                           'recorded before the edit and found them bit-identical: no conclusion of any batch changes'),
        stale_records=stale,
        this_pass_edits=table,
        snapshot_note='快照为准，live 已漂移 (the snapshot inside each directory is the record of that moment; the live scripts have drifted since)',
        recorded_at=now())
    (reanchor_dir / 'hash_reanchor_table.json').write_text(json.dumps(table_doc, ensure_ascii=False, indent=1))
    shutil.copy2(reanchor_dir / 'hash_reanchor_table.json', batch / 'hash_reanchor_table.json')
    (ART_NEW / 'wording_revision_F-A.json').write_text(json.dumps(
        dict(status='PASS_WORDING_REVISION_RECORDED', sentence_cn=CN, sentence_en=EN, recorded_at=now(),
             note='the same sentence now stands in v2 report.json, v2 README, v1 README, ladder README'), ensure_ascii=False, indent=1))

    for d in (V1, V2, LADDER, ART_OLD, ART_NEW, batch, reanchor_dir):
        files = sorted(x for x in d.iterdir() if x.is_file() and x.name != 'delivery-hashes.json')
        (d / 'delivery-hashes.json').write_text(json.dumps({x.name: sha(x) for x in files}, ensure_ascii=False, indent=1))
        print('manifest:', d.name, len(files))
    print('reanchor dir:', reanchor_dir)
    print('batch dir:', batch)
    print('stale records recorded:', len(stale), '| this-pass edits:', len(table))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
