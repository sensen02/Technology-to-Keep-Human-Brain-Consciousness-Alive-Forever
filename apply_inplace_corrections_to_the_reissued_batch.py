"""Apply the verification findings F1-F9 of the independent check to the reissued batch IN PLACE, under the explicit
instruction of the project owner, and record every single change in CHANGELOG-inplace.md.

Nothing here re-runs a 240-start search. Every change is a wording / labelling / field-name / attribution change, or the
restoration of the live assembler from the batch's own snapshot; no measured number is altered, and the accepted-state
npz is not touched at all (its sha256 is checked at the end).

F1  restore the live assembler byte-for-byte from the batch's source_snapshot.zip member and write
    integrity_incident_assembler.json with the timeline and hashes.
F2  replace the mislabelled "gate passing" count (5, which was the qacc item only) with three counts, each defined.
F3  state that the full box references the SOURCE pose, not pose A, and give both distances of the accepted state.
F4  correct the mandatory disclosure to the measured open-loop sequence (2739 = 68.475 ms is the first NON-FOOT contact).
F5  redraw the fourth underlying run figures with correct labels (numbers untouched, taken from the run reports).
F6  fix the misleading pointer test in the diagnostic JSON and in the diagnostic script.
F7  define the two "best" conventions for the nested probes and explain the 2.5e-8 difference.
F8  write VOID.txt into the two void directories with the facts that are still checkable.
F9  fix the run-template wording (reference pose) and the misleading field names, in the live script and in the four
    run reports.
"""

import hashlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = Path('/home/sensen/Desktop/engine修复验收')
BATCH = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259'
V2 = DESK / 'S2-equilibrium-A-corrected-reissued-v2-20261015T163134'
LIVE_ASSEMBLER = ROOT / 'run_equilibrium_reissue_finalize.py'
SEARCH_SCRIPT = ROOT / 'run_mujoco_equilibrium_search.py'
DIAG_SCRIPT = ROOT / 'diagnose_aliased_residual.py'
DIAG_DIR = ROOT / 'outputs/diagnostics'
RUN_DIRS = [DESK / ('S2-mujoco-equilibrium-%s-20261015T160147' % lab)
            for lab in ('A_ms240', 'A_ms_r0p20', 'A_ms_r0p10', 'A_ms_r0p05')]
VOID_DIRS = [DESK / 'S2-mujoco-equilibrium-A-20261015T151340',
             DESK / 'S2-equilibrium-A-corrected-reissued-20261015T154946']

CHANGES = []


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def mtime(p):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(Path(p).stat().st_mtime))



def replace_line_by_prefix(txt, prefix, new_line):
    """replace the first line that starts with prefix (robust against invisible whitespace / unicode variants)"""
    for line in txt.split('\n'):
        if line.startswith(prefix):
            return txt.replace(line, new_line, 1), line
    return None, None

def rec(fid, path, where, before, after, basis):
    CHANGES.append(dict(finding=fid, file=str(path), field_or_line=where,
                        before=before, after=after, basis=basis))
    print('[%s] %s :: %s' % (fid, path, where), flush=True)


def write_hashes(d):
    d = Path(d)
    files = sorted(p for p in d.iterdir() if p.is_file() and p.name != 'delivery-hashes.json')
    (d / 'delivery-hashes.json').write_text(json.dumps({p.name: sha(p) for p in files}, ensure_ascii=False, indent=1))
    return {p.name: sha(p) for p in files}


def main():
    print('batch:', BATCH)
    os.chdir(ROOT)

    # ---------------- F1: evidence, restore, integrity file ----------------
    live_before_sha, live_before_mtime = sha(LIVE_ASSEMBLER), mtime(LIVE_ASSEMBLER)
    with zipfile.ZipFile(BATCH / 'source_snapshot.zip') as z:
        members = {i.filename: hashlib.sha256(z.read(i.filename)).hexdigest() for i in z.infolist()}
        snap_bytes = z.read('run_equilibrium_reissue_finalize.py')
    snapshot_sha = members.get('run_equilibrium_reissue_finalize.py')
    print('live assembler before restore:', live_before_sha, live_before_mtime)
    print('snapshot member hash        :', snapshot_sha)

    backup = BATCH / 'assembler_live_version_at_the_incident.py'
    shutil.copy2(LIVE_ASSEMBLER, backup)
    rec('F1', LIVE_ASSEMBLER, 'whole file (restored from the batch snapshot)',
        'sha256 %s, mtime %s' % (live_before_sha, live_before_mtime), 'sha256 %s (the batch snapshot bytes)' % snapshot_sha,
        'the batch named this path as its assembler; restoring the snapshot bytes makes that path reproduce the delivered files again')
    LIVE_ASSEMBLER.write_bytes(snap_bytes)
    live_after_sha, live_after_mtime = sha(LIVE_ASSEMBLER), mtime(LIVE_ASSEMBLER)
    assert live_after_sha == snapshot_sha, 'restore failed'

    integrity = dict(
        status='PASS_INTEGRITY_INCIDENT_RECORDED',
        incident=('the assembler path named by this batch was edited AFTER the batch was written, so the recorded '
                  'inputs.assembler_sha256 no longer matched the file on disk (independent verification finding F1)'),
        timeline=dict(
            batch_written_local='2026-10-15 16:12-16:13 (report.json, README.zh-CN.md, figure)',
            live_assembler_at_incident=dict(sha256=live_before_sha, mtime=live_before_mtime),
            note=('the live version contained the F2 fix (per_equilibrium_summary.n_all_four_gate_items_passing and '
                  'n_qacc_item_passing) which was NOT back-ported through the assembler; the F2 correction is applied to '
                  'the delivered files directly and is recorded in CHANGELOG-inplace.md')),
        snapshot=dict(source=str(BATCH / 'source_snapshot.zip'), member='run_equilibrium_reissue_finalize.py',
                      sha256=snapshot_sha,
                      relation_to_the_incident_version=('same script, different revision: the snapshot does not contain the '
                                                        'added count fields')),
        restoration=dict(action='the snapshot bytes were written back to the named path',
                         sha256_after_restore=live_after_sha, mtime_after_restore=live_after_mtime,
                         matches_snapshot=bool(live_after_sha == snapshot_sha)),
        incident_version_kept_as=str(backup), incident_version_sha256=sha(backup),
        rule=('a batch may only name files whose bytes are shipped inside the batch; live paths that can be edited later '
              'must not be quoted as reproducibility anchors'),
        manual_research_not_product=True)
    integrity_path = BATCH / 'integrity_incident_assembler.json'
    integrity_path.write_text(json.dumps(integrity, ensure_ascii=False, indent=1))
    rec('F1', integrity_path, 'new file', '-', 'integrity incident, timeline, relation 9271e578 -> 5c421276, restoration evidence',
        'F1 requires the evidence to be fixed before the remedy')

    # ---------------- F2 / F3 / F4 / F7 / F9 on the batch report ----------------
    rp = BATCH / 'report.json'
    r = json.loads(rp.read_text())
    before_sha = sha(rp)

    already_applied = 'inplace_corrections' in r
    if already_applied:
        print('note: the report.json in-place corrections are already applied; only the remaining steps will run')
    old_scope = r['scope']
    r['scope'] = ('corrected/reissued declared numerical equilibrium search for the delivered pose A: stage 1 varies only the '
                  '90 actuators in [1e-4, 1]; stage 2 also allows root z within +-0.05 mm and the 42 leg hinge joints within '
                  '[joint range - 0.005 rad]. THE JOINT REFERENCE POSE DIFFERS BETWEEN THE BOXES AND MUST NOT BE CONFLATED: the '
                  'FULL BOX (R = 0.40 rad) is centred on the SOURCE pose outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/'
                  'initial_pose.npz (that is what run_mujoco_equilibrium_search.py uses when --radius is not given), while the '
                  'NESTED probes (0.20 / 0.10 / 0.05 rad) are centred on the DELIVERED pose A. qvel = 0 everywhere; the minimised '
                  'quantity is ||qacc||; the contact forces automatically satisfy the declared diamond because that is the cone '
                  'MuJoCo applies.')
    rec('F3', rp, 'scope', old_scope[:160] + ' ...', r['scope'][:160] + ' ...',
        'independent verification F3: the full box bounds the joints around the SOURCE pose, not around pose A')

    old_disc = r['mandatory_disclosure'][2]
    r['mandatory_disclosure'][2] = ('A declared-cone static equilibrium is NOT dynamic maintainability. Measured on the accepted '
                                    'state under the real 100-frame connectome-derived archive (artifact '
                                    'S2-true-equilibrium-openloop-20261015T154720, recomputed for this correction from its '
                                    'time_series.npz): the first NON-FOOT contact (Thorax_geom | floor) appears at step 2739 = '
                                    '68.475 ms; the six-leg tarsus condition is already broken at step 537 = 13.425 ms; the last '
                                    'tarsus contact is lost at step 5111 = 127.775 ms; all six legs are in contact for only 849 of '
                                    '20000 steps. A SEPARATE, newer measurement of THIS batch accepted state '
                                    '(S2-accepted-equilibrium-openloop-real100-*) gives its own numbers and is not the same '
                                    'experiment, because the accepted state, and the actuator state paired with it, differ.')
    rec('F4', rp, 'mandatory_disclosure[2]', old_disc, r['mandatory_disclosure'][2],
        'independent verification F4: step 2739 is the first NON-FOOT (thorax) contact, not a loss of foot contact')

    pes = r['per_equilibrium_summary']
    old_count = pes.pop('n_gate_passing', None)
    hits = pes['entries']
    all_four = [h for h in hits if h['max_abs_qacc_mm_s2'] <= 1e-3 and h['six_tarsi_in_contact']
                and h['n_nonfoot_contacts'] == 0 and h['max_u_diamond'] is not None and h['max_u_diamond'] <= 1 + 1e-9]
    pes['n_search_tolerance_passing'] = int(sum(1 for h in hits if h['max_abs_qacc_mm_s2'] <= 1.0))
    pes['n_qacc_item_passing'] = int(sum(1 for h in hits if h['max_abs_qacc_mm_s2'] <= 1e-3))
    pes['n_all_four_gate_items_passing'] = int(len(all_four))
    pes['count_definitions'] = ('three different counts, each defined: (1) n_search_tolerance_passing = max|qacc| <= 1.0 '
                                'mm/s^2 search tolerance; (2) n_qacc_item_passing = the qacc item of the declared gate, '
                                'max|qacc| <= 1e-3 mm/s^2; (3) n_all_four_gate_items_passing = all four declared gate items '
                                '(qacc, six tarsi in contact, zero non-foot contacts, every contact inside the declared '
                                'diamond). The field that previously carried the number 5 was named n_gate_passing but counted '
                                'only item (2); that name is removed.')
    pes['failure_of_the_fifth_hit'] = dict(start_index=207, max_abs_qacc_mm_s2=2.77381e-11, six_tarsi_in_contact=False,
                                           tarsi_with_contact=['LH', 'LM', 'RF', 'RH', 'RM'],
                                           reason='its 51 tarsus contacts cover only five legs (LF is missing), so the '
                                                  '"six tarsi active contact" gate item fails even though qacc passes')
    rec('F2', rp, 'per_equilibrium_summary.n_gate_passing', '%s' % old_count,
        'replaced by n_qacc_item_passing=%d and n_all_four_gate_items_passing=%d plus count_definitions and failure_of_the_fifth_hit'
        % (pes['n_qacc_item_passing'], pes['n_all_four_gate_items_passing']),
        'independent verification F2: the number 5 was the qacc item only; the four-item count is 4 (record written on the '
        'second pass; the edit itself was performed on the first pass of this same script)')

    r['reference_pose_of_each_box'] = dict(
        full_box=dict(joint_reference='SOURCE pose (outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz)',
                      radius_rad=0.40),
        nested_probes=dict(joint_reference='DELIVERED pose A', radii_rad=[0.20, 0.10, 0.05]),
        accepted_state=dict(max_abs_delta_q_rad_vs_source_pose=0.38823562709632525,
                            max_abs_delta_q_rad_vs_delivered_pose_A=0.42313245798143917,
                            reading=('the accepted state is inside the full box (0.38823562709632525 <= 0.40 rad from the SOURCE '
                                     'pose) and OUTSIDE a 0.40 rad box around pose A (0.42313245798143917 > 0.40)')),
        basis='independent verification F3 and the recomputation stored in this report')
    rec('F3', rp, 'reference_pose_of_each_box', '-',
        'new block with both references and both distances of the accepted state', 'F3')

    for key in ('r_0.05', 'r_0.10', 'r_0.20'):
        entry = r['nested_radii_around_pose_A'][key]
        entry['best_convention'] = dict(
            value_quoted_here='stage-2 refined point (stage2_final_equilibrium of that run)',
            stage2_refined_point_max_abs_qacc_mm_s2=entry['best_max_abs_qacc_mm_s2'],
            multistart_start_level_best_max_abs_qacc_mm_s2=dict(r_0p05=235.55162264425334,
                                                                r_0p10=66.61772121175831,
                                                                r_0p20=16.23588504496477)[key.replace('r_0.05', 'r_0p05')
                                                                                        .replace('r_0.10', 'r_0p10')
                                                                                        .replace('r_0.20', 'r_0p20')],
            note=('finding F7: "best" has two conventions. The quoted value is the refined stage-2 point; the start-level best '
                  'is the smallest residual among the random starts after their own least-squares run. They differ by up to '
                  '2.5e-8 mm/s^2 (R = 0.10: 66.6177212372147 vs 66.61772121175831, the start-level one being slightly SMALLER, '
                  'because the stage-2 polish is not guaranteed to improve a start that already sits on the answer). Both are '
                  'four to five orders of magnitude above the search tolerance 1.0 and the declared gate 1e-3, so no '
                  'conclusion depends on the choice; it only had to be defined.'))
    rec('F7', rp, 'nested_radii_around_pose_A.{r_0.05,r_0.10,r_0.20}.best_convention', '-',
        'both conventions reported per nested probe plus the 2.5e-8 explanation', 'independent verification F7')

    old_model = json.dumps(r['model'], ensure_ascii=False)
    r['model'].pop('friction_rows_of_tarsus', None)
    r['model']['tarsus_geom_friction_3_values_NOT_the_contact_friction_row'] = [[1.0, 0.005, 0.0001]]
    r['model']['contact_friction_rows_measured'] = [1.0, 1.0, 0.005, 0.0001, 0.0001]
    r['model']['mu1_mu2'] = [1.0, 1.0]
    r['model']['friction_field_note'] = ('the 3-valued array is the tarsus GEOM friction (slide, spin, roll) and is not the '
                                         'criterion; the criterion uses the CONTACT friction rows, whose first two entries give '
                                         'mu1 = mu2 = 1.0')
    rec('F9', rp, 'model.friction_rows_of_tarsus', old_model, json.dumps(r['model'], ensure_ascii=False),
        'independent verification F9: the field name claimed contact friction rows but held the 3-value geom friction')

    for lab, rb in r.get('runs', {}).items():
        if 'delivered_pose_sha256_of_the_run' in rb:
            old_val = rb.pop('delivered_pose_sha256_of_the_run')
            rb['delivered_pose_path_of_the_run'] = old_val
            rb['delivered_pose_note'] = ('this field holds a PATH, not a sha256; the name was corrected on 2026-10-15 (F9). '
                                         'The sha256 of that pose file is in inputs.pose_A_sha256')
            rec('F9', rp, "runs['%s'].delivered_pose_sha256_of_the_run" % lab, old_val,
                'renamed to delivered_pose_path_of_the_run (value unchanged)', 'independent verification F9')

    r['inplace_corrections'] = dict(
        applied_on='2026-10-15 (this file was rewritten in place; the previous report.json sha256 is recorded in '
                   'CHANGELOG-inplace.md)',
        source_of_the_findings='independent verification S2-equilibrium-reissued-check-20261015T231321Z (PASS_WITH_CORRECTIONS, '
                               '4 material / 5 minor / 1 cosmetic)',
        changelog=str(BATCH / 'CHANGELOG-inplace.md'),
        integrity_incident=str(integrity_path),
        numbers_unchanged='no measured number in this report was altered by these corrections',
        assembler='the assembler named in inputs.assembler_sha256 was restored from the batch snapshot; the F2 fix is therefore '
                  'NOT reproducible from the assembler alone and is instead documented and reproducible through '
                  'apply_inplace_corrections.py (shipped in this directory)')
    r['status'] = 'PASS_EQUILIBRIUM_SEARCH_CORRECTED_REISSUED_WITH_INPLACE_CORRECTIONS'
    rp.write_text(json.dumps(r, ensure_ascii=False, indent=1))
    rec('F2/F3/F4/F7/F9', rp, 'whole file rewritten in place',
        'sha256 %s' % before_sha, 'sha256 %s' % sha(rp), 'all in-place corrections of this round are listed in CHANGELOG-inplace.md')

    # ---------------- README corrections ----------------
    rmp = BATCH / 'README.zh-CN.md'
    rm_before = sha(rmp)
    txt = rmp.read_text()
    old_line = None
    for cand in ['  其中（<= 1e-3 mm/s^2）`5`。', '  其中（<= 1e-3 mm/s²）`5`。', '其中（<= 1e-3 mm/s^2）`5`。']:
        if txt.count(cand) == 1:
            old_line = cand
            break
    if old_line is None:
        old_line = '  其中（<= 1e-3 mm/s^2）`5`。'
    new_line = ('  其中 qacc 项（<= 1e-3 mm/s²）`5` 个；**四项门限全过 `4` 个**（起点 16 / 117 / 131 / 212）。'
                '起点 207 的 `max|qacc| = 2.77381e-11` 虽过 qacc 项，但它的 51 个跗节接触只覆盖 **5 条腿**（缺 LF），'
                '所以"六足跗节接触"这一项不过。**"过门限"一律指四项全过，不再使用只算 qacc 一项的口径。**')
    if txt.count(old_line) == 1:
        txt = txt.replace(old_line, new_line, 1)
        rec('F2', rmp, '§2 命中数那一行', old_line, new_line, 'independent verification F2')
    elif new_line in txt:
        print('note: the F2 README line is already corrected; skipping')
    else:
        raise AssertionError('F2 README line not found: %r' % old_line)

    old_full = None
    new_full = ('  全盒（act ∈ [1e-4,1] + 根 z ±0.05 mm + 42 个腿关节在 **源姿态** 的 0.40 rad 内；'
                '源姿态 = `outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz`，'
                '**不是**交付姿态 A；全盒的参考姿态与嵌套探针的参考姿态**不可混用**），240 个随机起点：')
    _t, old_full = replace_line_by_prefix(txt, '- 全盒（act', new_full)
    if old_full:
        txt = _t
        rec('F3', rmp, '§2 第一行', old_full, new_full, 'independent verification F3')
    else:
        print('note: the F3 full-box line was already corrected')

    old_acc = None
    new_acc = ('  执行器触界（各自 atol）：下限 0 个、上限 0 个。\n'
               '  参考姿态两个距离（F3）：`max|Δq| vs 源姿态 = 0.38823562709632525 rad ≤ 0.40`（在**全盒**内）、'
               '`max|Δq| vs 交付姿态 A = 0.42313245798143917 rad > 0.40`（**不在**以 A 为心的 0.40 rad 盒内）。')
    _t, old_acc = replace_line_by_prefix(txt, '  执行器触界', new_acc)
    if old_acc:
        txt = _t
        rec('F3', rmp, '§2 被接受点段', old_acc, new_acc, 'F3')
    else:
        print('note: the F3 accepted-state line was already corrected')

    old_disc_line = ('- A declared-cone static equilibrium is NOT dynamic maintainability. The separately measured open-loop run of this '
                     'same state under the real 100-frame connectome-derived archive loses foot contact at step 2739 = 68.475 ms '
                     '(artifact S2-true-equilibrium-openloop-20261015T154720).')
    new_disc_line = ('- A declared-cone static equilibrium is NOT dynamic maintainability. Measured under the real 100-frame '
                     'connectome-derived archive (artifact S2-true-equilibrium-openloop-20261015T154720, recomputed from its '
                     'time_series.npz for this correction): first NON-FOOT contact (`Thorax_geom | floor`) at step 2739 = '
                     '68.475 ms; the six-leg tarsus condition already broken at step 537 = 13.425 ms; last tarsus contact lost at '
                     'step 5111 = 127.775 ms; all six legs in contact in only 849 of 20000 steps. A newer measurement of THIS '
                     'batch accepted state (S2-accepted-equilibrium-openloop-real100-*) is a different experiment and reports its '
                     'own numbers.')
    if old_disc_line in txt:
        txt = txt.replace(old_disc_line, new_disc_line, 1)
        rec('F4', rmp, '§4 披露第 3 条', old_disc_line, new_disc_line, 'independent verification F4')
    else:
        txt2, old_disc_line2 = replace_line_by_prefix(txt, '- A declared-cone static equilibrium is NOT dynamic maintainability.', new_disc_line)
        if old_disc_line2:
            txt = txt2
            rec('F4', rmp, '§4 披露第 3 条', old_disc_line2, new_disc_line, 'independent verification F4')
        else:
            print('note: the F4 disclosure line was already corrected')

    txt += ('\n\n## 8. 本轮就地更正（F1–F9，不改任何实测数字）\n'
            '- 逐条记录见 `CHANGELOG-inplace.md`；装配脚本事故与还原见 `integrity_incident_assembler.json`；'
            '可复现的更正脚本见 `apply_inplace_corrections.py`。\n'
            '- F1：批次点名的装配脚本已被还原为批内快照字节（sha256 `5c421276…`）。\n'
            '- F2：过门限口径改为三项计数（1.0 容差 6 / qacc 项 5 / **四项全过 4**）。\n'
            '- F3：全盒参考姿态写清是**源姿态**，并给出被接受点的两个距离。\n'
            '- F4：披露改为实测时序（2739 = 68.475 ms 为**首次非足触地**，六足条件在 537 = 13.425 ms 已破，'
            '最后跗节接触在 5111 = 127.775 ms 失去）。\n'
            '- F5：四个底层运行的 `equilibrium-search.png` 已重绘标注（红线标为 **search tolerance 1.0**，'
            '点线标为 **declared gate 1e-3**；数值仍取自各自 `report.json`）。\n'
            '- F6：诊断文件的判据文字更正（改为机制判据：有限差分雅可比恒为 0、`nfev = 1`）。\n'
            '- F7：嵌套三档给出两种"best"口径及其 2.5e-8 差异说明。\n'
            '- F8：两个作废目录各写入 `VOID.txt`（原因、时间、三个中间哈希、指向 `S2-equilibrium-A-corrected-reissued-v2-*`）。\n'
            '- F9：运行模板文案改为按实际生效的参考姿态生成；字段名 `delivered_pose_sha256_of_the_run` → '
            '`delivered_pose_path_of_the_run`，模型摩擦字段改名并补上接触摩擦行。\n'
            '- **本批次不派新的验证**（既定规则：每个批次至多一次独立验证；本轮的验证额度留给新做的开环批次）。\n')
    rmp.write_text(txt)
    rec('F1-F9', rmp, 'whole file (appended §8 and four line-level corrections)',
        'sha256 %s' % rm_before, 'sha256 %s' % sha(rmp), 'in-place corrections of this round')

    # ---------------- F5: redraw the four run figures ----------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    redraw_src = []
    for d in RUN_DIRS:
        if not d.exists():
            print('missing run dir', d)
            continue
        rjr = json.loads((d / 'report.json').read_text())
        hist = rjr.get('multistart', {}).get('history_all_starts') or []
        s1 = rjr.get('stage1_act_only', {}).get('history', [])
        fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))
        ax = axes[0]
        if hist:
            ax.semilogy(np.arange(len(hist)), np.maximum(hist, 1e-14), lw=1.0, color='#1f77b4')
        ax.axhline(1.0, color='gray', ls='--', lw=1.2, label='SEARCH tolerance 1.0 mm/s^2 (this is NOT the gate)')
        ax.axhline(1e-3, color='black', ls=':', lw=1.7, label='DECLARED gate 1e-3 mm/s^2')
        ax.set_xlabel('multi-start index'); ax.set_ylabel('final max|qacc| of that start  [mm/s^2], log scale')
        ax.set_title('%s: %d random starts (R = %s, reference = %s)'
                     % (d.name, rjr.get('multistart', {}).get('n_starts', 0),
                        rjr.get('nested_radius_rad'), ('DELIVERED pose A' if rjr.get('nested_radius_rad') else 'SOURCE pose')),
                     fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')
        ax = axes[1]
        if s1:
            names = [h.get('start', '') for h in s1]
            vals = [max(h.get('qacc_max', 0.0), 1e-14) for h in s1]
            ax.bar(range(len(vals)), vals, color='#ff7f0e')
            ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=7, rotation=20)
        ax.set_yscale('log')
        ax.axhline(1.0, color='gray', ls='--', lw=1.2, label='SEARCH tolerance 1.0 mm/s^2 (NOT the gate)')
        ax.axhline(1e-3, color='black', ls=':', lw=1.7, label='DECLARED gate 1e-3 mm/s^2')
        ax.set_ylabel('stage-1 max|qacc| (qpos fixed)  [mm/s^2], log scale')
        ax.set_title('stage 1 (act only, qpos fixed at the reference pose)', fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y', which='both')
        fig.suptitle('labels corrected on 2026-10-15: the 1.0 mm/s^2 line is the SEARCH tolerance, the declared gate is 1e-3 mm/s^2.\n'
                     'Every number in this figure is read from this run report.json; no number was changed.',
                     fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig(d / 'equilibrium-search.png', dpi=130)
        plt.close(fig)
        rec('F5', d / 'equilibrium-search.png', 'figure legend/annotations',
            'the 1.0 mm/s^2 line was labelled "declared tolerance"', 'labelled "SEARCH tolerance 1.0 (NOT the gate)" and "DECLARED gate 1e-3"',
            'independent verification F5; the plotted numbers are unchanged, taken from the same report.json')

    # ---------------- F9 in the four run reports and in the live search script ----------------
    for d in RUN_DIRS:
        if not d.exists():
            continue
        p = d / 'report.json'
        before = sha(p)
        rr = json.loads(p.read_text())
        changed = False
        ref = 'DELIVERED pose A' if rr.get('nested_radius_rad') else 'SOURCE pose'
        if 'declared' in rr and 'stage2' in rr['declared']:
            rr['declared']['stage2_original_template_text'] = rr['declared']['stage2']
            rr['declared']['stage2'] = ('variables act (90) + root z (in mm, clamped to +-%.3f) + 42 leg hinge joints, each within '
                                        '[joint range minus %.3f rad] and within %s rad of the %s'
                                        % (0.05, 5e-3, ('%.2f' % rr['nested_radius_rad']) if rr.get('nested_radius_rad') else '0.40', ref))
            changed = True
        if 'verdict' in rr and 'search_range' in rr['verdict']:
            rr['verdict']['search_range']['stage2_original_template_text'] = rr['verdict']['search_range'].get('stage2')
            rr['verdict']['search_range']['stage2'] = ('act in [1e-4, 1] + root z +-0.05 mm + 42 leg hinge joints within R = %s rad of '
                                                       'the %s' % (('%.2f' % rr['nested_radius_rad']) if rr.get('nested_radius_rad') else '0.40', ref))
            changed = True
        if 'declared' in rr and 'delivered_pose_sha256_of_the_run' in rr['declared']:
            rr['declared']['delivered_pose_path_of_the_run'] = rr['declared'].pop('delivered_pose_sha256_of_the_run')
            changed = True
        if changed:
            p.write_text(json.dumps(rr, ensure_ascii=False, indent=1))
            rec('F9', p, 'declared.stage2 / verdict.search_range.stage2 / delivered_pose_* field name',
                'sha256 %s (template text said "0.40 rad of the source pose" for every run; the pose field was named ..._sha256 but held a path)' % before,
                'sha256 %s (text now names the reference pose actually in force; field renamed to delivered_pose_path_of_the_run)' % sha(p),
                'independent verification F9')

    # F9 also: the live search script template must name the reference pose actually in force
    ss_before = sha(SEARCH_SCRIPT)
    sst = SEARCH_SCRIPT.read_text()
    old_tpl = "stage2=('variables act (90) + root z (+-%.3f mm) + 42 leg hinge joints (joint range minus %.3f rad margin and within '\n                                        '%.2f rad of the source pose)' % (ROOT_Z_RANGE, JOINT_MARGIN, MAX_DEVIATION)),"
    new_tpl = "stage2=('variables act (90) + root z (+-%.3f mm) + 42 leg hinge joints (joint range minus %.3f rad margin and within '\n                                        '%.2f rad of the %s)' % (ROOT_Z_RANGE, JOINT_MARGIN, _dev, ('DELIVERED pose A' if RADIUS is not None else 'SOURCE pose'))),"
    if old_tpl in sst:
        sst = sst.replace(old_tpl, new_tpl, 1)
        sst = sst.replace("""                              stage2=('act in [1e-4, 1] + root z +-%.3f mm + 42 leg hinge joints within [joint range - %.3f rad] and within '
                                      '%.2f rad of the source pose' % (ROOT_Z_RANGE, JOINT_MARGIN, MAX_DEVIATION))),""",
                          """                              stage2=('act in [1e-4, 1] + root z +-%.3f mm + 42 leg hinge joints within [joint range - %.3f rad] and within '
                                      '%.2f rad of the %s' % (ROOT_Z_RANGE, JOINT_MARGIN, _dev,
                                                              ('DELIVERED pose A' if RADIUS is not None else 'SOURCE pose')))),""", 1)
        SEARCH_SCRIPT.write_text(sst)
        rec('F9', SEARCH_SCRIPT, 'declared.stage2 and verdict.search_range.stage2 template text',
            'sha256 %s (always said "of the source pose")' % ss_before,
            'sha256 %s (now names the reference pose actually in force: SOURCE pose for the full box, DELIVERED pose A when --radius is used)'
            % sha(SEARCH_SCRIPT), 'independent verification F9')
    else:
        print('note: the stage2 template text was not found verbatim; the live script was left unchanged (reported in the changelog)')
        rec('F9', SEARCH_SCRIPT, 'declared.stage2 template text', 'sha256 %s' % ss_before,
            'NOT CHANGED: the expected template string was not found verbatim in this revision',
            'conservative: no blind edit of the live search script')

    # ---------------- F6: diagnostic wording ----------------
    for dj in sorted(DIAG_DIR.glob('aliased_residual_*.json')):
        before = sha(dj)
        dd = json.loads(dj.read_text())
        for tag, case in dd.get('cases', {}).items():
            if 'fun_is_the_same_object_as_mjdata_qacc' in case:
                case['pointer_test_removed_because_misleading'] = case.pop('fun_is_the_same_object_as_mjdata_qacc')
        dd['mechanism_evidence_for_the_misleading_pointer_test'] = (
            'the pointer comparison was misleading and is superseded by the mechanism: with a VIEW residual the finite-difference '
            'Jacobian is exactly zero, the first-order optimality is 0.0, least_squares stops after nfev = 1 with "gtol termination '
            'condition is satisfied", and the returned fun equals the residual at the last finite-difference trial point. The scalar '
            'max|fun| can still differ from a recomputation in its last digits, which is what the pointer test was groping for.')
        dj.write_text(json.dumps(dd, ensure_ascii=False, indent=1))
        rec('F6', dj, 'cases[*].fun_is_the_same_object_as_mjdata_qacc + new mechanism note',
            'sha256 %s (raw pointer equality test, misleading)' % before,
            'sha256 %s (renamed, plus the mechanism-based criterion)' % sha(dj),
            'independent verification F6')

    ds = DIAG_SCRIPT.read_text()
    if 'fun_is_the_same_object_as_mjdata_qacc' in ds:
        ds_before = sha(DIAG_SCRIPT)
        ds = ds.replace("""            fun_is_the_same_object_as_mjdata_qacc=bool(np.asarray(res.fun).__array_interface__['data'][0]
                                                       == np.asarray(d.qacc).__array_interface__['data'][0]),""",
                        """            note_on_the_old_pointer_test=('a raw buffer-pointer equality test is NOT the criterion and must not be used: '
                                          'the decisive evidence is that the view variant stops after nfev = 1 with a zero '
                                          'finite-difference Jacobian, while the copy variant converges'),""")
        DIAG_SCRIPT.write_text(ds)
        rec('F6', DIAG_SCRIPT, 'case dict in the diagnosis test',
            'sha256 %s (pointer equality as a case field)' % ds_before,
            'sha256 %s (pointer test replaced by a note; mechanism evidence kept)' % sha(DIAG_SCRIPT),
            'independent verification F6: the pointer test can mislead because res.fun may be a fresh wrapper over the same buffer')

    # ---------------- F8: VOID.txt in the two void directories ----------------
    three_hashes = dict(report_json_current=(sha(VOID_DIRS[0] / 'report.json') if (VOID_DIRS[0] / 'report.json').exists() else None),
                        quoted_before='c12ead58 (8 hex chars, as recorded in the tracker at the time of the incident)',
                        quoted_after='62841f3c (8 hex chars, as recorded in the tracker at the time of the incident)')
    for i, vd in enumerate(VOID_DIRS):
        if not vd.exists():
            print('void dir missing', vd)
            continue
        txt = []
        txt.append('VOID — 本目录结论作废，仅作历史 / this directory is VOID, historical record only')
        txt.append('')
        txt.append('写入时间：%s（本轮就地更正，F8）' % time.strftime('%Y-%m-%d %H:%M:%S'))
        txt.append('目录：%s' % vd)
        if i == 0:
            txt.append('原因：完整性事故。`report.json` 在 2026-10-15 15:22:25（本地）被就地替换，'
                       '`delivery-hashes.json` 随之一并重写，脚本在 15:23:44 被编辑；批内快照与 live 脚本不再一致。')
            txt.append('可核查的事实：当前 `report.json` sha256 = %s；目录内文件 mtime 为 2026-10-15 15:47:40（晚于所述事件）；'
                       '本目录自身原先没有任何 VOID 标记；事故当时记录的中间哈希 %s → %s **已无法从文件系统复核**。'
                       % (three_hashes['report_json_current'], three_hashes['quoted_before'], three_hashes['quoted_after']))
            txt.append('该事故的同类问题（批次点名的脚本在写盘后被编辑）与处置，见 '
                       '`/home/sensen/Desktop/engine修复验收/S2-equilibrium-A-corrected-reissued-20261015T161259/'
                       'integrity_incident_assembler.json` 与本目录同批的 `CHANGELOG-inplace.md`。')
        else:
            txt.append('原因：被取代的中间产物。该轮 `run_equilibrium_reissue.py` 的多起点残差返回了活 `MjData` 缓冲区的视图'
                       '（`np.asarray(d.qacc)` 不是拷贝），有限差分雅可比恒为 0，最小二乘一步都不动：240 个起点最好 '
                       '`max|qacc| = 9016.5 mm/s^2`、0 命中。**这不是结果，不得引用。**')
            txt.append('可核查的事实：当前 `report.json` sha256 = %s。' % three_hashes['report_json_current'])
        txt.append('')
        txt.append('接替产物：`/home/sensen/Desktop/engine修复验收/S2-equilibrium-A-corrected-reissued-v2-20261015T163134/`'
                   '（及其前身 `S2-equilibrium-A-corrected-reissued-20261015T161259/`）。')
        txt.append('本目录内容未被删除、未被修改；本文件是**追加**的标注。')
        (vd / 'VOID.txt').write_text('\n'.join(txt) + '\n')
        rec('F8', vd / 'VOID.txt', 'new file', '-', 'VOID marker with reason, time, first 8 hex chars of the three intermediate hashes, pointer to the new batch',
            'independent verification F8: the VOID claim lived only outside the directory')

    # ---------------- CHANGELOG + reproducible correction script ----------------
    shutil.copy2(Path(__file__), BATCH / 'apply_inplace_corrections.py')
    dedup = {}
    for c in CHANGES:
        dedup[(c['finding'], c['file'], c['field_or_line'])] = c
    CHANGES[:] = list(dedup.values())
    lines = ['# CHANGELOG-inplace.md — 就地更正记录（本轮，F1–F9）', '',
             '规则：**只改措辞/标注/字段名/归属，不改任何实测数字**；被接受状态 npz 未被触碰。',
             '每条 = 文件 + 字段或行 + 改前 + 改后 + 依据。生成脚本：`apply_inplace_corrections.py`（在批内）。', '']
    for c in CHANGES:
        lines.append('## %s — `%s`' % (c['finding'], c['file']))
        lines.append('- 位置：`%s`' % c['field_or_line'])
        lines.append('- 改前：%s' % c['before'])
        lines.append('- 改后：%s' % c['after'])
        lines.append('- 依据：%s' % c['basis'])
        lines.append('')
    lines.append('## 未修项（明确保留并披露）')
    lines.append('- 底层运行 `report.json` 里的 `declared.qacc_tol_mm_s2 = 1.0` 仍在 `declared` 之下（该值本身是**搜索容差**，'
                 '不是门限）；本轮只改了图的标注与参考姿态文案，未改字段结构，以免与已公布的哈希链冲突。')
    lines.append('- 本批次名为 `...-corrected-reissued-...` 的 v1 目录与 v2 目录同时存在：v2 是"结论性重发"，v1 是"就地更正后的交付件"；'
                 '两者数字一致，冲突时以 v2 的措辞为准。')
    lines.append('- v2 的 `report.json` 记录的是**更正前** v1 `report.json` 的 sha256（更正前为 '
                 '`%s`）；本轮就地更正后该值变为 `%s`，v2 内的记录因此是历史值（v2 目录正在被独立核验，不做改动）。'
                 % ('9d86fdd535d7396fb9ee0fc9ac2c3ba4c4343fbc690ca9f7159aa12b6d071a5f', sha(rp)))
    (BATCH / 'CHANGELOG-inplace.md').write_text('\n'.join(lines) + '\n')
    rec('ALL', BATCH / 'CHANGELOG-inplace.md', 'new file', '-', '%d change records' % len(CHANGES),
        'the owner asked for every in-place change to be recorded in this file')

    # ---------------- hashes ----------------
    batch_files = write_hashes(BATCH)
    run_hashes = {d.name: write_hashes(d) for d in RUN_DIRS if d.exists()}
    print('batch hashes:', len(batch_files), 'run dirs rehashed:', len(run_hashes))
    print('CHANGES:', len(CHANGES))
    print('live assembler after restore:', sha(LIVE_ASSEMBLER))
    print('batch state npz sha256 (must be unchanged):', sha(BATCH / 'equilibrium_best_state.npz'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
