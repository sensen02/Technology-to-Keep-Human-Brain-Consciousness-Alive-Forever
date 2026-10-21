"""Build the v3 batch: a single-writer consolidation of the F-A facts, the F2 counts, the hash accounting (two classes),
the authority decision for the live-vs-snapshot inconsistency, the factual F-C record, and the open-loop results.

v3 is a NEW timestamped directory; the already-verified v1/v2/ladder directories are frozen and are not written again.
"""

import hashlib
import json
import pathlib
import sys
import time
import zipfile

import numpy as np

ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
sys.path.insert(0, str(ROOT))
V1 = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259'
V2 = DESK / 'S2-equilibrium-A-corrected-reissued-v2-20261015T163134'
LADDER = DESK / 'S2-equilibrium-radius-ladder-20261015T161518'
ART = DESK / 'S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164559'
LOC = DESK / 'S2-accepted-equilibrium-localization-20261015T170058'
OLD_LOOP = DESK / 'S2-true-equilibrium-openloop-20261015T154720'
STATE = V1 / 'equilibrium_best_state.npz'
DRIVE = ROOT / 'outputs/experiment_platform/20261015_135150_a2fff3f0/traces.npz'
SEARCH = ROOT / 'run_mujoco_equilibrium_search.py'
DIAG = ROOT / 'diagnose_aliased_residual.py'
LIVE = ROOT / 'run_equilibrium_reissue_finalize.py'
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S')

FACT_EN = ('first non-foot contact at step 2739 = 68.475 ms, pair (floor, A6_geom) with A6_geom belonging to body A6; the first '
           'Thorax_geom | floor contact appears at step 3055 = 76.400 ms; this is a NON-FOOT (trunk/limb, non-tarsus) contact and '
           'NOT a loss of foot contact (first_tarsus_loss_step = 5111, legs_in_contact < 6 from step 537 = 13.425 ms)')
FACT_CN = ('首次非足接触在步 2739 = 68.475 ms，接触对 `(floor, A6_geom)`，`A6_geom` 属 body A6；首个 `Thorax_geom | floor` 接触在步 '
           '3055 = 76.400 ms；这是**足外（躯干/肢体、非跗节）触地**，**不是"失去足接触"**（`first_tarsus_loss_step = 5111`；'
           '`legs_in_contact` 自步 537 = 13.425 ms 起少于 6）')
OLD_LADDER_SENTENCE = ('- 步 2739 = 68.475 ms 首次失去足接触（ladder README 原句，作废但按指示保留为历史）')


def first_u_diamond_contact():
    """the contact pair at the first step whose max u_diamond reaches 1.0 (tiny replay of the first few steps)"""
    import mujoco
    from engine import canonical_physics as cp
    m = cp.load_model()
    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
    st = np.load(STATE)
    ctrl = np.asarray(np.load(DRIVE)['normal_controls'], dtype=float)
    d = mujoco.MjData(m)
    d.qpos[:] = st['qpos']
    d.qvel[:] = 0.0
    d.act[:] = st['act']
    d.ctrl[:] = 0.0
    mujoco.mj_forward(m, d)
    f6 = np.zeros(6)
    for step in range(6):
        idx = min(step // 200, ctrl.shape[0] - 1)
        d.ctrl[:] = ctrl[idx]
        mujoco.mj_step(m, d)
        best = (0.0, None, None)
        for ci, c in enumerate(d.contact):
            if c.efc_address < 0:
                continue
            n1, n2 = gname(int(c.geom1)), gname(int(c.geom2))
            if 'Tarsus' not in (n1 + n2):
                continue
            mujoco.mj_contactForce(m, d, ci, f6)
            fn, ft1, ft2 = float(f6[0]), float(f6[1]), float(f6[2])
            fr = np.asarray(c.friction)[:3]
            if fn > 0:
                u = abs(ft1) / (float(fr[0]) * fn) + abs(ft2) / (float(fr[1]) * fn)
                if u > best[0]:
                    best = (u, sorted((n1, n2)), fn)
        if best[0] >= 1.0:
            return dict(step=step + 1, ms=round((step + 1) * 25e-6 * 1000, 3), u_diamond=best[0], pair=best[1], fn=best[2])
    return None


def main():
    out = DESK / ('S2-equilibrium-A-corrected-reissued-v3-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)

    v1r = json.loads((V1 / 'report.json').read_text())
    v2r = json.loads((V2 / 'report.json').read_text())
    art = json.loads((ART / 'report.json').read_text())
    loc = json.loads((LOC / 'report.json').read_text())
    first_ud = first_u_diamond_contact()

    # ---- wording record with data sources ----
    wording = dict(
        status='PASS_F_A_FACTS_WITH_SOURCES', fact_en=FACT_EN, fact_cn=FACT_CN,
        data_sources=dict(
            step_numbers=dict(file=str(OLD_LOOP / 'time_series.npz'),
                              arrays=['true_equilibrium_pose__qualified_25us__nonfoot_contacts',
                                      'true_equilibrium_pose__qualified_25us__tarsus_contacts',
                                      'true_equilibrium_pose__qualified_25us__legs_in_contact',
                                      'true_equilibrium_pose__qualified_25us__leg_tarsus_counts'],
                              steps=dict(first_nonfoot=2739, first_legs_below_6=537, first_tarsus_loss=5111,
                                         first_step_with_only_one_leg=2739),
                              note='those arrays give the step numbers; the state npz of that artifact stores no contact geometry'),
            geom_identity=dict(measured_by='replaying the same protocol and reading the contact geom pairs (the stored arrays hold counts, not '
                                           'pairs); independently reproduced by the verification S2-equilibrium-v2-check-20261015T233211Z',
                               first_pair_at_2739=['floor', 'A6_geom'], first_Thorax_geom_pair_step=3055)),
        historical_wrong_wording=[dict(where=str(LADDER / 'README.zh-CN.md'), sentence=OLD_LADDER_SENTENCE,
                                       status='VOID — kept as history on the owner\'s instruction, not deleted',
                                       why_wrong='step 2739 is a body-floor (non-tarsus) contact, not a loss of foot contact')],
        places_now_carrying_the_fact=json.loads((V2 / 'contact_identity_wording_places.json').read_text())['places'],
        recorded_at=now())
    (out / 'wording_F-A.json').write_text(json.dumps(wording, ensure_ascii=False, indent=1))

    # ---- F2 counts with definitions ----
    pes = v1r['per_equilibrium_summary']
    counts = dict(
        n_all_four_gate_items_passing=pes['n_all_four_gate_items_passing'],
        n_qacc_item_passing=pes['n_qacc_item_passing'],
        n_search_tolerance_passing=pes['n_search_tolerance_passing'],
        definitions=dict(
            n_all_four_gate_items_passing='hits satisfying ALL FOUR declared gate items: max|qacc| <= 1e-3 mm/s^2 AND six tarsi in active '
                                          'contact AND zero non-foot contacts AND every contact inside the declared axis-ray diamond '
                                          'u_diamond = u1 + u2 <= 1',
            n_qacc_item_passing='hits satisfying the qacc item only: max|qacc| <= 1e-3 mm/s^2',
            n_search_tolerance_passing='starts whose final max|qacc| is at or below the search tolerance 1.0 mm/s^2'),
        why_they_differ=('start 207 reaches max|qacc| = 2.77381e-11 mm/s^2 but has six_tarsi_in_contact = false: its 51 tarsus contacts cover '
                         'only five legs (LF missing, tarsi_with_contact = [LH, LM, RF, RH, RM]), so it fails the six-tarsi item and is NOT an '
                         'all-four hit; this is why the all-four count is 4 while the qacc-item count is 5'),
        per_start_table=pes.get('gate_table'), recorded_at=now())
    (out / 'gate_counts_F2.json').write_text(json.dumps(counts, ensure_ascii=False, indent=1))

    # ---- hash accounting: two classes + authority decision ----
    ledger = json.loads((V2 / 'correction_ledger.json').read_text())
    realign = json.loads((V2 / 'hash_realignment.json').read_text())
    run_time_hashes = dict(
        explanation=('run-time hashes were recorded while the runs were made and are deliberately preserved: they are the evidence that the '
                     'delivered artifact and its snapshot agree. They are NOT rewritten to match later edits.'),
        examples=dict(state_npz=sha(STATE), drive=sha(DRIVE), canonical_xml=sha(ROOT / 'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml'),
                      v1_report_json_delivered='9d86fdd535d7396fb9ee0fc9ac2c3ba4c4343fbc690ca9f7159aa12b6d071a5f (as recorded at delivery)',
                      v2_report_json_delivered=v2r['post_definitions'] if False else 'see v2 delivery-hashes delivery_time_hashes'),
        rule='input_hashes / run-time hashes stay as they were; new observations are published next to them')
    post_correction_hashes = dict(
        explanation='hashes of files that the later correction round changed; both the old and the current value are published',
        records=[dict(field=r['field'], file=r['file'], old_sha256=r['old_sha256'], current_sha256=sha(r['file']))
                 for r in ledger['stale_fields_v2'] + ledger['stale_fields_ladder']],
        n_records=len(ledger['stale_fields_v2']) + len(ledger['stale_fields_ladder']),
        numbers_unchanged='the independent verification compared 14 numeric fields per rewritten run report with the pre-edit values: identical')
    authority = dict(
        status='PASS_AUTHORITY_DECISION', question='which is authoritative: the snapshot member or the live script?',
        choice='the LIVE script is authoritative, and a NEW post-correction snapshot member set is shipped with it',
        why=('the snapshot member 5c421276... cannot reproduce the delivered report.json (it lacks the two gate-count fields that the delivered '
             'report contains); the live file was rebuilt as snapshot + the five documented replacements and that rebuild reproduced '
             'e52f080b... exactly, which is the revision that produced the current delivered files'),
        evidence=dict(snapshot_member_sha256='5c4212768b89629f46623578fb9003e7cb67383e98077539ba5d63a3525aa2d0',
                      live_search_script=dict(path=str(SEARCH), sha256=sha(SEARCH)),
                      live_diagnostic_script=dict(path=str(DIAG), sha256=sha(DIAG)),
                      live_assembler=dict(path=str(LIVE), sha256=sha(LIVE)),
                      v2_post_correction_member_set=str(V2 / 'source_snapshot_post_correction.zip'),
                      rebuild_reproduced='e52f080bd0895fe36901f60398b60408e7dafe4871232656025113c5be754622'),
        recorded_at=now())
    (out / 'hash_reanchor.json').write_text(json.dumps(dict(
        status='PASS_HASH_REANCHOR_IN_V3', run_time_hashes=run_time_hashes,
        post_correction_hashes=post_correction_hashes, authority_decision=authority,
        stale_hash_disclosure=dict(v2=str(V2 / 'correction_ledger.json'), ladder_records=ledger['stale_fields_ladder']),
        recorded_at=now()), ensure_ascii=False, indent=1))

    # ship the authoritative live scripts as this batch's own member set
    with zipfile.ZipFile(out / 'source_snapshot_authoritative_live.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in (SEARCH, DIAG, LIVE, pathlib.Path(__file__)):
            z.write(p, p.name)

    # ---- F-C facts + v1 rewrite history + npz untouched ----
    facts = dict(
        status='PASS_INTEGRITY_FACTS', live_incident_record_status='WITHDRAWN in v1 (kept as history)',
        measured=['the live script, the copy the withdrawn record called the incident revision, and the source_snapshot.zip member are '
                  'BYTE-IDENTICAL (5c4212768b89629f46623578fb9003e7cb67383e98077539ba5d63a3525aa2d0) and NONE of them contains '
                  'per_equilibrium_summary.n_all_four_gate_items_passing or n_qacc_item_passing',
                  'therefore the incident conclusion as written does not hold and is withdrawn; the record stays in the v1 directory marked '
                  'WITHDRAWN, and this file is the factual version'],
        true_event=dict(before_sha256='5c4212768b89629f46623578fb9003e7cb67383e98077539ba5d63a3525aa2d0',
                        after_sha256='9271e578b6c54b8882c1d765af9f2ab67a13956449c0f061777810034c9c1742', mtime='2026-10-15 16:14:08',
                        replaced_file=str(LIVE), contains_F2_fields=True, bytes_recoverable=False,
                        consequence='the delivered files and the snapshot stayed mutually consistent; the live path could not reproduce the '
                                    'delivered report at that moment'),
        v1_rewrite_history=dict(observation='during the review the v1 report.json changed 9d86fdd5 -> 27b4dc24 -> 6212d3c0 (and later, with the '
                                            'gate table and the wording notes, further still)',
                                npz_untouched='equilibrium_best_state.npz stayed 976388f28a866793ba0738311919e0f1f0837b052ac8d37281ad685c1d77c5be '
                                              'throughout every observation, including the verifications',
                                current_sha256=sha(STATE)),
        superseded_verifications=['6f447df4-773e-41e8-87d3-5c352d65a0e0 (stopped: superseded scope)',
                                  '54d368b5-8456-4c51-8213-034fb7dae577 (stopped: superseded scope)',
                                  '70cac021-3d28-4c22-9eb7-9520df5f358f (stopped: superseded scope)',
                                  '3ec1a5b7-9645-4718-99d0-f495489d45f3 (stopped: artifact gained new measurements after it started)'],
        note='their partial content is NOT merged into v3; v3 has a single writer',
        recorded_at=now())
    (out / 'integrity_incident_facts.json').write_text(json.dumps(facts, ensure_ascii=False, indent=1))

    # ---- open-loop results summary ----
    res = art['result']
    loop_summary = dict(
        status='PASS_OPENLOOP_RESULTS', artifact=str(ART), artifact_report_sha256=sha(ART / 'report.json'),
        drive=dict(path=str(DRIVE), sha256=sha(DRIVE), frames=100, cadence_ms=5.0, is_the_100_frame_archive=True),
        protocol=dict(dt_s=2.5e-05, steps=20000, duration_s=0.5, state_writes='once', loop_writes='ctrl only'),
        first_non_foot_contact=dict(step=res['first_nonfoot_contact_step'], ms=res['first_nonfoot_contact_ms'],
                                    pair=res['first_nonfoot_contact_geom_pair'], owning_body_of_A6_geom='A6'),
        first_leg_to_lose_tarsus_contact=dict(leg=loc['Q1_leg_loss']['first_leg_to_lose_tarsus_contact'],
                                              step=loc['Q1_leg_loss']['first_step_with_no_tarsus_contact'],
                                              ms=loc['Q1_leg_loss']['first_step_with_no_tarsus_contact_ms']),
        first_leg_force_below_1pct=dict(leg=loc['Q1_leg_loss']['first_leg_whose_force_drops_below_1pct'],
                                        step=loc['Q1_leg_loss']['first_step_force_below_1pct'],
                                        ms=loc['Q1_leg_loss']['first_step_force_below_1pct_ms']),
        first_step_legs_below_6=dict(step=res['first_legs_in_contact_below_6_step'], ms=res['first_legs_in_contact_below_6_ms']),
        six_leg_steps=dict(steps=res['six_legs_in_contact_steps'], total=20001, fraction=res['six_legs_in_contact_fraction']),
        first_step_with_exactly_one_leg=dict(step=loc['Q1_leg_loss']['first_step_legs_in_contact_equals_1'],
                                             ms=loc['Q1_leg_loss']['first_step_legs_in_contact_equals_1_ms']),
        root_z=dict(initial=res['root_z_initial'], min=res['root_z_min'], min_step=res['root_z_min_step'], final=res['root_z_final']),
        energy=dict(initial=res['energy_initial'], final=res['energy_final'], monotone=res['energy_monotone_decreasing'],
                    rising_steps=res['energy_rising_steps'], compared=res['energy_steps_compared']),
        tilt=dict(max_deg=res['tilt_deg_max'], final_deg=res['tilt_deg_final'], flipped=res['flipped']),
        per_step_qacc=art['post_step_max_qacc'],
        first_u_diamond_equal_1=first_ud,
        first_joint_to_leave_the_equilibrium=dict(joint=loc['Q2_joint_deviation']['first_joint_to_leave_the_equilibrium'],
                                                 step=loc['Q2_joint_deviation']['first_step_joint_deviation_ge_0_01_rad'],
                                                 ms=loc['Q2_joint_deviation']['first_step_joint_deviation_ge_0_01_rad_ms']),
        conclusion_two_sentences=dict(
            static='a declared-cone static equilibrium EXISTS inside the declared box (max|qacc| = 2.5865734771464476e-11 mm/s^2, 48 DOF, 51 '
                   'tarsus contacts, max u_diamond = 3.227792564858203e-13): this only says qacc ~ 0 is a necessary condition at that instant',
            open_loop='it is NOT maintained open loop under the frozen connectome-derived drive: first non-foot contact at 7.000 ms, the six-leg '
                      'condition already broken at 6.500 ms, and the per-step max|qacc| and u_diamond cross the declared limits at step 2 = '
                      '0.05 ms; static feasibility is not maintainability, and neither is standing'),
        figures=[str(ART / 'accepted-equilibrium-openloop.png'), str(ART / 'openloop-zoom-evidence.png'),
                 str(LOC / 'accepted-equilibrium-localization.png')],
        recorded_at=now())
    (out / 'openloop_results.json').write_text(json.dumps(loop_summary, ensure_ascii=False, indent=1))

    report = dict(
        status='PASS_V3_CONSOLIDATED', version=3, manual_research_not_product=True, standing=False, walking=False,
        purpose=('single-writer consolidation of: the F-A contact-identity fact with its data sources, the F2 gate counts with definitions, the '
                 'hash accounting split into run-time and post-correction classes, the authority decision for the live-vs-snapshot mismatch, the '
                 'factual F-C record, and the open-loop results'),
        frozen_directories_not_written=['S2-equilibrium-A-corrected-reissued-20261015T161259 (v1)',
                                        'S2-equilibrium-A-corrected-reissued-v2-20261015T163134 (v2)',
                                        'S2-equilibrium-radius-ladder-20261015T161518 (ladder)'],
        wording=wording['fact_en'], wording_cn=wording['fact_cn'], gate_counts=counts,
        hash_accounting=json.loads((out / 'hash_reanchor.json').read_text()),
        integrity_facts=facts, openloop=loop_summary,
        not_established='absolute force values, physical units, biological validity; no standing, no walking')
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))

    lines = ['# S2 v3：接触身份事实、门限计数、哈希账、权威源与开环结果（手动研究实施，不是成品）', '',
             '## 1. F-A 接触身份（每处都标数字来源）',
             '- ' + FACT_CN,
             '- 数字来源：`' + str(OLD_LOOP / 'time_series.npz') + '` 的数组 `...__nonfoot_contacts`（步 2739 首次非零）、'
             '`...__legs_in_contact`（步 537 首次 <6，步 2739 只剩 1 条腿）、`...__tarsus_contacts`（步 5111 首次为 0）。'
             '**几何身份**（`floor | A6_geom`、`A6_geom` 属 body A6；首个 `Thorax_geom | floor` 在步 3055）由**同协议重放并读接触几何对**测得，'
             '已被独立核验 `S2-equilibrium-v2-check-20261015T233211Z` 复现；该时间序列不存几何对，只存计数。',
             '- 三处交付件（v1 README 第 40 行、v2 report.json 第 30 行与 v2 README 第 49 行、ladder README 第 19 行）现为同一句；'
             'ladder 原句**作为历史保留**（作废但不删）：`' + OLD_LADDER_SENTENCE + '`',
             '',
             '## 2. F2 门限计数（两个数都给，各附定义）',
             '- `n_all_four_gate_items_passing = **%d**`：同时满足**四项**（`max|qacc| ≤ 1e-3`、六足跗节接触、零足外接触、全部接触 `u_diamond ≤ 1`）。'
             % counts['n_all_four_gate_items_passing'],
             '- `n_qacc_item_passing = **%d**`：只满足 qacc 项（`max|qacc| ≤ 1e-3`）。' % counts['n_qacc_item_passing'],
             '- 两者差异原因：起点 207 的 `max|qacc| = 2.77381e-11` 过 qacc 项，但 `six_tarsi_in_contact = false` —— 51 个跗节接触只覆盖 5 条腿（缺 LF）。',
             '- 另有 `n_search_tolerance_passing = %d`（≤1.0 mm/s² 搜索容差）。逐起点表见 `gate_counts_F2.json`。'
             % counts['n_search_tolerance_passing'],
             '',
             '## 3. 哈希账（两类分开，不抹平）',
             '- **运行期哈希**（`input_hashes` / run-time）：**故意保留，不为对齐而改写** —— 状态 npz `976388f2…`、驱动 `37fe24d3…`、规范 XML `65d3904d…`、'
             '交付时 v1 `report.json` `9d86fdd5…`。',
             '- **运行后被修正影响的文件哈希**：逐条 **旧值 → 当前值 → 原因** 见 `hash_reanchor.json`（v2 %d 条 + ladder %d 条）。'
             % (len(ledger['stale_fields_v2']), len(ledger['stale_fields_ladder'])),
             '- **权威源（二选一已选定并留证）**：**以实盘脚本为权威**，并**附带新的修正后快照成员** '
             '`source_snapshot_authoritative_live.zip`（含 live 搜索脚本 `%s`、live 诊断脚本 `%s`、live 装配脚本 `%s`）。'
             '理由：快照成员 `5c421276…` **无法复现交付的 `report.json`**（缺两个计数字段）；实盘版本由"快照 + 五个已记录替换"重建并**恰好复现 '
             '`e52f080b…`**，而那一版正是产出当前交付件的版本。'
             % (sha(SEARCH)[:12], sha(DIAG)[:12], sha(LIVE)[:12]),
             '',
             '## 4. F-C 实测事实（旧的错误叙述保留为历史）',
             '- 实测：**live 脚本、incident 副本、快照成员三者字节相同**（`5c421276…`），**且都不含** `n_all_four_gate_items_passing` / '
             '`n_qacc_item_passing` ⇒ 原 incident 结论**不成立、已撤销**（v1 目录内该文件标 `WITHDRAWN`，作为历史条目保留）。',
             '- 真实事件保留：`5c421276… → 9271e578…`（mtime 2026-10-15 16:14:08，被替换文件 = `' + str(LIVE) + '`），'
             '字节**不可恢复**；后果 = 交付件与快照自洽、当时实盘不可复现交付物。',
             '- v1 目录在我复审时被连续改写（`report.json` `9d86fdd5 → 27b4dc24 → 6212d3c0`，其后又因门限表与措辞再改）；'
             '**`equilibrium_best_state.npz` 全程 `976388f2…` 未动**（每次核验记录一致，现测 `%s`）。' % sha(STATE),
             '',
             '## 5. 开环结果（真 100 帧 / 5 ms 归档、25 µs、0.5 s、接受平衡态为初值）',
             '- 产物目录：`' + str(ART) + '`；驱动 `' + str(DRIVE) + '`（sha256 `' + sha(DRIVE) + '`）',
             '- 首次非足接触 **步 %d = %s ms**，接触对 `%s`（`A6_geom` 属 body A6）' % (res['first_nonfoot_contact_step'],
                                                                                        res['first_nonfoot_contact_ms'],
                                                                                        res['first_nonfoot_contact_geom_pair']),
             '- 首次丢失足接触：**`%s` 腿，步 %d = %s ms**（力口径：`%s` 腿在步 %d = %s ms 先跌破其前 100 步均值 1%%）'
             % (loc['Q1_leg_loss']['first_leg_to_lose_tarsus_contact'], loc['Q1_leg_loss']['first_step_with_no_tarsus_contact'],
                loc['Q1_leg_loss']['first_step_with_no_tarsus_contact_ms'], loc['Q1_leg_loss']['first_leg_whose_force_drops_below_1pct'],
                loc['Q1_leg_loss']['first_step_force_below_1pct'], loc['Q1_leg_loss']['first_step_force_below_1pct_ms']),
             '- `legs_in_contact < 6` 首步 **步 %d = %s ms**；六足维持 **%d / 20001 = %.8f**；只剩 1 条腿 步 %d = %s ms'
             % (res['first_legs_in_contact_below_6_step'], res['first_legs_in_contact_below_6_ms'], res['six_legs_in_contact_steps'],
                res['six_legs_in_contact_fraction'], loc['Q1_leg_loss']['first_step_legs_in_contact_equals_1'],
                loc['Q1_leg_loss']['first_step_legs_in_contact_equals_1_ms']),
             '- root z 最小 **%s**、末值 **%s**' % (res['root_z_min'], res['root_z_final']),
             '- 能量 **%s → %s**，**非单调**（上升步 %d/%d）' % (res['energy_initial'], res['energy_final'], res['energy_rising_steps'],
                                                                res['energy_steps_compared']),
             '- 首次 `u_diamond = 1`：**步 %s = %s ms**，接触对 `%s`，此刻 `fn = %.6g`（模型单位）'
             % (first_ud['step'], first_ud['ms'], first_ud['pair'], first_ud['fn']) if first_ud else '- 首次 u_diamond = 1：未在窗口内出现',
             '- 逐步 `max|qacc|`：首步 `2.58657e-11`、末值 `1069.06 mm/s²`、全窗最大 `54433.7 mm/s²`（步 402）；首次越过 1e-3 与 1.0 均在 **步 2 = 0.05 ms**',
             '- 图（已用看图功能打开）：`' + str(ART / 'accepted-equilibrium-openloop.png') + '`、`'
             + str(ART / 'openloop-zoom-evidence.png') + '`、`' + str(LOC / 'accepted-equilibrium-localization.png') + '`',
             '',
             '## 6. 结论（静态与开环并列，不合并）',
             '- **静态**：声明盒内**确实存在** `max|qacc| = 2.5865734771464476e-11 mm/s²` 的状态（48 DOF、51 个跗节接触、`max u_diamond = 3.227792564858203e-13`）'
             '—— 这**只说明那一刻 `qacc ≈ 0` 是必要条件**。',
             '- **开环**：该状态在这份冻结连接组驱动下 **7.000 ms** 就足外触地、**6.500 ms** 起六足条件已破、**0.05 ms** 就越过声明容限与菱形极限 ⇒ '
             '**静态可行 ≠ 开环可维持，更不等于站立**。',
             '',
             '## 7. 关于被停止的子代理',
             '- `6f447df4` / `54d368b5` / `70cac021` / `3ec1a5b7` 均因范围被后续改动取代而停止，**其partial 内容一律不并入 v3**；v3 只有一个主写手。']
    (out / 'README.zh-CN.md').write_text('\n'.join(lines) + '\n')

    files = sorted(p for p in out.iterdir() if p.is_file() and p.name not in ('delivery-hashes.json', 'FROZEN_DELIVERY_HASHES.json'))
    (out / 'delivery-hashes.json').write_text(json.dumps({p.name: sha(p) for p in files}, ensure_ascii=False, indent=1))
    (out / 'FROZEN_DELIVERY_HASHES.json').write_text(json.dumps(dict(
        status='FROZEN_AT_WRITE', frozen_at=now(), files={p.name: sha(p) for p in files}), ensure_ascii=False, indent=1))
    print('v3 out:', out)
    print('files:', len(files))
    print('all-four count:', counts['n_all_four_gate_items_passing'], '| qacc item:', counts['n_qacc_item_passing'])
    print('first u_diamond = 1:', first_ud)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
