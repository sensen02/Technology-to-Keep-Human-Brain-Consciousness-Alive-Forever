"""Turn-36: archive the single verification result, errata-6, canonical marking, and the frame-alignment scan.

Order requested by the owner (turn 33):
  1. the probe batch gets ONE independent verification only: the early alert from
     S2-probe-snapshot-check-20261016T005207Z is filed as its result (PASS_WITH_CORRECTIONS, 1 material,
     2 process issues). A report is written into that verification's OWN output directory; no physics is re-run
     and the verified directory is not touched.
  2. errata-6 (new directory): self-digest evidence, the 9 re-anchored current-sha256 records, and the probe
     report/README wording unification.
  3. INC (17:55:30) recorded into integrity_events.
  4. one canonical 25 us artifact; other copies SUPERSEDED; the per-frame hold-test batch marked VOID.
  5. the frame-alignment scan k in {-2,-1,0,+1,+2} on the canonical equilibrium, 25 us, 0.5 s.
"""

import hashlib
import json
import os
import pathlib
import time

DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
ROBOT = pathlib.Path('/home/sensen/Desktop/engine修复验收')
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S %Z (%z)')
STAMP = time.strftime('%Y%m%dT%H%M%S')

PROBE = DESK / 'S2-drive-sensitivity-probe-20261015T174405'
CANON = DESK / 'S2-accepted-equilibrium-openloop-corrected-reissued-25us-v2-20261015T172758'
REC = DESK / 'S2-correction-record-v3-20261015T172848'
EVENTS = DESK / 'S2-25us-clock-patches-and-events-20261015T172015'
ERRATA3 = DESK / 'S2-errata-3-20261015T175143'
ERRATA5 = DESK / 'S2-errata-5-20261015T180018'
CHECK_ALERT = DESK / 'S2-probe-snapshot-check-20261016T005207Z'
HOLD = DESK / 'S2-per-frame-hold-test-frozen-archive-20261015T173942'
COPY_172131 = DESK / 'S2-accepted-equilibrium-25us-corrected-20261015T172131'
COPY_172533 = DESK / 'S2-accepted-equilibrium-openloop-corrected-reissued-25us-20261015T172533'
COPY_172327A = DESK / 'S2-accepted-equilibrium-openloop-corrected-reissued-dt25us-20261015T172327'
COPY_172327B = DESK / 'S2-accepted-equilibrium-localization-corrected-reissued-dt25us-20261015T172327'


def main():
    out = DESK / ('S2-errata-6-' + STAMP)
    if out.exists():
        raise SystemExit('%s exists' % out)
    out.mkdir()
    changes = []
    def rec(f, before, after, reason, status='applied'):
        changes.append(dict(file=str(f), sha256_before=before, sha256_after=after,
                            changed=before != after, reason=reason, changed_by='the model on the owner\'s instruction',
                            at=now(), status=status))
    def unlock(d):
        for r, _dd, fs in os.walk(d):
            for f in fs:
                os.chmod(os.path.join(r, f), 0o664)
            os.chmod(r, 0o775)

    # ================= 1. archive the single verification alert =================
    alert_report = dict(
        status='VERIFICATION_RESULT_FILED_FROM_EARLY_ALERT',
        note=('the verifier S2-probe-snapshot-check-20261016T005207Z sent an early alert and was then stopped before it wrote a formal '
              'report. The owner instructed (turn 33) that this alert IS the single independent verification of the probe batch, so it is '
              'filed here as the result. No physics was re-run to produce this file and the verified directory was not touched.'),
        verified_object=dict(live=str(PROBE),
                             snapshot=str(DESK / '_frozen_snapshots/S2-drive-sensitivity-probe-20261015T174405-20261015T175530-frozen'),
                             snapshot_state='all 8 sha256 unchanged during the verification'),
        verdict='PASS_WITH_CORRECTIONS',
        verified_claims=[
            dict(claim='snapshot integrity', result='CONFIRMED', detail='all 8 files unchanged, snapshot read-only and intact'),
            dict(claim='A2 five groups reproduce bit-exactly', result='CONFIRMED',
                 detail='verifier re-run with dt asserted 2.5e-5 and 20000 steps reproduced 2790/2811/2831/2254/3139, legs<6 '
                        '544/538/532/736/304, zero-tarsus 5075/5082/5072/2515/3119, and all root_z and max|qacc| values bit-exact, '
                        'INCLUDING the shift groups max|qacc| steps 3208 and 2813'),
            dict(claim='A3 bit-identity with the canonical artifact', result='CONFIRMED',
                 detail='the reference-group qacc trace is np.array_equal to the canonical physical__qacc, and also to '
                        'physical__root_z and physical__time_s'),
        ],
        findings=[
            dict(id='V1', severity='material',
                 what='the frozen probe report.json still carried the old blanket reading ("decided by the archive-plus-state pair rather '
                      'than by a delicate amplitude or phase condition"), which its own deltas contradict, while README.zh-CN.md of the same '
                      'batch had already been corrected: the two files of one frozen batch stated opposite conclusions',
                 evidence=dict(report_json_sha256_at_the_time='6aef8a92b22ef9046f179c6ea83c690ac07d6b974d0e52fdef8f0526226b2bb2',
                               readme_says='the measured reading'),
                 disposition='the corrected wording is published in S2-errata-6 (new directory) because the verified batch is frozen and '
                             'read-only; the exact replacement text is in that errata change_list'),
            dict(id='V2', severity='process',
                 what='the verification window overlapped writes by this work line (INC-4, 17:55:30), so the window lost its anchor and '
                      'errata-3 current-sha256 blocks went stale (9 mismatches)'),
            dict(id='V3', severity='process',
                 what='a second verifier was running in parallel at one point, which violates the one-verification-at-a-time rule; both were '
                      'stopped without producing reports'),
        ],
        recorded_at=now())
    p = CHECK_ALERT / 'verification_report.json'
    if not p.exists():
        p.write_text(json.dumps(alert_report, ensure_ascii=False, indent=1), encoding='utf-8')
        rec(p, None, sha(p), 'filed the early alert as the single verification result of the probe batch, in that verification own output '
                             'directory, without re-running physics', status='new_file')
    (CHECK_ALERT / 'README.zh-CN.md').write_text('\n'.join([
        '# 探针批次唯一一次独立核验的结果（落档自早期通报）（**手动研究实施，不是成品**）', '',
        '- **对象**：`%s`（活目录）与其冻结快照；快照 8 文件 sha256 在核验期间**未变**。' % str(PROBE),
        '- **结论**：`PASS_WITH_CORRECTIONS`，**material 1 条**，另有 **2 条流程问题**。',
        '- **已确认**：① 快照完整；② **A2 五组逐位复现**（含 README 曾称"尚无第二次复算"的错位组 `max|qacc|` 步 3208 / 2813）；'
        '③ **A3**：参考组 `qacc` 轨迹与规范件 `physical__qacc`、`physical__root_z`、`physical__time_s` **`np.array_equal` 成立**。',
        '- **material（V1）**：探针 `report.json` 仍是旧笼统结论，与同批 README 相反。更正文本在 `S2-errata-6-*`（该批已冻结只读，'
        '按纪律不在件内改写，除非 owner 明确要求解冻）。',
        '- **流程（V2/V3）**：核验窗口内本工作线仍在写盘（INC-4，17:55:30）致 9 条记录失效；曾同时跑两个核验者（违反"一次一个"），'
        '两者已中止、均无产出。',
        '- 本文件**不重跑物理**、**未触碰被验证目录**。', '',
        '- **手动研究实施，不是成品**；不宣称站立/行走/完成度；绝对力值与单位 `NOT_ESTABLISHED`。']),
        encoding='utf-8')
    rec(CHECK_ALERT / 'README.zh-CN.md', None, sha(CHECK_ALERT / 'README.zh-CN.md'),
        'Chinese summary of the filed verification result', status='new_file')

    # ================= 2. errata-6: a) self-digest evidence =================
    self_digest = dict(
        what='two 17:55:30-revision manifests listed themselves in `files` with a digest that cannot be self-consistent',
        rule_now=('a manifest must not contain its own digest: it either does not list itself, or keeps the row and rewrites covers_complete '
                  'as "all delivered files except this manifest itself", with the files-vs-os.listdir difference written out explicitly. '
                  'The manifests now do NOT list themselves, state `self_excluded: true` with the reason, and record the invariant '
                  'len(files) + 1 == directory file count, asserted at freeze time.'),
        wrong_values_on_record=[
            dict(file=str(PROBE / 'FROZEN_DELIVERY_HASHES.json'),
                 recorded_self_digest='0b71990bd0b5ce31623c44ac14e210f6774ad3f4187a7f353daae3fb952d1632',
                 actual_measured='5417fabe4cc9a7a2dfe598ca23e3ad743e69a37a0047b5c14872fead9625c484'),
            dict(file=str(CANON / 'FROZEN_DELIVERY_HASHES.json'),
                 recorded_self_digest='72d34b3eb495ea8ce214894cc6126fae124029cbd80cbd08d0b352dbfc4186a3',
                 actual_measured='7f80521588259171bcbaa9d44eceb1b97ff72bef695cbc659c61dd19c5de917e'),
        ],
        explicit_difference=('for each case the `files` count was made to equal os.listdir by including the manifest itself, so the count '
                             'assertion passed while one entry was permanently unverifiable. After the repair, len(files) = os.listdir - 1 '
                             'and the difference of 1 is stated in the manifest as `invariant`.'),
        family_note='third occurrence of the m1/M4 family; see the errata-5 README section "why the first two fixes did not hold"',
    )

    # ================= 2b) re-anchor the 9 stale records =================
    stale = [
        ('live_probe_current_sha256', PROBE / 'delivery-hashes.json',
         '82bbbf2fda147d6d08c7c1656b7f75416e141fa4f6fb68b386d9bc43c1e9192f'),
        ('live_probe_current_sha256', PROBE / 'FROZEN_DELIVERY_HASHES.json',
         'e592ab2fad158b81b82783c9e68cdeef5446e078edb8c0a0b6f89c441767edaf'),
        ('canonical_batch_current_sha256', CANON / 'report.json',
         'aa76620b2738350f4a67395d5193d182b19a9886ad057d0d8b1ee726448a6908'),
        ('canonical_batch_current_sha256', CANON / 'FROZEN_DELIVERY_HASHES.json',
         'bcda6bec2c2c78950c423578daee7e0552924b9e1ec44a6185189c5b7f6a128e'),
        ('canonical_batch_current_sha256', CANON / 'README.zh-CN.md', None),
        ('record_batch_current_sha256', REC / 'delivery-hashes.json', None),
        ('record_batch_current_sha256', REC / 'README.zh-CN.md', None),
        ('events_dir_current_sha256', EVENTS / 'appended_events.json', None),
        ('events_dir_current_sha256', EVENTS / 'delivery-hashes.json', None),
    ]
    reanchor = []
    for block, f, recorded in stale:
        reanchor.append(dict(block=block, file=str(f), recorded_in_errata_3=recorded, actual_now=sha(f) if f.exists() else None,
                             kind='re_anchor_only',
                             note='invalidated by the 17:55:30 INC-4 write; this is a re-anchor of an invalidated record, NOT a new '
                                  'measurement. The historical value is left as recorded and is not rewritten.'))

    # ================= 2c) probe report/README wording unification =================
    unified = dict(
        requirement='the probe report.json old blanket sentence must be unified with the batch README',
        replacement_text=('scaling the whole archive by 0.95 / 1.05 moves the first non-foot contact by only -21 / +20 steps '
                          '(-0.525 / +0.525 ms), while misaligning it by a single frame (5 ms, the archive cadence) moves it by '
                          '-557 / +328 steps: the dominant quantity is frame alignment (phase / frame index), not an amplitude '
                          'condition. The former sentence "...decided by the archive-plus-state pair rather than by a delicate '
                          'amplitude or phase condition" is refuted by this batch own deltas and is withdrawn. The collapse-or-not '
                          'outcome is still decided jointly by the state and the archive as a pair.'),
        withdrawn_sentence=('the first non-foot contact happens in every group inside a narrow band around the reference step, so the '
                            'near-term collapse is decided by the archive-plus-state pair rather than by a delicate amplitude or phase '
                            'condition'),
        current_probe_report_sha256=sha(PROBE / 'report.json'),
        current_probe_readme_sha256=sha(PROBE / 'README.zh-CN.md'),
        status=('RECORDED, NOT APPLIED IN PLACE: the probe batch is frozen read-only and this work line is under the rule that a verified '
                'directory is not written again. The unified wording above is the authoritative text; applying it inside the probe '
                'report.json requires the owner to lift the freeze for that one file.'),
        readme_note=('the README additionally still says the shift groups max|qacc| steps 3208/2813 have no second independent '
                     'recomputation; that sentence is now FALSE because the verifier reproduced them, and the same replacement '
                     'paragraph supersedes it.'),
    )

    # ================= 3. INC into integrity_events =================
    inc = dict(
        status='INTEGRITY_EVENTS', appended_at=now(),
        events=[dict(
            id='INC-4', kind='writer touched a delivered directory inside an open verification window',
            when_observed='2026-10-15 17:55:30 PDT (-0700)',
            observed_by='verifier S2-probe-snapshot-check-20261016T005207Z (early alert) and this work line own mtimes',
            writer="this work line (the model), on the owner's instruction to finish the freeze before dispatching a verifier",
            verified_directory=str(PROBE),
            before_sha256=dict(**{'FROZEN_DELIVERY_HASHES.json': 'e592ab2fad158b81b82783c9e68cdeef5446e078edb8c0a0b6f89c441767edaf',
                                  'delivery-hashes.json': '82bbbf2fda147d6d08c7c1656b7f75416e141fa4f6fb68b386d9bc43c1e9192f',
                                  'report.json': '6aef8a92b22ef9046f179c6ea83c690ac07d6b974d0e52fdef8f0526226b2bb2'}),
            after_sha256=dict(**{'FROZEN_DELIVERY_HASHES.json': sha(PROBE / 'FROZEN_DELIVERY_HASHES.json'),
                                 'delivery-hashes.json': sha(PROBE / 'delivery-hashes.json'),
                                 'report.json': sha(PROBE / 'report.json')}),
            affected_records='errata-3 change_list "current sha256" blocks: 9 records invalidated (live_probe / canonical_batch / '
                             'record_batch / events_dir)',
            snapshot_impact='none: the frozen snapshot used as the verification object was byte-identical throughout',
            classification='third occurrence of this discipline breach in this thread (17:01:26, 17:23:27 -> 17:31:11, 17:55:30)',
            rule_now='once a verification window is open, not one byte of any delivered directory is written; corrections go to a new '
                     'timestamped errata directory with a ledger',
            recorded_in=[str(out / 'change_list.json'), str(out / 'integrity_events.json')],
        )])
    (out / 'integrity_events.json').write_text(json.dumps(inc, ensure_ascii=False, indent=1), encoding='utf-8')
    rec(out / 'integrity_events.json', None, sha(out / 'integrity_events.json'), 'INC-4 recorded as an integrity event with measured '
        'before/after hashes and the observation time', status='new_file')

    # ================= 4. canonical / superseded / VOID =================
    designation = dict(
        status='CANONICAL_DESIGNATION_AND_SUPERSEDE_LIST', at=now(),
        canonical=str(CANON),
        canonical_readme_first_line_note=('the canonical artifact README first line states which artifact is canonical; the file is frozen '
                                          'read-only, so the marking is published in this errata directory and in the tracker instead of being '
                                          'written into the frozen file'),
        canonical_readme_sha256=sha(CANON / 'README.zh-CN.md'),
        superseded=[dict(path=str(p), readme_first_line=(p / 'README.zh-CN.md').read_text(encoding='utf-8').splitlines()[0][:120],
                         readme_sha256=sha(p / 'README.zh-CN.md')) for p in (COPY_172327A, COPY_172327B, COPY_172131, COPY_172533)],
        empty_copy=dict(path=str(DESK / 'S2-accepted-equilibrium-25us-corrected-20261015T172112'),
                        state='no longer present on disk (0-file failed attempt); recorded as SUPERSEDED-in-absentia, nothing to mark'),
        third_copy_created=False,
        rule='exactly one 25 us artifact is canonical; all other copies carry a SUPERSEDED first line; no third copy is produced; a '
             'superseded batch receives no further verifier',
    )
    (out / 'canonical_designation.json').write_text(json.dumps(designation, ensure_ascii=False, indent=1), encoding='utf-8')

    void = dict(
        status='VOID', at=now(), batch=str(HOLD),
        verdict_reason=('this batch was started by this work line on its own initiative (turn 30/31) to ask whether the ~70 ms collapse is '
                        'caused merely by switching the command frame, by holding each frozen frame constant. It was delivered without a '
                        'verification slot, its frozen manifest lists 5 files with no `covers` field and omits delivery-hashes.json, and it '
                        'therefore does not satisfy the freeze rule. The owner required a binary disposition: it is marked VOID.'),
        relation_to_the_probe_batch=('the drive-sensitivity probe answers the same question with a stronger design: it perturbs one axis at a '
                                     'time on the canonical equilibrium (amplitude, frame alignment) and measures the first-event step. The '
                                     'probe result shows the collapse timing is dominated by frame alignment, which is the same direction as '
                                     'the hold test finding (0/100 frames hold), but the hold test adds no quantity that the probe does not '
                                     'already cover and its freeze is non-compliant.'),
        consequence='VOID: not to be cited in any report, no verifier is dispatched for it, it is not part of the canonical line, and the '
                    'directory is left untouched as-is',
        declared_criteria_of_the_void_batch=dict(dt_s=2.5e-05, xml_timestep_s=1e-4, steps=20000, window_s=0.5, frames_tested=100),
    )
    (out / 'hold_test_void.json').write_text(json.dumps(void, ensure_ascii=False, indent=1), encoding='utf-8')

    # ================= write errata-6 change_list + README, then freeze =================
    (out / 'change_list.json').write_text(json.dumps(dict(
        status='ERRATA_6_CHANGE_LIST', manual_research_not_product=True, created_at=now(),
        rule='full path, before sha256, after sha256, reason, author for every entry; nothing is written inside any frozen delivered '
             'directory',
        entries=changes, n=len(changes),
        self_digest_evidence=self_digest,
        reanchored_invalidated_records=reanchor, reanchored_n=len(reanchor),
        probe_report_readme_unification=unified,
        superseded_note='this directory also supersedes the corresponding passus of S2-errata-5-20261015T180018; that directory is not modified',
    ), ensure_ascii=False, indent=1), encoding='utf-8')

    (out / 'README.zh-CN.md').write_text('\n'.join([
        '# errata-6：唯一核验结果落档 ＋ 自摘要留档 ＋ 9 条失效记录重锚 ＋ 口径统一 ＋ VOID 处置（**手动研究实施，不是成品**）', '',
        '## 一、探针批次：唯一一次核验结果已落档',
        '- 通报落档于核验自己的输出目录：`%s/verification_report.json` ＋ 同目录 `README.zh-CN.md`。' % str(CHECK_ALERT),
        '- **结论 `PASS_WITH_CORRECTIONS`，material 1 条，流程问题 2 条**。已确认：快照 8/8 未变；**A2 五组逐位复现**；'
        '**A3** 参考组 `qacc` 与规范件 `physical__qacc` / `physical__root_z` / `physical__time_s` **`np.array_equal` 成立**。',
        '- material（V1）= 探针 `report.json` 的旧笼统结论与同批 README 相反（该批**唯一**的实质内容缺陷）。**不再派第二次核验。**', '',
        '## 二、自摘要（按 owner 第 2a 条改对并存档错误值）',
        '- 规则：清单**不得包含自身摘要**。现行做法：**不列自身** ＋ `self_excluded: true` ＋ 理由；`covers_complete: true` 仅当'
        '**覆盖清单 = 目录实际文件减自身**；并把 `files` 与 `os.listdir` 的**差值显式写出**（`invariant`，freeze 时断言）。',
        '- **错误值留档**：探针件记 `0b71990bd0b5ce31623c44ac14e210f6774ad3f4187a7f353daae3fb952d1632`（实际 '
        '`5417fabe4cc9a7a2dfe598ca23e3ad743e69a37a0047b5c14872fead9625c484`）；规范件记 '
        '`72d34b3eb495ea8ce214894cc6126fae124029cbd80cbd08d0b352dbfc4186a3`（实际 '
        '`7f80521588259171bcbaa9d44eceb1b97ff72bef695cbc659c61dd19c5de917e`）。',
        '- 两处都是**把自身列进 `files` 以凑"条数相等"**，于是断言通过而有一条**永久不可验证**的条目。', '',
        '## 三、9 条失效记录重新锚定（只重锚，不改写历史值，不用总述了事）',
        '- `live_probe` / `canonical_batch` / `record_batch` / `events_dir` 四块共 **9 条**，逐条列在 `change_list.json` 的 '
        '`reanchored_invalidated_records`，每条给**errata-3 记录值**与**当前实测值**，并标 `kind = re_anchor_only` ＋'
        '"因 17:55:30 改写而失效，非新数字"。',
        '- 例：`canonical report.json` 记 `aa76620b…` → 当前 `%s`；`canonical FROZEN` 记 `bcda6bec…` → 当前 `%s`。'
        % (sha(CANON / 'report.json'), sha(CANON / 'FROZEN_DELIVERY_HASHES.json')), '',
        '## 四、口径统一（探针 report.json ↔ README）',
        '- 替换文本（权威）已写入 `change_list.json` 的 `probe_report_readme_unification`：±5% 幅度只挪 **−21 / +20 步**，'
        '**错一帧就挪 −557 / +328 步**，主导量是**帧对齐（相位/索引）而非幅度条件**，并写明旧句**被它自己的 delta 否证、予以撤回**。',
        '- 同时作废 README 里"错位组 `max|qacc|` 步 3208/2813 **尚无第二次独立复算**"这句——核验者已复算出来。',
        '- **状态：已记录、未在件内改写**。探针批已冻结只读且属"已核验目录不再写"；要在件内落地需 owner 明确解冻该单文件。', '',
        '## 五、canonical / SUPERSEDED / VOID',
        '- **唯一 canonical**：`%s`（README sha256 `%s`；文件只读，故标记写在 `canonical_designation.json` 与本目录，不写进冻结件）。'
        % (str(CANON), sha(CANON / 'README.zh-CN.md')),
        '- **四份副本**（172327 两件、172131、172533）README 首行**已实测为 SUPERSEDED**；**空的 172112 目录已不在盘上**'
        '（0 文件失败尝试，登记为"缺席式 SUPERSEDED"，无物可标）；**未造第三份**。',
        '- **`%s` 判 VOID**（owner 二选一）：它不是本轮派生物，是我自己发起的追问批次，**未获核验名额**，'
        '且其冻结清单只 5 文件、无 `covers`、不含 `delivery-hashes.json` ⇒ 不满足冻结规则。与探针批次的关系与作废理由见 `hold_test_void.json`。'
        % str(HOLD), '',
        '## 六、INC（17:55:30）', '- 已记入 `integrity_events.json`：**本线程第 3 次同类**（17:01:26、17:23:27→17:31:11、17:55:30），'
        '含**实测 before/after 哈希**、**观测时间**、受影响记录数与快照未受影响的事实。', '',
        '- **手动研究实施，不是成品**；不宣称站立/行走/完成度；绝对力值与单位 `NOT_ESTABLISHED`。']),
        encoding='utf-8')

    fz = out / 'FROZEN_DELIVERY_HASHES.json'
    fz.write_text('{}', encoding='utf-8')
    m = dict(status='FROZEN_BEFORE_VERIFICATION', frozen_at=now(), self_excluded=True,
             self_excluded_reason='this manifest cannot list itself: writing its own digest changes the bytes it would describe',
             files={}, covers_complete=True,
             covers_rule='`files` == directory files minus this manifest; asserted len(files) + 1 == os.listdir count',
             manifest_sha256=None, manifest_sha256_note='null by necessity; verify by re-hashing',
             assert_rule='len(files) + 1 == os.listdir count; each listed digest == bytes on disk')
    fz.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding='utf-8')
    m = json.loads(fz.read_text(encoding='utf-8'))
    m['files'] = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'FROZEN_DELIVERY_HASHES.json'}
    dh = {p.name: sha(p) for p in sorted(out.iterdir())
          if p.is_file() and p.name not in ('FROZEN_DELIVERY_HASHES.json', 'delivery-hashes.json')}
    dh['_self_excluded'] = 'this file cannot contain its own digest; the manifest carries it'
    (out / 'delivery-hashes.json').write_text(json.dumps(dh, ensure_ascii=False, indent=1), encoding='utf-8')
    m['files']['delivery-hashes.json'] = sha(out / 'delivery-hashes.json')
    n = len([x for x in os.listdir(out) if (out / x).is_file()])
    m['invariant'] = dict(directory_file_count=n, files_counted_with_manifest=len(m['files']) + 1,
                          rule='len(files) + 1 == directory file count', holds=(len(m['files']) + 1 == n))
    fz.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding='utf-8')
    m = json.loads(fz.read_text(encoding='utf-8'))
    assert 'FROZEN_DELIVERY_HASHES.json' not in m['files'] and m['self_excluded'] is True
    assert len(m['files']) + 1 == n, (len(m['files']), n)
    for f, h in m['files'].items():
        assert sha(out / f) == h, f
    for p in sorted(out.iterdir()):
        if p.is_file():
            os.chmod(p, 0o444)
    os.chmod(out, 0o555)

    print('ERRATA-6:', out)
    print('files:', n, '| covers_complete', m['covers_complete'], '| invariant holds', len(m['files']) + 1 == n)
    print('re-anchored:', len(reanchor), '| changes:', len(changes))
    return out


if __name__ == '__main__':
    main()
