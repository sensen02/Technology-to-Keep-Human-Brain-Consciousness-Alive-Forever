"""Turn 29: correction-record batch (v3) that closes the findings of the two verification reports on the already-superseded
batches, and fixes the one material finding (EV-02's "before hash untraceable" claim).

No physics is recomputed here: this batch is a record. It also lists which directories are current and which are historical,
because the last verification noted that the "single current batch" statement had become literally false.
"""

import hashlib
import json
import pathlib
import time

DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
V2BATCH = DESK / 'S2-accepted-equilibrium-openloop-corrected-reissued-25us-v2-20261015T172758'
V1 = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259'
V3OLD = DESK / 'S2-equilibrium-A-corrected-reissued-v3-20261015T170319'
PAIRCHECK = DESK / 'S2-openloop-pair-check-20261015T235120Z'
CHECK_171644 = DESK / 'S2-corrected-dt-check-20261016T001721Z'
CHECK_172131 = DESK / 'S2-25us-corrected-check-20261016T002225Z'
STRAY = DESK / 'S2-accepted-equilibrium-25us-corrected-20261015T172112'
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S %Z (%z)')


def main():
    out = DESK / ('S2-correction-record-v3-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)

    # ---- the material finding: EV-02's before-state IS partially recorded ----
    v1_manifest = json.loads((V1 / 'delivery-hashes.json').read_text())
    pair_manifest_path = PAIRCHECK / 'hash_manifest_check.json'
    pair_claimed = pair_actual = None
    if pair_manifest_path.exists():
        txt = json.dumps(json.loads(pair_manifest_path.read_text()), ensure_ascii=False)
        import re
        hits = re.findall(r'[0-9a-f]{64}', txt)
        pair_claimed, pair_actual = (hits[0], hits[1]) if len(hits) >= 2 else (None, None)
    ev02 = dict(
        status='EV_02_BEFORE_STATE_CORRECTED', manual_research_not_product=True,
        what_was_wrong=('the earlier event record said the pre-rewrite hash was "NOT RECORDED — untraceable". That is incomplete: two different pre-rewrite '
                        'hashes are on disk, so at least two revisions existed before the rewrite while only one rewrite is documented.'),
        before_hashes_found=[
            dict(value=v1_manifest.get('delivery_time_hashes', {}).get('integrity_incident_assembler.json'),
                 source=str(V1 / 'delivery-hashes.json'), field='delivery_time_hashes["integrity_incident_assembler.json"]',
                 recorded_at=v1_manifest.get('recorded_at'), meaning='the value carried as the delivery-time column, i.e. the revision in place before the '
                 '17:01:26 rewrite'),
            dict(value=pair_actual, source=str(pair_manifest_path),
                 meaning='the value measured by the pair-check verification at 16:55; it differs from the delivery-time column, so an unlogged '
                         'intermediate write happened between the delivery-time record and the 17:01:26 rewrite')],
        after_hash=sha(V1 / 'integrity_incident_assembler.json'),
        changed_at='2026-10-15 17:01:26 (the rewrite that changed the status from INVALID to WITHDRAWN)',
        unresolved='which revision corresponds to the text read at 16:59 (INVALID) cannot be settled from the two hashes alone; the intermediate write is '
                   'unpaired and is recorded as such rather than reconstructed',
        consequence='the earlier "untraceable" wording is retracted here; the before-state is PARTIALLY traceable (two candidate hashes) and one write remains '
                    'unaccounted for',
        detected_by=str(CHECK_172131), recorded_at=now())
    (out / 'ev02_before_state_correction.json').write_text(json.dumps(ev02, ensure_ascii=False, indent=1))

    # ---- disposition of every finding from both verification reports ----
    disposition = dict(
        status='FINDINGS_DISPOSITION', manual_research_not_product=True,
        reports=[dict(path=str(CHECK_171644), verdict='PASS_WITH_CORRECTIONS', object=str(DESK / 'S2-accepted-equilibrium-openloop-corrected-dt-20261015T171644'),
                      counts='1 material / 4 minor / 3 cosmetic / 0 blocking'),
                 dict(path=str(CHECK_172131), verdict='PASS_WITH_CORRECTIONS', object=str(DESK / 'S2-accepted-equilibrium-25us-corrected-20261015T172131'),
                      counts='1 material / 6 minor / 4 cosmetic / 0 blocking')],
        items=[
            dict(id='M1(171644)', severity='material', what='the F-A sentence mixed numbers of another run into the batch without a source label',
                 disposition='FIXED in the current physical batch', evidence=str(V2BATCH / 'report.json') + ' -> f_a_sentence_by_source (this_batch vs '
                 'quoted_other_artifact); this batch moreover shows its own first Thorax_geom|floor contact at step 3134'),
            dict(id='M2(171644)', severity='minor', what='claim that the 76.375 ms correction landed in that batch only',
                 disposition='RETRACTED in the current batch', evidence='the v3 batch already carried the corrected sentence four minutes earlier'),
            dict(id='M3(171644)', severity='minor', what='the self-check could not fail on its own and the cadence check was self-referential',
                 disposition='FIXED', evidence='the current batch asserts that the declared step differs from the XML step and measures the command cadence '
                 'from the d.time samples at command boundaries (measured 0.005 s; the 100 us side branch is checked against its own 20 ms cadence)'),
            dict(id='M4(171644)', severity='minor', what='the pose-A comparison was a hard-coded constant',
                 disposition='FIXED', evidence='the current batch recomputes pose A in the same run (step 2399 = 59.975 ms, act initialised to zeros) and '
                 'records the external cross-check (2398 = 59.950 ms) beside it'),
            dict(id='M5(171644)', severity='minor', what='max u_diamond was taken over tarsus contacts only without disclosure',
                 disposition='FIXED', evidence='both readings are reported: tarsus-only 1.0000000000000002 at 18143/20001 steps, all-contacts '
                 '1.0000000000000002 at 19549/20001 steps'),
            dict(id='C1(171644)', severity='cosmetic', what='the frozen manifest did not cover every file',
                 disposition='FIXED', evidence='the current manifest covers every delivered file including delivery-hashes.json and states its own exclusion'),
            dict(id='C2(171644)', severity='cosmetic', what='in-place VOID-line writes to two frozen READMEs with no logged hash',
                 disposition='RECORDED as EV-03', evidence=str(V2BATCH / 'correction_ledger.json')),
            dict(id='C3(171644)', severity='cosmetic', what='timezone label wrong',
                 disposition='FIXED', evidence='timestamps now use the machine zone (%s)' % now()),
            dict(id='EV02(172131)', severity='material', what='"before hash untraceable" was incomplete',
                 disposition='CORRECTED here', evidence=str(out / 'ev02_before_state_correction.json')),
            dict(id='m1(172131)', severity='minor', what='the report said the final time equals steps x dt while the assertion uses a 1e-9 tolerance',
                 disposition='DISCLOSED', evidence='measured final d.time 0.49999999999994904 s, difference -5.096e-14 s, tolerance 1e-9; stated in the current '
                 'batch and in this record'),
            dict(id='m2(172131)', severity='minor', what='the read-back evidence is masked by an unconditional reset of m.opt.timestep',
                 disposition='DISCLOSED', evidence='the decisive evidence is the measured d.time increment and the bit-identical reprojection, both of which '
                 'the verification reproduced; the reset only restores the declared value between runs'),
            dict(id='m3(172131)', severity='minor', what='the two VOID batches own frozen manifests went stale when the VOID lines were written',
                 disposition='DISCLOSED, not repaired', evidence='repairing them would be another in-place write to frozen directories; EV-03 records the write'),
            dict(id='m4(172131)', severity='minor', what='the two VOID reports still carry a PASS_* top-level status',
                 disposition='DISCLOSED, not repaired', evidence='the same reason; protocol_void governs their reading'),
            dict(id='m5(172131)', severity='minor', what='the localization figure bottom-right axis does not say it spans a 0.5 ms lookback window',
                 disposition='DISCLOSED', evidence='the panel title states the 20-step lookback; the axis label is a cosmetic gap in a historical batch'),
            dict(id='m6(172131)', severity='minor', what='the "single current batch" statement became literally false as newer batches appeared',
                 disposition='FIXED by this record', evidence=str(out / 'current_directories.json')),
            dict(id='c1..c4(172131)', severity='cosmetic', what='figure title formatting (peak 1), the reused sentence only in report.json, a stray empty '
                 'directory, and the displacement field name', disposition='DISCLOSED / partially cleaned',
                 evidence='the stray empty directory was removed after recording its (empty) state; the rest are cosmetic and are not repaired in frozen files'),
        ],
        recorded_at=now())
    (out / 'verification_findings_disposition.json').write_text(json.dumps(disposition, ensure_ascii=False, indent=1))

    # ---- current vs historical directories ----
    dirs = dict(
        status='DIRECTORY_STATUS', manual_research_not_product=True,
        current=[dict(path=str(V2BATCH), role='the one current physical batch: open loop + localization at the enforced 25 us step, with both figures, the '
                                             'strengthened self-check, the source-labelled F-A sentence and the pose-A comparison computed in-run')],
        records_of_this_round=[dict(path=str(out), role='correction record: EV-02 before-state, findings disposition, directory status')],
        superseded_or_historical=[
            dict(path=str(DESK / 'S2-accepted-equilibrium-openloop-corrected-dt-20261015T171644'), why='superseded by the current batch (no side-branch '
                 'fields, mixed F-A sentence)'), dict(path=str(DESK / 'S2-accepted-equilibrium-localization-20261015T170833'), why='never created'),
            dict(path=str(DESK / 'S2-accepted-equilibrium-25us-corrected-20261015T172131'), why='superseded: contains the mixed F-A sentence and the '
                 'self-referential cadence check'), dict(path=str(DESK / 'S2-accepted-equilibrium-openloop-corrected-reissued-dt25us-20261015T172327'),
                 why='superseded pair'), dict(path=str(DESK / 'S2-accepted-equilibrium-localization-corrected-reissued-dt25us-20261015T172327'),
                 why='superseded pair'), dict(path=str(DESK / 'S2-accepted-equilibrium-openloop-corrected-reissued-25us-20261015T172533'),
                 why='superseded: mixed F-A sentence'), dict(path=str(V3OLD), why='historical: assembled from the 100 us runs; its millisecond labels are void')],
        void=[dict(path=str(DESK / 'S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164559'), why='B1: protocol declaration false'),
              dict(path=str(DESK / 'S2-accepted-equilibrium-localization-20261015T170058'), why='B1: protocol declaration false'),
              dict(path=str(DESK / 'S2-mujoco-equilibrium-A-20261015T151340'), why='integrity incident'),
              dict(path=str(DESK / 'S2-equilibrium-A-corrected-reissued-20261015T154946'), why='defective multi-start')],
        historical_only_refuted=[str(p) for p in sorted(DESK.glob('S2-batch-*'))],
        kept_read_only=[str(V1), str(DESK / 'S2-equilibrium-A-corrected-reissued-v2-20261015T163134'),
                        str(DESK / 'S2-equilibrium-radius-ladder-20261015T161518')],
        recorded_at=now())
    (out / 'current_directories.json').write_text(json.dumps(dirs, ensure_ascii=False, indent=1))

    # ---- clean the stray empty directory after recording it ----
    stray_note = None
    if STRAY.exists():
        contents = sorted(p.name for p in STRAY.iterdir())
        stray_note = dict(path=str(STRAY), contents=contents, was_empty=(len(contents) == 0),
                          action='removed (empty, created by a failed run; nothing lost)')
        if not contents:
            STRAY.rmdir()

    readme = '\n'.join([
        '# correction record v3：两份核验的发现处置 ＋ EV-02 前置哈希更正（手动研究实施，不是成品）', '',
        '## 本轮性质', '- **不重算物理**：物理结论以当前批次 `/home/sensen/Desktop/engine修复验收/'
        'S2-accepted-equilibrium-openloop-corrected-reissued-25us-v2-20261015T172758/` 为准；本批只做记录与溯源更正。', '',
        '## 唯一 material：EV-02 的前置状态其实部分可查',
        '- 旧记录写"before 哈希未记录、不可追溯"，**不完整**：磁盘上存在两个改写前哈希 —— '
        '`%s`（v1 `delivery-hashes.json` 的 `delivery_time_hashes` 列）与 `%s`（pair-check 在 16:55 的实测）。'
        % (ev02['before_hashes_found'][0]['value'], ev02['before_hashes_found'][1]['value']),
        '- 两者不同 ⇒ **至少存在两个改写前修订，而只记录了一次改写**；中间那次无配对记录，**如实标为 unpaired 而不去重建**。'
        '`after = %s`，改动时刻 17:01:26。' % ev02['after_hash'], '',
        '## 两份核验的发现处置', '- 见 `verification_findings_disposition.json`（M1–M5 / C1–C3（171644）与 material+6 minor+4 cosmetic（172131）逐条：'
        'FIXED / RETRACTED / RECORDED / DISCLOSED）。其中 **M1 已修**（F-A 句按来源分离）、**M3 已修**（cadence 由实测 `d.time` 差值校验，'
        '实测 0.005 s）、**M4 已修**（姿态 A 本批实算 步 2399 = 59.975 ms，外部交叉校验 2398 = 59.950 ms）、**M5 已修**（u_diamond 两口径并列）。', '',
        '## 目录状态', '- 见 `current_directories.json`：**current 只有 1 个**（上述 v2 物理批次）；其余按 superseded / void / historical-only 分类。',
        '- 清理：空目录 `%s`（0 文件，失败尝试留下）已于记录后删除。' % (stray_note['path'] if stray_note else '—'), '',
        '## 纪律', '- 冻结件只读；更正一律以新时间戳记录文件发布；验证期间不写入项目根与交付目录。', '',
        '- **手动研究实施，不是成品**；不宣称站立/行走/完成度；绝对力值与单位 `NOT_ESTABLISHED`。'])
    (out / 'README.zh-CN.md').write_text(readme)
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != 'FROZEN_DELIVERY_HASHES.json')
    (out / 'delivery-hashes.json').write_text(json.dumps({p.name: sha(p) for p in files}, ensure_ascii=False, indent=1))
    (out / 'FROZEN_DELIVERY_HASHES.json').write_text(json.dumps(dict(status='FROZEN_AT_WRITE', frozen_at=now(),
                                                                    covers='every file including delivery-hashes.json; excludes only itself',
                                                                    files={p.name: sha(p) for p in files}), ensure_ascii=False, indent=1))
    print('out:', out)
    print('EV-02 before candidates:', ev02['before_hashes_found'][0]['value'], '|', ev02['before_hashes_found'][1]['value'])
    print('after:', ev02['after_hash'])
    print('stray dir cleaned:', bool(stray_note and stray_note['was_empty']))
    print('files:', len(files))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
