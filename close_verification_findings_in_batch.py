"""Close the last discrepancies the v2 verification reported, inside the batch that this round is allowed to touch:

F-A  the first non-foot contact at step 2739 is A6_geom | floor, not Thorax_geom | floor (the first Thorax_geom | floor
     contact follows at step 3055 = 76.400 ms). Fixed in the batch report.json and README.
F-C  integrity_incident_assembler.json was self-contradictory after the earlier restore: it described a copy that was in
     fact the snapshot revision. Rewritten to state only what is checkable, with the preserved file renamed accordingly.
F-B/F-L  the stale v2 / ladder hash records and the assembler revisions are documented in the CHANGELOG instead of being
     silently re-anchored (v2 and the ladder are verified archives and are not touched).
"""

import hashlib
import json
import pathlib
import shutil
import time

ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
BATCH = pathlib.Path('/home/sensen/Desktop/engine修复验收/S2-equilibrium-A-corrected-reissued-20261015T161259')
LIVE = ROOT / 'run_equilibrium_reissue_finalize.py'
MISLEADING_COPY = BATCH / 'assembler_live_version_at_the_incident.py'
CORRECT_COPY_NAME = BATCH / 'assembler_snapshot_revision_5c421276.py'
AUTH = 'e52f080bd0895fe36901f60398b60408e7dafe4871232656025113c5be754622'
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
mtime = lambda p: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(pathlib.Path(p).stat().st_mtime))


def main():
    assert sha(LIVE) == AUTH, 'the live assembler is not the authoritative revision: %s' % sha(LIVE)
    changes = []

    # ---------------- F-A: geom identity ----------------
    rp = BATCH / 'report.json'
    r = json.loads(rp.read_text())
    old_disc = r['mandatory_disclosure'][2]
    r['mandatory_disclosure'][2] = (
        'A declared-cone static equilibrium is NOT dynamic maintainability. Measured on the accepted state under the real '
        '100-frame connectome-derived archive (artifact S2-true-equilibrium-openloop-20261015T154720, recomputed from its '
        'time_series.npz), the failure sequence is three separate facts and must not be compressed into one sentence: (a) step '
        '2739 = 68.475 ms is the first NON-FOOT contact and its geometry pair is A6_geom | floor (A6_geom belongs to body A6; '
        'the first Thorax_geom | floor contact only follows at step 3055 = 76.400 ms); (b) legs_in_contact is already below 6 '
        'from step 537 = 13.425 ms; (c) first_tarsus_loss_step = 5111 = 127.775 ms, i.e. when the last tarsus contact is lost. '
        'An earlier wording of this paragraph named Thorax_geom for step 2739, which was wrong and was corrected on 2026-10-15 '
        '(the same wrong wording survives in the v2 batch and in the ladder README; those directories are verified archives and '
        'are deliberately not edited, so this batch is the authoritative wording). A separate open-loop measurement of THIS '
        'batch accepted state (S2-accepted-equilibrium-openloop-real100-20261015T163532) is a different experiment.')
    changes.append(('F-A', str(rp), 'mandatory_disclosure[2]', old_disc, r['mandatory_disclosure'][2],
                    'independent verification of the v2 batch: at index 2739 the pair is (floor, A6_geom); Thorax_geom|floor first at 3055'))

    r['inputs']['assembler_sha256'] = AUTH
    r['assembler_provenance']['authoritative_version'] = dict(
        path=str(LIVE), sha256=AUTH, mtime=mtime(LIVE),
        role='this exact revision produced the report.json / README.zh-CN.md in this directory (run at 16:41)',
        byte_identity_with_the_run='the revision that ran and the file on disk now are byte-identical: sha256(live) == inputs.assembler_sha256 == %s' % AUTH,
        hash_chain_check='verified on 2026-10-15 16:43 by rebuilding the live file as snapshot bytes + the five documented F2 replacements; the rebuild reproduced this hash exactly')
    r['assembler_provenance']['previous_mutated_revisions'] = {
        'd94b8379c6b08aef99d6963567b498c7e8b7329b3ffd5ecaa48c40ab260c9c61':
            'my failed finalisation passes inserted the same `hit4=` keyword twice (Python accepted the duplicate in the report '
            'but the assembler then failed to compile)',
        '8ef3e33fd0dd8cd6c4f29e7a7cc42cb63a8096d4c0cc5fc6969ad463fea44a01':
            'an intermediate deduplication attempt that removed overlapping blocks and damaged the README substitution dict (KeyError n05)',
        '3806dbe545e86b80c550dad7089901de407ed96e1d6df325f6a17cd9341a9c81':
            'the state after cleaning the duplicates; it never produced a delivered file',
        'resolution': 'the file was rebuilt from the snapshot plus the five documented replacements, which reproduced %s exactly' % AUTH}
    rp.write_text(json.dumps(r, ensure_ascii=False, indent=1))

    rmp = BATCH / 'README.zh-CN.md'
    txt = rmp.read_text()
    old_en = ('- A declared-cone static equilibrium is NOT dynamic maintainability. Three separate measured facts, never '
              'fused into one clause: (a) step 2739 = 68.475 ms is the first NON-FOOT contact (`Thorax_geom | floor`); '
              '(b) `legs_in_contact` is below 6 from step 537 = 13.425 ms; (c) `first_tarsus_loss_step = 5111` = '
              '127.775 ms. Source artifact S2-true-equilibrium-openloop-20261015T154720, recomputed this round. A newer '
              'measurement of THIS batch accepted state (S2-accepted-equilibrium-openloop-real100-20261015T163532) is a '
              'different experiment.')
    new_en = ('- 静态平衡**不等于**可动态维持，下面三件事**必须分开写**（不得压缩成"步 2739 失去足接触"一句）：'
              '(a) **步 2739 = 68.475 ms 是首次足外触地**，几何对 **`A6_geom | floor`**（`A6_geom` 属于 body A6；'
              '首个 `Thorax_geom | floor` 接触要到 **步 3055 = 76.400 ms** 才出现）；'
              '(b) `legs_in_contact` **自步 537 = 13.425 ms 起已少于 6**；'
              '(c) `first_tarsus_loss_step = 5111 = 127.775 ms`（最后一个跗节接触失去）。'
              '（引自 `S2-true-equilibrium-openloop-20261015T154720`，并从其 `time_series.npz` 重算；'
              '此前把步 2739 写成 `Thorax_geom` 是**错的**，已于 2026-10-15 更正；同样的错句仍留在 v2 批次与阶梯 README —— '
              '那两个目录是已核验归档，按既定规则不去改动，**以本批次为准**。）')
    if old_en in txt:
        txt = txt.replace(old_en, new_en, 1)
        changes.append(('F-A', str(rmp), '§4 披露第 3 条', old_en, new_en, 'A6_geom, not Thorax_geom'))
    old_zh_thorax = '步 2739 = 68.475 ms 是首次足外触地**，几何对 `Thorax_geom | floor`'
    if old_zh_thorax in txt:
        txt = txt.replace(old_zh_thorax, '步 2739 = 68.475 ms 是首次足外触地**，几何对 **`A6_geom | floor`**（首个 `Thorax_geom | floor` 在步 3055 = 76.400 ms）', 1)
        changes.append(('F-A', str(rmp), '§4/§9 中文句里的几何名', 'Thorax_geom | floor', 'A6_geom | floor (+ Thorax 在 3055 ms)',
                        'same finding, Chinese copy'))
    txt = txt.replace('Thorax_geom | floor）', 'A6_geom | floor）')
    rmp.write_text(txt)

    # ---------------- F-C: honest incident record + rename the misleading copy ----------------
    if MISLEADING_COPY.exists():
        if sha(MISLEADING_COPY) == '5c4212768b89629f46623578fb9003e7cb67383e98077539ba5d63a3525aa2d0':
            shutil.move(str(MISLEADING_COPY), str(CORRECT_COPY_NAME))
            changes.append(('F-C', str(CORRECT_COPY_NAME), 'file name', 'assembler_live_version_at_the_incident.py',
                            'assembler_snapshot_revision_5c421276.py (the file is the SNAPSHOT revision, not the incident revision)',
                            'the old name asserted something false'))
    integrity = dict(
        status='PASS_INTEGRITY_INCIDENT_RECORDED_AND_RESOLVED',
        incident=('the assembler path named by this batch was edited after the batch was written, so the declared '
                  'inputs.assembler_sha256 no longer matched the file on disk (independent verification finding F1)'),
        checkable_facts=dict(
            declared_originally='5c4212768b89629f46623578fb9003e7cb67383e98077539ba5d63a3525aa2d0 (the bytes shipped in '
                                'source_snapshot.zip)',
            live_at_incident='9271e578b6c54b8882c1d765af9f2ab67a13956449c0f061777810034c9c1742, mtime 2026-10-15 16:14:08 '
                             '(recorded by the earlier independent verification, which also confirms that revision already '
                             'contained the F2 count fields)',
            incident_bytes_availability='LOST: my own remedy in the previous round wrote the backup file twice, so the '
                                        '9271e578 bytes exist nowhere on disk any more. This is stated plainly; no sentence in '
                                        'this batch claims the snapshot proves otherwise.',
            authoritative_now=AUTH + ', mtime %s, byte-identical to the revision that ran and produced the delivered files' % mtime(LIVE),
            verification_of_the_chain='sha256(live assembler) == report.json.inputs.assembler_sha256 == %s; the live file was '
                                      'rebuilt as snapshot bytes + the five documented F2 replacements and the rebuild reproduced '
                                      'this hash exactly' % AUTH),
        preserved_file=dict(name=CORRECT_COPY_NAME.name, sha256=sha(CORRECT_COPY_NAME) if CORRECT_COPY_NAME.exists() else None,
                            what_it_really_is='the snapshot revision, mislabelled in the previous round as the incident revision; '
                                              'renamed so the record cannot mislead'),
        rule=('a batch may only quote bytes it ships; a remedy must never overwrite the only copy of the version it documents; '
              'and a file name must not assert a provenance that has not been checked'))
    (BATCH / 'integrity_incident_assembler.json').write_text(json.dumps(integrity, ensure_ascii=False, indent=1))
    changes.append(('F-C', str(BATCH / 'integrity_incident_assembler.json'), 'whole file',
                    'self-contradictory claims about the live/copy/snapshot revisions',
                    'checkable facts only, with the lost-bytes statement and the closed hash chain',
                    'independent verification finding F-C'))

    # ---------------- CHANGELOG: this round's records + residual disclosures ----------------
    lines = ['', '', '---', '', '# 收口轮之二（%s）：F-A / F-C 落地 + 残余项披露' % time.strftime('%Y-%m-%d %H:%M:%S'), '',
             '权威装配脚本：`%s` sha256 `%s`（mtime %s）。**它与产出交付件的那一版字节相同**，'
             '因此 `inputs.assembler_sha256 == sha256(live)` 成立。' % (LIVE, AUTH, mtime(LIVE)), '']
    for c in changes:
        lines += ['## %s — `%s`' % (c[0], c[1]), '- 位置：`%s`' % c[2], '- 改前：%s' % c[3], '- 改后：%s' % c[4],
                  '- 依据：%s' % c[5], '']
    lines += [
        '## 残余项（明确不修，附理由）',
        '- **F-B（v2 与阶梯的哈希记录过期）**：本轮的合法就地更正改写了 4 个运行目录（`report.json` 与图）、实盘搜索脚本、'
        '诊断脚本与诊断 JSON，因此 v2 批次里 9 条运行哈希、阶梯里 1 条运行哈希已过期，3 个快照成员不再等于 live 文件。'
        '**没有任何数字变化**（v2 的核验方逐字段核对 14 个数值字段全部一致）。v2 与阶梯是**已核验归档**，按规则不就地改，'
        '故只在此披露。',
        '- **F9 覆盖范围**：模板文案更正只落在 4 个主运行（`A_ms240` / `A_ms_r0p20` / `A_ms_r0p10` / `A_ms_r0p05`）。'
        '5 个阶梯运行与 2 个补测运行的 `report.json` 里仍写着模板原句"within 0.40 rad of the source pose"，'
        '但它们的 `nested_radius_rad` 字段写明真实半径（0.45/0.40/0.35/0.30/0.25/0.20/0.15），且参考姿态是姿态 A。'
        '不改的理由：那些目录属于本轮另一份核验（刚完成），就地改动会让它的哈希记录再次失效。',
        '- **F-L（多版本共存）**：本轮期间实盘装配脚本先后出现 `9271e578`（事故时，字节已失）、`d94b8379`、`8ef3e33f`、`3806dbe5` 与'
        '最终 `e52f080b`（= 快照 + 五个已记录替换，且正是产出交付件的版本）。批内 `assembler_snapshot_revision_5c421276.py` 是快照版本，'
        '名称已改正确。',
        '- **CHANGELOG 的历史条目**：多次重跑使得若干 before/after 对退化为相同值（before == after）；'
        '可复核的锚点是本文件每次记录里的 sha256 与 `delivery-hashes.json`，而不是 before 字段。',
        '- **v2 / 阶梯里的 Thorax_geom 错句**：不在本批次，按规则不改；本批次 §4 的 A6_geom 措辞与开环批次的 `A6_geom | floor` 一致。',
    ]
    with (BATCH / 'CHANGELOG-inplace.md').open('a') as fh:
        fh.write('\n'.join(lines) + '\n')

    files = sorted(p for p in BATCH.iterdir() if p.is_file() and p.name != 'delivery-hashes.json')
    (BATCH / 'delivery-hashes.json').write_text(json.dumps({p.name: sha(p) for p in files}, ensure_ascii=False, indent=1))
    r2 = json.loads(rp.read_text())
    print('changes applied:', len(changes))
    print('chain closed:', r2['inputs']['assembler_sha256'] == sha(LIVE) == AUTH)
    print('batch files:', len(files))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
