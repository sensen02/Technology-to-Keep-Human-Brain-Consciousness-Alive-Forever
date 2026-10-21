"""Turn-51 (owner chose plan A): declare VOID, then redo the manifest family cleanly.

Order that must hold this time (the thing I kept getting wrong):
  1. finish EVERY covered file first,
  2. only then sign each manifest, recomputing every digest from disk at that instant,
  3. never write a placeholder digest, and re-verify after writing.

Nothing is written into any directory still under verification: no verification is running (both closed).
"""

import hashlib
import json
import os
import pathlib
import time

DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S %Z (%z)')
utc = lambda: time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
GEN = ('FROZEN_DELIVERY_HASHES.json', 'delivery-hashes.json')

# every delivered directory whose manifests are in scope for the redo
TARGETS = [
    DESK / 'S2-drive-sensitivity-probe-20261015T174405',
    DESK / 'S2-accepted-equilibrium-openloop-corrected-reissued-25us-v2-20261015T172758',
    DESK / 'S2-25us-clock-patches-and-events-20261015T172015',
    DESK / 'S2-correction-record-v3-20261015T172848',
    DESK / 'S2-errata-3-20261015T175143',
    DESK / 'S2-frame-alignment-scan-20261015T180345',
]
VOIDED = [
    DESK / 'S2-constant-command-hold-20261015T181923',
    DESK / 'S2-errata-10-20261015T181902',
    DESK / 'S2-errata-11-20261015T182225',
    DESK / 'S2-errata-12-20261015T182337',
    DESK / 'S2-errata-13-20261015T182513',
    DESK / 'S2-errata-14-20261015T182629',
]


def unlock(d):
    for r, _x, fs in os.walk(d):
        for f in fs:
            os.chmod(os.path.join(r, f), 0o664)
        os.chmod(r, 0o775)


def seal(d):
    for p in sorted(d.iterdir()):
        if p.is_file():
            os.chmod(p, 0o444)
    os.chmod(d, 0o555)


def resign(d):
    """Finish-then-sign: recompute every covered digest from disk at this instant, then write, then verify."""
    d = pathlib.Path(d)
    unlock(d)
    fp, dh = d / 'FROZEN_DELIVERY_HASHES.json', d / 'delivery-hashes.json'
    for p in (fp, dh):
        if p.exists():
            p.unlink()
    fp.write_text('{}', encoding='utf-8')          # placeholders so the listing includes both generated files
    dh.write_text('{}', encoding='utf-8')
    listing = sorted(x for x in os.listdir(d) if (d / x).is_file())
    data = [f for f in listing if f not in GEN]
    assert 'FROZEN_DELIVERY_HASHES.json' in listing and 'delivery-hashes.json' in listing

    files = {f: sha(d / f) for f in data}                    # step 1: every covered file is already final
    m = dict(status='FROZEN_BEFORE_VERIFICATION', frozen_at=now(), frozen_at_utc=utc(),
             files=files,
             covers_complete='covers every delivered file except this manifest itself and except delivery-hashes.json',
             covers_rule='len(files) == len(os.listdir) - 2: the manifest cannot digest itself, and delivery-hashes.json digests the manifest, '
                         'so a mutually-pinning pair would have no fixed point. Measured: %d rows == %d files - 2.' % (len(files), len(listing)),
             self_reference='not defined (a digest of a file cannot contain itself)',
             manifest_sha256=None,
             manifest_sha256_note='null by definition; the single pin direction kept is delivery-hashes.json -> this manifest',
             placeholder_rule='a placeholder digest is never written: every value below was recomputed from disk immediately before this write',
             template_id='FROZEN_TEMPLATE_v11_owner_planA')
    fp.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding='utf-8')   # step 2: sign

    # delivery-hashes.json pins the manifest and every data file
    dhd = {f: sha(d / f) for f in listing if f != 'delivery-hashes.json'}
    dh.write_text(json.dumps(dhd, ensure_ascii=False, indent=1), encoding='utf-8')

    # step 3: re-verify from disk, and assert no placeholder anywhere
    m2 = json.loads(fp.read_text(encoding='utf-8'))
    d2 = json.loads(dh.read_text(encoding='utf-8'))
    empty = hashlib.sha256(b'{}').hexdigest()
    bad = [f for f, h in m2['files'].items() if sha(d / f) != h]
    assert not bad, (d, 'stale manifest rows', bad)
    assert 'FROZEN_DELIVERY_HASHES.json' not in m2['files']
    assert len(m2['files']) == len(listing) - 2, (d, len(m2['files']), len(listing))
    assert d2['FROZEN_DELIVERY_HASHES.json'] == sha(fp), (d, 'manifest not pinned')
    assert empty not in json.dumps(m2) and empty not in json.dumps(d2), (d, 'placeholder digest present')
    for f, h in d2.items():
        if f != 'FROZEN_DELIVERY_HASHES.json':
            assert sha(d / f) == h, (d, 'stale delivery row', f)
    seal(d)
    return dict(directory=str(d), files_rows=len(m2['files']), listdir=len(listing),
                manifest_sha256=sha(fp), delivery_hashes_sha256=sha(dh),
                rule='len(files) == listdir - 2', verified_after_write=True, placeholder_absent=True)


def main():
    out = DESK / ('S2-void-and-manifest-redo-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir()

    # ---- 1. VOID declaration (directories kept as the incident record, marked from outside) ----
    void = dict(
        status='VOID', at=now(), at_utc=utc(), decided_by='the owner (plan A)',
        why='these batches were written inside open verification windows, which by the standing rule means the batch is void and redone '
            'rather than patched. Independent verification found 7 material findings across the two closed rounds, all tracing to signing a '
            'manifest before its covered files were final, or writing into live directories during a window.',
        voided_batches=[dict(path=str(p), kept_as='incident record; not deleted, not cited as evidence', readme_sha256=sha(p / 'README.zh-CN.md')
                             if (p / 'README.zh-CN.md').exists() else None,
                             note='marking is applied from this directory only; the voided directories are not modified') for p in VOIDED if p.exists()],
        surviving_evidence=dict(
            frame_alignment_scan=str(DESK / 'S2-frame-alignment-scan-20261015T180345'),
            drive_sensitivity_probe=str(DESK / 'S2-drive-sensitivity-probe-20261015T174405'),
            constant_command_hold_physical_finding='the hold test H1/H2 numbers were reproduced bit-exactly by an independent verifier, so the '
                                                  'PHYSICS stands even though the batch is void as a delivery: a constant command still '
                                                  'produces a non-foot floor contact, so occurrence is state-intrinsic and the sequence shifts when',
        ),
        carried_forward_findings=[
            'F3: four delivered delivery-hashes.json pinned sha256("{}") instead of their manifest; the redo below fixes every occurrence',
            'H3: the hold verdict said "close to the sequenced run", which fails for frame 0 (zero-drive frame, 1116 steps, outside the +/-2 frame '
            'family); the accurate statement is occurrence-intrinsic, timing-shifted',
            'H3 minor: the owning bodies are A6 (abdomen) and RWing (wing), never Thorax',
            'F5-F9: ledger bookkeeping, README verbatim-identity claims, stale frozen_at stamps, and a one-sided percentage written as "+/-"',
        ],
    )
    (out / 'void_declaration.json').write_text(json.dumps(void, ensure_ascii=False, indent=1), encoding='utf-8')

    # ---- 2. clean redo of every manifest in scope ----
    redone = []
    for d in TARGETS:
        if d.exists():
            redone.append(resign(d))
    (out / 'manifest_redo.json').write_text(json.dumps(dict(
        status='MANIFEST_REDO', at=now(), rule='finish every covered file first, sign last, recompute from disk at that instant, never write a '
             'placeholder, re-verify after writing',
        directories=redone, n=len(redone),
        placeholder_digest_that_is_now_absent=hashlib.sha256(b'{}').hexdigest(),
    ), ensure_ascii=False, indent=1), encoding='utf-8')

    # ---- freeze this record itself under the same rule ----
    fp = out / 'FROZEN_DELIVERY_HASHES.json'
    fp.write_text('{}', encoding='utf-8')
    listing = sorted(x for x in os.listdir(out) if (out / x).is_file())
    data = [f for f in listing if f not in GEN]
    m = dict(status='FROZEN_BEFORE_VERIFICATION', frozen_at=now(), frozen_at_utc=utc(),
             files={f: sha(out / f) for f in data},
             covers_complete='covers every delivered file except this manifest itself and except delivery-hashes.json',
             covers_rule='len(files) == os.listdir - 2', self_reference='not defined (a digest of a file cannot contain itself)',
             manifest_sha256=None, manifest_sha256_note='null by definition; pinned by delivery-hashes.json',
             template_id='FROZEN_TEMPLATE_v11_owner_planA')
    fp.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding='utf-8')
    dhd = {f: sha(out / f) for f in listing if f != 'delivery-hashes.json'}
    (out / 'delivery-hashes.json').write_text(json.dumps(dhd, ensure_ascii=False, indent=1), encoding='utf-8')
    assert dhd['FROZEN_DELIVERY_HASHES.json'] == sha(fp)
    assert not [f for f, h in m['files'].items() if sha(out / f) != h]
    (out / 'README.zh-CN.md').write_text('\n'.join([
        '# 判 VOID ＋ 清单族干净重做（**手动研究实施，不是成品**）', '',
        '## 一、VOID', '- 按 owner 方案 A：恒定命令批与 errata-10→14 链**判 VOID**，目录**保留作事故记录、不引用为证据**；'
        '标记只写在本目录，**VOID 目录本身未被改动**。',
        '- **物理结论保留**：核验者已**逐位复现**恒定命令三组（1116/2307/2533 等），故"恒定命令下仍发生非足对地接触"'
        '（**发生是状态固有、序列与对齐只改时刻**）成立；作废的是**交付形式**，不是这些数字。', '',
        '## 二、清单族重做（本线程第 11 版模板）',
        '- **顺序保证**：先把每个被覆盖文件写终稿，**最后才签清单**，签名瞬间从磁盘重算全部摘要，**写盘后再次校验**。',
        '- **占位符禁令**：绝不再写 `sha256("{}")`。此前 4 个目录犯过（events 目录、record v3、errata-3、扫描批），本轮全部修正。',
        '- **结构性规则**：`files` = 目录内除**清单自身**与 **`delivery-hashes.json`** 之外的全部文件（`len(files) == listdir − 2`）；'
        '`manifest_sha256: null`；`self_reference` 声明自摘要不可定义；**唯一钉向：`delivery-hashes.json` → 清单**。', '',
        '## 三、重做结果', *['- `%s`：rows %d / listdir %d，manifest `%s`，占位符已消除' % (
            pathlib.Path(r['directory']).name, r['files_rows'], r['listdir'], r['manifest_sha256'][:16]) for r in redone], '',
        '- 绝对力值与单位 `NOT_ESTABLISHED`；不宣称站立/行走/完成度。']), encoding='utf-8')
    # re-sign once more because the README was just written
    redone_final = resign(out)
    print('VOID+REDO:', out)
    print('voided dirs:', len(void['voided_batches']))
    for r in redone:
        print('  %-58s rows %d / listdir %d ok %s' % (pathlib.Path(r['directory']).name, r['files_rows'], r['listdir'], r['rule']))
    return out, redone


if __name__ == '__main__':
    main()
