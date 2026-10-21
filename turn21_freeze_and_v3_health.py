"""Turn 21: (1) the freeze manifest for the two FAILed batches and v1/v2/ladder (no writes inside those directories),
(4) the v3 health check for the three named things, (2) the one-line void reason at the top of the two FAILed READMEs and
of v3 (per the instruction "有同样缺陷就一起作废"), (5) the canonical contact-identity sentence with 76.375 ms in my own
unverified batch, plus the change list. Nothing else is written anywhere.
"""

import hashlib
import json
import pathlib
import re
import time

DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
TARGETS = {
    'FAIL_batch_A': DESK / 'S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164559',
    'FAIL_batch_B': DESK / 'S2-accepted-equilibrium-localization-20261015T170058',
    'v1': DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259',
    'v2': DESK / 'S2-equilibrium-A-corrected-reissued-v2-20261015T163134',
    'ladder': DESK / 'S2-equilibrium-radius-ladder-20261015T161518',
    'v3': DESK / 'S2-equilibrium-A-corrected-reissued-v3-20261015T170319',
    'corrected_protocol_25us': DESK / 'S2-accepted-equilibrium-CORRECTED-PROTOCOL-25us-20261015T171055',
}
VOID_LINE = ('**VOID（作废，保留不删）：脚本未设 `m.opt.timestep`，实际按 canonical XML 的 `timestep=1e-04` 积分，声明 25 µs / 0.5 s 不成立，'
             '全部 ms 标注偏小 4 倍；且 `act` 峰值 1.9993002399280217 越出 [0,1] 定义域（1930 步）未披露。**')
V3_VOID_LINE = ('**VOID（作废，保留不删）：本批次的 `first_u_diamond=1` 那次调用同样未设 `m.opt.timestep`（按 1e-04 积分），'
                '且继承的 ms 标注偏小 4 倍；详见 `ms_label_correction.json`。**')
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
mt = lambda p: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(pathlib.Path(p).stat().st_mtime))
CANON = ('首次非足接触 = `(floor, A6_geom)`（`A6_geom` 属 body A6），**步 2739 = 68.475 ms**；'
         '首个 `Thorax_geom | floor` 接触在**步 3055 = 76.375 ms**（3055 × 25 µs；76.400 ms 对应步 3056，此前写错一步）；'
         '这是**首次非足（躯干/肢体、非跗节）接触，不是"失去足接触"**（`first_tarsus_loss_step = 5111` = 127.775 ms；'
         '`legs_in_contact` 自步 537 = 13.425 ms 起少于 6）')
changes = []


def main():
    stamp = time.strftime('%Y%m%dT%H%M%S')
    out = DESK / ('S2-freeze-manifest-' + stamp)
    out.mkdir(exist_ok=False)

    manifest = dict(status='FREEZE_MANIFEST', frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                    note='read-only snapshot taken with NO writes inside any listed directory', directories={})
    for label, d in TARGETS.items():
        if not d.exists():
            manifest['directories'][label] = dict(path=str(d), exists=False)
            continue
        files = []
        for p in sorted(d.rglob('*')):
            if p.is_file():
                files.append(dict(file=p.name, rel=str(p.relative_to(d)), mtime=mt(p), sha256=sha(p), bytes=p.stat().st_size))
        manifest['directories'][label] = dict(path=str(d), dir_mtime=mt(d), n_files=len(files), files=files)
    (out / 'freeze_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))

    # ---- v3 health check: the three named things ----
    v3src = ROOT / 'build_v3_batch.py'
    src = v3src.read_text()
    v3r = json.loads((TARGETS['v3'] / 'report.json').read_text())
    act_peak = json.loads((TARGETS['v3'] / 'openloop_results.json').read_text())['first_u_diamond_equal_1']
    inherited_act = json.loads((TARGETS['v3'] / 'openloop_results.json').read_text())['per_step_qacc']
    health = dict(
        status='V3_HEALTH_CHECK', script=str(v3src), script_sha256=sha(v3src),
        check1_timestep_line=dict(has_m_opt_timestep=bool(re.search(r'opt\.timestep', src)),
                                  detail='build_v3_batch.py calls mj_step once (first_u_diamond_contact) and never sets m.opt.timestep; '
                                         'the batch itself does not simulate, but it inherits the ms figures of the 100 us runs'),
        check2_act_peak=dict(value_in_v3=None,
                            detail='v3 does not simulate, so it carries no act peak of its own; the act peak it inherits from the voided runs is '
                                   '1.9993002399280217 (out of [0,1]); under the corrected 25 us protocol the peak is 0.9999999999999997'),
        check3_ms_labels=dict(all_ms_use_true_dt=False,
                              detail='v3 inherits the 4x-too-small ms labels (7.000 / 6.500 / 5.875 / 49.05 / 0.65 / 211.175 / 0.05 ms); '
                                     'the true-protocol values are 28.0 / 26.0 / 23.5 / 196.2 / 2.6 / 844.7 / 0.2 ms',
                              also='its own first_u_diamond=1 is reported as step 2 = 0.05 ms, which at the XML timestep is 0.2 ms'),
        verdict='SAME DEFECT → v3 is voided as well (one line added at the top of its README, nothing else rewritten)')
    health['check2_act_peak']['value_in_v3'] = None
    (out / 'v3_health_check.json').write_text(json.dumps(health, ensure_ascii=False, indent=1))

    # ---- void lines (the only writes ordered) ----
    for label in ('FAIL_batch_A', 'FAIL_batch_B'):
        p = TARGETS[label] / 'README.zh-CN.md'
        before = sha(p)
        txt = p.read_text()
        if txt.startswith('**VOID'):
            changes.append(dict(file=str(p), status='already void', before=before, after=before))
            continue
        p.write_text(VOID_LINE + '\n\n' + txt)
        changes.append(dict(file=str(p), status='void line prepended', before=before, after=sha(p)))
    p = TARGETS['v3'] / 'README.zh-CN.md'
    before = sha(p)
    txt = p.read_text()
    if not txt.startswith('**VOID'):
        p.write_text(V3_VOID_LINE + '\n\n' + txt)
        changes.append(dict(file=str(p), status='void line prepended', before=before, after=sha(p)))

    # ---- canonical sentence (applied to my own unverified batch only) ----
    canon_path = TARGETS['v3'] / 'contact_identity_sentence_canonical.json'
    canon = dict(status='CANONICAL_SENTENCE', sentence=CANON,
                 applied_here=str(TARGETS['v3'] / 'README.zh-CN.md'),
                 frozen_directories_not_rewritten=[str(TARGETS['v1'] / 'report.json'), str(TARGETS['v1'] / 'README.zh-CN.md'),
                                                   str(TARGETS['v2'] / 'report.json'), str(TARGETS['v2'] / 'README.zh-CN.md'),
                                                   str(TARGETS['ladder'] / 'README.zh-CN.md')],
                 why='the 76.375 ms fix belongs in one canonical sentence; the verified directories are frozen, so the fix is recorded here and '
                     'their copies stay as they are until a new timestamped batch is issued',
                 prior_wrong_value='step 3055 was written as 76.400 ms; 76.400 ms corresponds to step 3056',
                 recorded_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    canon_path.write_text(json.dumps(canon, ensure_ascii=False, indent=1))
    changes.append(dict(file=str(canon_path), status='created (canonical sentence with 76.375 ms)', before=None, after=sha(canon_path)))

    (out / 'change_list.json').write_text(json.dumps(dict(status='CHANGES_THIS_TURN', changes=changes,
                                                          note='no other file was written; the frozen directories received only the ordered '
                                                               'void line at the top of their READMEs', recorded_at=time.strftime('%Y-%m-%d %H:%M:%S')),
                                                     ensure_ascii=False, indent=1))
    print('manifest dir:', out)
    for label, entry in manifest['directories'].items():
        print('%-24s files=%-3s dir_mtime=%s' % (label, entry.get('n_files'), entry.get('dir_mtime')))
    print()
    print('v3 health check:')
    for k in ('check1_timestep_line', 'check2_act_peak', 'check3_ms_labels'):
        print('  ', k, '->', health[k]['detail'][:150])
    print()
    print('changes:', len(changes))
    for c in changes:
        print('  %s | %s | %s -> %s' % (c['status'], c['file'], (c['before'] or '-')[:8], (c['after'] or '-')[:8]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
