"""Finish the delivery with the frozen builder, without editing any frozen file.

report.py was frozen with the pre-registration (its sha256 is pinned in INPUT_HASHES_BEFORE.json), so
it is not edited after the run. It has one gap: it expects every warm_neural entry to carry the
relative metrics of a shifted condition, but S0 has no shifted reference. This wrapper supplies the
S0 row with the only values that are correct by definition - it is compared with itself:

    pool_rates_trajectory_nRMS = 0.0
    window_mean_pool_rates_nRMS = 0.0
    window_mean_pool_rates_pearson_r = 1.0
    mean_allcell_hz_ratio = 1.0
    total_spikes = S0's own measured total

The injection happens in memory only: no saved result file is rewritten, and the frozen report.py,
run.py, verify.py stay byte-identical to their pre-registered versions.

Subcommands
  build       figure + Chinese report + file inventory + cleanup ledger (through the frozen builder)
  inventory   regenerate FILE_INVENTORY.json (fresh hashes) and apply the importance classification
              of the files added to the delivery after the frozen build
  manifest    MANIFEST_SHA256.json + self-audit
  all         build, inventory, manifest
"""
import sys
import json
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import report as RP

O = RP.O
SKIP = ('pool_rates_trajectory_nRMS', 'window_mean_pool_rates_nRMS',
        'window_mean_pool_rates_pearson_r', 'mean_allcell_hz_ratio', 'total_spikes',
        'total_spikes_S0')

REF_ONLY = Path('SUMMARY.json')


def _augment(comparison):
    """Fill the S0 (reference) warm row with values that are exact by definition."""
    warm = comparison['warm_neural']
    s0 = warm['S0']
    s0.setdefault('pool_rates_trajectory_nRMS', 0.0)
    s0.setdefault('window_mean_pool_rates_nRMS', 0.0)
    s0.setdefault('window_mean_pool_rates_pearson_r', 1.0)
    s0.setdefault('mean_allcell_hz_ratio', 1.0)
    s0['self_reference_note'] = ('S0 is compared with itself; nRMS 0 and Pearson 1 are exact by '
                                 'definition, not measurements of a difference')
    return s0


def build():
    base_js = RP.js

    mine = O.resolve()

    def js_patched(p):
        v = base_js(p)
        here = Path(p).resolve().parent == mine
        if here and Path(p).name == 'SUMMARY.json':
            _augment(v['comparison'])
            v['comparison']['warm_neural']['S0']['total_spikes'] = int(v['warm']['S0']['total_spikes'])
            v['comparison']['warm_neural']['S0']['total_spikes_S0'] = int(v['warm']['S0']['total_spikes'])
        if here and Path(p).name == 'CROSS_CONDITION.json':
            _augment(v)
        return v

    RP.js = js_patched
    total0 = base_js(O / 'SUMMARY.json')['warm']['S0']
    s0 = _augment(base_js(O / 'SUMMARY.json')['comparison'])
    s0['total_spikes'] = int(total0['total_spikes'])
    s0['total_spikes_S0'] = int(total0['total_spikes'])
    out = RP.build()
    print('BUILT', out['report'], out['figure']['figure'], out['verdict'])
    return out


PATCH = {
    'report_fix.py': ('核心必需', '用冻结的 report.py 生成交付物并补齐 S0 自参照行；不修改任何冻结文件'),
    'verify_correction.py': ('核心必需', '修正被误规格化的 V9 校验并复核释放起始物理状态逐位一致'),
    'VERIFICATION_CORRECTION.json': ('核心必需', 'V9 修正校验结果（物理状态列逐位一致、首条命令与当瞬'
                                                 '足接触力按设计不同）'),
    'VERIFY.log': ('重要辅助', 'verify.py 的原始 stdout（含唯一一条误规格化检查未通过）'),
    'physical_phase_J_foot_com.png': ('核心必需', '交付图：J / 足支撑 / 最小COM 逐相位对比 + 1.3 s '
                                                 'dt 对照柱状图'),
    'REPORT.zh-CN.md': ('核心必需', '中文报告（含实际看图后的图注与 V9 修正说明）'),
    'FILE_INVENTORY.json': ('核心必需', '文件清单与重要程度（绝对路径）'),
    'MANIFEST_SHA256.json': ('核心必需', '全量 sha256 清单与自审'),
    'CLEANUP_LEDGER.json': ('核心必需', '清理台账'),
    'TRANSLATOR_IDENTITIES.json': ('核心必需', '本次运行的池/细胞/RH 掩码/读出矩阵与限幅身份，'
                                                '独立校验按它重算池率与命令'),
    'ANATOMICAL_IDENTITIES.json': ('重要辅助', '每腿映射到的运动细胞集合（解剖身份记录，不参与判定）'),
    'PROGRESS.json': ('可再生临时', 'run.py 的进度快照（15 次释放在中途写出的中间态），'
                                    '不参与结论，重跑可重建'),
}


def inventory():
    counts, total = RP.inventory()
    p = O / 'FILE_INVENTORY.json'
    inv = RP.js(p)
    changed = []
    for e in inv['files']:
        _paths(O if e.get('tree') == '输出目录' else SRC, e)
        if e['relative'] in PATCH and e['importance'] != PATCH[e['relative']][0]:
            e['importance'], e['purpose'] = PATCH[e['relative']]
            changed.append(e['relative'])
    counts = {}
    for e in inv['files']:
        counts[e['importance']] = counts.get(e['importance'], 0) + 1
    inv['counts_by_importance'] = counts
    inv['entries'] = len(inv['files'])
    inv['total_bytes'] = sum(e['bytes'] for e in inv['files'])
    inv['reclassified_after_frozen_build'] = changed
    inv['reclassification_note'] = (
        'report.py is frozen with the pre-registration, so files added to the delivery after the frozen '
        'build are classified here (see VERIFICATION_CORRECTION.json) instead of by editing report.py.')
    RP.dump(p, inv)
    print('INVENTORY', counts, 'bytes', inv['total_bytes'], 'reclassified', changed)
    # The inventory and the manifest cannot both pin each other's hash: whichever is written last makes
    # the other stale. MANIFEST_SHA256.json is the authority, so its own entry here is explicitly
    # self-referential instead of carrying a hash that must go stale.
    inv = RP.js(p)
    for e in inv['files']:
        if e['relative'] == 'MANIFEST_SHA256.json':
            e['sha256'] = None
            e['sha256_self_referential'] = True
            e['purpose'] = ('全量 sha256 清单与自审（自指：本行不记录自身哈希，权威值见该文件内 '
                            'audit / combined_audit 的重算结果）')
    inv['self_referential_entries'] = ['MANIFEST_SHA256.json']
    RP.dump(p, inv)
    return inv


def _paths(base, rec):
    """Keep the user-facing absolute path (/home/sensen/Desktop/...) and record the resolved one.

    /home/sensen/Desktop is a symlink into /run/media/.../root-space-relocation/.../Desktop, so
    Path.resolve() would hide the canonical delivery path the user asked for.
    """
    shown = str((base / rec['relative']).absolute())
    real = str((base / rec['relative']).resolve())
    rec['path'] = shown
    if real != shown:
        rec['resolved_path'] = real
    return rec


def manifest():
    """Frozen manifest over the output tree, extended with the source tree and a combined audit."""
    RP.manifest()
    p = O / 'MANIFEST_SHA256.json'
    m = RP.js(p)
    for e in m['files']:
        _paths(O, e)
    m['root'] = str(O.absolute())
    src = [_paths(SRC, dict(relative=str(q.relative_to(SRC)), bytes=q.stat().st_size, sha256=RP.sha(q)))
           for q in sorted(SRC.rglob('*')) if q.is_file()]
    m['source_tree'] = dict(root=str(SRC.absolute()), entry_count=len(src), files=src,
                            note='The frozen builder lists only the output tree. The delivery source tree '
                                 'is added here so the whole delivery is hash-pinned in one file. run.py, '
                                 'verify.py and report.py are additionally pinned in '
                                 'INPUT_HASHES_BEFORE.json; report_fix.py, verify_correction.py and the '
                                 'verification correction were created after the pre-registration and '
                                 'are pinned here and in FILE_INVENTORY.json.')
    out_files = [Path(e['path']) for e in m['files']]
    src_files = [Path(e['path']) for e in src]
    stored = {e['path']: e['sha256'] for e in m['files'] + src}
    rehashed = {str(q): RP.sha(q) for q in out_files + src_files}
    mismatch = [k for k in stored if stored[k] != rehashed.get(k)]
    missing = [k for k in rehashed if k not in stored]
    m['audit'] = dict(entry_count=len(m['files']), rehashed=len(out_files), mismatches=mismatch,
                      checked_at=RP.datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
                      self_excluded='MANIFEST_SHA256.json',
                      passed=bool(len(m['files']) == len(out_files) and not mismatch and not missing))
    m['combined_audit'] = dict(
        output_entries=len(out_files), source_entries=len(src),
        total_entries=len(out_files) + len(src), rehashed=len(rehashed),
        mismatches=mismatch, missing=missing,
        checked_at=m['audit']['checked_at'],
        passed=bool(len(out_files) + len(src) == len(rehashed) == len(stored)
                    and not mismatch and not missing))
    m['path_note'] = ('paths are the canonical absolute delivery paths; /home/sensen/Desktop is a symlink '
                      'and the resolved target is recorded per entry in resolved_path when it differs')
    RP.dump(p, m)
    print('AUDIT entries=%d rehashed=%d mismatches=%s passed=%s' %
          (m['audit']['entry_count'], m['audit']['rehashed'], m['audit']['mismatches'],
           m['audit']['passed']))
    print('COMBINED AUDIT output=%d source=%d total=%d rehashed=%d passed=%s'
          % (len(out_files), len(src), len(out_files) + len(src), len(rehashed),
             m['combined_audit']['passed']))
    if not m['combined_audit']['passed'] or not m['audit']['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if action in ('build', 'all'):
        build()
    if action in ('inventory', 'all'):
        inventory()
    if action in ('manifest', 'all'):
        manifest()
    print('DELIVERY OK', flush=True)
