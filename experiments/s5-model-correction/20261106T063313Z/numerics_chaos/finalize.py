"""File inventory, manifest and cleanup ledger for the numerics_chaos stage. No simulation."""
from pathlib import Path
import run as r

CORE = '核心必需'
HELP = '重要辅助'
TEMPORARY = '可再生临时'
UNSURE = '待确认'


def inventory():
    out = r.OUT
    rows = []

    def add(path, importance, purpose):
        fp = Path(path)
        rows.append(dict(path=str(fp), importance=importance, purpose=purpose,
                         exists=fp.exists(), bytes=(fp.stat().st_size if fp.exists() else None)))

    add(out / 'PREREGISTRATION.json', CORE, '运行前的设计、判定规则与目标细胞推导，冻结后不再修改')
    add(out / 'INPUT_HASHES_BEFORE.json', CORE, '运行前后均校验的封存输入哈希，证明未写封存目录')
    add(out / 'INPUT_HASHES_AFTER.json', CORE, '运行后同一批输入的哈希，与 BEFORE 相同')
    add(out / 'CHANGE_LEDGER.json', CORE, '旧文件/旧 sha256 与新文件/新 sha256 台账')
    add(out / 'SOURCE_FREEZE.json', CORE, '冻结源码哈希')
    add(out / 'MECHANISM_TESTS.json', CORE, '合成(非12000细胞)机制与注入精确性测试')
    add(out / 'CONTROL_BITWISE_CHECK.json', CORE, '对照与封存 numerics_followup 的逐位比较')
    add(out / 'SUMMARY.json', CORE, '4 个病例结果、三次微扰比较与预先登记判定的汇总')
    for label, _ in r.RUNS:
        folder = out / 'RUNS' / label
        add(folder / 'NEURAL.npz', CORE, '全 12000 细胞 packed 尖峰、每 1 ms 全状态/计数、命令、输入与时间轴')
        add(folder / 'RESULT.json', CORE, '该病例汇总与实测量')
        add(folder / 'INITIAL_STATE.pkl', HELP, '该病例初始完整网络与池率（含 pending ring）')
        add(folder / 'FINAL_STATE.pkl', HELP, '该病例末态完整网络与池率')
        add(folder / 'TARGET_TRACE.npz', HELP, '目标细胞逐步 v/ge/gi/adaptation 轨迹')
        site = folder / 'SITE_STATE.npz'
        if site.exists():
            add(site, HELP, 'P3 注入区间上的全网 ge/gi/v 快照，用于注入精确性校验')
    add(out / 'VERIFICATION.json', CORE, '独立重算的验证结论')
    add(out / 'REPORT.zh-CN.md', HELP, '中文报告')
    add(out / 'MANIFEST_SHA256.json', HELP, '本目录逐文件哈希清单')
    add(out / 'FILE_INVENTORY.json', HELP, '本清单')
    add(out / 'CLEANUP_LEDGER.json', HELP, '清理台账')
    add(out / 'PROGRESS.json', TEMPORARY, '运行中途进度快照，完成后可由 SUMMARY.json 替代')
    add(out / 'RUN.log', TEMPORARY, '运行日志')
    for f in sorted(out.glob('FAILURE_*.json')):
        add(f, UNSURE, '失败轨迹留档，按项目规则保留，不删除')
    return dict(folder=str(out), labels=dict(core=CORE, helper=HELP, temporary=TEMPORARY, unsure=UNSURE), files=rows)


def manifest():
    out = r.OUT
    files = {}
    for p in sorted(out.rglob('*')):
        if p.is_file():
            files[str(p)] = dict(sha256=r.sha(p), bytes=p.stat().st_size)
    r.dump(out / 'MANIFEST_SHA256.json', dict(folder=str(out), file_count=len(files), files=files))
    return files


def cleanup():
    path = r.OUT / 'CLEANUP_LEDGER.json'
    if path.exists():
        return r.js(path)
    rec = dict(deleted=[], kept_unsure=[dict(path=str(r.OUT / f), reason='失败轨迹留档')
                                       for f in sorted(p.name for p in r.OUT.glob('FAILURE_*.json'))],
               policy='只清理本阶段自己产生的、确认已被替代且无需回滚的临时文件；封存目录一律不动',
               note='本阶段未删除任何输出文件')
    r.dump(path, rec)
    return rec


def audit():
    s = r.js(r.OUT / 'SUMMARY.json')
    v = r.js(r.OUT / 'VERIFICATION.json')
    inv = inventory()
    r.dump(r.OUT / 'FILE_INVENTORY.json', inv)
    files = manifest()
    cw = r.js(r.OUT / 'CONTROL_BITWISE_CHECK.json')
    missing = [x['path'] for x in inv['files'] if not x['exists']]
    ok = dict(passed=bool(not missing and v['passed'] and cw['all_exact'] and s['physical_runs'] == 0),
              missing_inventory_files=missing,
              runs=len(s['results']), neural_steps=sum(x['neural_steps'] for x in s['results']),
              physical_runs=0, physical_steps=0, new_warm_runs=0,
              control_bitwise=cw['all_exact'], verification_checks=len(v['checks']),
              verdict=s['verdict']['reading'], selected_timestep=None, model_selection=None,
              biological_validity='NOT_ESTABLISHED',
              sealed_inputs_unchanged=r.js(r.OUT / 'INPUT_HASHES_AFTER.json') == r.checkhash(),
              source_freeze_exact=all(r.sha(p) == h for p, h in r.js(r.OUT / 'SOURCE_FREEZE.json').items()),
              manifest_file_count=len(files), cleanup=cleanup(), image=None,
              note='本阶段无图像产物；结论只涉及放大 vs 偏差与验收目标')
    r.dump(r.OUT / 'FINAL_AUDIT.json', ok)
    print('FINAL_AUDIT', ok['passed'], len(files), flush=True)


if __name__ == '__main__':
    import sys
    {'inventory': inventory, 'manifest': manifest, 'cleanup': cleanup, 'audit': audit}[sys.argv[1]]()
