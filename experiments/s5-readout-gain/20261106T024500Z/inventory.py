"""Inventory and hash manifest for the parameter audit and the gain validation."""
import json
import hashlib
from pathlib import Path

PA = Path('/home/sensen/Desktop/engine修复验收/s5-neural-parameter-audit/20261106T022431Z')
GA = Path('/home/sensen/Desktop/engine修复验收/s5-readout-gain/20261106T024500Z')
CPA = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-neural-parameter-audit/20261106T022431Z')
CGA = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-readout-gain/20261106T024500Z')

GRADE = {
    'PARAMETER_SENSITIVITY.json': ('核心必需', '13个电参数的合法有界扫描全部数值'),
    'READOUT_GAIN_SCAN.json': ('核心必需', '解码增益-饱和比例扫描'),
    'TORQUE_BUDGET.json': ('核心必需', '算法预热实际力矩预算与分级读出量级对比'),
    'TORQUE_TRACES.npz': ('重要辅助', '参考与参数设置下的力矩轨迹'),
    'REPORT.zh-CN.md': ('重要辅助', '参数微调评估报告'),
    'CLEANUP_LEDGER.json': ('重要辅助', '清理台账'),
    'FILE_INVENTORY.json': ('核心必需', '重要程度分级'),
    'MANIFEST_SHA256.json': ('核心必需', '哈希封存'),
    'TRANSLATOR_IDENTITIES.json': ('核心必需', '细胞/池/旧读出矩阵与执行器限幅身份，读出与评分复算依赖'),
}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def build(root, grade, code_root, extra_runs=True):
    inv = []
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.name in ('MANIFEST_SHA256.json',):
            continue
        rel = str(p.relative_to(root))
        if rel.startswith('RUNS/'):
            g, why = '核心必需', '单次释放的原生接触/身体/神经记录与评分，供独立复算'
        else:
            g, why = grade.get(rel, ('待确认', '未登记用途，待人工确认'))
        inv.append(dict(path=str(p), importance=g, purpose=why))
    code = [dict(path=str(p), importance='核心必需', purpose='本轮实验代码，可复验')
            for p in sorted(code_root.rglob('*.py'))]
    pending = [i for i in inv + code if i['importance'] == '待确认']
    (root / 'FILE_INVENTORY.json').write_text(json.dumps(dict(files=inv, code=code, pending_confirmation=pending),
                                                          ensure_ascii=False, indent=2) + '\n')
    manifest = {i['path']: sha(i['path']) for i in inv}
    manifest[str(root / 'FILE_INVENTORY.json')] = sha(root / 'FILE_INVENTORY.json')
    for i in code:
        manifest[i['path']] = sha(i['path'])
    (root / 'MANIFEST_SHA256.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    bad = [p for p, h in manifest.items() if sha(p) != h]
    return dict(root=str(root), entries=len(manifest), mismatches=bad, pending=[i['path'] for i in pending])


def main():
    GA_GRADE = {
        'PREREGISTRATION.json': ('核心必需', '预登记：唯一变化是解码标量g，判据与保护门'),
        'GAIN_SELECTION.json': ('核心必需', '按冻结规则得到的增益排序与选择'),
        'VERDICT.json': ('核心必需', '结论、崩溃记录、机制量化'),
        'SUMMARY.json': ('重要辅助', '崩溃前写入的中间汇总'),
        'INPUT_HASHES_BEFORE.json': ('核心必需', '冻结输入哈希'),
        'INPUT_HASHES_AFTER.json': ('重要辅助', '执行后输入哈希复核'),
        'CLEANUP_LEDGER.json': ('重要辅助', '清理台账'),
        'MODELS/zero.npz': ('核心必需', '零残差分支模型，移除翻译器贡献'),
        'FILE_INVENTORY.json': ('核心必需', '重要程度分级'),
        'MANIFEST_SHA256.json': ('核心必需', '哈希封存'),
        'TRANSLATOR_IDENTITIES.json': ('核心必需', '细胞/池/旧读出矩阵与执行器限幅身份，读出与评分复算依赖'),
        'REPORT.zh-CN.md': ('重要辅助', '增益验证报告'),
    }
    a = build(PA, GRADE, CPA)
    b = build(GA, GA_GRADE, CGA)
    print(json.dumps(dict(parameter_audit=a, gain_validation=b), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
