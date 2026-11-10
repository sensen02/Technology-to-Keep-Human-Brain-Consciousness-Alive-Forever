"""Write the graded file inventory and hash manifest for the audit delivery."""
import json
import hashlib
from pathlib import Path

ROOT = Path('/home/sensen/Desktop/engine修复验收/s5-sensory-neural-audit/20261106T015244Z')
CODE = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-sensory-neural-audit/20261106T015244Z')

GRADE = {
    'AUDIT_SCOPE.json': ('核心必需', '审计范围与边界'),
    'SUMMARY.zh-CN.md': ('重要辅助', '四问总答复'),
    'FILE_INVENTORY.json': ('核心必需', '本文件的重要程度分级'),
    'MANIFEST_SHA256.json': ('核心必需', '交付哈希封存，清单本身不自哈希'),
    'tactile/EVIDENCE.json': ('核心必需', '实际12000子集的感觉覆盖与静态探针证据'),
    'tactile/REPORT.zh-CN.md': ('核心必需', '触觉接口审计报告'),
    'tactile/VERIFICATION.json': ('重要辅助', '触觉审计独立复核'),
    'provenance/REPORT.zh-CN.md': ('核心必需', '神经真实性与来源审计报告'),
    'provenance/PROVENANCE_TABLE.json': ('核心必需', '证据分层表'),
    'provenance/GRAPH_METADATA.json': ('核心必需', '逐边静态核对结果'),
    'provenance/SOURCE_SHA256.json': ('核心必需', '源文件身份'),
    'provenance/DELIVERY_SHA256.json': ('重要辅助', '来源侧交付身份'),
    'neural/NEURAL_DIAGNOSTICS_V2.json': ('核心必需', '神经时间稳定性/敏感性/维度全部数值'),
    'neural/VERIFICATION_V2.json': ('核心必需', '数值诊断独立复核'),
    'neural/SENSORY_CURRENTS.npz': ('核心必需', '真实轨迹重建的感觉电流样本'),
    'neural/REPLAY_TRACES_V2.npz': ('核心必需', '群体率/运动池率/奇异值轨迹'),
    'neural/REPORT.zh-CN.md': ('重要辅助', '神经侧数值诊断报告'),
    'neural/NEURAL_DIAGNOSTICS.json': ('重要辅助', '初版诊断，含速率单位错误，绝对值以V2为准'),
    'neural/CLEANUP_LEDGER.json': ('重要辅助', '清理台账'),
    'neural/TRANSLATOR_IDENTITIES.json': ('重要辅助', '构造器产生的身份映射，可复现'),
}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


files = sorted(p for p in ROOT.rglob('*') if p.is_file())
inventory = []
for p in files:
    rel = str(p.relative_to(ROOT))
    if rel == 'MANIFEST_SHA256.json':
        continue
    grade, purpose = GRADE.get(rel, ('待确认', '未登记用途，需人工确认后方可处置'))
    inventory.append(dict(path=str(p), importance=grade, purpose=purpose))
code = []
for p in sorted(CODE.rglob('*')):
    if p.is_file():
        code.append(dict(path=str(p), importance='核心必需' if p.suffix == '.py' else '待确认',
                         purpose='审计代码，可复验' if p.suffix == '.py' else '未登记用途'))
pending = [i for i in inventory + code if i['importance'] == '待确认']
(ROOT / 'FILE_INVENTORY.json').write_text(json.dumps(dict(files=inventory, audit_code=code,
                                                          pending_confirmation=pending,
                                                          note='待确认项未删除，等待人工确认用途后处理。'),
                                                     ensure_ascii=False, indent=2) + '\n')
manifest = {i['path']: sha(i['path']) for i in inventory}
manifest[str(ROOT / 'FILE_INVENTORY.json')] = sha(ROOT / 'FILE_INVENTORY.json')
for i in code:
    manifest[i['path']] = sha(i['path'])
(ROOT / 'MANIFEST_SHA256.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
bad = [p for p, h in manifest.items() if sha(p) != h]
print('files', len(manifest), 'mismatches', len(bad), 'pending', [i['path'] for i in pending])
