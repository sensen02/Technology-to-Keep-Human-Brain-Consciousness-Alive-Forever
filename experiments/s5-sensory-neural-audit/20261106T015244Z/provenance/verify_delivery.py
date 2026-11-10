"""Verify static audit delivery without importing project code."""
from pathlib import Path
import hashlib,json
O=Path('/home/sensen/Desktop/engine修复验收/s5-sensory-neural-audit/20261106T015244Z/provenance')
S=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-sensory-neural-audit/20261106T015244Z/provenance')
assert O.resolve().is_relative_to(O.parent.resolve())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
m=json.loads((O/'SOURCE_SHA256.json').read_text())
assert all(sha(Path(x['path']))==x['sha256'] for x in m['files'])
g=json.loads((O/'GRAPH_METADATA.json').read_text());assert g['all_audited_sources_unchanged'] and g['excluded_inside_pairs']==113428
assert sum(v['nnz'] for v in g['matrices'].values())==329415
json.loads((O/'PROVENANCE_TABLE.json').read_text())
r=(O/'REPORT.zh-CN.md').read_text();assert '%' not in r and '百分' not in r
paths=[O/x for x in ['REPORT.zh-CN.md','GRAPH_METADATA.json','SOURCE_SHA256.json','PROVENANCE_TABLE.json']]+[S/'audit_metadata.py',Path(__file__)]
rows=[dict(path=str(p),resolved_path=str(p.resolve()),sha256=sha(p),importance='重要辅助' if p.suffix=='.py' else '核心必需') for p in paths]
(O/'DELIVERY_SHA256.json').write_text(json.dumps(dict(verified=True,source_files_verified=len(m['files']),files=rows,cleanup='无废弃文件；未删除任何文件',calls=dict(neural_step=0,physics_step=0)),ensure_ascii=False,indent=2)+'\n')
assert all(sha(Path(x['path']))==x['sha256'] for x in rows)
print('PASS source hashes, static counts, structured JSON, report constraints, delivery hashes; no stepping')
