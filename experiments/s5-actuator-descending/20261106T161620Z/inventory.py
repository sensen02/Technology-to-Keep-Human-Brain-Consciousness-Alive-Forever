import json,hashlib
from pathlib import Path
O=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z')
S=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-actuator-descending/20261106T161620Z')
excluded={'FILE_INVENTORY.json','MANIFEST_SHA256.json','MANIFEST_CHECK.json'}
paths=sorted(p for root in [O,S] for p in root.rglob('*') if p.is_file() and p.name not in excluded)
entries=[dict(path=str(p),importance='重要辅助' if p.name=='IMPLEMENTATION_REVIEW.zh-CN.md' else '核心必需',purpose='implementation, raw evidence, protocol or verification; failed versions retained',bytes=p.stat().st_size) for p in paths]
def put(name,x):(O/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
put('FILE_INVENTORY.json',entries)
paths.append(O/'FILE_INVENTORY.json')
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
put('MANIFEST_SHA256.json',hashes)
bad=[str(p) for p in paths if hashlib.sha256(p.read_bytes()).hexdigest()!=hashes[str(p)]]
put('MANIFEST_CHECK.json',dict(files=len(paths),mismatches=bad,passed=not bad,excludes='manifest itself and its check report to avoid recursive hashing',cleanup='none: all retained failures needed for provenance'))
print(json.dumps(dict(files=len(paths),mismatches=len(bad))))
