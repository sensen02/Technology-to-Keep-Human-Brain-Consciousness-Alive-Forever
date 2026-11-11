"""Independent known sealed-manifest audit; no model import or training."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
MANIFEST=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/recovery_data/ARTIFACT_SHA256.json')
EXPECTED='46e44c767cf9a3ad8d00a340072921307efe815940063dfcce9893776e95f1a2'
def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
if digest(MANIFEST)!=EXPECTED:raise ValueError('known sealed corrective manifest changed')
m=json.loads(MANIFEST.read_text());checks={p:digest(p)==v['sha256'] and Path(p).stat().st_size==v['size'] for p,v in m['files'].items()}
report=dict(all_passed=all(checks.values()),manifest=str(MANIFEST),manifest_sha256=EXPECTED,file_count=len(checks),checks=checks,physical_steps=0,optimizer_updates=0)
with (HERE/'SEALED_DATA_CHECK.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(dict(all_passed=report['all_passed'],file_count=len(checks))))
if not report['all_passed']:raise ValueError('sealed artifact changed')
