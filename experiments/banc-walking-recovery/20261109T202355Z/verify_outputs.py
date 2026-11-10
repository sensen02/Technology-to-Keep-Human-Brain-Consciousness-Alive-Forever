from pathlib import Path
import hashlib,json,shutil
HERE=Path(__file__).resolve().parent
DEST=Path('/home/sensen/Desktop/engine修复验收/banc-walking-recovery/20261109T202355Z')
f=json.loads((HERE/'FRAME_AUDIT.json').read_text());w=json.loads((HERE/'WRENCH_CHECK.json').read_text());p=json.loads((HERE/'SUPPORT_PROBE_V2.json').read_text())
checks=dict(relative_jacobian=max(r['relative_error'] for r in f['jacobian_checks'])<1e-5,adhesion_wrench=w['wrench_error']<1e-10,original_sources_unchanged=all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha for path,sha in f['source_hashes'].items()),no_probe_mislabelled_success=not any(r['summary']['support_safe'] and r['summary']['advance']>=.03 and r['summary']['max_lift']>=.015 and r['summary']['final_load']>=.1 for r in p['runs']))
trainer=json.loads((HERE/'TRAINER_CHECKS.json').read_text())
checks['trainer_identity_matches_tested_file']=hashlib.sha256((HERE/'train_v2.py').read_bytes()).hexdigest()==trainer['trainer_sha256']
checks['recorded_trainer_regression_exit_zero']=trainer['exit_code']==0
assert all(checks.values()),checks
DEST.mkdir(parents=True,exist_ok=True)
files=[p for p in HERE.iterdir() if p.is_file() and p.name!='OUTPUT_MANIFEST.json']
manifest={}
for p in files:
    shutil.copy2(p,DEST/p.name);sha=hashlib.sha256(p.read_bytes()).hexdigest()
    assert hashlib.sha256((DEST/p.name).read_bytes()).hexdigest()==sha
    manifest[p.name]=sha
out=dict(checks=checks,files=manifest,desktop=str(DEST),note='Numerical checks and artifact integrity, not walking acceptance.')
(HERE/'OUTPUT_MANIFEST.json').write_text(json.dumps(out,indent=2)+'\n')
shutil.copy2(HERE/'OUTPUT_MANIFEST.json',DEST/'OUTPUT_MANIFEST.json')
print(json.dumps(dict(checks=checks,files=len(manifest),desktop=str(DEST))))
