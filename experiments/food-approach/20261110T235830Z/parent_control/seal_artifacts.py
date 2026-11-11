"""Seal completed owned control artifacts; no simulation or optimization."""
from pathlib import Path
import json
from evaluate_parent import sha256,HERE

old=json.loads((HERE/'SUMMARY.json').read_text())
for p,h in old['source_hashes'].items():
    if sha256(p)!=h:raise ValueError('captured source changed '+p)
if sha256(HERE/'SUMMARY.json')!=sha256(HERE/'SUMMARY_INTERRUPTED_ORIGINAL.json'):raise ValueError('original interrupted summary changed')
r=json.loads((HERE/'RECOVERED_SUMMARY.json').read_text())
if not r['saved_trajectory_verification'] or r['all_conditions_eligible']:raise ValueError('unexpected recovered outcome')
files={str(p.resolve()):dict(sha256=sha256(p),size=p.stat().st_size) for p in sorted(HERE.glob('**/*')) if p.is_file() and p.name!='ARTIFACT_SHA256.json'}
if any(p.endswith('.tmp') for p in files):raise ValueError('temporary artifact remains')
report=dict(format_version=1,files=files,file_count=len(files),matched_controls=2,source_freeze_unchanged=True,all_saved_trajectory_audits_pass=True,all_controls_engineering_pass=False,new_recovery_physics_steps=0,original_job_terminal='unknown after interruption; original saved subprocess results recorded only where available',recovery_job_exit_code=0)
out=HERE/'ARTIFACT_SHA256.json'
with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
for filename,meta in files.items():
    if sha256(filename)!=meta['sha256']:raise ValueError('seal changed artifact '+filename)
print(json.dumps(dict(file_count=len(files),all_hashes_verified=True,manifest=str(out),manifest_sha256=sha256(out))))
