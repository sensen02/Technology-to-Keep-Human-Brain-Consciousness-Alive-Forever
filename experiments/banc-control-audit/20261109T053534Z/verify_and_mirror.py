"""Verify audit evidence and mirror only deliverables; no simulation."""
from pathlib import Path
import json, hashlib, shutil
HERE=Path(__file__).resolve().parent
DEST=Path('/home/sensen/Desktop/engine修复验收/banc-control-audit/20261109T053534Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((HERE/'PROBE_RESULTS.json').read_text())
checks={}
for p,h in r['source_sha256'].items():checks['unchanged:'+p]=sha(Path(p))==h
checks['parameters']=r['actor_parameters']==83114 and r['all_optimizer_parameters']==83285
checks['step_reconstruction']=r['training_decisions_recorded']*80==r['training_native_steps_reconstructed']==3932160
checks['rate_limit']=r['actual_max_rate_range']==[2.,2.]
checks['schedule_difference']=r['schedule_differences']['evaluation_vs_training']['target_max_abs']>.25
checks['local_saturation']=r['run_diagnostics']['training_deterministic']['effective_rate_unchanged_if_actor_removed_at_same_states_count']==1210
report=(HERE/'控制链与训练接口审计.zh-CN.md').read_text()
checks['report_complete']='下节补入' not in report and 'RESULT_CALIBRATION.json' in report
assert all(checks.values()),checks
DEST.mkdir(parents=True,exist_ok=True)
files=['audit_probe.py','PROBE_RESULTS.json','控制链与训练接口审计.zh-CN.md','verify_and_mirror.py']
manifest=[]
for name in files:
    src=HERE/name; dst=DEST/name;shutil.copy2(src,dst)
    assert sha(src)==sha(dst)
    manifest.append({'source':str(src),'desktop':str(dst),'sha256':sha(src)})
result={'checks':checks,'all_passed':True,'mirrored':manifest,'note':'audit integrity checks only; not behavioral acceptance'}
text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
(HERE/'VERIFICATION.json').write_text(text)
(DEST/'VERIFICATION.json').write_text(text)
print(json.dumps({'passed_checks':len(checks),'source_hashes_verified':len(r['source_sha256']),'desktop':str(DEST)},ensure_ascii=False))
