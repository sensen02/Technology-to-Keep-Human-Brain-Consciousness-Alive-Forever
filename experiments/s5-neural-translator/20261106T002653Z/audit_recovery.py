"""Audit abandoned implementation; quarantine only positively identified accidental additions.
Never rewrite historical manifests or their original tracked files.
"""
from pathlib import Path
import hashlib,json,shutil
BASE=Path('/home/sensen/Desktop/engine修复验收/s5-neural-translator')
BAD=BASE/'20261106T002653Z';REC=BAD/'RECOVERY';REC.mkdir(exist_ok=False)
CODE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-neural-translator/20261106T002653Z')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def js(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
archives=[]
for stamp in ['20261105T200021Z','20261105T231405Z','20261105T234531Z']:
 root=BASE/stamp;manifest=js(root/'MANIFEST_SHA256.json');mismatch=[p for p,h in manifest.items() if not Path(p).is_file() or sha(p)!=h]
 extras=[str(p) for p in sorted(root.rglob('*')) if p.is_file() and str(p) not in manifest and p.name!='MANIFEST_SHA256.json']
 archives.append(dict(root=str(root),manifest_sha256=sha(root/'MANIFEST_SHA256.json'),tracked=len(manifest),mismatches=mismatch,extra_files_before=extras))
assert not any(a['mismatches'] for a in archives),'Original archived evidence changed: stop without altering it'
runfiles=sorted((BAD/'RUNS').glob('*/PREREGISTRATION.json'));runs=[js(p) for p in runfiles];results=[js(p.parent/'RESULT.json') for p in runfiles if (p.parent/'RESULT.json').exists()]
models={r['model_file']:r['model_sha256'] for r in runs};moved=[]
for archive in archives:
 root=Path(archive['root'])
 for name in archive['extra_files_before']:
  src=Path(name)
  if src.parent.parent!=root/'CONFIGS' or not src.parent.name.endswith('_regression_zero') or src.name not in ['MODEL.npz','PREREGISTRATION.json']:continue
  cfg=src.parent/'PREREGISTRATION.json'
  # Check declarations before moving either file; persist them in memory.
  if cfg.exists():meta=js(cfg)
  else:meta=next(m['config'] for m in moved if m['original_parent']==str(src.parent))
  assert meta['name']=='regression_zero' and meta['gain']==0
  model_file=str(src.parent/'MODEL.npz');assert model_file in models
  if src.name=='MODEL.npz':assert sha(src)==models[model_file]
  dst=REC/'UNINTENDED_ADDITIONS'/root.name/src.relative_to(root);dst.parent.mkdir(parents=True,exist_ok=True);digest=sha(src);shutil.copy2(src,dst);assert sha(dst)==digest
  moved.append(dict(old_path=str(src),new_path=str(dst),sha256=digest,bytes=src.stat().st_size,original_parent=str(src.parent),config=meta,classification='重要辅助',reason='Created by failed new attempt through old.make_variant using stale global output path; not an original manifest entry. Preserved for failed-run model references, removed only accidental addition from frozen directory.'))
  src.unlink()
  if not any(src.parent.iterdir()):src.parent.rmdir()
for archive in archives:
 root=Path(archive['root']);manifest=js(root/'MANIFEST_SHA256.json');assert sha(root/'MANIFEST_SHA256.json')==archive['manifest_sha256'];assert all(Path(p).is_file() and sha(p)==h for p,h in manifest.items())
 archive['extra_files_after']=[str(p) for p in sorted(root.rglob('*')) if p.is_file() and str(p) not in manifest and p.name!='MANIFEST_SHA256.json']
 archive['tracked_content_unchanged_after']=True
incident=dict(status='ABANDONED_INVALID_ATTEMPT; do not execute regression.py or repair_prereg.py again',classification='重要辅助',
 commands_failed=[dict(stage='startup1',reason='translator import path missing'),dict(stage='startup2',reason='translator path still points to wrong source directory'),dict(stage='startup3',reason='timing import path missing'),dict(stage='startup4',reason='torch interop thread setup repeated'),dict(job='bash-28',reason='wrong DATA/warm_dev path after5 baseline releases'),dict(job='bash-29',reason='source hash assertion after code edits; no physical runs'),dict(job='bash-30',reason='wrong DATA/warm_train path after5 further baseline releases')],
 recovery_command_failures=[dict(exit_code=127,reason='Bare python executable unavailable; audit script did not start. Corrected to existing absolute venv interpreter.')],successful_physical_releases=len(results),simulated_release_seconds=sum(r['duration_s'] for r in results),trained_candidates=0,
 reporting_corrections=['Earlier statement physical execution had not begun was false: baseline releases did execute.','Earlier statement repair was ledgered was false: repair_prereg.py overwrote INPUT_HASHES_BEFORE.json directly without retaining its original bytes.','Original overwritten BEFORE snapshot is not reconstructed or presented as original evidence. Current repaired file is retained as incident evidence only.'],
 structural_defects=['Imported helper old.make_variant wrote into frozen prior output directory because old.O was not redirected.','Data paths incomplete; fit would penalize all warm neural features instead of actual1.300s old release trajectory.','10/40/100ms initialization was paired with5/20/50ms input features/runtime: filter mismatch.','Translator.export would drop residual_gain metadata after predeclaring model hash.','Named cut controls did not pass cut flags at all.','FINAL phases were already evaluated in previous round; would not be fresh.'],
 retained_runs=[dict(folder=r['folder'],J=r['J'],model_file=r['model_file']) for r in results],historical_archives=archives,quarantined_additions=moved,
 cleanup='Only verified accidental additions moved out of historical directories; model/config bytes preserved under RECOVERY for provenance. No original historical file or manifest rewritten. Broken scripts retained solely as failure evidence; they will not be reused.',
 followup='Independent corrected implementation in a new timestamped directory; preflight before training and append-only amendments if needed')
dump(REC/'INCIDENT_AND_INTEGRITY.json',incident)
manifest={str(p):sha(p) for p in sorted(BAD.rglob('*')) if p.is_file()};manifest.update({str(p):sha(p) for p in CODE.glob('*.py')});dump(REC/'RETAINED_EVIDENCE_SHA256.json',manifest)
print(json.dumps(dict(physical_releases=len(results),original_archive_mismatches=sum(len(a['mismatches']) for a in archives),quarantined_files=len(moved),extras_after={a['root']:a['extra_files_after'] for a in archives}),ensure_ascii=False),flush=True)
