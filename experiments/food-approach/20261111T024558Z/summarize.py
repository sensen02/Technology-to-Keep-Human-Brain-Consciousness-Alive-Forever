"""Read-only full-window pad-speed decomposition and artifact sealing."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent

def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
plan=json.loads((ROOT/'PREREG.json').read_text());summary=json.loads((ROOT/'FINAL_SUMMARY.json').read_text());prep=json.loads((ROOT/'PREPARATION.json').read_text())
for p,h in prep['source_hashes'].items():
 if digest(p)!=h:raise ValueError('source changed '+p)
metrics={}
for c in plan['candidates']:
 folder=ROOT/c['id']/'EVAL_left';r=json.loads((folder/'RESULT.json').read_text());a=json.loads((ROOT/c['id']/'VERIFY_left.json').read_text())
 if not a['integrity_all_pass'] or not all(a['checks'].values()):raise ValueError('audit failed')
 with np.load(folder/'trace_1ms.npz',allow_pickle=False) as z:t={k:z[k].copy() for k in ('time','loads','speed','com','control_source')}
 mask=t['control_source']==3;loads=t['loads'][mask];speed=t['speed'][mask];com=t['com'][mask];time=t['time'][mask]
 vt=(loads*speed).sum(0)/loads.sum(0);den=np.linalg.norm(com[-1,:2]-com[0,:2])/(time[-1]-time[0]);ratio=vt/den
 if not np.array_equal(ratio,np.asarray(list(r['autonomous']['per_leg_slip_ratio'].values()))):raise ValueError('metric recomputation mismatch')
 w=24.446354755878005;num=loads*speed
 loaded=(num*(loads>=.1*w)).sum(0)/num.sum(0)
 metrics[c['id']]=dict(alpha=c['alpha'],load_weighted_pad_tangent_speed=vt.tolist(),net_xy_speed=float(den),per_leg_slip=dict(zip(('LF','LH','LM','RF','RH','RM'),ratio.tolist())),loaded_ge_point1W_numerator_fraction=loaded.tolist(),scope='whole9.6s descriptive; no source-stratified causal claim')
(ROOT/'PAD_SPEED_DIAGNOSIS.json').write_text(json.dumps(metrics,indent=2)+'\n')
manifest={str(p):dict(sha256=digest(p),size=p.stat().st_size) for p in ROOT.rglob('*') if p.is_file() and p.name!='ARTIFACT_SHA256.json' and '__pycache__' not in p.parts}
(ROOT/'ARTIFACT_SHA256.json').write_text(json.dumps(dict(all_original_final_sources_unchanged=True,files=manifest),indent=2)+'\n')
print(json.dumps(dict(files=len(manifest),selected=summary['selected'],all_left_failed=all(not x['engineering_pass'] for x in summary['left_results'].values()))))
