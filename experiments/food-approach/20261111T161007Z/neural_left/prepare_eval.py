"""Post-training offline preparation only, explicitly gated by parent release."""
import argparse,json
from pathlib import Path
import model_io as io
import evaluate_motor as ev
import verify_actual
HERE=io.HERE

def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-resource-released',action='store_true');args=p.parse_args()
 if not args.parent_resource_released:raise ValueError('parent resource release required')
 import torch
 torch.set_num_threads(1);checks={}
 for step in (2000,4000):
  path=HERE/'TRAIN'/f'checkpoint_{step:06d}.pt';v=json.loads((HERE/'TRAIN'/f'VERIFY_{step:06d}.json').read_text());live=json.loads((HERE/'TRAIN'/f'LIVE_SAVED_{step:06d}.json').read_text())
  checks['saved_audit_'+str(step)]=v['all_passed'] and v['sha256']==io.sha256(path);checks['live_saved_'+str(step)]=live['all_exact'] and live['sha256']==io.sha256(path) and all(live['checks'].values())
 checks['preflight']=json.loads((HERE/'PREFLIGHT.json').read_text())['all_passed']
 frozen=json.loads((HERE/'CODEONLY.json').read_text());io.verify_hashes(frozen['source_hashes']);io.verify_hashes(frozen['evidence_hashes']);checks['CODEONLY_source_freeze']=frozen['all_static_checks_passed']
 checks['evaluation_adapter_builds']=callable(ev.build());checks['independent_actual_audit_builds']=callable(verify_actual.build())
 m,payload=ev.checked_load(io,ev.CHECKPOINT,io.sha256(ev.CHECKPOINT));checks['strict_final4000']=payload['curve'][-1]['step']==4000 and payload['format_version']==io.FORMAT
 checks['2730_head_only']=sum(p.numel() for p in m.parameters() if p.requires_grad)==2730
 checks['neutral_exact']=all(io.neutral(m).values())
 paths=ev.source_paths(include_readiness=False);checks['all_preCore_paths_exist']=all(p.is_absolute() and p.is_file() for p in paths)
 r=dict(all_passed=all(checks.values()),checks=checks,source_hashes={str(p):io.sha256(p) for p in paths},checkpoint_sha256=io.sha256(ev.CHECKPOINT),Core_constructions=0,physics_steps=0)
 out=HERE/'READINESS.json'
 if out.exists():raise FileExistsError(out)
 out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
 if not r['all_passed']:raise ValueError('LEFT offline evaluation readiness failed')
 ev.source_paths();print(json.dumps(r),flush=True)
if __name__=='__main__':main()
