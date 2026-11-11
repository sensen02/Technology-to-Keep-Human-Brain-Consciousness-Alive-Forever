"""Post-training strict refinement readiness; no physical construction/integration."""
import argparse,json
from pathlib import Path
import model_io as io
import evaluate_motor as ev
import verify_actual
HERE=io.HERE

def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-resource-released',action='store_true');args=p.parse_args()
 if not args.parent_resource_released:raise ValueError('parentresource release required')
 import torch
 torch.set_num_threads(1);checks={}
 for step in (1000,2000):
  ck=HERE/'TRAIN'/f'checkpoint_{step:06d}.pt';audit=json.loads((HERE/'TRAIN'/f'VERIFY_{step:06d}.json').read_text());live=json.loads((HERE/'TRAIN'/f'LIVE_SAVED_{step:06d}.json').read_text())
  checks['saved_audit_'+str(step)]=audit['all_passed'] and audit['sha256']==io.sha256(ck);checks['live_saved_'+str(step)]=live['all_exact'] and all(live['checks'].values()) and live['sha256']==io.sha256(ck)
 pre=json.loads((HERE/'PREFLIGHT.json').read_text());checks['actualfullgradientpreflight']=pre['all_passed']
 checks['evaluation_adapter_builds']=callable(ev.build());checks['independent_audit_builds']=callable(verify_actual.build())
 m,payload=ev.checked_load(io,ev.CHECKPOINT,io.sha256(ev.CHECKPOINT));checks['strict_final2000']=payload['format_version']==io.FORMAT and payload['curve'][-1]['step']==2000
 checks['adapter43242_8tensors']=sum(p.numel() for p in m.parameters() if p.requires_grad)==43242 and len(m.adapter_parameters())==8
 checks['baseonlyfrozen']=all(not p.requires_grad for p in m.base.parameters()) and all(p.requires_grad for p in m.adapter_parameters())
 checks['neutralexact']=all(io.neutral(m).values())
 paths=ev.source_paths(include_readiness=False);checks['allpreCorepathsexist']=all(p.is_absolute() and p.is_file() for p in paths)
 r=dict(all_passed=all(checks.values()),checks=checks,source_hashes={str(p):io.sha256(p) for p in paths},checkpoint_sha256=io.sha256(ev.CHECKPOINT),Core_constructions=0,physics_steps=0)
 out=HERE/'READINESS.json'
 if out.exists():raise FileExistsError(out)
 out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
 if not r['all_passed']:raise ValueError('refinement readinessfailed')
 ev.source_paths();print(json.dumps(dict(all_passed=True,checks=checks,source_paths=len(paths),checkpoint_sha256=r['checkpoint_sha256'])),flush=True)
if __name__=='__main__':main()
