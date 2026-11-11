"""Explicit derived copy of immutable step3000 evaluator: high-level command ramp only.

The old evaluator source is transformed with exact, AST-audited substitutions and
compiled in a private namespace. New reference association and source capture are
explicit. No training or physics executes on import.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import sys
import types
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
CONT=ROOT/'continuation3000'
BASE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
COMMANDS={'left':[0.,1.],'right':[0.,-1.],'speed085':[-.15,0.],'speed090':[-.10,0.]}


def sha256(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def command_at(condition,decision):
    if condition not in COMMANDS or not isinstance(decision,(int,np.integer)) or not 0<=decision<2880:raise ValueError('fixed condition and decision0..2879 required')
    if decision<960:return np.zeros(2,dtype=np.float64)
    u=np.clip((decision-960)*.005,0.,1.)
    return np.asarray(COMMANDS[condition],dtype=np.float64)*(3.*u*u-2.*u*u*u)


def source_paths():
    paths=[ROOT/n for n in ('evaluate_fast.py','verify_fast_actual.py','EVAL_PREREG.json','PREREG.json','fast_motor.py')]
    paths += [CONT/n for n in ('eval3000.py','verify_actual3000.py','pipeline3000.py','EVAL_PREREG.json','PREREG.json')]
    paths += [HERE/n for n in ('eval_ramp.py','verify_ramp.py','pipeline_ramp.py','check_ramp.py','EVAL_PREREG.json','READINESS.json')]
    return paths


def checked_load(fm,path,digest):
    if Path(path).resolve()!=CONT/'TRAIN/checkpoint_003000.pt':raise ValueError('only exact continuation3000 final checkpoint')
    model,p=fm.load(path,digest)
    if not p.get('curve') or p['curve'][-1]['step']!=3000:raise ValueError('requires actual3000step checkpoint')
    return model,p


def neutral_reference(path,args,identity,fm,current_ident):
    if path is None or not Path(path).is_absolute():raise ValueError('absolute passed actualstep3000 neutral RESULT required')
    path=Path(path).resolve();r=json.loads(path.read_text())
    if not path.is_relative_to(CONT):raise ValueError('neutral must be recorded original continuation3000 evaluation')
    if r.get('condition')!='neutral' or not r.get('engineering_pass') or not all(r['criteria'].values()):raise ValueError('qualified neutral required')
    if r['checkpoint_sha256']!=args.sha256 or Path(r['checkpoint']).resolve()!=Path(args.checkpoint).resolve():raise ValueError('same3000 checkpoint required')
    manifest=json.loads((path.parent/'IDENTITY.json').read_text());identity.verify(manifest)
    if manifest['compiled_arrays']!=current_ident['compiled_arrays']:raise ValueError('matchedneutral compiled runtime mismatch')
    fm.verify_hashes(r['source_hashes'])
    for filename,key in [('trace_1ms.npz','trace'),('neural_decisions.npz','neural_decisions'),('runtime_geometry.npz','runtime_geometry')]:
        if fm.sha256(path.parent/filename)!=r[key]['sha256']:raise ValueError('neutral artifact drift')
    if not r.get('neutral_baseline_neural_parity') or not all(r['neutral_baseline_neural_parity'].values()):raise ValueError('neutral original bitexact proof required')
    # Recovered pipeline may locate the independent audit differently; bind explicit CLI path.
    audit_path=Path(args.neutral_audit).resolve()
    if not audit_path.is_relative_to(CONT):raise ValueError('neutral audit outside continuation')
    audit=json.loads(audit_path.read_text())
    if not audit.get('integrity_all_pass') or not audit.get('engineering_pass') or audit['condition']!='neutral' or audit['checkpoint_sha256']!=args.sha256 or Path(audit['run']).resolve()!=path.parent:raise ValueError('qualified independently audited neutral required')
    return dict(path=str(path),sha256=fm.sha256(path),forward_signed_native_per_s=r['autonomous']['forward_signed_native_per_s'],audit=str(audit_path),audit_sha256=fm.sha256(audit_path),
                original_source_hashes=r['source_hashes'],association='NEW ramp sources differ; neutral zero-command step and smoothstep are mathematically identical; no claim identical evaluator source identities',
                neutral_command_equivalence='0*(3u^2-2u^3)=0 at every decision',checkpoint_same=True,compiled_runtime_same=True)


def substitutions():
    return [
      ("ROOT=HERE.parent\n","ROOT=HERE.parent\n"),
      ("command=np.zeros(2) if warm else np.asarray(COMMANDS[args.condition],dtype=np.float64)","command=command_at(args.condition,k)"),
      ("kind='fixed3000 continued fast neural motor actual evaluation'","kind='fixed3000 matched1s smoothstep neural command DIAGNOSTIC'"),
      ("ref=neutral_reference(args.neutral_reference,args,identity,fm,ident) if args.condition.startswith('speed') else None","ref=neutral_reference(args.neutral_reference,args,identity,fm,ident)"),
      ("status='ENGINEERING_ONLY_NOT_FULL_NATIVE_CERTIFICATION'","status='DIAGNOSTIC_ENGINEERING_ONLY_NOT_FOOD_QUALIFICATION'"),
    ]


def derived_source():
    source=(CONT/'eval3000.py').read_text()
    for old,new in substitutions():
        if source.count(old)!=1:raise ValueError('old evaluator shape changed: '+old)
        source=source.replace(old,new)
    return source


def derived_module():
    readiness=json.loads((HERE/'READINESS.json').read_text())
    for filename,digest in readiness['frozen_sources'].items():
        if sha256(filename)!=digest:raise ValueError('source readiness drift '+filename)
    module=types.ModuleType('_food_ramp3000_derived_eval')
    module.__dict__.update(__file__=str(HERE/'eval_ramp.py'),__name__=module.__name__)
    exec(compile(derived_source(),str(HERE/'eval_ramp.py')+'::derived_eval3000','exec'),module.__dict__)
    # Only explicitly declared metadata/identity/reference adapters, not physics/scoring.
    module.COMMANDS=COMMANDS
    module.source_paths=source_paths
    module.checked_load=checked_load
    module.neutral_reference=neutral_reference
    module.command_at=command_at
    return module


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--sha256',required=True);p.add_argument('--condition',choices=list(COMMANDS),required=True);p.add_argument('--out',required=True)
    p.add_argument('--neutral-reference',required=True);p.add_argument('--neutral-audit',required=True);p.add_argument('--native',action='store_true');p.add_argument('--launch-authorized',action='store_true')
    args=p.parse_args()
    if not args.launch_authorized:raise ValueError('parent explicit launch authorization required; code-only preparation')
    return derived_module().evaluate(args)

if __name__=='__main__':main()
