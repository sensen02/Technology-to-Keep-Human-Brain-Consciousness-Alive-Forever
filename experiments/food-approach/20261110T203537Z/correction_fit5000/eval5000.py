"""Fixed5000 corrective-fit evaluation; no model or physics construction on import."""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import types
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
CONT=ROOT/'continuation3000'
RAMP=ROOT/'ramp3000'
BASE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
CHECKPOINT=HERE/'TRAIN/checkpoint_005000.pt'
COMMANDS={'neutral':[0.,0.],'left':[0.,1.],'right':[0.,-1.],'speed085':[-.15,0.],'speed090':[-.10,0.]}
SOURCE_NAMES=('eval5000.py','verify_actual5000.py','pipeline5000.py','EVAL_PREREG.json')

def sha256(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def old_ramp():
    spec=importlib.util.spec_from_file_location('_immutable_old_ramp',RAMP/'eval_ramp.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def command_at(condition,decision):
    if condition not in COMMANDS:raise ValueError('only five fixed conditions')
    if condition=='neutral':
        if not isinstance(decision,(int,np.integer)) or not 0<=decision<2880:raise ValueError('decision0..2879 required')
        return np.zeros(2,dtype=np.float64)
    return old_ramp_cached.command_at(condition,decision)

old_ramp_cached=old_ramp()

def check_sources():
    prereg=json.loads((HERE/'EVAL_PREREG.json').read_text())
    for path,digest in prereg['frozen_old_sources'].items():
        if sha256(path)!=digest:raise ValueError('immutable old source drift '+path)
    return prereg

def source_paths():
    prereg=check_sources()
    return [Path(p) for p in prereg['frozen_old_sources']]+[HERE/n for n in SOURCE_NAMES]

def checked_load(fm,path,digest):
    if not Path(path).is_absolute() or Path(path).resolve()!=CHECKPOINT:raise ValueError('only exact correction_fit5000/TRAIN/checkpoint_005000.pt')
    if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('explicit lowercase SHA256 required')
    model,payload=fm.load(path,digest)
    if not payload.get('curve') or payload['curve'][-1]['step']!=5000:raise ValueError('requires actual final5000 checkpoint')
    return model,payload

def neutral_reference(path,args,identity,fm,current_ident):
    if path is None or not Path(path).is_absolute() or Path(path).resolve()!=HERE/'EVAL_neutral/RESULT.json':raise ValueError('new actual neutral5000 required')
    path=Path(path).resolve();r=json.loads(path.read_text())
    if r.get('condition')!='neutral' or not r.get('engineering_pass') or not all(r['criteria'].values()):raise ValueError('neutral must actually pass first')
    if r['checkpoint_sha256']!=args.sha256 or Path(r['checkpoint']).resolve()!=CHECKPOINT:raise ValueError('same5000 checkpoint SHA required')
    manifest=json.loads((path.parent/'IDENTITY.json').read_text());identity.verify(manifest)
    if manifest['compiled_arrays']!=current_ident['compiled_arrays'] or r['source_hashes']!=current_ident['config']['evaluation_source_hashes']:raise ValueError('neutral runtime/source identity mismatch')
    fm.verify_hashes(r['source_hashes'])
    for filename,key in [('trace_1ms.npz','trace'),('neural_decisions.npz','neural_decisions'),('runtime_geometry.npz','runtime_geometry')]:
        if fm.sha256(path.parent/filename)!=r[key]['sha256']:raise ValueError('neutral saved evidence drift')
    audit_path=HERE/'VERIFY_neutral.json'
    if Path(args.neutral_audit).resolve()!=audit_path:raise ValueError('fixed new neutral audit required')
    audit=json.loads(audit_path.read_text())
    if not audit.get('integrity_all_pass') or not audit.get('engineering_pass') or audit['condition']!='neutral' or audit['checkpoint_sha256']!=args.sha256 or Path(audit['run']).resolve()!=path.parent:raise ValueError('independently audited neutral5000 must pass first')
    return dict(path=str(path),sha256=sha256(path),forward_signed_native_per_s=r['autonomous']['forward_signed_native_per_s'],audit=str(audit_path),audit_sha256=sha256(audit_path),association='new actual neutral5000, identical checkpoint SHA, evaluation sources and compiled runtime')

def replace_once(source,old,new):
    if source.count(old)!=1:raise ValueError('frozen source shape changed '+old)
    return source.replace(old,new)

def derived_source():
    source=old_ramp_cached.derived_source()
    pairs=[("prereg['checkpoint_step']!=3000","prereg['checkpoint_step']!=5000"),
           ("kind='fixed3000 matched1s smoothstep neural command DIAGNOSTIC'","kind='fixed5000 corrective-fit matched1s smoothstep actual evaluation'"),
           ("status='DIAGNOSTIC_ENGINEERING_ONLY_NOT_FOOD_QUALIFICATION'","status='ENGINEERING_ONLY_NOT_FULL_NATIVE_CERTIFICATION'"),
           ('ref=neutral_reference(args.neutral_reference,args,identity,fm,ident)','ref=neutral_reference(args.neutral_reference,args,identity,fm,ident) if args.condition!=\'neutral\' else None'),
           ("native_audit='NOT_PERFORMED',food_success=False)","native_audit='NOT_PERFORMED',food_success=False,continuation_metadata=dict(checkpoint_config=payload['config'],final_curve_row=payload['curve'][-1],evaluation_does_not_certify_training=True))")]
    for old,new in pairs:source=replace_once(source,old,new)
    return source

def derived_module():
    check_sources()
    mod=types.ModuleType('_correction5000_eval')
    mod.__dict__.update(__file__=str(HERE/'eval5000.py'),__name__=mod.__name__)
    exec(compile(derived_source(),str(HERE/'eval5000.py')+'::derived','exec'),mod.__dict__)
    mod.COMMANDS=COMMANDS;mod.source_paths=source_paths;mod.checked_load=checked_load;mod.neutral_reference=neutral_reference;mod.command_at=command_at
    return mod

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--sha256',required=True);p.add_argument('--condition',choices=list(COMMANDS),required=True);p.add_argument('--out',required=True)
    p.add_argument('--neutral-reference');p.add_argument('--neutral-audit');p.add_argument('--native',action='store_true');p.add_argument('--training-complete',action='store_true');a=p.parse_args()
    if not a.training_complete:raise ValueError('code-only until parent final checkpoint authorization: --training-complete --sha256 SHA')
    if Path(a.out).resolve()!=HERE/('EVAL_'+a.condition):raise ValueError('fixed fresh condition output required')
    return derived_module().evaluate(a)

if __name__=='__main__':main()
