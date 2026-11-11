"""Explicit read-only primary-code binding for own two-trajectory control identity."""
from pathlib import Path
import hashlib
import json
import types

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PRIMARY=ROOT/'evaluation'
TRAIN=ROOT/'train_two_only_control'
CHECKPOINT=TRAIN/'TRAIN/checkpoint_001500.pt'


def sha256(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def verify_primary():
    prereg=json.loads((HERE/'EVAL_PREREG.json').read_text())
    for path,digest in prereg['frozen_primary_sources'].items():
        if sha256(path)!=digest:raise ValueError('frozen primary source drift '+path)
    return prereg


def derived_source():
    verify_primary();source=(PRIMARY/'evaluate_motor.py').read_text()
    old="kind='round2 fixed1500 slow residual actual physical evaluation'"
    if source.count(old)!=1:raise ValueError('frozen metadata source shape changed')
    return source.replace(old,"kind='round2 original-two-only budgetmatched fixed1500 control actual evaluation'")


def build():
    module=types.ModuleType('_two_only_evaluation_scoped')
    module.__dict__.update(__file__=str(HERE/'evaluate_motor.py'),__name__=module.__name__)
    exec(compile(derived_source(),str(HERE/'evaluate_motor.py')+'::frozen-derived','exec'),module.__dict__)
    # These explicit bindings change paths/identity only; original inference/physics/scoring remain intact.
    module.TRAIN=TRAIN;module.CHECKPOINT=CHECKPOINT
    original_paths=module.source_paths
    def source_paths():
        prereg=verify_primary()
        return list(dict.fromkeys(original_paths()+[ROOT/'TWO_TRAJECTORY_CONTROL_AMENDMENT.json']+[Path(p) for p in prereg['frozen_primary_sources']]))
    module.source_paths=source_paths
    return module


_MODULE=build()
for _name in ('BASE','ORIGINAL','COMMANDS','HEAD_AUDIT','BASE_CHECKPOINT','jsonable','command_at','score','pin','flatten','checked_load','parser','evaluate','source_paths'):
    globals()[_name]=getattr(_MODULE,_name)

if __name__=='__main__':evaluate(parser().parse_args())
