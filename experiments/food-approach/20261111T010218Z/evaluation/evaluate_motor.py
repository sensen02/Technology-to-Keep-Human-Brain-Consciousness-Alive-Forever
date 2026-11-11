"""Explicit immutable evaluator reuse for a final2000 positive-command neural readout."""
from pathlib import Path
import hashlib
import json
import types
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OLD=ROOT.parent/'20261110T235830Z'
PRIMARY=OLD/'evaluation'
TRAIN=ROOT/'train'
CHECKPOINT=TRAIN/'TRAIN/checkpoint_002000.pt'
PARENT_CHECKPOINT=OLD/'train_two_only_control/TRAIN/checkpoint_001500.pt'
PARENT_SHA='c6b395f29676ef933e2f55ca5a92b7c5b55183a2d432226487b39e916b51acdb'
PARENT_RIGHT=OLD/'two_only_evaluation/EVAL_right'
PARENT_AUDIT=OLD/'two_only_evaluation/VERIFY_right.json'


def sha256(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def verify_primary():
    prereg=json.loads((HERE/'EVAL_PREREG.json').read_text())
    for p,h in prereg['frozen_primary_sources'].items():
        if sha256(p)!=h:raise ValueError('immutable evaluator source drift '+p)
    return prereg


def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def parent_paths():
    return [PARENT_CHECKPOINT,PARENT_AUDIT]+[PARENT_RIGHT/n for n in ('RESULT.json','IDENTITY.json','trace_1ms.npz','neural_decisions.npz','geometry_1ms.npz','runtime_geometry.npz')]


def verify_parent_right(t,d,g,current_ident,identity,check=None):
    """Compare saved parent arrays only. Never construct a body or execute a model."""
    checks={}
    def require(name,value):
        checks[name]=bool(value)
        if check is not None:check('right_parent_'+name,value)
        elif not value:raise ValueError('right preservation failed: '+name)
    r=json.loads((PARENT_RIGHT/'RESULT.json').read_text());audit=json.loads(PARENT_AUDIT.read_text());ident=json.loads((PARENT_RIGHT/'IDENTITY.json').read_text())
    identity.verify(ident)
    require('parent_checkpoint_digest',sha256(PARENT_CHECKPOINT)==PARENT_SHA)
    require('passed_right_binding',r['condition']=='right' and r['engineering_pass'] and r['checkpoint_sha256']==PARENT_SHA and Path(r['checkpoint']).resolve()==PARENT_CHECKPOINT and r['identity_sha256']==ident['identity_sha256'])
    require('independent_parent_right_audit',audit['integrity_all_pass'] and audit['engineering_pass'] and all(audit['checks'].values()) and audit['condition']=='right' and Path(audit['run']).resolve()==PARENT_RIGHT and audit['checkpoint_sha256']==PARENT_SHA)
    require('same_compiled_runtime',ident['compiled_arrays']==current_ident['compiled_arrays'])
    for filename,key in [('trace_1ms.npz','trace'),('neural_decisions.npz','neural_decisions'),('geometry_1ms.npz','geometry'),('runtime_geometry.npz','runtime_geometry')]:
        require(filename+'_digest',sha256(PARENT_RIGHT/filename)==r[key]['sha256'])
    oldt=arrays(PARENT_RIGHT/'trace_1ms.npz');oldd=arrays(PARENT_RIGHT/'neural_decisions.npz');oldg=arrays(PARENT_RIGHT/'geometry_1ms.npz')
    for group,new,old in [('physical',t,oldt),('neural',d,oldd),('geometry',g,oldg)]:
        for key in old:
            require(group+'_'+key,key in new and new[key].dtype==old[key].dtype and new[key].shape==old[key].shape and np.array_equal(new[key],old[key]))
    return dict(all_exact=all(checks.values()),checks=checks,reference_run=str(PARENT_RIGHT),reference_result_sha256=sha256(PARENT_RIGHT/'RESULT.json'),reference_audit_sha256=sha256(PARENT_AUDIT),parent_checkpoint_sha256=PARENT_SHA,parent_identity_sha256=ident['identity_sha256'],new_identity_sha256=current_ident['identity_sha256'],source_identity_policy='Different frozen evaluation sources are preserved separately; only checkpoint ancestry, compiled physical runtime and measured arrays are compared.',parent_evaluation_sources=r['source_hashes'])


def replace_once(source,old,new):
    if source.count(old)!=1:raise ValueError('immutable source shape changed: '+old)
    return source.replace(old,new)


def derived_source():
    verify_primary();source=(PRIMARY/'evaluate_motor.py').read_text()
    for old,new in [
        ('checkpoint_001500.pt','checkpoint_002000.pt'),
        ("payload['curve'][-1]['step']!=1500 or payload['config']['physical_eval_only_step']!=1500","payload['curve'][-1]['step']!=2000 or payload['config']['physical_eval_only_step']!=2000"),
        ("prereg['checkpoint_step']!=1500","prereg['checkpoint_step']!=2000"),
        ("ROOT/'ROUND2_PLAN.json'","ROOT/'PREREG.json'"),
        ("kind='round2 fixed1500 slow residual actual physical evaluation'","kind='fixed2000 positive-neural-readout actual physical evaluation'"),
        ("        identity.verify(ident);identity.verify_runtime(ident,c)","        if args.condition=='right':\n            result['right_parent_preservation']=verify_parent_right(\n                {key:np.asarray([row[key] for row in tr.rows]) for key in tr.rows[0]},\n                {key:np.asarray([row[key] for row in rows]) for key in rows[0]},\n                {key:np.asarray([row[key] for row in geometry]) for key in geometry[0]},ident,identity)\n        identity.verify(ident);identity.verify_runtime(ident,c)")]:
        source=replace_once(source,old,new)
    # Error/doc labels follow the explicitly changed checkpoint identity.
    source=source.replace('final1500','final2000').replace('fixed1500','fixed2000')
    return source


def build():
    module=types.ModuleType('_positive_readout_actual_evaluation')
    module.__dict__.update(__file__=str(HERE/'evaluate_motor.py'),__name__=module.__name__)
    exec(compile(derived_source(),str(HERE/'evaluate_motor.py')+'::frozen-derived','exec'),module.__dict__)
    original_paths=module.source_paths
    def source_paths():
        prereg=verify_primary()
        return list(dict.fromkeys(original_paths()+parent_paths()+[Path(p) for p in prereg['frozen_primary_sources']]))
    module.source_paths=source_paths;module.verify_parent_right=verify_parent_right
    return module


_MODULE=build()
for _name in ('BASE','ORIGINAL','COMMANDS','HEAD_AUDIT','BASE_CHECKPOINT','jsonable','command_at','score','pin','flatten','checked_load','parser','evaluate','source_paths'):
    globals()[_name]=getattr(_MODULE,_name)

if __name__=='__main__':evaluate(parser().parse_args())
