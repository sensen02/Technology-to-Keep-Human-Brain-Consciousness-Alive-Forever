"""Explicit matched parent500 binding of frozen round2 evaluator; no old edits."""
from pathlib import Path
import importlib.util
import json
import sys
import types

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
EVALUATION=ROOT/'evaluation'
LEGACY=ROOT.parent/'20261110T191449Z/residual_candidate'
BASE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
ORIGINAL=BASE.parent/'20261110T152754Z/NATIVE_ROLE_2500'
CHECKPOINT=LEGACY/'TRAIN_RESIDUAL_500/checkpoint_000500.pt'
SHA='040b26b7ff0e48329c6f6f8f3461f5ced783f108db3d48c264cdf78c5e35e7e3'
HEAD_AUDIT=ROOT.parent/'20261110T191449Z/odor/HEAD_GEOMETRY_AUDIT.json'
COMMANDS={'left':[0.,1.],'right':[0.,-1.]}
NEUTRAL=LEGACY/'EVAL_RESIDUAL500_neutral'

def module_at(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

old=module_at('_parent_control_original_evaluator',EVALUATION/'evaluate_motor.py')
sha256=old.sha256;jsonable=old.jsonable;flatten=old.flatten;score=old.score
command_at=old.command_at

def pin():
    for name in ('physics','static_teacher','student_wave','models','identity','cycle_reward','wave_reference'):
        if name in sys.modules and Path(sys.modules[name].__file__).resolve()!=BASE/(name+'.py'):raise ValueError('conflicting baseline '+name)
    sys.path.insert(0,str(BASE))
    import student_wave as sw
    import identity
    io=module_at('_parent_control_legacy_model',LEGACY/'residual_motor.py')
    return sw,identity,io


def checked_load(io,path,digest):
    if Path(path).resolve()!=CHECKPOINT or digest!=SHA:raise ValueError('exact preregistered parent500 required')
    if sha256(path)!=SHA:raise ValueError('parent500 checkpoint changed')
    model,p=io.load_candidate(path)
    cfg=p['config']
    if cfg['steps']!=500 or cfg['eval_only_step']!=500 or len(cfg['curve'])!=501 or cfg['curve'][-1]['step']!=500:raise ValueError('actual parent500 identity/curve required')
    for group in ('source_hashes','data_hashes'):
        if not cfg.get(group):raise ValueError('missing frozen ancestry '+group)
        for filename,d in cfg[group].items():
            if not Path(filename).is_absolute() or sha256(filename)!=d:raise ValueError('parent ancestry drift '+filename)
    return model.eval(),p


def neutral_association(identity,core=None):
    r=json.loads((NEUTRAL/'RESULT.json').read_text());a=json.loads((LEGACY/'ACTUAL_VERIFICATION.json').read_text());ident=json.loads((NEUTRAL/'IDENTITY.json').read_text())
    identity.verify(ident)
    if r['checkpoint_sha256']!=SHA or not r['candidate_engineering_pass'] or not r['autonomous']['all_gates'] or not r['autonomous_cycles']['all_legs_completed'] or r['autonomous']['nonfoot_force_max_native']!=0:raise ValueError('parent neutral must pass')
    ar=a['reports']['neutral']
    if not a['integrity_all_pass'] or not ar['integrity_pass'] or not ar['engineering_pass'] or not all(ar['checks'].values()):raise ValueError('old independent neutral audit failed')
    if sha256(NEUTRAL/'RESULT.json')!=ar['result_sha256']:raise ValueError('old neutral result changed')
    if sha256(NEUTRAL/'trace_1ms.npz')!=r['trace']['sha256']:raise ValueError('old neutral trace changed')
    if core is not None:identity.verify_runtime(ident,core)
    return dict(checkpoint_sha256=SHA,result=str(NEUTRAL/'RESULT.json'),result_sha256=sha256(NEUTRAL/'RESULT.json'),audit=str(LEGACY/'ACTUAL_VERIFICATION.json'),audit_sha256=sha256(LEGACY/'ACTUAL_VERIFICATION.json'),identity=str(NEUTRAL/'IDENTITY.json'),identity_sha256=sha256(NEUTRAL/'IDENTITY.json'),trace_sha256=sha256(NEUTRAL/'trace_1ms.npz'),neural_sha256=sha256(NEUTRAL/'neural_decisions.npz'),association='same parent500 checkpoint and original compiled runtime, neutral command remains zero, sources explicitly different; no repeated neutral physics',original_source_hashes={str(p):sha256(p) for p in (LEGACY/'residual_motor.py',LEGACY/'evaluate_candidate.py',LEGACY/'PREREG.json')})


def source_paths():
    files=[HERE/n for n in ('evaluate_parent.py','verify_actual.py','pipeline.py','check_prepared.py','READINESS.json','PREREG.json')]
    files += [ROOT/'MATCHED_PARENT_CONTROL_AMENDMENT.json',HEAD_AUDIT,CHECKPOINT,NEUTRAL/'RESULT.json',NEUTRAL/'IDENTITY.json',LEGACY/'ACTUAL_VERIFICATION.json']
    files += [EVALUATION/n for n in ('evaluate_motor.py','verify_actual.py')]
    files += [LEGACY/n for n in ('residual_motor.py','evaluate_candidate.py','PREREG.json')]
    return list(dict.fromkeys(p.resolve() for p in files))


def replace_once(source,oldtext,new):
    if source.count(oldtext)!=1:raise ValueError('frozen source pattern changed '+oldtext)
    return source.replace(oldtext,new)


def derived_source():
    s=(EVALUATION/'evaluate_motor.py').read_text()
    replacements=[("prereg=json.loads((HERE/'EVAL_PREREG.json').read_text())","prereg=json.loads((HERE/'PREREG.json').read_text())"),
      ("prereg['checkpoint_step']!=1500","prereg['checkpoint_step']!=500"),
      ("kind='round2 fixed1500 slow residual actual physical evaluation'","kind='preregistered matched parent500 one-second-ramp physical control'"),
      ("        if mesh.sum()!=71 or not np.all(c.plant.geom_rgba[mesh,3]==1.):raise ValueError('complete71 opaque body required')","        if mesh.sum()!=71 or not np.all(c.plant.geom_rgba[mesh,3]==1.):raise ValueError('complete71 opaque body required')\n        parent_neutral=neutral_association(identity,c)"),
      ("geometry_diagnostic_only=True)","geometry_diagnostic_only=True,parent_control_neutral=parent_neutral,optimization_steps=0)")]
    for x,y in replacements:s=replace_once(s,x,y)
    return s


def derived_module():
    m=types.ModuleType('_matched_parent500_physical');m.__dict__.update(__file__=str(HERE/'evaluate_parent.py'),__name__=m.__name__)
    exec(compile(derived_source(),str(HERE/'evaluate_parent.py')+'::derived','exec'),m.__dict__)
    for key,value in dict(HERE=HERE,ROOT=ROOT,BASE=BASE,ORIGINAL=ORIGINAL,CHECKPOINT=CHECKPOINT,HEAD_AUDIT=HEAD_AUDIT,COMMANDS=COMMANDS,pin=pin,checked_load=checked_load,source_paths=source_paths,neutral_association=neutral_association).items():setattr(m,key,value)
    # Preregistration filename is the only file-access binding change besides model/path metadata.
    m.evaluate.__globals__['__parent_control__']=True
    return m


def main():
    m=derived_module();a=m.parser().parse_args();return m.evaluate(a)

if __name__=='__main__':main()
