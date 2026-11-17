"""Single namespace over frozen v1 helpers; standard library until parent release."""
from pathlib import Path
import importlib.util
import sys
import types

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'pivot_teacher_design'
spec=importlib.util.spec_from_file_location('_stopshadow_frozen_common',OLD/'common.py')
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
g=dict(old.__dict__)
g.update(ROOT=ROOT,FILES=('teacher.py','common.py','adapter.py','preflight.py','runner.py','audit_saved.py','run.py','PREREG.json'))
for name,value in list(g.items()):
    if isinstance(value,types.FunctionType):g[name]=types.FunctionType(value.__code__,g,value.__name__,value.__defaults__,value.__closure__)
globals().update({k:v for k,v in g.items() if not k.startswith('__')})
# Preserve fresh globals ROOT after copying old module contents.
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'pivot_teacher_design'


def archived_reference():
    p=read(ROOT/'PREREG.json');folder=OLD/'trials/stop'
    if sha(folder/'RESULT.json')!=p['ancestor_result_sha256'] or sha(folder/'VERIFY.json')!=p['ancestor_verify_sha256'] or sha(folder/'HANDOVER.json')!=p['ancestor_handover_sha256']:
        raise ValueError('Fixed archived actual-shadow reference identity drift')
    result=read(folder/'RESULT.json');audit=read(folder/'VERIFY.json')
    if not audit['integrity_all_pass'] or audit['result_sha256']!=sha(folder/'RESULT.json') or result['condition']!='stop' or not result['no_neural_models']:
        raise ValueError('Actual archived teacher stop independentaudit required')
    verify(result['source_hashes'])
    for key in ('trace','geometry','teacher_samples','diagnostics'):
        item=result[key]
        if sha(item['path'])!=item['sha256']:raise ValueError('Archived savedartifact drift')
    return folder,result,audit


def frozen():
    folder,result,audit=archived_reference()
    paths=[ROOT/n for n in g['FILES']]+[OLD/n for n in old.FILES]+[old.BASE/'student_wave.py',old.BASE/'physics.py',old.BASE/'static_teacher.py',old.BASE/'wave_reference.py',old.BASE/'identity.py',old.PEAK_SOURCE]
    paths += [folder/n for n in ('RESULT.json','VERIFY.json','HANDOVER.json','IDENTITY.json','BODY_LENGTH.json')]
    paths += [Path(result[k]['path']) for k in ('trace','geometry','teacher_samples','diagnostics')]
    return {str(p):sha(p) for p in paths}


# Cloned v1 entry helpers must resolve the new source ledger, not old function.
g.update(ROOT=ROOT,frozen=frozen,archived_reference=archived_reference)
