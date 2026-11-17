"""One namespace over immutable v1 helpers, with archived controls and stopproof."""
from pathlib import Path
import importlib.util
import sys
import types

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'pivot_teacher_design'
STOP=ROOT.parent/'stop_shadow_control'
spec=importlib.util.spec_from_file_location('_pivotfixed_frozen_common_v1',OLD/'common.py')
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
g=dict(old.__dict__)
g.update(ROOT=ROOT,FILES=('teacher.py','common.py','adapter.py','preflight.py','runner.py','audit_saved.py','run.py','PREREG.json'))
for name,value in list(g.items()):
    if isinstance(value,types.FunctionType):g[name]=types.FunctionType(value.__code__,g,value.__name__,value.__defaults__,value.__closure__)
globals().update({k:v for k,v in g.items() if not k.startswith('__')})
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'pivot_teacher_design';STOP=ROOT.parent/'stop_shadow_control'


def fixed_stop_proof():
    p=read(ROOT/'PREREG.json')['fixed_stop_support_proof']
    if sha(STOP/'SUMMARY.json')!=p['summary_sha256'] or sha(STOP/'MATCHED_COMPARISON.json')!=p['matched_comparison_sha256']:
        raise ValueError('Actual fixedstop proofSHA drift')
    summary=read(STOP/'SUMMARY.json');comparison=read(STOP/'MATCHED_COMPARISON.json');result=read(STOP/'trials/stop/RESULT.json');audit=read(STOP/'trials/stop/VERIFY.json')
    if sha(STOP/'trials/stop/RESULT.json')!=p['fixed_result_sha256'] or not comparison['integrity_all_pass'] or not all(comparison['checks'].values()) or not summary['new_support_full_window_qualified'] or summary['stop_speed_PASS'] is not None:
        raise ValueError('Actual matched fixedstop support proofrequired, not speedPASS')
    if not audit['integrity_all_pass'] or audit['result_sha256']!=p['fixed_result_sha256'] or not all(result['candidate_metrics']['gates'].values()) or result['candidate_metrics']['task_pass_candidate'] is not None:
        raise ValueError('Actual stopaudit and supportfullwindow required')
    verify(summary['source_hashes'])
    return summary,comparison


def archived_reference(condition):
    p=read(ROOT/'PREREG.json')['archived_actual_shadow_controls']
    if condition not in p:raise ValueError('Only fixed preregistered left/right references')
    folder=OLD/'trials'/condition;entry=p[condition]
    if sha(folder/'RESULT.json')!=entry['result_sha256'] or sha(folder/'VERIFY.json')!=entry['verify_sha256']:
        raise ValueError('Same-direction archived pivotidentity drift')
    result=read(folder/'RESULT.json');audit=read(folder/'VERIFY.json')
    if not audit['integrity_all_pass'] or audit['result_sha256']!=entry['result_sha256'] or result['condition']!=condition or not result['no_neural_models']:
        raise ValueError('Archived pivot actualsaved auditrequired')
    verify(result['source_hashes'])
    for key in ('trace','geometry','teacher_samples','diagnostics'):
        item=result[key]
        if sha(item['path'])!=item['sha256']:raise ValueError('Archived pivotartifact drift')
    return folder,result,audit


def frozen():
    proof,comparison=fixed_stop_proof()
    hashes=dict(proof['source_hashes'])
    paths=[ROOT/n for n in g['FILES']]+[OLD/n for n in old.FILES]+[STOP/n for n in ('teacher.py','common.py','adapter.py','preflight.py','runner.py','audit_saved.py','run.py','PREREG.json','SUMMARY.json','MATCHED_COMPARISON.json')]
    paths += [ROOT/'DESIGN.zh-CN.md',STOP/'DESIGN.zh-CN.md',ROOT.parent/'pivot_teacher_failure_diagnostic/REPORT.zh-CN.md']
    paths += [STOP/'trials/stop'/n for n in ('RESULT.json','VERIFY.json','IDENTITY.json','BODY_LENGTH.json','HANDOVER.json')]
    stop_result=read(STOP/'trials/stop/RESULT.json')
    for key in ('trace','geometry','teacher_samples','diagnostics'):
        item=stop_result[key]
        if sha(item['path'])!=item['sha256']:raise ValueError('Fixedstop proofartifact drift')
        paths.append(Path(item['path']))
    for condition in ('left_pivot','right_pivot'):
        folder,result,audit=archived_reference(condition)
        paths += [folder/n for n in ('RESULT.json','VERIFY.json','HANDOVER.json','IDENTITY.json','BODY_LENGTH.json')]
        paths += [Path(result[k]['path']) for k in ('trace','geometry','teacher_samples','diagnostics')]
    hashes.update({str(p):sha(p) for p in paths})
    return hashes


g.update(ROOT=ROOT,frozen=frozen,fixed_stop_proof=fixed_stop_proof,archived_reference=archived_reference)
