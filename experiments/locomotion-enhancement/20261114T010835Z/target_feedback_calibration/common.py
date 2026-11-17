"""Exact parent strictstorage helpers, explicit newfcbd/source/feedback identity.
STDLIB ONLY on import. Numerical construction only inside public parent calls.
"""
from pathlib import Path
import ast
import sys

_PARENT_SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T230738Z/local_mean_corrective_calibration/common.py')
_g={'__name__':'_targetfeedback_exact_storage_helpers','__file__':__file__}
exec(compile(ast.parse(_PARENT_SOURCE.read_text()),str(_PARENT_SOURCE)+'::EXACT_STORAGE_HELPERS_NEW_GLOBALS','exec'),_g)
globals().update({k:v for k,v in _g.items() if not k.startswith('__')})
ROOT=Path(__file__).resolve().parent
PARENT_DIR=_PARENT_SOURCE.parent
PARENT_CHECKPOINT=PARENT_DIR/'TRAIN/checkpoint_000600.pt'
PARENT_SHA='fcbd818f9d6caf3431d6c545356ecd03c09369aab9eeae29363a7ccf128ce49b'
FORMAT='INDEPENDENT_MOTOR_TARGET_FEEDBACK_V1'
ACCUMULATOR=PARENT_DIR.parent/'accumulator_replay_diagnostic'
EXECUTED_DIAGNOSE_SHA='363678b72efa4420a55897042b64afc56423861b9f9c9cbc87b5dba214a61827'


def parent_qualification_hashes():
    """Actual212 storage and executed accumulator proof, NEVER postrunCODECHECK."""
    if sha(PARENT_CHECKPOINT)!=PARENT_SHA or sha(OLD_CHECKPOINT)!=OLD_SHA:raise ValueError('Actualfcbd/original789f identities required')
    tr=_read(PARENT_DIR/'TRAIN/RESULT.json');stage=_read(PARENT_DIR/'STAGE_VERIFICATION.json');jobs=_read(PARENT_DIR/'JOBS.json');review=_read(PARENT_DIR/'PARENT_CALIBRATION_REVIEW.json');ev=_read(PARENT_DIR/'EVALUATION.json');audit=_read(PARENT_DIR/'EVALUATION_VERIFY.json')
    if tr['updates']!=600 or tr['checkpoint']!=str(PARENT_CHECKPOINT) or tr['sha256']!=PARENT_SHA or review['sha256']!=PARENT_SHA or review['updates']!=600 or not review['source_and_data_hashes_rechecked']:raise ValueError('Actualfcbd final600 parentreview required')
    if not stage['all_stages_exit0'] or stage['train_updates']!=600 or stage['checkpoint_sha256']!=PARENT_SHA or len(jobs)!=3 or any(j.get('exit_code')!=0 or j.get('status')!='COMPLETED' for j in jobs):raise ValueError('All3 actualparent stages required')
    if not audit['integrity_all_pass'] or not all(audit['checks'].values()) or len(audit['checks'])!=58 or review['independent_metric_checks']!=58 or audit['evaluation_sha256']!=sha(PARENT_DIR/'EVALUATION.json') or ev['checkpoint_sha256']!=PARENT_SHA:raise ValueError('Actual58 metricproof required')
    paths=[PARENT_DIR/n for n in ('model.py','common.py','data.py','sources.py','TRAIN/RESULT.json','STAGE_VERIFICATION.json','JOBS.json','PARENT_CALIBRATION_REVIEW.json','EVALUATION.json','EVALUATION_VERIFY.json')]+[PARENT_CHECKPOINT,OLD_CHECKPOINT,Path(ev['arrays']['path'])]
    if sha(ev['arrays']['path'])!=ev['arrays']['sha256']:raise ValueError('Actualparent arrayartifact SHA')
    for step in SAVE_STEPS:
        path=PARENT_DIR/'TRAIN'/f'VERIFY_{step:06d}.json';v=_read(path)
        if not v['integrity_all_pass'] or not all(v['checks'].values()) or len(v['checks'])!=45 or v['step']!=step or sha(v['checkpoint'])!=v['checkpoint_sha256'] or len(v['rows'])!=20:raise ValueError('Actual12TRAIN8DEV freshparent storageproof required')
        bound=next(r for r in review['storage_proofs'] if r['step']==step)
        if bound['path']!=str(path) or bound['sha256']!=sha(path):raise ValueError('Parentreviewstorage SHA')
        paths.extend((path,Path(v['checkpoint'])))
    stat=_read(ACCUMULATOR/'STAT.json');proof=_read(ACCUMULATOR/'VERIFY.json');wording=_read(ACCUMULATOR/'POSTRUN_WORDING_AUDIT.json')
    if not proof['integrity_all_pass'] or not all(proof['checks'].values()) or proof['STAT_sha256']!=sha(ACCUMULATOR/'STAT.json') or proof['report_sha256']!=sha(ACCUMULATOR/'REPORT.zh-CN.md'):raise ValueError('Executedaccumulator proof required')
    executed=str(ACCUMULATOR/'diagnose.py')
    if proof['source_hashes'].get(executed)!=EXECUTED_DIAGNOSE_SHA or sha(executed)!=EXECUTED_DIAGNOSE_SHA or wording['executed_diagnose_sha256']!=EXECUTED_DIAGNOSE_SHA or wording['numerical_reruns']!=0 or not wording['numerical_results_unchanged']:raise ValueError('Executed363678identity only; postrunstatic CODECHECK not evidence')
    if stat['checkpoint_sha256']!=PARENT_SHA or len(stat['phaseA'])!=13 or any(r['actual_replay']['max_abs_target_error']!=0. or not r['actual_replay']['passed'] or not r['encoding']['passed'] or not r['encoding']['target_error_float32_exact'] or not r['encoding']['target_position_float32_exact'] for r in stat['phaseA']):raise ValueError('Actual13targetzero/float32exact necessaryproof')
    verify_hashes(proof['source_hashes']);verify_hashes(tr['source_hashes']);verify_hashes(tr['data_hashes'])
    paths += [ACCUMULATOR/n for n in ('metadata.py','analysis.py','diagnose.py','STAT.json','VERIFY.json','REPORT.zh-CN.md','POSTRUN_WORDING_AUDIT.json')]
    result=dict(proof['source_hashes']);result.update({str(p):sha(p) for p in paths});return result


def _backend():
    name='_independent_target_feedback_numeric_model';cached=sys.modules.get(name)
    if cached is not None:
        if Path(cached.__file__).resolve()!=ROOT/'model.py':raise ValueError('Targetfeedback backendnamespacecollision')
        return cached
    return _module(name,ROOT/'model.py')


# Each inherited function resolves these exact same globals, never an old ROOT.
_g.update(ROOT=ROOT,PARENT_DIR=PARENT_DIR,PARENT_CHECKPOINT=PARENT_CHECKPOINT,PARENT_SHA=PARENT_SHA,FORMAT=FORMAT,parent_qualification_hashes=parent_qualification_hashes,_backend=_backend)
