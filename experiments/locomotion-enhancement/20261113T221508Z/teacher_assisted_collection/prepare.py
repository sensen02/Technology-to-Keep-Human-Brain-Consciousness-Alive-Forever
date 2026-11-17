"""Stdlib-only exclusive binding of actual007d and recorded query/recovery proof."""
from pathlib import Path
import argparse
import json
from common import ROOT,CAL,FORMAT,FILES,ACTUAL_SHA,read,save,sha,verify

PREVIOUS=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T195245Z')
BODY=PREVIOUS/'fastpath_motor_body_component'
QUERY=PREVIOUS/'teacher_student_state_query'
RECOVERY=PREVIOUS/'teacher_query_report_recovery'
MOTOR_FORMAT='INDEPENDENT_SPEED_YAW_MOTOR_FASTPATH_REPAIR_V1'
MASK_NAMES=('actual_COM','actual_NF_1ms','IK','unbacked_plannedload','teacher_supported','qdot_clip','rate_clip','target_bounds','tau_ff_clip','finite42','actual_restore')
REPRO_KEYS=('action_max_abs_error','private_qdes_max_abs_error','desired_target_max_abs_error','IK_error_max_abs_error','contact_count_max_abs_difference')


def require_checks(value,key='integrity_all_pass',count=None):
    checks=value.get('checks')
    if value.get(key) is not True or not isinstance(checks,dict) or not checks or any(v is not True for v in checks.values()) or (count is not None and len(checks)!=count):raise ValueError('Actual successful proof required: '+key)


def merge(*maps):
    result={}
    for mapping in maps:
        if not isinstance(mapping,dict) or not mapping:raise ValueError('Nonempty actual identity dictionary required')
        for path,digest in mapping.items():
            if not Path(path).is_absolute() or len(digest)!=64:raise ValueError('Absolute SHA identity required')
            if path in result and result[path]!=digest:raise ValueError('Conflicting immutable identity: '+path)
            result[path]=digest
    return result


def qualification():
    if ACTUAL_SHA!='007d0c74118715eacd03d49c7e6847d8349e55db2793dd9d461a4b9c311e0fcd' or CAL!=PREVIOUS/'fastpath_motor_calibration':raise ValueError('Actual fixed007d calibration identity')
    inherited=read(BODY/'INPUT_BINDINGS.json')
    for key in ('source_hashes','data_hashes','proof_hashes'):verify(inherited[key])
    target=CAL/'TRAIN/checkpoint_000600.pt'
    if inherited['kind']!='INDEPENDENT_FASTPATH_MOTOR_COMPONENT_BODY_V1' or inherited['checkpoint']!=str(target) or inherited['checkpoint_sha256']!=ACTUAL_SHA or inherited['motor_format']!=MOTOR_FORMAT or inherited['final_step']!=600 or inherited['state_dims']!=[128,64,64,96] or sha(target)!=ACTUAL_SHA:raise ValueError('Existing strict actual007d final600 binding required')
    def pinned(path):
        path=Path(path);digest=sha(path)
        if inherited['proof_hashes'].get(str(path))!=digest:raise ValueError('Prior body actual qualification hash missing/drift: '+str(path))
        return read(path)
    trained=pinned(CAL/'TRAIN/RESULT.json');jobs=pinned(CAL/'JOBS.json');stage=pinned(CAL/'STAGE_VERIFICATION.json');ev=pinned(CAL/'EVALUATION.json');ea=pinned(CAL/'EVALUATION_VERIFY.json')
    expected=('TRAIN_FIXED600_AND_3FRESH_SAVED_VERIFICATIONS','FINAL600_EVALUATION','INDEPENDENT_SAVED_METRIC_AUDIT')
    if len(jobs)!=3 or tuple(j['stage'] for j in jobs)!=expected or any(j.get('status')!='COMPLETED' or j.get('exit_code')!=0 for j in jobs):raise ValueError('Actual calibration3stagesexit0 required')
    if trained['updates']!=600 or trained['checkpoint']!=str(target) or trained['sha256']!=ACTUAL_SHA:raise ValueError('Actual TRAIN600 result required')
    for key in ('source_hashes','data_hashes'):verify(trained[key])
    if stage['all_stages_exit0'] is not True or stage['train_updates']!=600 or stage['checkpoint']!=str(target) or stage['checkpoint_sha256']!=ACTUAL_SHA or stage['evaluation_sha256']!=sha(CAL/'EVALUATION.json') or stage['new_neck_deployed'] is not False or stage['body_behavior_pass'] is not None:raise ValueError('Actual completed calibration STAGE qualification')
    require_checks(ea)
    if ev['step']!=600 or ev['checkpoint_sha256']!=ACTUAL_SHA or ea['evaluation_sha256']!=sha(CAL/'EVALUATION.json') or sha(ev['arrays']['path'])!=ev['arrays']['sha256'] or inherited['proof_hashes'].get(ev['arrays']['path'])!=ev['arrays']['sha256']:raise ValueError('Actual independent EVAL artifact identity')
    for step in (0,300,600):
        v=pinned(CAL/'TRAIN'/f'VERIFY_{step:06d}.json');d=pinned(CAL/'TRAIN'/f'DEV_VERIFY_{step:06d}.json');pt=CAL/'TRAIN'/f'checkpoint_{step:06d}.pt';digest=sha(pt)
        require_checks(v,count=29);require_checks(d,count=4)
        if v['step']!=step or v['checkpoint']!=str(pt) or v['checkpoint_sha256']!=digest or inherited['proof_hashes'].get(str(pt))!=digest or d['step']!=step or d['checkpoint_sha256']!=digest or d['physical_steps']!=0 or d['teacher_labels_available'] is not False or len(d['rows'])!=4:raise ValueError('Actual0/300/600 fresh VERIFY plus DEV proof')
        for key in ('source_hashes','data_hashes'):verify(v[key])
    dev=pinned(CAL/'DEV_EVALUATION.json')
    if dev['checkpoint_sha256']!=ACTUAL_SHA or dev['teacher_labels_available'] is not False or len(dev['rows'])!=4:raise ValueError('Actual final DEV evaluation identity')
    qp=read(QUERY/'PREFLIGHT.json');stat=read(QUERY/'STAT.json')
    require_checks(qp,'all_passed',9)
    if len(stat.get('checks',{}))!=7 or any(v is not True for v in stat['checks'].values()) or stat['status']!='LABEL_QUERY_ONLY_NOT_PHYSICAL_SUCCESS' or stat['Core_constructions']!=9 or stat['physical_integration_steps']!=0 or stat['NN_forwards']!=0 or stat['optimizer_updates']!=0:raise ValueError('Actual query STAT7checks and recorded zero-integration scope')
    if qp['Core_constructions']!=1 or qp['physical_native_steps']!=0 or qp['NN_forwards']!=0:raise ValueError('Actual query preflight scope')
    verify(qp['source_hashes']);verify(stat['source_hashes'])
    qualifications=stat['qualification'];students=stat['students']
    if len(qualifications)!=3 or {r['condition'] for r in qualifications}!={'stop','left_pivot','right_pivot'} or len(students)!=6:raise ValueError('Actual query3teacher plus6student cases')
    artifacts=[]
    for row in qualifications+students:
        if tuple(row['label_mask_names'])!=MASK_NAMES or sha(row['artifact']['path'])!=row['artifact']['sha256']:raise ValueError('Actual query artifact/mask identity')
        query_json=QUERY/'queries'/(row['id']+'.json')
        if read(query_json)!=row:raise ValueError('Actual per-query JSON must match STAT row')
        artifacts.extend((Path(row['artifact']['path']),query_json))
    for row in qualifications:
        if row['teacher_reproduction_pass'] is not True or set(row['teacher_reproduction'])!=set(REPRO_KEYS) or any(row['teacher_reproduction'][k]!=0. for k in REPRO_KEYS) or row['errors']:raise ValueError('All3 archived teacher reproduction errors must be actual zero')
    # Read exact outputs of recover.py/audit_arrays.py; never import query.main or
    # run the recovery producer. Missing/failed proof blocks binding.
    rv=read(RECOVERY/'VERIFY.json');ra=read(RECOVERY/'AUDIT.json')
    require_checks(rv)
    if rv['original_main_exit_code']!=1 or rv['original_failure_kind']!='NameError_json_REPORT_EXPORT_ONLY' or rv['original_STAT_sha256']!=sha(QUERY/'STAT.json') or rv['original_FAILURE_sha256']!=sha(QUERY/'FAILURE.json') or rv['recovered_report_sha256']!=sha(RECOVERY/'REPORT.zh-CN.md') or rv['independent_saved_array_audit_sha256']!=sha(RECOVERY/'AUDIT.json'):raise ValueError('Actual report-only recovery artifact identity')
    if any(rv[k]!=0 for k in ('Core_constructions','teacher_queries','NN_forwards','physical_steps')):raise ValueError('Recovery must not reexecute query/NN/physics')
    if ra['kind']!='INDEPENDENT_SAVED_TEACHER_QUERY_ARRAY_AUDIT_V1' or ra['all_checks_passed'] is not True or ra['case_count']!=9 or len(ra['cases'])!=9 or ra['checks']!=rv['checks'] or ra['physical_success_inferred'] is not False:raise ValueError('Actual saved-array independent audit schema')
    verify(rv['source_hashes']);verify(ra['source_hashes'])
    ids={r['id'] for r in qualifications+students}
    if {r['id'] for r in ra['cases']}!=ids:raise ValueError('Actual saved-array audit all9case coverage')
    for row in ra['cases']:
        if tuple(row['label_mask_names'])!=MASK_NAMES:raise ValueError('Recovery actual11mask schema')
        if row['teacher_qualification']:
            reproduction=row['teacher_reproduction_independent']
            if reproduction['passes_original_1e6_tolerance'] is not True or reproduction['source_clock_exact'] is not True or reproduction['missing_numeric'] is not False or any(reproduction['maxima'][k]!=0. for k in REPRO_KEYS):raise ValueError('Actual independent3teacher allvector reproduction zero')
    code=read(ROOT/'CODECHECK.json')
    if not code.get('checks') or any(v is not True for v in code['checks'].values()):raise ValueError('Completed collection static freeze required')
    verify(code['source_hashes'])
    own={str(ROOT/name):sha(ROOT/name) for name in FILES}
    if any(code['source_hashes'].get(p)!=digest for p,digest in own.items()):raise ValueError('Actual allFILES frozen source hashes required')
    source=merge(inherited['source_hashes'],trained['source_hashes'],qp['source_hashes'],stat['source_hashes'],rv['source_hashes'],ra['source_hashes'],own)
    proof_paths=[BODY/'INPUT_BINDINGS.json',QUERY/'PREFLIGHT.json',QUERY/'STAT.json',QUERY/'FAILURE.json',RECOVERY/'VERIFY.json',RECOVERY/'AUDIT.json',RECOVERY/'REPORT.zh-CN.md',RECOVERY/'recover.py',RECOVERY/'audit_arrays.py',ROOT/'CODECHECK.json']+artifacts
    proof=merge(inherited['proof_hashes'],{str(p):sha(p) for p in proof_paths})
    data=merge(inherited['data_hashes'],trained['data_hashes'])
    verify(source);verify(data);verify(proof)
    return dict(kind=FORMAT,checkpoint=str(target),checkpoint_sha256=ACTUAL_SHA,motor_format=MOTOR_FORMAT,final_step=600,state_dims=[128,64,64,96],state_total=352,source_hashes=source,data_hashes=data,proof_hashes=proof,teacher_query_recovery_verified=True,training_updates=0,new_neck_deployed=False,body_behavior_pass=None,label='TEACHER_ASSISTED_DATA_NOT_NEURAL_ACCEPTANCE',scope='BodyANNcomponent_NOTnewbrainneckdeployment')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bind-completed-final600',action='store_true');parser.add_argument('--sha256',required=True);args=parser.parse_args()
    if not args.bind_completed_final600:parser.error('Explicit actual qualification binding required')
    if args.sha256!=ACTUAL_SHA:raise ValueError('Only actual fixed007d SHA accepted')
    if (ROOT/'INPUT_BINDINGS.json').exists():raise FileExistsError('No overwrite/rebinding')
    binding=qualification()
    for key in ('source_hashes','data_hashes','proof_hashes'):verify(binding[key])
    save(ROOT/'INPUT_BINDINGS.json',binding)
    print(json.dumps(dict(binding=str(ROOT/'INPUT_BINDINGS.json'),checkpoint_sha256=ACTUAL_SHA,label=binding['label'],body_behavior_pass=None)))


if __name__=='__main__':main()
