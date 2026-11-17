"""Stdlib metadata/NPY-header qualification for saved handover parity only.
No array values, numerical backends, checkpoint deserialization or entry main.
qualify() is parent-runtime-only after the complete nine-stage body cohort.
"""
from pathlib import Path
import ast
import hashlib
import json
import struct
import zipfile

ROOT=Path(__file__).resolve().parent
CAL=ROOT.parent/'target_feedback_order_repair_v4'
BODY=ROOT.parent/'target_feedback_motor_body_v3'
FILES=('PREREG.json','metadata.py','analyze.py','static_check.py','README.zh-CN.md')
SHA='41fa1e2232f5143fec0a3ebb5d506c819af0ba084b3a05041e829fd1477b906b'
CONDITIONS=('walk','stop','left_pivot','right_pivot')
TRAINING_IDS=('stop','left_pivot','right_pivot','walk')
BODY_KIND='TARGET_FEEDBACK_ORDER_REPAIR_MOTOR_COMPONENT_BODY_V1'
MOTOR_FORMAT='INDEPENDENT_MOTOR_TARGET_FEEDBACK_V1'


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verify(hashes):
    if not isinstance(hashes,dict) or not hashes:raise ValueError('Nonempty absolute identity map required')
    for path,digest in hashes.items():
        if not isinstance(path,str) or not Path(path).is_absolute() or not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest) or sha(path)!=digest:raise ValueError('Immutable source/input drift: '+str(path))
    return True


def header(path,key):
    """Read ZIP member NPY magic/version/header ONLY, never its array payload."""
    if not isinstance(key,str) or not key or key.endswith('.npy'):raise ValueError('Bare NPY field name required')
    with zipfile.ZipFile(path) as archive:
        members=[item for item in archive.infolist() if item.filename==key+'.npy']
        if len(members)!=1:raise ValueError('Exactly one actual NPY field required: '+key)
        with archive.open(members[0]) as stream:
            if stream.read(6)!=b'\x93NUMPY':raise ValueError('NPY magic mismatch')
            version=tuple(stream.read(2))
            if version not in ((1,0),(2,0),(3,0)):raise ValueError('Unsupported NPY header version')
            width=2 if version==(1,0) else 4;raw=stream.read(width)
            if len(raw)!=width:raise ValueError('Truncated NPY header length')
            size=struct.unpack('<H' if width==2 else '<I',raw)[0]
            if size>65536:raise ValueError('Oversize NPY header')
            text=stream.read(size)
            if len(text)!=size:raise ValueError('Truncated NPY header')
            value=ast.literal_eval(text.decode('utf8' if version==(3,0) else 'latin1'))
    if not isinstance(value,dict) or set(value)!={'descr','fortran_order','shape'} or type(value['fortran_order']) is not bool or not isinstance(value['shape'],tuple) or any(type(n) is not int or n<0 for n in value['shape']):raise ValueError('Invalid NPY header schema')
    if value['fortran_order'] or not isinstance(value['descr'],str) or 'O' in value['descr']:raise ValueError('Row-major nonobject NPY required')
    return dict(shape=list(value['shape']),descr=value['descr'],fortran_order=False)


def _members(path):
    with zipfile.ZipFile(path) as archive:return {item.filename[:-4] for item in archive.infolist() if item.filename.endswith('.npy')}


def _checks(proof,count=None,key='integrity_all_pass'):
    checks=proof.get('checks')
    if proof.get(key) is not True or not isinstance(checks,dict) or not checks or any(value is not True for value in checks.values()) or (count is not None and len(checks)!=count):raise ValueError('Actual successful saved proof required')


def _merge(*maps):
    out={}
    for values in maps:
        if not isinstance(values,dict):raise ValueError('Identity dictionary required')
        for path,digest in values.items():
            if path in out and out[path]!=digest:raise ValueError('Conflicting immutable identity: '+path)
            out[path]=digest
    return out


def _headers(path,fields):
    values={name:header(path,name) for name in fields}
    for name,shape in fields.items():
        if values[name]['shape']!=list(shape):raise ValueError('Actual saved field shape mismatch: '+str(path)+' '+name)
    return values


def _artifact(item,expected):
    path=Path(expected)
    if item['path']!=str(path) or sha(path)!=item['sha256']:raise ValueError('Actual saved artifact identity mismatch: '+str(path))
    return path


def _source(entry,index):
    condition=entry['id'];folder=Path(entry['folder']);samples=Path(entry['path'])
    paths=dict(samples=str(samples),trace=entry['trace_path'],identity=entry['identity_path'],initial_state=entry['initial_state_path'],result=str(folder/'RESULT.json'),verify=str(folder/'VERIFY.json'),handover=str(folder/'HANDOVER.json') if (folder/'HANDOVER.json').is_file() else None)
    if not folder.is_absolute() or any(not Path(path).is_absolute() for path in paths.values() if path is not None):raise ValueError('Absolute actual source paths required')
    teacher=entry['teacher'] is True;n=2880 if teacher else 3080;start=960 if teacher else 1160;warm=0 if teacher else 200
    if entry['collection'] is not False or entry['start']!=start or entry['warm_start']!=warm or entry['warm_length']!=960 or entry['length']!=1920:raise ValueError('Actual original teacher/NN source offsets required')
    result=read(paths['result']);proof=read(paths['verify']);_checks(proof)
    if sha(paths['result'])!=entry['result_sha256'] or sha(paths['verify'])!=entry['audit_sha256'] or proof['result_sha256']!=entry['result_sha256'] or result['actual_ms']!=15400:raise ValueError('Actual original full source proof required')
    trace_item=_artifact(result['trace'],paths['trace']);sample_item=_artifact(result['teacher_samples' if teacher else 'neural_decisions'],samples)
    count_key='teacher_decisions' if teacher else 'neural_records'
    if result[count_key]!=n:raise ValueError('Actual source decision count mismatch')
    sample_fields={'X238':(n,238),'time':(n,)}
    sample_fields.update({'Y42':(n,42)} if teacher else {'mean':(n,42),'policy_action':(n,42),'actually_applied_action':(n,42),'hidden_before':(n,256),'hidden_after':(n,256),'raw_turn':(n,),'command':(n,2)})
    # Actual archived motor hidden fields are256; full_hidden fields are separate.
    # Teacher hidden is NOT_RECORDED, never inferred or zero-filled.
    available=_members(samples)
    sh=_headers(samples,sample_fields)
    th=_headers(trace_item,{'time':(15400,),'qpos':(15400,49),'qvel':(15400,48),'target':(15400,42),'actualctrl':(15400,42),'action':(15400,42),'control_source':(15400,)})
    optional={name:header(samples,name) for name in ('physical_ms','physical_step_ms','raw_turn','command','control_source','candidate_phase','elapsed_teacher_s') if name in available}
    hidden_status='NOT_RECORDED' if teacher else 'RECORDED_ACTUAL_SOURCE_ABI'
    return dict(condition=condition,entry=entry,index_in_training=index,sourcefolder=str(folder),sourcepaths=paths,source_headers=dict(samples=sh,trace=th,optional_samples=optional),source_samples_path=str(sample_item),source_trace_path=str(trace_item),source_identity_path=paths['identity'],source_initial_state_path=paths['initial_state'],source_start=start,source_warm_start=warm,source_rows=n,source_hidden_status=hidden_status,source_handover_status='RECORDED' if paths['handover'] is not None else 'NOT_RECORDED',source_command_rule='[0,entry.turn]' if teacher else '[1,raw_turn[source_start]] from actual saved source row',source_hidden_fields=[] if teacher else ['hidden_before','hidden_after'],source_result=result,source_verify=proof)


def static_metadata(completed_conditions=('walk','stop')):
    """Optional header preview of only already completed cases; no qualification.
    Missing previews are reported explicitly and never imply all four are ready.
    """
    rows=[]
    for condition in completed_conditions:
        if condition not in CONDITIONS:raise ValueError('Registered body condition required')
        folder=BODY/'trials'/condition
        if not (folder/'RESULT.json').is_file():
            rows.append(dict(condition=condition,status='NOT_AVAILABLE',qualified=False));continue
        result=read(folder/'RESULT.json');path=folder/'neural_decisions.npz';n=result['neural_records']
        rows.append(dict(condition=condition,status='HEADER_PREVIEW_ONLY',qualified=False,bodyfolder=str(folder),headers=_headers(path,{'X238':(n,238),'hidden_before':(n,352),'hidden_after':(n,352),'mean':(n,42)})))
    return dict(status='OPTIONAL_COMPLETED_HEADERS_NOT_FULL_COHORT_QUALIFICATION',cases=rows,all_four_qualified=False,arrays_loaded=False)


def qualify():
    """Parent-only saved metadata admission, only AFTER all nine stages complete."""
    code=read(ROOT/'CODECHECK.json')
    if not code.get('checks') or any(value is not True for value in code['checks'].values()):raise ValueError('Completed own standard-library source audit required')
    own={str(ROOT/name):sha(ROOT/name) for name in FILES}
    if any(code['source_hashes'].get(path)!=digest for path,digest in own.items()):raise ValueError('Frozen own source identities required')
    verify(code['source_hashes'])
    jobs=read(BODY/'JOBS.json')
    expected=[(condition,stage) for condition in CONDITIONS for stage in ('PHYSICS','FRESH_NN_SAVED_AUDIT')]+[(None,'SUMMARY')]
    if len(jobs)!=9 or any(row.get('exit_code')!=0 or row.get('status')!='COMPLETED' for row in jobs) or [(row.get('condition'),row['stage']) for row in jobs]!=expected:raise ValueError('Actual complete fixed nine-stage body cohort required before diagnostic')
    body_binding=read(BODY/'INPUT_BINDINGS.json');cal_binding=read(CAL/'INPUT_BINDINGS.json');trained=read(CAL/'TRAIN/RESULT.json');stage=read(CAL/'STAGE_VERIFICATION.json');evaluation=read(CAL/'EVALUATION.json');ev=read(CAL/'EVALUATION_VERIFY.json');fresh=read(CAL/'TRAIN/VERIFY_000600.json');caljobs=read(CAL/'JOBS.json');summary=read(BODY/'SUMMARY.json')
    target=CAL/'TRAIN/checkpoint_000600.pt'
    if sha(target)!=SHA or trained['sha256']!=SHA or trained['checkpoint']!=str(target) or trained['total_updates']!=600 or trained['new_updates']!=300:raise ValueError('Actual fixed41fa final600 resume qualification required')
    if body_binding['kind']!=BODY_KIND or body_binding['checkpoint']!=str(target) or body_binding['checkpoint_sha256']!=SHA or body_binding['motor_format']!=MOTOR_FORMAT or body_binding['final_step']!=600 or body_binding['state_dims']!=[128,64,64,96] or body_binding['control_source_candidate']!=3:raise ValueError('Actual same final600 body input binding required')
    if len(caljobs)!=3 or any(row.get('exit_code')!=0 or row.get('status')!='COMPLETED' for row in caljobs) or stage['all_stages_exit0'] is not True or stage['train_updates']!=600 or stage['new_updates']!=300 or stage['checkpoint_sha256']!=SHA or stage['evaluation_sha256']!=sha(CAL/'EVALUATION.json'):raise ValueError('Actual three-stage calibration completion required')
    _checks(ev);_checks(fresh)
    if fresh['step']!=600 or fresh['checkpoint_sha256']!=SHA or ev['checkpoint_sha256']!=SHA or ev['evaluation_sha256']!=sha(CAL/'EVALUATION.json') or evaluation['checkpoint_sha256']!=SHA or evaluation['step']!=600:raise ValueError('Actual final fresh/evaluation proof required')
    mapping=cal_binding['normalizer_mapping']
    if any(len(mapping[key])!=42 for key in ('qpos_indices','halfspan42','q0_42')):raise ValueError('Actual recorded42 normalizer mapping required')
    entries=cal_binding['entries']
    if tuple(entry['id'] for entry in entries[:4])!=TRAINING_IDS or len(entries)!=12:raise ValueError('Actual first four training source identities required')
    source=_merge(own,cal_binding['source_hashes'],trained['source_hashes'],evaluation['source_hashes'],fresh['source_hashes'],body_binding['source_hashes'])
    data=_merge(cal_binding['data_hashes'],trained['data_hashes'],evaluation['data_hashes'],fresh['data_hashes'],body_binding['data_hashes'])
    proofs=_merge(body_binding['proof_hashes'])
    evalpath=_artifact(evaluation['arrays'],CAL/'EVALUATION_ARRAYS.npz');conditionalpath=_artifact(fresh['conditional_arrays'],CAL/'TRAIN/CONDITIONAL_000600.npz')
    evalheaders=_headers(evalpath,{'conditional_mean':(1920,12,42),'conditional_action':(1920,12,42),'conditional_post_targets':(1920,12,42)})
    cases=[]
    for index,entry in enumerate(entries[:4]):
        row=_source(entry,index);condition=entry['id'];folder=BODY/'trials'/condition;r=read(folder/'RESULT.json');v=read(folder/'VERIFY.json');_checks(v,42)
        if r['condition']!=condition or r['kind']!=BODY_KIND or r['checkpoint_sha256']!=SHA or r['state_total']!=352 or r['state_dims']!=[128,64,64,96] or r['final_step']!=600 or v['checkpoint_sha256']!=SHA or v['result_sha256']!=sha(folder/'RESULT.json'):raise ValueError('Actual body result/fresh42 source replay identity required')
        bodypaths=dict(samples=str(folder/'neural_decisions.npz'),trace=str(folder/'trace_1ms.npz'),identity=str(folder/'IDENTITY.json'),initial_state=str(folder/'INITIAL_STATE.json'),result=str(folder/'RESULT.json'),verify=str(folder/'VERIFY.json'),handover=str(folder/'HANDOVER.json'))
        n=r['neural_records'];ms=r['actual_ms']
        if not (1160<n<=3080 and 5800<ms<=15400):raise ValueError('Saved body must include its actual candidate window; no padding')
        _artifact(r['neural_decisions'],bodypaths['samples']);_artifact(r['trace'],bodypaths['trace'])
        bh=_headers(bodypaths['samples'],{'time':(n,),'physical_step_ms':(n,),'X238':(n,238),'command':(n,2),'mean':(n,42),'residual':(n,42),'hidden_before':(n,352),'hidden_after':(n,352),'policy_action':(n,42),'actually_applied_action':(n,42),'control_source':(n,)})
        bth=_headers(bodypaths['trace'],{'time':(ms,),'qpos':(ms,49),'qvel':(ms,48),'target':(ms,42),'actualctrl':(ms,42),'action':(ms,42),'control_source':(ms,)})
        tag=condition+str(index);fields={name:tag+'_'+name for name in ('mean','action','pretarget','posttarget','new96','residual')};ch=_headers(conditionalpath,{key:(1920,96 if name=='new96' else 42) for name,key in fields.items()})
        if fresh['checks'].get(tag+'_actualX_production_cache') is not True or fresh['checks'].get(tag+'_conditional_fullproduction') is not True or fresh['checks'].get(tag+'_conditional_finite') is not True:raise ValueError('Case actualX/conditional final600 qualifications required')
        row.update(bodyfolder=str(folder),bodypaths=bodypaths,body_headers=dict(samples=bh,trace=bth),body_samples_path=bodypaths['samples'],body_trace_path=bodypaths['trace'],body_identity_path=bodypaths['identity'],body_initial_state_path=bodypaths['initial_state'],body_rows=n,body_actual_ms=ms,body_start=1160,body_warm_start=200,body_candidate_rows=n-1160,body_result=r,body_verify=v,conditional_tag=tag,conditional_fields=fields,conditional_headers=ch,evaluation_index=index,alignment=dict(physical_handover_ms=5800,preaction_trace_row=5799,first_postdecision_trace_row=5804,source_candidate_sample=entry['start'],body_candidate_sample=1160,source_warm_start=entry['warm_start'],warm_length=960),teacher_hidden_status='NOT_RECORDED' if entry['teacher'] else 'NOT_APPLICABLE_SOURCE_IS_NN')
        source=_merge(source,r['source_hashes'],row['source_result']['source_hashes'])
        for path_map in (row['sourcepaths'],bodypaths):
            for path in path_map.values():
                if path is not None:proofs[str(path)]=sha(path)
        for key in ('initial_state','handover','geometry','runtime_geometry','body_length_artifact','rng_initial'):
            if key in r:
                item=r[key]
                if sha(item['path'])!=item['sha256']:raise ValueError('Actual saved body artifact drift')
                proofs[item['path']]=item['sha256']
        cases.append(row)
    if len(summary['rows'])!=4 or summary['checkpoint_sha256']!=SHA:raise ValueError('Actual completed four-case summary identity required')
    for row in summary['rows']:
        folder=BODY/'trials'/row['condition']
        if row['result_sha256']!=sha(folder/'RESULT.json') or row['audit_sha256']!=sha(folder/'VERIFY.json'):raise ValueError('Actual summary result/audit identity required')
    additional=[ROOT/'CODECHECK.json',BODY/'JOBS.json',BODY/'SUMMARY.json',BODY/'INPUT_BINDINGS.json',CAL/'INPUT_BINDINGS.json',CAL/'TRAIN/RESULT.json',CAL/'TRAIN/VERIFY_000600.json',CAL/'STAGE_VERIFICATION.json',CAL/'JOBS.json',CAL/'EVALUATION.json',CAL/'EVALUATION_VERIFY.json',evalpath,conditionalpath,target]
    proofs=_merge(proofs,{str(path):sha(path) for path in additional})
    verify(source);verify(data);verify(proofs)
    return dict(kind='BODY_TRAINING_HANDOVER_PARITY_SAVED_METADATA_V1',scope='Saved-array handover parity diagnostic only; not body PASS or newbrainneckdeployment',checkpoint=str(target),checkpoint_sha256=SHA,cases=cases,normalizer_mapping=mapping,cal_binding=cal_binding,evaluation_arrays=dict(path=str(evalpath),sha256=evaluation['arrays']['sha256'],sha=evaluation['arrays']['sha256'],headers=evalheaders),conditional_arrays=dict(path=str(conditionalpath),sha256=fresh['conditional_arrays']['sha256'],sha=fresh['conditional_arrays']['sha256']),source_hashes=source,data_hashes=data,proof_hashes=proofs,arrays_loaded=False,body_behavior_pass=None,training_updates=0,NN_forwards=0,Core_constructions=0,physical_steps=0)
