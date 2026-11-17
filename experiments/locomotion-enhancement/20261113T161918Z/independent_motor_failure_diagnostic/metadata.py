"""Stdlib-only source/result/header qualification; no NPZ array/model/backend load."""
from pathlib import Path
import ast
import hashlib
import json
import struct
import zipfile

ROOT=Path(__file__).resolve().parent
EXP=ROOT.parent
BODY=EXP/'independent_motor_body_component'
CAL=EXP/'independent_motor_calibration'
SHA='454581a41034f92e253a2cb31215b5ddd1e09f8795ae54fb89053fd3187d1398'
CASES=('walk','stop','left_pivot','right_pivot')


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
def verify(hashes):
    for path,digest in hashes.items():
        if sha(path)!=digest:raise ValueError('Inputidentitydrift: '+path)
def header(path,key):
    with zipfile.ZipFile(path) as archive,archive.open(key+'.npy') as stream:
        if stream.read(6)!=b'\x93NUMPY':raise ValueError('NPY magic')
        version=tuple(stream.read(2))
        if version not in ((1,0),(2,0),(3,0)):raise ValueError('NPYversion')
        size=struct.unpack('<H' if version==(1,0) else '<I',stream.read(2 if version==(1,0) else 4))[0]
        if size>65536:raise ValueError('Oversizeheader')
        value=ast.literal_eval(stream.read(size).decode('utf8' if version==(3,0) else 'latin1'))
    if value['fortran_order'] or 'O' in value['descr']:raise ValueError('Rowmajor nonobjectrequired')
    return dict(shape=list(value['shape']),descr=value['descr'])

def reference(condition):
    if condition=='walk':return EXP/'terrain_neural_baseline/trials/plane_control',False
    if condition=='stop':return EXP/'stop_shadow_control/trials/stop',True
    return EXP/'pivot_fixed_shadow_v2/trials'/condition,True

def qualify():
    prereg=read(ROOT/'PREREG.json')
    if prereg['checkpoint_sha256']!=SHA:raise ValueError('Frozenactual600SHA')
    checkpoint=CAL/'TRAIN/checkpoint_000600.pt'
    if sha(checkpoint)!=SHA:raise ValueError('Actualcheckpointdrift')
    summary=read(BODY/'SUMMARY.json');jobs=read(BODY/'JOBS.json')
    if summary['actual_new_native_steps']!=4928000 or len(jobs)!=9 or any(j.get('exit_code')!=0 for j in jobs):raise ValueError('Completeactual4case9stagesrequired')
    hashes={str(checkpoint):SHA};rows=[]
    paths=[ROOT/'PREREG.json',ROOT/'metadata.py',ROOT/'stats.py',ROOT/'diagnose.py',BODY/'SUMMARY.json',BODY/'JOBS.json',CAL/'INPUT_BINDINGS.json',CAL/'STAGE_VERIFICATION.json',CAL/'EVALUATION.json',CAL/'EVALUATION_VERIFY.json',CAL/'model.py',CAL/'common.py',CAL/'data.py',CAL/'sources.py']
    for c in CASES:
        folder=BODY/'trials'/c;r=read(folder/'RESULT.json');audit=read(folder/'VERIFY.json')
        if r['checkpoint_sha256']!=SHA or r['kind']!='INDEPENDENT_MOTOR_COMPONENT_BODY_DIAGNOSTIC_V1' or not audit['integrity_all_pass'] or not all(audit['checks'].values()) or audit['result_sha256']!=sha(folder/'RESULT.json'):raise ValueError('Actualsavedbodyaudit required')
        neural=folder/'neural_decisions.npz';trace=folder/'trace_1ms.npz';geometry=folder/'geometry_1ms.npz'
        expected={'X238':[3080,238],'command':[3080,2],'mean':[3080,42],'rawmean':[3080,42],'oldmean':[3080,42],'residual':[3080,42],'hidden_before':[3080,352],'hidden_after':[3080,352],'policy_action':[3080,42],'actually_applied_action':[3080,42],'time':[3080]}
        headers={k:header(neural,k) for k in expected}
        if any(headers[k]['shape']!=shape or headers[k]['descr']!='<f8' for k,shape in expected.items()):raise ValueError('Actualcomponentheader schema')
        if header(trace,'qpos')['shape']!=[15400,49] or header(trace,'target')['shape']!=[15400,42] or header(geometry,'time')['shape']!=[14400]:raise ValueError('Actualtrace/geometryheader schema')
        for key in ('trace','neural_decisions','geometry'):
            item=r[key]
            if sha(item['path'])!=item['sha256']:raise ValueError('Actualartifactidentity')
            paths.append(Path(item['path']))
        paths += [folder/n for n in ('RESULT.json','VERIFY.json','HANDOVER.json','IDENTITY.json','BODY_LENGTH.json')]
        hashes.update(r['source_hashes'])
        ref,is_teacher=reference(c);rr=read(ref/'RESULT.json');ra=read(ref/'VERIFY.json')
        if not ra['integrity_all_pass'] or ra['result_sha256']!=sha(ref/'RESULT.json'):raise ValueError('Matchedreference actualauditrequired')
        ref_samples=ref/('teacher_samples_5ms.npz' if is_teacher else 'neural_decisions.npz')
        paths += [ref/n for n in ('RESULT.json','VERIFY.json','HANDOVER.json','trace_1ms.npz')]+[ref_samples]
        hashes.update(rr['source_hashes'])
        rows.append(dict(condition=c,folder=str(folder),neural_headers=headers,reference=str(ref),reference_is_teacher=is_teacher,reference_sample=str(ref_samples),actual_result_summary=dict(candidate_metrics=r['candidate_metrics'],native=r['native'],original_five=r['original_five'],actual_ms=r['actual_ms'])))
    cb=read(CAL/'INPUT_BINDINGS.json');hashes.update(cb['source_hashes']);hashes.update(cb['data_hashes'])
    for path in paths:hashes[str(path)]=sha(path)
    verify(hashes)
    return dict(kind='QUALIFIED_ACTUAL_FOUR_COMPONENT_FAILURE_METADATA_V1',checkpoint=str(checkpoint),checkpoint_sha256=SHA,rows=rows,source_hashes=hashes,array_values_read=False)
