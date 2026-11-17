"""Stdlib-only stageB schema/source proof; no arrays/Core/NN loaded."""
from pathlib import Path
import ast
import hashlib
import json
import struct
import zipfile

ROOT=Path(__file__).resolve().parent
OLD=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z')
BASE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
FILES=('PREREG.json','metadata.py','restore.py','common.py','preflight.py','query.py')


def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,value):
    with Path(p).open('x') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
def verify(hashes):
    for p,h in hashes.items():
        if sha(p)!=h:raise ValueError('Sourceidentitydrift: '+p)
def header(p,key):
    with zipfile.ZipFile(p) as archive,archive.open(key+'.npy') as stream:
        if stream.read(6)!=b'\x93NUMPY':raise ValueError('NPYmagic')
        version=tuple(stream.read(2));n=2 if version==(1,0) else 4
        if version not in ((1,0),(2,0),(3,0)):raise ValueError('NPYversion')
        size=struct.unpack('<H' if n==2 else '<I',stream.read(n))[0]
        if size>65536:raise ValueError('NPYheaderoversize')
        d=ast.literal_eval(stream.read(size).decode('utf8' if version==(3,0) else 'latin1'))
    if d['fortran_order'] or 'O' in d['descr']:raise ValueError('Rowmajornonobjectrequired')
    return dict(shape=list(d['shape']),descr=d['descr'])

def teacher_folder(condition):return OLD/('stop_shadow_control/trials/stop' if condition=='stop' else 'pivot_fixed_shadow_v2/trials/'+condition)
def factory_path(condition):return OLD/('stop_shadow_control/teacher.py' if condition=='stop' else 'pivot_fixed_shadow_v2/teacher.py')

def qualify():
    prereg=read(ROOT/'PREREG.json');hashes={};teachers=[];students=[];walks=[]
    paths=[ROOT/n for n in FILES]+[BASE/n for n in ('physics.py','static_teacher.py','wave_reference.py','identity.py','student_wave.py','models.py')]+list((BASE/'sources').glob('*.py'))+[OLD/'pivot_teacher_design/common.py',OLD/'pivot_teacher_design/teacher.py',OLD/'stop_shadow_control/teacher.py',OLD/'pivot_fixed_shadow_v2/teacher.py',OLD/'pivot_fixed_shadow_v2/PREFLIGHT.json']
    def source(folder,teacher):
        result=read(folder/'RESULT.json');audit=read(folder/'VERIFY.json');identity=read(folder/'IDENTITY.json');initial=read(folder/'INITIAL_STATE.json')
        if not audit['integrity_all_pass'] or not all(audit['checks'].values()) or audit['result_sha256']!=sha(folder/'RESULT.json'):raise ValueError('Actualindependentarchiveproofrequired')
        needed=('qpos','qvel','ctrl','xfrc_applied','qfrc_applied','target','buffer')
        if any(k not in initial for k in needed) or len(initial['buffer'])!=40 or any(len(row)!=42 for row in initial['buffer']):raise ValueError('Sourceinitialactualbuffer/state missing; nozerofill')
        trace=folder/'trace_1ms.npz';count=result['actual_ms']
        fields={'qpos':[count,49],'qvel':[count,48],'target':[count,42],'actualctrl':[count,42],'padz':[count,6],'loads':[count,6],'nonfoot':[count],'time':[count],'action':[count,42],'actual_requested_action':[count,42],'external_body_force_max':[count],'external_generalized_force_max':[count]}
        headers={k:header(trace,k) for k in fields}
        if any(headers[k]['shape']!=shape for k,shape in fields.items()):raise ValueError('ActualTracefields/shapes required')
        if teacher:
            samples=folder/'teacher_samples_5ms.npz';n=result['teacher_decisions'];samplefields={'X238':[n,238],'Y42':[n,42],'physical_ms':[n],'elapsed_teacher_s':[n]}
            if n!=2880 or count!=15400 or not all(result['candidate_metrics']['gates'].values()):raise ValueError('Qualifiedfullteacher only')
        else:
            samples=folder/'neural_decisions.npz';n=result['neural_records'];samplefields={'X238':[n,238],'actually_applied_action':[n,42],'oldmean':[n,42],'physical_step_ms':[n],'command':[n,2]}
        sampleheaders={k:header(samples,k) for k in samplefields}
        if any(sampleheaders[k]['shape']!=shape for k,shape in samplefields.items()):raise ValueError('Saveddecisionfield/shapes')
        for key in ('trace','teacher_samples' if teacher else 'neural_decisions','geometry'):
            item=result[key]
            if sha(item['path'])!=item['sha256']:raise ValueError('Actualsavedartifactdrift')
            paths.append(Path(item['path']))
        paths.extend(folder/n for n in ('RESULT.json','VERIFY.json','IDENTITY.json','INITIAL_STATE.json','HANDOVER.json','BODY_LENGTH.json'))
        hashes.update(result['source_hashes'])
        for p,record in identity['files'].items():hashes[p]=record['sha256']
        if teacher:
            diagnostic=folder/'teacher_diagnostics.jsonl'
            if sha(diagnostic)!=result['diagnostics']['sha256']:raise ValueError('Actualqualifiedteacherdiagnostic SHA')
            paths.append(diagnostic)
        return dict(folder=str(folder),result=result,trace=str(trace),samples=str(samples),headers=headers,sample_headers=sampleheaders,decision_rows=n,actual_ms=count)
    for condition in ('stop','left_pivot','right_pivot'):
        item=source(teacher_folder(condition),True);item.update(id='qualification_'+condition,condition=condition,teacher_qualification=True,factory=str(factory_path(condition)),query_start=960,query_rows=1920);teachers.append(item)
    for cohort,spec in prereg['cohorts'].items():
        root=Path(spec['root']);jobs=read(root/'JOBS.json')
        if len(jobs)!=9 or any(j.get('exit_code')!=0 for j in jobs):raise ValueError('Allcohort9stagescomplete required')
        paths += [root/'SUMMARY.json',root/'JOBS.json',root/'INPUT_BINDINGS.json']
        for condition in ('walk','stop','left_pivot','right_pivot'):
            item=source(root/'trials'/condition,False)
            if item['result']['checkpoint_sha256']!=spec['checkpoint_sha256']:raise ValueError('Actualcohortcheckpointidentity')
            item.update(id=cohort+'_'+condition,cohort=cohort,condition=condition,teacher_qualification=False,factory=None if condition=='walk' else str(factory_path(condition)),query_start=1160,query_rows=item['decision_rows']-1160)
            (walks if condition=='walk' else students).append(item)
    for p in paths:hashes[str(p)]=sha(p)
    verify(hashes)
    return dict(kind=prereg['kind'],teachers=teachers,students=students,walks=walks,total_teacher_queries=sum(e['query_rows'] for e in teachers+students),source_hashes=hashes,arrays_loaded=False)
