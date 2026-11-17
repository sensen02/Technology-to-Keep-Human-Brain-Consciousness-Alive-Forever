"""Stdlib actual12 dataset manifest: immutable8 plus audited masked collection4.
NPY headers/byte hashes only. No array values or numerical backend on import.
"""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import struct
import sys
import zipfile

ROOT=Path(__file__).resolve().parent
OLD=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z')
PARENT=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T195245Z/fastpath_motor_calibration')
COLLECTION=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T221508Z/teacher_assisted_collection')
PARENT_SHA='007d0c74118715eacd03d49c7e6847d8349e55db2793dd9d461a4b9c311e0fcd'
OLD_MOTOR_DIR=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/combined_turning')
OLD_MOTOR_SHA='789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd'
FILES=('PREREG.json','sources.py','data.py','model.py','common.py','learning.py','adapter.py','preflight.py','train.py','verify_saved.py','evaluate.py','audit_eval.py','run.py')
COLLECTION_CASES=(('old_walk_control',1920,True),('assisted_stop',1920,False),('assisted_left_pivot',1280,False),('assisted_right_pivot',1293,False))


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
def verify(hashes):
    for path,digest in hashes.items():
        if sha(path)!=digest:raise ValueError('Frozen source/input drift: '+path)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value

def npy_header(path,key):
    with zipfile.ZipFile(path) as archive,archive.open(key+'.npy') as stream:
        if stream.read(6)!=b'\x93NUMPY':raise ValueError('NPYmagic')
        version=tuple(stream.read(2))
        if version not in ((1,0),(2,0),(3,0)):raise ValueError('NPYversion')
        width=2 if version==(1,0) else 4;size=struct.unpack('<H' if width==2 else '<I',stream.read(width))[0]
        if size>65536:raise ValueError('OversizeNPYheader')
        data=ast.literal_eval(stream.read(size).decode('utf8' if version==(3,0) else 'latin1'))
    if data['fortran_order'] or 'O' in data['descr']:raise ValueError('Rowmajor nonobject source arrays required')
    return dict(shape=list(data['shape']),descr=data['descr'])

def _proof(path,result_sha):
    proof=read(path)
    if proof.get('integrity_all_pass') is not True or not proof.get('checks') or not all(v is True for v in proof['checks'].values()) or proof.get('result_sha256')!=result_sha:
        raise ValueError('Actual independent source proof required: '+str(path))
    return proof

def bindings():
    original=module('_localmean_original_actual8_sources',OLD/'independent_motor_calibration/sources.py').bindings()
    io=module('_localmean_new_strict_checkpoint_io',ROOT/'common.py');parent_proofs=io.parent_qualification_hashes()
    parent_checkpoint=PARENT/'TRAIN/checkpoint_000600.pt'
    if sha(parent_checkpoint)!=PARENT_SHA or sha(OLD_MOTOR_DIR/'COMBINED.pt')!=OLD_MOTOR_SHA:raise ValueError('Fixed actual007d and original789f identities required')
    data_hashes={**original['data_hashes'],**original['source_hashes'],**parent_proofs,str(parent_checkpoint):PARENT_SHA,str(OLD_MOTOR_DIR/'COMBINED.pt'):OLD_MOTOR_SHA}
    qualification_path=COLLECTION/'PARENT_DATASET_QUALIFICATION.json';qualification=read(qualification_path);jobs=read(COLLECTION/'JOBS.json');summary=read(COLLECTION/'SUMMARY.json')
    if qualification.get('status')!='ELIGIBLE_FOR_MASKED_SUPERVISED_CORRECTIVE_LEARNING_NOT_NEURAL_ACCEPTANCE' or qualification.get('eligible_for_masked_supervised_learning') is not True or qualification.get('all_raw_labels_eligible') is not False or qualification.get('all12stages_exit0') is not True or qualification.get('all_sources_reverified') is not True or qualification.get('neural_acceptance') is not False:
        raise ValueError('Actual parent dataset masked-only qualification required')
    if qualification['student_checkpoint_sha256']!=PARENT_SHA or qualification['student_mix']!=.25 or qualification['teacher_mix']!=.75 or qualification['assisted_teacher_valid_labels']!=4493 or qualification['old_control_actual_suffix_rows']!=1920:
        raise ValueError('Fixed teacher-assisted collection qualification counts/mixing')
    if len(jobs)!=12 or any(j.get('exit_code')!=0 or j.get('status')!='COMPLETED' for j in jobs) or summary.get('stages')!=12 or summary.get('neural_acceptance') is not False or summary.get('new_neck_deployed') is not False:
        raise ValueError('All12 collection stages complete, no neural acceptance')
    if summary['actual_new_native_steps']!=4928000 or qualification['actual_new_native_steps']!=4928000:raise ValueError('Actual fixed4 collection completion')
    qualification_rows={row['case']:row for row in qualification['rows']}
    if set(qualification_rows)!={row[0] for row in COLLECTION_CASES}:raise ValueError('Exactly4 actual qualified collection sources')
    entries=[]
    for old in original['entries']:
        entry=dict(old);entry.update(source_kind='original_teacher' if entry['teacher'] else 'original_neural',collection=False,v1_prior=not entry['teacher'],expected_valid_count=1920,labels_available=True,training_input=True)
        entries.append(entry)
    collection_headers={key:([3080,238] if key=='X238' else [3080,352] if key in ('hidden_before','hidden_after') else [3080,11] if key=='label_mask' else [3080,2] if key=='command' else [3080,42] if key in ('mean','rawmean','oldmean_direct','oldmean_compat','residual','supervision_target','teacher_action','actually_applied_action','mixed_applied_action','policy_action','student_action') else [3080]) for key in ('X238','command','mean','rawmean','oldmean_direct','oldmean_compat','residual','hidden_before','hidden_after','supervision_target','teacher_action','teacher_available','label_mask','label_qualified','actually_applied_action','mixed_applied_action','policy_action','student_action','control_source','physical_step_ms','time')}
    boolean_keys={'teacher_available','label_mask','label_qualified'};integer_keys={'control_source','physical_step_ms'}
    shared=[qualification_path,COLLECTION/'SUMMARY.json',COLLECTION/'JOBS.json',COLLECTION/'INPUT_BINDINGS.json',COLLECTION/'collector.py',COLLECTION/'audit_nn.py',COLLECTION/'audit_teacher.py',COLLECTION/'common.py']
    for path in shared:data_hashes[str(path)]=sha(path)
    for case,valid_count,v1_prior in COLLECTION_CASES:
        folder=COLLECTION/'trials'/case;result_path=folder/'RESULT.json';result=read(result_path);result_sha=sha(result_path);row=qualification_rows[case];assisted=case!='old_walk_control'
        if result['kind']!='TEACHER_ASSISTED_COLLECTION_FIXED4_V1' or result['label']!='TEACHER_ASSISTED_DATA_NOT_NEURAL_ACCEPTANCE' or result['case']!=case or result['checkpoint_sha256']!=PARENT_SHA or result['neural_acceptance'] is not False or result['teacher_assisted'] is not assisted or not result['full_duration'] or result['actual_ms']!=15400 or result['neural_records']!=3080:
            raise ValueError('Actual complete collection kind/identity/scope')
        if row['result']!=str(result_path) or row['result_sha256']!=result_sha or row['teacher_assisted'] is not assisted:raise ValueError('Parentqualified row identity')
        if assisted:
            if row['qualified_teacher_labels']!=valid_count or result['qualified_teacher_labels']!=valid_count or row['supervision_kind']!='unmixed_full_teacher_action' or result['teaching_label_target']!='unmixed_full_teacher_action' or result['teacher_weight']!=.75 or result['student_weight']!=.25:
                raise ValueError('Unmixed teacher targets and exact masked-label counts required')
        elif row['supervision_kind']!='direct_frozen_oldmean_tanh' or result['teaching_label_target']!='direct_frozen_oldmean_tanh' or not qualification['old_control_original5_cycle_PASS'] or not result['candidate_metrics']['task_pass_candidate']:
            raise ValueError('Positivecontrol targets must be direct oldmean tanh, not shadownewNN')
        nnpath=folder/'NN_VERIFY.json';nn=_proof(nnpath,result_sha);proof_paths=[result_path,nnpath]
        if assisted:
            teacher_path=folder/'TEACHER_VERIFY.json';teacher=_proof(teacher_path,result_sha)
            if teacher['replayed_rows']!=1920 or teacher['physical_integration_steps']!=0 or teacher['NN_forward_calls']!=0:raise ValueError('Actual independent teacher-label replay proof')
            proof_paths.append(teacher_path)
        declared={p['path']:p['sha256'] for p in row['proofs']}
        for path in proof_paths[1:]:
            if declared.get(str(path))!=sha(path):raise ValueError('Parent proof SHA binding mismatch')
        artifact=result['neural_decisions'];path=folder/'neural_decisions.npz'
        if artifact['path']!=str(path) or artifact['rows']!=3080 or sha(path)!=artifact['sha256']:raise ValueError('Actual collection neural sample SHA')
        headers={key:npy_header(path,key) for key in collection_headers}
        for key,shape in collection_headers.items():
            dtype='|b1' if key in boolean_keys else '<i8' if key in integer_keys else '<f8'
            if headers[key]['shape']!=shape or headers[key]['descr']!=dtype:raise ValueError('Collection actual field/header schema: '+key)
        for p in proof_paths+[path]:data_hashes[str(p)]=sha(p)
        data_hashes.update(result['source_hashes']);data_hashes.update(nn['source_hashes'])
        entries.append(dict(id=case,folder=str(folder),path=str(path),source_kind='assisted_unmixed_teacher' if assisted else 'collection_old_walk_control',collection=True,teacher=assisted,turn=float(result['command'][1]),command=list(result['command']),v1_prior=v1_prior,expected_valid_count=valid_count,labels_available=True,training_input=True,warm_start=200,start=1160,warm_length=960,length=1920,headers=headers,result_sha256=result_sha,audit_sha256=sha(nnpath),neural_acceptance=False))
    if len(entries)!=12 or sum(bool(e['v1_prior']) for e in entries)!=6:raise ValueError('Exactly12 equal-mass sequences and6 v1prior sources')
    dev=[]
    for cohort,body in (('v1',OLD/'independent_motor_body_component'),('v2',PARENT.parent/'fastpath_motor_body_component')):
        for case in ('walk','stop','left_pivot','right_pivot'):
            folder=body/'trials'/case;result=read(folder/'RESULT.json');audit_path=folder/'VERIFY.json';_proof(audit_path,sha(folder/'RESULT.json'));path=folder/'neural_decisions.npz';n=result['neural_records']
            if n<1161 or result['state_total']!=352 or sha(path)!=result['neural_decisions']['sha256'] or npy_header(path,'X238')['shape']!=[n,238] or npy_header(path,'command')['shape']!=[n,2]:raise ValueError('Actual variable-length unlabeledDEV source')
            for p in (folder/'RESULT.json',audit_path,path):data_hashes[str(p)]=sha(p)
            dev.append(dict(id=cohort+'_'+case,cohort=cohort,case=case,path=str(path),warm_start=200,start=1160,warm_length=960,length=n-1160,labels_available=False,training_input=False))
    verify(data_hashes)
    paths=[ROOT/name for name in FILES]+[PARENT/'model.py',PARENT/'common.py',OLD/'independent_motor_calibration/sources.py',OLD/'independent_motor_calibration/preflight.py',OLD/'independent_motor_calibration/learning.py',OLD_MOTOR_DIR/'model.py',OLD_MOTOR_DIR/'model_io.py']
    return dict(kind='LOCAL_MEAN_CORRECTIVE_12_ACTUAL_MASKED_SOURCES_V1',entries=entries,dev_entries=dev,data_hashes=data_hashes,source_hashes={str(p):sha(p) for p in paths},arrays_loaded=False,training_sequences=12,v1_prior_sequences=6,collection_mask_counts={case:count for case,count,_ in COLLECTION_CASES},neural_acceptance=False,parent_dataset_qualification_sha256=sha(qualification_path),invalid_history_retained=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--prepare-source-metadata',action='store_true');a=p.parse_args()
    if not a.prepare_source_metadata:p.error('Explicitmetadata preparation flag required')
    save(ROOT/'INPUT_BINDINGS.json',bindings());print('LOCAL_MEAN_ACTUAL12_MASKED_SOURCES_BOUND_NO_ARRAY_LOAD',flush=True)
