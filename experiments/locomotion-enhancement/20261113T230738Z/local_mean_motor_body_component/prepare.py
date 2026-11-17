"""Stdlib actual localmean final600 qualification; unknown SHA never bound."""
from pathlib import Path
import argparse
import json
from common import ROOT,CAL,FORMAT,FILES,SOURCE_CANDIDATE,read,save,sha,verify
from adapter import MOTOR_FORMAT


def passed(value):
    return value.get('integrity_all_pass') is True and bool(value.get('checks')) and all(v is True for v in value['checks'].values())


def merge(*maps):
    result={}
    for values in maps:
        if not isinstance(values,dict) or not values:raise ValueError('Nonempty source identity map required')
        for path,digest in values.items():
            if not Path(path).is_absolute() or len(digest)!=64:raise ValueError('Absolute SHA identity required')
            if path in result and result[path]!=digest:raise ValueError('Conflicting immutable identity: '+path)
            result[path]=digest
    return result


def qualification(digest):
    if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Actual lowercase64hex finalSHA required')
    code=read(ROOT/'CODECHECK.json');verify(code['source_hashes'])
    if code['status']!='READYSTD_FROZEN_UNBOUND_FINAL600_BEHAVIOR_UNKNOWN' or not all(code['checks'].values()):raise ValueError('Frozen READYSTD required')
    old=read(ROOT/'OLD_SOURCE_SNAPSHOT.json');verify(old['source_hashes'])
    cal_binding=read(CAL/'INPUT_BINDINGS.json');verify(cal_binding['source_hashes']);verify(cal_binding['data_hashes'])
    trained=read(CAL/'TRAIN/RESULT.json');identity=read(CAL/'TRAIN/IDENTITY.json');stage=read(CAL/'STAGE_VERIFICATION.json');evaluation=read(CAL/'EVALUATION.json');audit=read(CAL/'EVALUATION_VERIFY.json');jobs=read(CAL/'JOBS.json');pre=read(CAL/'PREFLIGHT.json')
    target=CAL/'TRAIN/checkpoint_000600.pt'
    if trained['updates']!=600 or trained['checkpoint']!=str(target) or trained['sha256']!=digest or sha(target)!=digest:raise ValueError('Actual TRAIN600/PT SHA required')
    expected=('TRAIN_FIXED600_3FRESH_SAVEDVERIFY','FINAL600_MASKED12_FIT_UNLABELED8DEV','INDEPENDENT_MASKED_METRIC_AUDIT')
    if len(jobs)!=3 or tuple(j['stage'] for j in jobs)!=expected or any(j.get('exit_code')!=0 or j.get('status')!='COMPLETED' for j in jobs):raise ValueError('Actual exact3mainstagesexit0 required')
    if not stage['all_stages_exit0'] or stage['train_updates']!=600 or stage['checkpoint']!=str(target) or stage['checkpoint_sha256']!=digest or stage['evaluation_sha256']!=sha(CAL/'EVALUATION.json') or stage['independent_audit_sha256']!=sha(CAL/'EVALUATION_VERIFY.json') or stage['new_neck_deployed'] is not False or stage['body_behavior_pass'] is not None:raise ValueError('Completed final600 qualification only')
    if not passed(audit) or audit['evaluation_sha256']!=sha(CAL/'EVALUATION.json') or audit['checkpoint_sha256']!=digest or evaluation['checkpoint_sha256']!=digest or evaluation['step']!=600 or evaluation['behavior_pass'] is not None or evaluation['new_neck_deployed'] is not False:raise ValueError('Actual independent final evaluation proof required')
    if not pre['all_passed'] or pre['timing_status']!='READY_FIXED600_3600S' or pre['source_hashes']!=cal_binding['source_hashes'] or pre['data_hashes']!=cal_binding['data_hashes']:raise ValueError('Actual same-source/data calibration preflight required')
    architecture=identity['architecture']
    if identity['steps']!=600 or architecture['format_version']!=MOTOR_FORMAT or architecture['state_dims']!=[128,64,64,96] or architecture['feature_dim']!=412 or architecture['encoder']!=[412,96] or architecture['head']!=[192,42] or architecture['new_trainable_tensor_count']!=8 or architecture['old_checkpoint_sha256']!='789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd':raise ValueError('Actual localmean412/352/old789f architecture identity')
    entries=cal_binding['entries'];dev_entries=cal_binding['dev_entries']
    if len(entries)!=12 or len(dev_entries)!=8:raise ValueError('Actual12TRAIN+8unlabeledDEV manifest required')
    proofs=[CAL/n for n in ('INPUT_BINDINGS.json','PREFLIGHT.json','TRAIN/RESULT.json','TRAIN/IDENTITY.json','STAGE_VERIFICATION.json','EVALUATION.json','EVALUATION_VERIFY.json','JOBS.json')]+[target,ROOT/'CODECHECK.json']
    # ACTUAL producer layout: each VERIFY contains BOTH12training and8DEV rows;
    # no separate DEV_VERIFY or DEV_EVALUATION files are produced by this cal.
    for step in (0,300,600):
        path=CAL/'TRAIN'/f'VERIFY_{step:06d}.json';v=read(path);checkpoint=CAL/'TRAIN'/f'checkpoint_{step:06d}.pt';actual=sha(checkpoint)
        if not passed(v) or v['step']!=step or v['checkpoint']!=str(checkpoint) or v['checkpoint_sha256']!=actual or v['physical_steps']!=0 or v['DEV_teacher_labels_available'] is not False or len(v['rows'])!=20:raise ValueError('Actual fresh combined12TRAIN8DEV0/300/600 proof required')
        for index,(row,entry) in enumerate(zip(v['rows'],entries+dev_entries)):
            if row['id']!=entry['id'] or row['training'] is not (index<12) or row['labels_available'] is not (index<12) or row['actual_rows']!=entry['length'] or row['old256exact'] is not True:raise ValueError('Fresh20sequence DEV/training identities mismatch')
            tag=entry['id']+str(index)
            if v['checks'].get(tag+'_production_cache') is not True or v['checks'].get(tag+'_412directoldmeanfeature') is not True:raise ValueError('Actual productioncache/directoldmean412 persequence proof')
        verify(v['source_hashes']);verify(v['data_hashes']);proofs.extend((path,checkpoint))
    if len(evaluation['DEV'])!=8 or len(evaluation['rows'])!=12:raise ValueError('Actual final12maskedfit plus8DEV required')
    for row,entry in zip(evaluation['DEV'],dev_entries):
        if row['id']!=entry['id'] or row['actual_suffix_rows']!=entry['length'] or row['labels_available'] is not False or row['training_input'] is not False or row['teacher_MSE'] is not None:raise ValueError('Actual unlabeled DEV embedded evaluation required')
    for row,entry in zip(evaluation['rows'],entries):
        if row['id']!=entry['id'] or row['valid_rows']!=entry['expected_valid_count']:raise ValueError('Actual masked12evaluation identity')
    arrays=evaluation['arrays']
    if sha(arrays['path'])!=arrays['sha256']:raise ValueError('Actual evaluation arrays artifact drift')
    proofs.append(Path(arrays['path']))
    for value in (trained,identity,evaluation):verify(value['source_hashes']);verify(value['data_hashes'])
    source=merge(old['source_hashes'],cal_binding['source_hashes'],trained['source_hashes'],code['source_hashes'])
    for name in FILES:
        path=str(ROOT/name)
        if source.get(path)!=sha(path):raise ValueError('Actual body freeze source mismatch')
    data=merge(cal_binding['data_hashes'],trained['data_hashes']);before={str(path):sha(path) for path in proofs}
    verify(source);verify(data);verify(before)
    return dict(kind=FORMAT,checkpoint=str(target),checkpoint_sha256=digest,motor_format=MOTOR_FORMAT,final_step=600,state_dims=[128,64,64,96],feature_dim=412,control_source_candidate=SOURCE_CANDIDATE,source_hashes=source,data_hashes=data,proof_hashes=before,unknownSHA_admitted=False,body_behavior_pass=None,new_neck_deployed=False,proof_layout='combined_VERIFY12TRAIN8DEV_and_EVALUATION_DEV',scope='BodyANNcomponent_NOTnewbrainneckdeployment')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bind-completed-final600',action='store_true');parser.add_argument('--sha256',required=True);args=parser.parse_args()
    if not args.bind_completed_final600:parser.error('Completed actual final600 qualification required')
    if (ROOT/'INPUT_BINDINGS.json').exists():raise FileExistsError('No overwrite/rebinding')
    binding=qualification(args.sha256)
    for key in ('source_hashes','data_hashes','proof_hashes'):verify(binding[key])
    save(ROOT/'INPUT_BINDINGS.json',binding)
    print(json.dumps(dict(binding=str(ROOT/'INPUT_BINDINGS.json'),checkpoint_sha256=args.sha256,body_behavior_pass=None)))


if __name__=='__main__':main()
