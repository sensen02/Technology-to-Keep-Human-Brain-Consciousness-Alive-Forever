"""Stdlib metadata binding of completed resume300 order repair; no numerics.
The failed ancestor300 audit stays failure evidence. No new step0 is required.
"""
from pathlib import Path
import argparse
import json
import math
from common import ROOT,CAL,FORMAT,FILES,SOURCE_CANDIDATE,read,save,sha,verify
from adapter import OLD,MOTOR_FORMAT

SCOPE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z')
ANCESTOR=SCOPE/'target_feedback_calibration_env_recovery_v1'
RESUME_CHECKPOINT=ANCESTOR/'TRAIN/checkpoint_000300.pt'
RESUME_SHA='b562fcc3e39c2aa56381a87b17f1e80124f203fa1a73e3cdf8888778de119f14'
LINEAGE='RESUME300_ORDER_REPAIR_TO600'
PYTHON='/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python'
BODY_FORMAT='TARGET_FEEDBACK_ORDER_REPAIR_MOTOR_COMPONENT_BODY_V1'
TARGET_FORMAT='INDEPENDENT_MOTOR_TARGET_FEEDBACK_V1'


def passed(value):
    checks=value.get('checks')
    return value.get('integrity_all_pass') is True and isinstance(checks,dict) and bool(checks) and all(v is True for v in checks.values())


def merge(*maps):
    out={}
    for values in maps:
        if not isinstance(values,dict) or not values:raise ValueError('Nonempty immutable identity map required')
        for filename,digest in values.items():
            if not isinstance(filename,str) or not Path(filename).is_absolute() or not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Absolute path and lowercase SHA256 required')
            if filename in out and out[filename]!=digest:raise ValueError('Conflicting source/artifact identity: '+filename)
            out[filename]=digest
    return out


def artifact(item,expected):
    expected=Path(expected)
    if item['path']!=str(expected) or sha(expected)!=item['sha256']:raise ValueError('Actual saved artifact identity required: '+str(expected))
    return expected


def qualification(digest):
    if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Actual completed final600 lowercase SHA256 required')
    if CAL!=SCOPE/'target_feedback_order_repair_v2' or FORMAT!=BODY_FORMAT or MOTOR_FORMAT!=TARGET_FORMAT or SOURCE_CANDIDATE!=3:raise ValueError('Orderrepair body/calibration/source3 identity required')
    code=read(ROOT/'CODECHECK.json')
    if code['status']!='READYSTD_FROZEN_UNBOUND_FINAL600_BEHAVIOR_UNKNOWN' or not code.get('checks') or any(v is not True for v in code['checks'].values()):raise ValueError('Completed frozen READYSTD required')
    verify(code['source_hashes'])
    own={str(ROOT/name):sha(ROOT/name) for name in FILES}
    if any(code['source_hashes'].get(path)!=value for path,value in own.items()):raise ValueError('All body FILES must match frozen CODECHECK')
    b=read(CAL/'INPUT_BINDINGS.json');pre=read(CAL/'PREFLIGHT.json')
    verify(b['source_hashes']);verify(b['data_hashes'])
    lineage=b['resume_lineage']
    expected_lineage=dict(kind=LINEAGE,checkpoint=str(RESUME_CHECKPOINT),checkpoint_sha256=RESUME_SHA,start_update=300,total_final_update=600,new_updates=300,save_steps=[300,450,600],optimizer='RESTORE_ORIGINAL8ADAM_COUNTERS300_NO_FRESH_ADAM',initial_anchor='ORIGINAL_PARENT_PAYLOAD_CONFIG_INITIAL_ANCHOR_NO_REANCHOR',RNG='EXPLICIT_RESTORE_PAYLOAD_RNG_AFTER_ALL_SETUP_AND_OPTIMIZERLOAD_NO_RESEED',failed_old_VERIFY_preserved=True,old0qualification_passed=True,qualified_python=PYTHON)
    if any(lineage.get(key)!=value or (isinstance(value,bool) and lineage.get(key) is not value) for key,value in expected_lineage.items()):raise ValueError('Actual original Adam/RNG/anchor resume lineage required')
    if b['resume_checkpoint']!=str(RESUME_CHECKPOINT) or b['resume_checkpoint_sha256']!=RESUME_SHA or sha(RESUME_CHECKPOINT)!=RESUME_SHA or b['data_hashes'].get(str(RESUME_CHECKPOINT))!=RESUME_SHA:raise ValueError('Actual immutable b562 step300 resume checkpoint required')
    if pre['all_passed'] is not True or pre['timing_status']!='READY_RESUME300_TO600' or not pre.get('checks') or any(v is not True for v in pre['checks'].values()) or pre['source_hashes']!=b['source_hashes'] or pre['data_hashes']!=b['data_hashes'] or pre['physical_steps']!=0 or pre['training_updates']!=0 or pre['new_training_updates']!=0:raise ValueError('Actual same-source resume300 preflight required')
    if pre['checks'].get('resume_strict300') is not True or pre['checks'].get('300weights_Adam_RNG_restored') is not True or pre['checks'].get('oldweights_unchanged') is not True:raise ValueError('Actual resume optimizer/RNG/frozenold numerical preflight evidence required')
    ancestor_zero_path=ANCESTOR/'TRAIN/VERIFY_000000.json';ancestor_failed_path=ANCESTOR/'TRAIN/VERIFY_000300.json'
    zero=read(ancestor_zero_path);failed=read(ancestor_failed_path)
    if not passed(zero) or zero['step']!=0 or zero['checkpoint']!=str(ANCESTOR/'TRAIN/checkpoint_000000.pt') or sha(zero['checkpoint'])!=zero['checkpoint_sha256']:raise ValueError('Actual ancestor step0 PASS required')
    if failed['step']!=300 or failed['checkpoint']!=str(RESUME_CHECKPOINT) or failed['checkpoint_sha256']!=RESUME_SHA or failed['integrity_all_pass'] is not False or not any(value is False for value in failed['checks'].values()):raise ValueError('Original failed300 VERIFY must remain FAILURE evidence')
    for value in (zero,failed):verify(value['source_hashes']);verify(value['data_hashes'])
    proofs=[CAL/name for name in ('INPUT_BINDINGS.json','PREFLIGHT.json','TRAIN/IDENTITY.json','TRAIN/RESULT.json','JOBS.json','STAGE_VERIFICATION.json','EVALUATION.json','EVALUATION_VERIFY.json')]+[ROOT/'CODECHECK.json',RESUME_CHECKPOINT,Path(zero['checkpoint']),ancestor_zero_path,ancestor_failed_path]
    for value,step in ((zero,0),(failed,300)):
        proofs.append(artifact(value['conditional_arrays'],ANCESTOR/'TRAIN'/f'CONDITIONAL_{step:06d}.npz'))
    # Failure/proof hashes must already be inherited by actual resume metadata.
    for path in (ancestor_zero_path,ancestor_failed_path):
        if b['data_hashes'].get(str(path))!=sha(path):raise ValueError('Inherited ancestor audit hash required')
    jobs=read(CAL/'JOBS.json');tr=read(CAL/'TRAIN/RESULT.json');ident=read(CAL/'TRAIN/IDENTITY.json');stage=read(CAL/'STAGE_VERIFICATION.json');evaluation=read(CAL/'EVALUATION.json');ev=read(CAL/'EVALUATION_VERIFY.json')
    names=('RESUME301_TO600_AND_FRESH300_450_600','FINAL600_EVALUATION','INDEPENDENT_SAVED_METRICS')
    if len(jobs)!=3 or tuple(row['stage'] for row in jobs)!=names or any(row.get('status')!='COMPLETED' or row.get('exit_code')!=0 for row in jobs):raise ValueError('Actual exact3 resume stages exit0 required')
    target=CAL/'TRAIN/checkpoint_000600.pt'
    if tr['kind']!='RESUME300_ORDER_REPAIR_TO600_COMPLETED_NOT_BODY_VERIFIED' or tr['updates']!=600 or tr['total_updates']!=600 or tr['new_updates']!=300 or tr['already_completed_updates']!=300 or tr['checkpoint']!=str(target) or tr['sha256']!=digest or sha(target)!=digest or tr['physical_steps']!=0 or tr['new_neck_deployment'] is not False or tr['final']['step']!=600:raise ValueError('Actual total600 ancestor300 plus new300 result required')
    identity_expected=dict(lineage=LINEAGE,resume_step=300,total_updates=600,new_updates=300,parent_checkpoint_sha256=RESUME_SHA,original_anchor_preserved=True,originalAdamRNGrestored=True)
    if any(ident.get(key)!=value or (isinstance(value,bool) and ident.get(key) is not value) for key,value in identity_expected.items()):raise ValueError('TRAIN IDENTITY original Adam/RNG/anchor lineage required')
    if stage['kind']!=LINEAGE or stage['all_stages_exit0'] is not True or stage['train_updates']!=600 or stage['new_updates']!=300 or stage['checkpoint']!=str(target) or stage['checkpoint_sha256']!=digest or stage['evaluation_sha256']!=sha(CAL/'EVALUATION.json') or stage['physical_steps']!=0 or stage['body_behavior_pass'] is not None or stage['new_neck_deployed'] is not False:raise ValueError('Actual resume STAGE_VERIFICATION required')
    entries=b['entries'];dev_entries=b['dev_entries']
    if len(entries)!=12 or len(dev_entries)!=8 or len({row['id'] for row in entries})!=12 or len({row['id'] for row in dev_entries})!=8 or len(b['target_selector'])!=12 or sum(b['target_selector'])!=9:raise ValueError('Actual12 conditional/20actualX/8DEV manifest required')
    for step in (300,450,600):
        path=CAL/'TRAIN'/f'VERIFY_{step:06d}.json';v=read(path);checkpoint=CAL/'TRAIN'/f'checkpoint_{step:06d}.pt'
        if not passed(v) or v['step']!=step or v['checkpoint']!=str(checkpoint) or v['checkpoint_sha256']!=sha(checkpoint) or v['physical_steps']!=0 or v['conditional_notphysicalcounterfactual'] is not True or len(v['rows'])!=20:raise ValueError('Actual fresh300/450/600 combined production proof required')
        for key in ('strictload_RNG_unchanged','oldweights_frozen','canonical8Adamstates','all_model_weights_unchanged','no_physical_or_teacherbackend'):
            if v['checks'].get(key) is not True:raise ValueError('Saved strict loader/eight Adam/frozen RNG proof required: '+key)
        for index,(row,entry) in enumerate(zip(v['rows'],entries+dev_entries)):
            if row['id']!=entry['id'] or row['training'] is not (index<12) or row['actual_rows']!=entry['length'] or row['old256exact'] is not True or row['DEV_labels_available'] is not (None if index<12 else False):raise ValueError('Actual20 sequence row identities required')
            tag=entry['id']+str(index)
            if v['checks'].get(tag+'_actualX_production_cache') is not True:raise ValueError('ActualX production replay proof required')
            for key in ('actualX_mean_error','new96_error'):
                if not isinstance(row[key],(int,float)) or not math.isfinite(row[key]) or not 0.<=row[key]<=1e-12:raise ValueError('ActualX numerical error metadata outside declared tolerance')
            if index<12:
                errors=row['conditional_output_errors']
                if len(errors)!=6 or any(not isinstance(x,(int,float)) or not math.isfinite(x) or not 0.<=x<=1e-12 for x in errors) or v['checks'].get(tag+'_conditional_fullproduction') is not True or v['checks'].get(tag+'_conditional_finite') is not True:raise ValueError('Actual12 conditional six-output production proof required')
        proofs.extend((path,checkpoint,artifact(v['conditional_arrays'],CAL/'TRAIN'/f'CONDITIONAL_{step:06d}.npz')))
        verify(v['source_hashes']);verify(v['data_hashes'])
    if evaluation['kind']!='CONDITIONAL_TARGETFEEDBACK_FINAL600_EVALUATION_V1' or evaluation['step']!=600 or evaluation['checkpoint']!=str(target) or evaluation['checkpoint_sha256']!=digest or evaluation['behavior_pass'] is not None or evaluation['new_neck_deployed'] is not False or evaluation['physical_steps']!=0 or evaluation['conditional_notphysicalcounterfactual'] is not True or len(evaluation['conditional_rows'])!=12 or len(evaluation['actualX_rows'])!=12 or len(evaluation['DEV'])!=8:raise ValueError('Actual conditional/actualX/DEV evaluation schema required')
    for index,entry in enumerate(entries):
        conditional=evaluation['conditional_rows'][index];actual=evaluation['actualX_rows'][index]
        if conditional['id']!=entry['id'] or actual['id']!=entry['id'] or conditional['valid_rows']!=entry['expected_valid_count'] or actual['valid_rows']!=entry['expected_valid_count'] or conditional['total_history_rows']!=1920 or conditional['target_supervision_selected'] is not bool(b['target_selector'][index]):raise ValueError('Actual12 conditional and actualX identity/valid mask required')
        if not b['target_selector'][index] and conditional['target_normalized_MSE'] is not None:raise ValueError('Assisted mixed target must not gain realized-target supervision')
    for row,entry in zip(evaluation['DEV'],dev_entries):
        if row['id']!=entry['id'] or row['actual_rows']!=entry['length'] or row['labels_available'] is not False or row['teacher_MSE'] is not None:raise ValueError('Actual8 unlabeledDEV evaluation required')
    if not passed(ev) or ev['evaluation_sha256']!=sha(CAL/'EVALUATION.json') or ev['checkpoint_sha256']!=digest or ev['physical_steps']!=0 or ev['neural_forwards']!=0 or ev['conditional_notphysicalcounterfactual'] is not True:raise ValueError('Actual independent saved metric audit required')
    proofs.append(artifact(evaluation['arrays'],CAL/'EVALUATION_ARRAYS.npz'))
    for value in (tr,ident,evaluation):verify(value['source_hashes']);verify(value['data_hashes'])
    source=merge(b['source_hashes'],tr['source_hashes'],ident['source_hashes'],evaluation['source_hashes'],code['source_hashes'],own)
    data=merge(b['data_hashes'],tr['data_hashes'],ident['data_hashes'],evaluation['data_hashes'])
    proof={str(path):sha(path) for path in proofs}
    verify(source);verify(data);verify(proof)
    return dict(kind=FORMAT,checkpoint=str(target),checkpoint_sha256=digest,motor_format=MOTOR_FORMAT,final_step=600,state_dims=[128,64,64,96],control_source_candidate=SOURCE_CANDIDATE,source_hashes=source,data_hashes=data,proof_hashes=proof,resume_lineage=lineage,total_updates=600,new_updates=300,ancestor_updates=300,resume_checkpoint=str(RESUME_CHECKPOINT),resume_checkpoint_sha256=RESUME_SHA,original_failed300_VERIFY_preserved=True,proof_layout='NEW300_450_600_VERIFY_ACTUAL20_CONDITIONAL12_PLUS_ANCESTOR0PASS_300FAIL',eight_Adam600_metadata_proof=True,PT_payload_and_original_anchor_validation='DEFERRED_TO_STRICT_COMMON_LOAD_CHECKPOINT_FINAL_TRUE',unknownSHA_admitted=False,body_behavior_pass=None,new_neck_deployed=False,scope='BodyANNcomponent_NOTnewbrainneckdeployment')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bind-completed-final600',action='store_true');parser.add_argument('--sha256',required=True);args=parser.parse_args()
    if not args.bind_completed_final600:parser.error('Explicit completed resume final600 binding required')
    if (ROOT/'INPUT_BINDINGS.json').exists():raise FileExistsError('No overwrite or rebinding')
    binding=qualification(args.sha256)
    for key in ('source_hashes','data_hashes','proof_hashes'):verify(binding[key])
    save(ROOT/'INPUT_BINDINGS.json',binding)
    print(json.dumps(dict(binding=str(ROOT/'INPUT_BINDINGS.json'),checkpoint_sha256=args.sha256,total_updates=600,new_updates=300,body_behavior_pass=None)))


if __name__=='__main__':main()
