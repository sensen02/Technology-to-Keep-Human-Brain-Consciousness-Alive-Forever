"""External metadata-only body binding; original CAL/science files never modified."""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import sys

ROOT=Path(__file__).resolve().parent
SCOPE=ROOT.parent
BODY=SCOPE/'state_aligned_motor_body'
CAL=SCOPE/'state_aligned_action_calibration/wall_replan_9000'
POSTRUN=SCOPE/'postrun_saved900_qualification_ONLYCODE'
BATCH=SCOPE/'postrun_final_state_aligned_batch_audit'
PYTHON='/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python'
FILES=('prepare.py','staticcheck.py','PREREG.json','README.zh-CN.md')
FORMAT='STATE_ALIGNED_ACTION_MOTOR_COMPONENT_BODY_V1'
MOTOR_FORMAT='INDEPENDENT_MOTOR_STATE_ALIGNED_ACTION_V1'
BODY_CODECHECK_SHA='3d0e7bcd31d01b0272d82222d6657d508dd49e6e184d2392a17c870502110386'
POSTRUN_CODECHECK_SHA='be1c712cc422cd8b764a3e7675a0be34162f909de904d93ae528eb1bef1ecb6a'
BATCH_FROZEN_SHA='d271afc9ff5f87bf40e2e55f6ce20c276eaec65fea007c33b3c2e962734f537b'


def read(path):return json.loads(Path(path).read_text())

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')

def verify(values):
    if not isinstance(values,dict) or not values:raise ValueError('Nonempty absolute SHA mapping required')
    for path,digest in values.items():
        if not Path(path).is_absolute() or not isinstance(digest,str) or len(digest)!=64 or sha(path)!=digest:raise ValueError('Source/proof drift: '+str(path))

def merge(*maps):
    out={}
    for mapping in maps:
        for path,digest in mapping.items():
            if path in out and out[path]!=digest:raise ValueError('Conflicting immutable identity: '+path)
            out[path]=digest
    return out


def check_body_unexecuted():
    if sha(BODY/'CODECHECK.json')!=BODY_CODECHECK_SHA:raise ValueError('Frozen original body CODECHECK changed')
    code=read(BODY/'CODECHECK.json')
    if len(code['checks'])!=32 or not all(v is True for v in code['checks'].values()) or len(code['source_hashes'])!=11 or code['actual_CAL']!=str(CAL):raise ValueError('Original32/11 frozen body and CAL identity required')
    verify(code['source_hashes'])
    if any((BODY/n).exists() for n in ('INPUT_BINDINGS.json','PREFLIGHT.json','JOBS.json','SUMMARY.json')) or any(BODY.glob('*.log')) or ((BODY/'trials').exists() and any((BODY/'trials').iterdir())):raise FileExistsError('Original body already bound or executed; no rebind/overwrite')
    return code


def check_checks(value,key='integrity_all_pass',count=None):
    if value.get(key) is not True or not value.get('checks') or not all(v is True for v in value['checks'].values()) or (count is not None and len(value['checks'])!=count):raise ValueError('Actual complete successful qualification proof required')


def original_deadline(digest):
    if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Explicit actual final900 SHA required; unknown refused')
    jobs=read(CAL/'JOBS.json')
    if not isinstance(jobs,list) or not jobs or any(row.get('status')=='RUNNING' for row in jobs):raise ValueError('Original pipeline still running or unavailable')
    if len(jobs)==3 and all(row.get('status')=='COMPLETED' and row.get('exit_code')==0 for row in jobs):raise ValueError('Postrun mode unnecessary for successful original pipeline')
    last=jobs[-1]
    if last.get('status')!='FAILED' or last.get('exit_code')!=124:raise ValueError('Original terminalFAILED124 must remain recorded')
    target=CAL/'TRAIN/checkpoint_000900.pt'
    if not target.is_file() or sha(target)!=digest:raise ValueError('Original actual900 PT SHA required; no750 admission')
    return dict(path=str(CAL/'JOBS.json'),sha256=sha(CAL/'JOBS.json'),status='FAILED',exit_code=124,last_stage=last['stage'])


def postrun_qualification(digest):
    failure=original_deadline(digest)
    if sha(POSTRUN/'CODECHECK.json')!=POSTRUN_CODECHECK_SHA:raise ValueError('Frozen postrun qualifier source identity changed')
    frozen=read(POSTRUN/'CODECHECK.json');verify(frozen['source_hashes'])
    if (POSTRUN/'FAILURE.json').exists():raise ValueError('Postrun failure bookkeeping never qualifies')
    jobs=read(POSTRUN/'JOBS.json');stage=read(POSTRUN/'STAGE_VERIFICATION.json');completion=read(POSTRUN/'RESOURCE_COMPLETION.json');tr=read(POSTRUN/'TRAIN/RESULT.json');fresh=read(POSTRUN/'TRAIN/VERIFY_000900.json');evaluation=read(POSTRUN/'EVALUATION.json');audit=read(POSTRUN/'EVALUATION_VERIFY.json');ledger=read(POSTRUN/'SOURCE_LEDGER.json')
    names=('FRESH900_SAVED_AUDIT','SAVED900_EVALUATION','INDEPENDENT_SAVED_METRICS')
    if len(jobs)!=3 or tuple(row['stage'] for row in jobs)!=names or any(row.get('status')!='COMPLETED' or row.get('exit_code')!=0 or not 0<=row['pipeline_elapsed_s']<=3600 for row in jobs):raise ValueError('Actual three postrun stagesexit0/shared3600 required')
    if stage['qualification_mode']!='POSTRUN_SAVED_MODEL_ONLY' or stage['all_postrun_stages_exit0'] is not True or stage['original_pipeline_all3exit0'] is not False or stage['checkpoint']!=str(CAL/'TRAIN/checkpoint_000900.pt') or stage['checkpoint_sha256']!=digest or stage['postrun_training_updates']!=0:raise ValueError('Postrun stage identity must not claim original success')
    if completion['pass'] is not True or completion['status']!='PASS' or completion['qualification_mode']!='POSTRUN_SAVED_MODEL_ONLY' or completion['wall_limit_s']!=3600 or completion['RSS_limit_MiB']!=1800 or not 0<=completion['wall_s']<=3600 or any(not 0<=rss<=1800 for rss in completion['stage_RSS_MiB'].values()) or completion['original_failure_preserved'] is not True or completion['no_false_original_pipeline_PASS'] is not True or completion['science_and_success_output_included'] is not True or completion['failure_bookkeeping_never_qualifies'] is not True or completion['requires_no_FAILURE_and_actual_run_exit0'] is not True:raise ValueError('Actual finalized postrun resource commit required')
    for key in ('postrun_training_updates','optimizer_operations','Core_constructions','physical_steps'):
        if completion[key]!=0:raise ValueError('Postrun qualification must not train or integrate')
    verify(stage['outputs']);verify(completion['artifacts'])
    if completion['artifacts'].get(str(POSTRUN/'STAGE_VERIFICATION.json'))!=sha(POSTRUN/'STAGE_VERIFICATION.json'):raise ValueError('Resource commit must include actual stage proof')
    if tr['kind']!='SAVED900_QUALIFIED_NO_NEW_UPDATES' or tr['qualification_mode']!='POSTRUN_SAVED_CHECKPOINT_ONLY' or tr['updates']!=900 or tr['total_updates']!=900 or tr['new_updates']!=300 or tr['already_completed_updates']!=600 or tr['qualification_new_updates']!=0 or tr['sha256']!=digest or tr['checkpoint']!=str(CAL/'TRAIN/checkpoint_000900.pt') or tr['original_pipeline_success_claim'] is not False:raise ValueError('Actual postrun saved900 TRAINRESULT required')
    for value in (tr,stage,ledger):
        record=value['original_deadline_failure']
        if record['path']!=failure['path'] or record['sha256']!=failure['sha256'] or record['status']!='FAILED' or record['exit_code']!=124:raise ValueError('Originalfailed124 proof differs')
    observer=tr['strict_payload_observation']
    if observer['step']!=900 or observer['optimizer_states']!=8 or observer['all_Adam_counters900'] is not True or observer['original_logged_update_rows']!=300 or observer['curve_logged_fields_exact'] is not True or observer['observer_training_updates']!=0 or observer['observer_optimizer_operations']!=0:raise ValueError('Actual same-load curve/Adam900 observation required')
    check_checks(fresh,count=81);check_checks(audit)
    if fresh['step']!=900 or fresh['checkpoint_sha256']!=digest or fresh['checkpoint']!=str(CAL/'TRAIN/checkpoint_000900.pt') or evaluation['checkpoint_sha256']!=digest or evaluation['step']!=900 or audit['checkpoint_sha256']!=digest or audit['evaluation_sha256']!=sha(POSTRUN/'EVALUATION.json'):raise ValueError('Postrun actual fresh/EVAL identity')
    if sha(fresh['conditional_arrays']['path'])!=fresh['conditional_arrays']['sha256'] or sha(evaluation['arrays']['path'])!=evaluation['arrays']['sha256']:raise ValueError('Postrun arrays SHA mismatch')
    # Strict model source is the actual saved payload config reported by fresh
    # verification, not the wider postrun TRAINRESULT qualification provenance.
    model_source=dict(fresh['source_hashes']);model_data=dict(fresh['data_hashes'])
    if evaluation['source_hashes']!=model_source or evaluation['data_hashes']!=model_data:raise ValueError('Actual payload model config differs between fresh and EVAL')
    verify(model_source);verify(model_data)
    qualification_source=merge(frozen['source_hashes'],tr['source_hashes'],stage['source_hashes'],ledger['source_hashes'])
    qualification_data=merge(tr['data_hashes'],stage['data_hashes'],ledger['data_hashes'])
    proofs=merge(tr['proof_hashes'],stage['proof_hashes'],ledger['proof_hashes'],completion['artifacts'])
    for name in names:
        done=read(POSTRUN/(name+'_COMPLETION.json'))
        if done['passed'] is not True or done['training_updates']!=0 or done['optimizer_operations']!=0 or done['physical_steps']!=0 or done['Core_constructions']!=0 or done['source_and_input_unchanged'] is not True:raise ValueError('Actual completed postrun source/zero-update stage required')
    for path in [POSTRUN/n for n in ('CODECHECK.json','JOBS.json','SOURCE_LEDGER.json','STAGE_VERIFICATION.json','RESOURCE_COMPLETION.json','INPUT_BINDINGS.json','TRAIN/RESULT.json','TRAIN/VERIFY_000900.json','EVALUATION.json','EVALUATION_VERIFY.json')]+[CAL/'JOBS.json']:
        proofs[str(path)]=sha(path)
    for values in (qualification_source,qualification_data,proofs):verify(values)
    return dict(failure=failure,model_source_hashes=model_source,model_data_hashes=model_data,qualification_source_hashes=qualification_source,qualification_data_hashes=qualification_data,proof_hashes=proofs,train_result=tr)


def batch_qualification(digest):
    # Ordinary import of the actual STD-only batch contract. No producer_context
    # or NN entry is called; final_qualification and accept_run are real exports.
    if sha(BATCH/'FROZEN.json')!=BATCH_FROZEN_SHA:raise ValueError('Actual FIRSTREADY batch frozen identity changed')
    frozen=read(BATCH/'FROZEN.json');verify(frozen['source_hashes']);verify(frozen['frozen_science_dependencies'])
    spec=importlib.util.spec_from_file_location('_external_body_postrun_batch_contracts',BATCH/'contracts.py');contract=importlib.util.module_from_spec(spec);sys.modules[spec.name]=contract;spec.loader.exec_module(contract)
    if contract.ROOT!=BATCH or contract.PRODUCER!=CAL or contract.POSTRUN!=POSTRUN or contract.MODE!='ORIGINAL_PIPELINE_FAILED124_SAVED900_POSTRUN_QUALIFIED' or contract.FORMAT!=MOTOR_FORMAT or contract.FINAL_STEP!=900 or contract.WALL_S!=1800 or contract.RSS_MIB!=1800:raise ValueError('Actual postrun batch source/format/resource identity required')
    proof,view=contract.final_qualification(CAL/'TRAIN/checkpoint_000900.pt',digest)
    folder=BATCH/('RUN_'+digest)
    if contract.accept_run(folder) is not True:raise ValueError('Actual finalized postrun batch must be accepted by its own contract')
    result=read(folder/'VERIFY.json');completion=read(folder/'RESOURCE_COMPLETION.json');rows=read(folder/'ROWS.json');ledger=read(folder/'SOURCE_LEDGER.json')
    if result['qualification_basis']!=contract.MODE or result['checkpoint_sha256']!=digest or result['status']!='PASS' or result['qualified_final900'] is not True or result['conditional_groups']!=36 or result['actualTF_groups']!=36 or result['checks'].get('all72_conditional_and_actual_comparisons_complete') is not True or result['weights_digest_before']!=result['weights_digest_after'] or result['physical_behavior_qualified'] is not False:raise ValueError('Actual postrun72group qualification identity required')
    if any(result[key]!=0 for key in ('training_updates','optimizer_operations','gradient_operations','Core_constructions','physical_steps')):raise ValueError('Batch qualification must not train or integrate')
    binding=read(CAL/'INPUT_BINDINGS.json')
    if len(rows)!=12 or [row['id'] for row in rows]!=[entry['id'] for entry in binding['entries']]:raise ValueError('All12actual batch source identities required')
    for row in rows:
        if row['actual_rows']!=1920 or row['old256_conditional_exact'] is not True or row['old256_actual_exact'] is not True:raise ValueError('Actual batch fullhistory/old256 proof required')
        for branch,names in (('conditional',contract.COND_NAMES),('actualX_TF',contract.TF_NAMES)):
            if set(row[branch])!={'B1','B12','micro4'}:raise ValueError('Allthree batch modes required')
            for errors in row[branch].values():
                if set(errors)!=set(names) or any(not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=contract.GATE for value in errors.values()):raise ValueError('Actual72 groups must all satisfy registered1e-12 gate')
    for before,after in (('source_before','source_after'),('final_qualification_before','final_qualification_after')):
        if ledger[before]!=ledger[after]:raise ValueError('Batch source/qualification hashes changed')
        verify(ledger[after])
    verify(completion['artifacts']);verify(proof)
    for value in result['artifacts'].values():
        if sha(value['path'])!=value['sha256']:raise ValueError('Batch success artifact SHA differs')
    sources=merge(frozen['source_hashes'],ledger['source_after'],view['source_hashes'],view['qualification_source_hashes'])
    data=merge(view['data_hashes'],view['qualification_data_hashes'])
    proofs=merge(proof,ledger['final_qualification_after'],completion['artifacts'])
    for path in (BATCH/'FROZEN.json',BATCH/'contracts.py',BATCH/'audit.py',folder/'VERIFY.json',folder/'RESOURCE_COMPLETION.json',folder/'ROWS.json',folder/'SOURCE_LEDGER.json',folder/'REPORT.zh-CN.md'):proofs[str(path)]=sha(path)
    if (folder/'FINALIZATION_FAILURE.json').exists():raise ValueError('Batch finalization failure invalidates binding')
    for values in (sources,data,proofs):verify(values)
    return dict(source_hashes=sources,data_hashes=data,proof_hashes=proofs,mode=contract.MODE,checkpoint_sha256=digest)


def qualification(digest):
    body=check_body_unexecuted();own=read(ROOT/'CODECHECK.json')
    if not own.get('checks') or not all(v is True for v in own['checks'].values()):raise ValueError('Own frozen metadata source check required')
    verify(own['source_hashes'])
    post=postrun_qualification(digest);batch=batch_qualification(digest)
    source=merge(body['source_hashes'],post['model_source_hashes'],post['qualification_source_hashes'],batch['source_hashes'],own['source_hashes'])
    data=merge(post['model_data_hashes'],post['qualification_data_hashes'],batch['data_hashes'])
    proof=merge(post['proof_hashes'],batch['proof_hashes'],{str(BODY/'CODECHECK.json'):sha(BODY/'CODECHECK.json'),str(ROOT/'CODECHECK.json'):sha(ROOT/'CODECHECK.json')})
    for values in (source,data,proof):verify(values)
    check_body_unexecuted();original_deadline(digest)
    return dict(kind=FORMAT,motor_format=MOTOR_FORMAT,checkpoint=str(CAL/'TRAIN/checkpoint_000900.pt'),checkpoint_sha256=digest,final_step=900,state_dims=[128,64,64,96],state_total=352,control_source_candidate=3,source_hashes=source,data_hashes=data,proof_hashes=proof,model_source_hashes=post['model_source_hashes'],model_data_hashes=post['model_data_hashes'],qualification_mode='POSTRUN_SAVED_CHECKPOINT_ONLY',original_pipeline_all3exit0=False,original_failed_124=post['failure'],postrun_all3stages_exit0=True,postrun_batch72_qualified=True,postrun_training_updates=0,original_completed_new_updates=300,total_updates=900,unknownSHA_admitted=False,body_behavior_pass=None,new_neck_deployed=False,external_binding_source=str(ROOT/'prepare.py'),CAL_unchanged=str(CAL),scope='BodyANNcomponent_NOTnewbrainneckdeployment')


def main():
    p=argparse.ArgumentParser();p.add_argument('--bind-qualified-postrun-final900',action='store_true');p.add_argument('--sha256',required=True);a=p.parse_args()
    if not a.bind_qualified_postrun_final900:p.error('Explicit external postrun binding required')
    if sys.executable!=PYTHON:raise ValueError('Qualified125712 Python required')
    binding=qualification(a.sha256)
    for key in ('source_hashes','data_hashes','proof_hashes'):verify(binding[key])
    check_body_unexecuted();save(BODY/'INPUT_BINDINGS.json',binding)
    print(json.dumps(dict(binding=str(BODY/'INPUT_BINDINGS.json'),checkpoint_sha256=a.sha256,qualification_mode=binding['qualification_mode'],original_failed_124_preserved=True,body_behavior_pass=None)),flush=True)


if __name__=='__main__':main()
