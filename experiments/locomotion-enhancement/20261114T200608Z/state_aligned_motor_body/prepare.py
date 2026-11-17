"""Stdlib-only final900 body binding; resource plan is read-only and NN-free.
resource_plan()->(planned_binding,planned_preflight,planned_receipt), even before
new cal metadata exists. qualification(digest) requires actual completed900 proof.
Neither helper calls any numerical loader or a calibration entry main.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import struct
import sys
import zipfile
from common import ROOT,CAL,FORMAT,FILES,SOURCE_CANDIDATE,read,save,sha,verify
from adapter import MOTOR_FORMAT,FINAL_STEP,OLD

EXPECTED_CAL=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T200608Z/state_aligned_action_calibration/wall_replan_9000')
RESUME_PARENT=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_order_repair_v4')
DATA_PARENT=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T230738Z/local_mean_corrective_calibration')
PARENT_SHA='41fa1e2232f5143fec0a3ebb5d506c819af0ba084b3a05041e829fd1477b906b'
BODY_KIND='STATE_ALIGNED_ACTION_MOTOR_COMPONENT_BODY_V1'
MODEL_FORMAT='INDEPENDENT_MOTOR_STATE_ALIGNED_ACTION_V1'
LINEAGE='STATE_ALIGNED_ACTION_RESUME600_TO900'
PYTHON='/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python'
NUMERICAL_MODULES=('torch','numpy','mujoco','physics','flygym','student_wave')


def _passed(value):
    checks=value.get('checks')
    return value.get('integrity_all_pass') is True and isinstance(checks,dict) and bool(checks) and all(v is True for v in checks.values())


def _merge(*maps):
    result={}
    for mapping in maps:
        if not isinstance(mapping,dict) or not mapping:raise ValueError('Nonempty absolute source identity map required')
        for filename,digest in mapping.items():
            if not isinstance(filename,str) or not Path(filename).is_absolute() or not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Absolute path and lowercase SHA256 required')
            if filename in result and result[filename]!=digest:raise ValueError('Conflicting immutable identity: '+filename)
            result[filename]=digest
    return result


def _subset(actual,planned,label):
    if not isinstance(actual,dict) or any(key not in actual or actual[key]!=value for key,value in planned.items()):raise ValueError('Actual metadata does not match read-only resource plan: '+label)


def _load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


def _cal_call(parent_proofs=False):
    """Ordinary modules; preserve public aliases and own temporary namespaces."""
    if CAL!=EXPECTED_CAL:raise ValueError('Exact wall_replan_9000 CAL required')
    if any(name in sys.modules for name in NUMERICAL_MODULES):raise ValueError('Fresh stdlib-only metadata process required')
    names=('sources','equivalence','common','model','adapter','prepare','_body900_plan_sources','_body900_plan_equivalence','_body900_plan_prepare','_body900_parent_io')
    sentinel=object();prior={name:sys.modules.get(name,sentinel) for name in names};paths=list(sys.path)
    try:
        sys.path.insert(0,str(CAL))
        sources=_load('_body900_plan_sources',CAL/'sources.py');sys.modules['sources']=sources
        if sources.ROOT!=CAL or sources.RESUME_PARENT!=RESUME_PARENT or sources.PARENT!=DATA_PARENT:raise ValueError('Data cache parent and strict resume parent must remain distinct')
        if parent_proofs:
            io=_load('_body900_parent_io',CAL/'common.py')
            if io.ROOT!=CAL or io.PARENT!=RESUME_PARENT or io.PARENT_SHA!=PARENT_SHA or io.FORMAT!=MODEL_FORMAT or tuple(io.SAVE_STEPS)!=(600,750,900):raise ValueError('Strict final900 storage/resume source identity')
            result=io.parent_qualification_hashes()
        else:
            equivalence=_load('_body900_plan_equivalence',CAL/'equivalence.py');sys.modules['equivalence']=equivalence
            planner=_load('_body900_plan_prepare',CAL/'prepare.py')
            if planner.ROOT!=CAL or planner.ORIGINAL!=CAL.parent:raise ValueError('Exact original132 resource-plan provenance required')
            result=planner.plan()
        if any(name in sys.modules for name in NUMERICAL_MODULES):raise ValueError('Resource metadata helper imported a numerical backend')
    finally:
        sys.path[:]=paths
        for name,value in prior.items():
            if value is sentinel:sys.modules.pop(name,None)
            else:sys.modules[name]=value
        if sys.path!=paths or any(sys.modules.get(name,sentinel) is not value for name,value in prior.items()):raise ValueError('Calibration metadata aliases/path restoration failed')
    return result


def resource_plan():
    """Call exact cal.prepare.plan only; never main/find_spec/prepare/save/NN.

    The producer itself verifies original132 all_passed=False solely because of
    timing, all96 correctness checks, exact science, old7200<estimate<=new9000,
    unchanged reference3072/training1800 and original before-training evidence.
    """
    planned=_cal_call()
    if not isinstance(planned,tuple) or len(planned)!=3:raise ValueError('Exact producer three-object resource plan required')
    return planned


def _artifact(item,expected):
    path=Path(expected)
    if item['path']!=str(path) or sha(path)!=item['sha256']:raise ValueError('Actual artifact path/SHA mismatch: '+str(path))
    return path


def _header(path,key,shape):
    """NPY header only; no array payload, pickle or numerical library."""
    with zipfile.ZipFile(path) as archive,archive.open(key+'.npy') as stream:
        if stream.read(6)!=b'\x93NUMPY':raise ValueError('NPY magic')
        version=tuple(stream.read(2))
        if version not in ((1,0),(2,0),(3,0)):raise ValueError('NPY version')
        width=2 if version==(1,0) else 4;length=struct.unpack('<H' if width==2 else '<I',stream.read(width))[0]
        if length>65536:raise ValueError('NPY header size cap')
        value=ast.literal_eval(stream.read(length).decode('utf8' if version==(3,0) else 'latin1'))
    if value['fortran_order'] or value['descr']!='<f8' or list(value['shape'])!=list(shape):raise ValueError('Saved float64 row-major array header mismatch: '+key)


def qualification(digest):
    if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Actual final900 lowercase SHA256 required; unknown refused')
    if CAL!=EXPECTED_CAL or FORMAT!=BODY_KIND or MOTOR_FORMAT!=MODEL_FORMAT or FINAL_STEP!=900 or SOURCE_CANDIDATE!=3:raise ValueError('Exact state-aligned final900 body identity required')
    code=read(ROOT/'CODECHECK.json')
    if code['status']!='READYSTD_FROZEN_UNBOUND_FINAL900_BEHAVIOR_UNKNOWN' or not code.get('checks') or not all(v is True for v in code['checks'].values()):raise ValueError('Frozen final900 READYSTD source qualification required')
    verify(code['source_hashes']);own={str(ROOT/name):sha(ROOT/name) for name in FILES}
    if any(code['source_hashes'].get(path)!=value for path,value in own.items()):raise ValueError('Every body FILES identity must match freeze')
    planned_binding,planned_pre,planned_receipt=resource_plan()
    b=read(CAL/'INPUT_BINDINGS.json');pre=read(CAL/'PREFLIGHT.json');receipt=read(CAL/'RESOURCE_REPLAN.json')
    _subset(b,planned_binding,'INPUT_BINDINGS');_subset(pre,planned_pre,'PREFLIGHT');_subset(receipt,planned_receipt,'RESOURCE_REPLAN')
    verify(b['source_hashes']);verify(b['data_hashes'])
    if b['resume_checkpoint']!=str(RESUME_PARENT/'TRAIN/checkpoint_000600.pt') or b['resume_checkpoint_sha256']!=PARENT_SHA or b['data_hashes'].get(b['resume_checkpoint'])!=PARENT_SHA or sha(b['resume_checkpoint'])!=PARENT_SHA:raise ValueError('Actual41fa parent600 resume required')
    if pre['all_passed'] is not True or pre['timing_status']!='READY_STATE_ALIGNED600_TO900' or pre['source_hashes']!=b['source_hashes'] or pre['data_hashes']!=b['data_hashes'] or pre['wall_limit_s']!=9000 or pre['preflight_RSS_limit_MiB']!=3072 or pre['training_RSS_limit_MiB']!=1800:raise ValueError('Actual amended readiness with unchanged phase RSS limits required')
    ref=pre['reused_actual_numeric_preflight'];old_path=CAL.parent/'PREFLIGHT.json';old=read(old_path)
    if ref['path']!=str(old_path) or ref['sha256']!=sha(old_path) or ref['original_all_passed'] is not False or ref['actual_correctness_checks']!=96 or old['all_passed'] is not False or old['timing_status']!='BLOCKED_BEFORE_TRAINING_TIMING' or len(old['checks'])!=96 or not all(v is True for v in old['checks'].values()):raise ValueError('Actual132 correctness PASS/timing-only FAIL must be preserved')
    if pre['new_numeric_preflight_runs']!=0 or pre['new_NN_forwards']!=0 or pre['new_optimizer_probe_updates']!=0 or pre['original_all_passed_false_preserved'] is not True or pre['resource_replan_pretraining'] is not True or pre['corrected_readiness_against9000'] is not True or pre['science_equivalence']['scientific_equivalence'] is not True:raise ValueError('No new numerical preflight or scientific changes admitted by resource replan')
    if receipt['kind']!='PRETRAINING_MEASURED_WALL_REPLAN_7200_TO9000' or receipt['new_wall_s']!=9000 or not 7200<receipt['old_estimate_s']<=9000 or any(receipt[key]!=0 for key in ('NN_forwards','model_loads','Core_constructions','physical_steps')):raise ValueError('Actual pretraining wall-only receipt required')
    jobs=read(CAL/'JOBS.json');tr=read(CAL/'TRAIN/RESULT.json');identity=read(CAL/'TRAIN/IDENTITY.json');stage=read(CAL/'STAGE_VERIFICATION.json');evaluation=read(CAL/'EVALUATION.json');audit=read(CAL/'EVALUATION_VERIFY.json')
    stage_specs=(('RESUME601_TO900_AND_FRESH600_750_900','train.py','--execute-parent-resume600-to900'),('FINAL900_STATE_ALIGNED_EVALUATION','evaluate.py','--execute-parent-evaluation'),('INDEPENDENT_SAVED_METRICS','audit_eval.py','--execute-parent-saved-audit'))
    if len(jobs)!=3:raise ValueError('Exactly three actual completed calibration stages required')
    elapsed=[]
    for job,(name,script,flag) in zip(jobs,stage_specs):
        value=job.get('pipeline_elapsed_s')
        if job.get('status')!='COMPLETED' or job.get('exit_code')!=0 or job['stage']!=name or job['command']!=[PYTHON,'-B',str(CAL/script),flag] or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=9000:raise ValueError('Actual stage exit0 and cumulative9000 pipeline deadline required')
        elapsed.append(value)
    if elapsed!=sorted(elapsed):raise ValueError('Pipeline cumulative clock must remain monotonic')
    target=CAL/'TRAIN/checkpoint_000900.pt'
    if tr['kind']!='STATE_ALIGNED_ACTION900_TRAINED_NOT_BODY_VERIFIED' or tr['updates']!=900 or tr['total_updates']!=900 or tr['new_updates']!=300 or tr['already_completed_updates']!=600 or tr['checkpoint']!=str(target) or tr['sha256']!=digest or sha(target)!=digest or tr['final']['step']!=900 or tr['physical_steps']!=0 or tr['new_neck_deployment'] is not False or not 0<=tr['wall_s']<=9000:raise ValueError('Actual total900/new300/ancestor600 TRAIN result and SHA required')
    expected=dict(lineage=LINEAGE,resume_step=600,total_updates=900,new_updates=300,parent_checkpoint_sha256=PARENT_SHA,original_anchor_preserved=True,originalAdamRNGrestored=True,action_supervision='actualXTFonly',conditionalactionlabels_supervised=False)
    if any(identity.get(key)!=value or (isinstance(value,bool) and identity.get(key) is not value) for key,value in expected.items()):raise ValueError('Actual original8Adam/RNG/anchor and TF-only supervision identity required')
    if stage['kind']!=LINEAGE or stage['all_stages_exit0'] is not True or stage['train_updates']!=900 or stage['new_updates']!=300 or stage['checkpoint']!=str(target) or stage['checkpoint_sha256']!=digest or stage['evaluation_sha256']!=sha(CAL/'EVALUATION.json') or stage['physical_steps']!=0 or stage['body_behavior_pass'] is not None or stage['new_neck_deployed'] is not False:raise ValueError('Actual final900 stage verification required')
    entries=b['entries'];dev_entries=b['dev_entries']
    if len(entries)!=12 or len(dev_entries)!=8 or len({x['id'] for x in entries})!=12 or len({x['id'] for x in dev_entries})!=8 or sum(b['target_selector'])!=9:raise ValueError('Actual12TRAIN/8DEV/9realizedtarget manifest required')
    proofs=[CAL/name for name in ('INPUT_BINDINGS.json','PREFLIGHT.json','RESOURCE_REPLAN.json','TRAIN/RESULT.json','TRAIN/IDENTITY.json','STAGE_VERIFICATION.json','JOBS.json','EVALUATION.json','EVALUATION_VERIFY.json')]+[target,ROOT/'CODECHECK.json',old_path]
    for step in (600,750,900):
        path=CAL/'TRAIN'/f'VERIFY_{step:06d}.json';v=read(path);checkpoint=CAL/'TRAIN'/f'checkpoint_{step:06d}.pt'
        if not _passed(v) or v['step']!=step or v['checkpoint']!=str(checkpoint) or v['checkpoint_sha256']!=sha(checkpoint) or len(v['rows'])!=20 or v['physical_steps']!=0 or v['training_RSS_limit_MiB']!=1800 or v['conditional_notphysicalcounterfactual'] is not True:raise ValueError('Actual fresh600/750/900 saved proof required')
        for key in ('strictloader_RNGunchanged','oldweights_frozen','canonical8Adamstates','all_model_weights_unchanged','no_Core_physics_teacherbackend'):
            if v['checks'].get(key) is not True:raise ValueError('Actual saved8Adam step-bound/RNG/frozenold proof missing: '+key)
        conditional=_artifact(v['conditional_arrays'],CAL/'TRAIN'/f'CONDITIONAL_{step:06d}.npz')
        for index,(row,entry) in enumerate(zip(v['rows'],entries+dev_entries)):
            if row['id']!=entry['id'] or row['training'] is not (index<12) or row['labels_available'] is not (index<12) or row['actual_rows']!=entry['length']:raise ValueError('Actual20 saved source row identities required')
            tag=entry['id']+str(index)
            if any(v['checks'].get(tag+suffix) is not True for suffix in ('_production_finite','_actualX_cachedproduction20')):raise ValueError('All20 actualX production/cache proofs required')
            if index<12:
                errors=row['conditional_output_errors']
                if len(errors)!=6 or any(not isinstance(x,(int,float)) or not math.isfinite(x) or not 0<=x<=1e-12 for x in errors) or any(v['checks'].get(tag+suffix) is not True for suffix in ('_TFproduction','_conditional_fullproduction','_conditional_finite')):raise ValueError('All12 TF and conditional six-output production proofs required')
                for field in ('mean','residual','new96','pretarget','posttarget','action','TFmean','TFaction'):_header(conditional,tag+'_'+field,(1920,96 if field=='new96' else 42))
        verify(v['source_hashes']);verify(v['data_hashes']);proofs.extend((path,checkpoint,conditional))
    if evaluation['kind']!='STATE_ALIGNED_ACTION_FINAL900_EVALUATION_V1' or evaluation['step']!=900 or evaluation['checkpoint']!=str(target) or evaluation['checkpoint_sha256']!=digest or evaluation['primary_supervision_metric']!='actualX_actionfit' or evaluation['conditional_action_label_distance']!='COUNTERFACTUAL_DISTANCE_NOT_SUPERVISION' or evaluation['behavior_pass'] is not None or evaluation['new_neck_deployed'] is not False or evaluation['physical_steps']!=0 or evaluation['conditional_notphysicalcounterfactual'] is not True or len(evaluation['actualX_rows'])!=12 or len(evaluation['conditional_rows'])!=12 or len(evaluation['DEV'])!=8:raise ValueError('Actual TF-primary final900 evaluation semantics required')
    for index,entry in enumerate(entries):
        tf=evaluation['actualX_rows'][index];conditional=evaluation['conditional_rows'][index]
        if tf['id']!=entry['id'] or conditional['id']!=entry['id'] or tf['valid_rows']!=entry['expected_valid_count'] or tf['qualified_rows']!=entry['expected_valid_count'] or tf['total_history_rows']!=1920 or conditional['valid_rows']!=entry['expected_valid_count'] or conditional['total_history_rows']!=1920 or conditional['target_supervision_selected'] is not bool(b['target_selector'][index]) or conditional['conditional_action_label_loss_used'] is not False or conditional['distance_semantics']!='COUNTERFACTUAL_DISTANCE_NOT_SUPERVISION':raise ValueError('TF-supervised12 and conditional-not-supervision identities required')
        if not b['target_selector'][index] and conditional['target_normalized_MSE'] is not None:raise ValueError('Mixed realized targets must remain unsupervised')
    for row,entry in zip(evaluation['DEV'],dev_entries):
        if row['id']!=entry['id'] or row['actual_rows']!=entry['length'] or row['labels_available'] is not False or row['teacher_MSE'] is not None:raise ValueError('Actual8 unlabeled DEV evaluation required')
    if not _passed(audit) or audit['evaluation_sha256']!=sha(CAL/'EVALUATION.json') or audit['checkpoint_sha256']!=digest or audit['physical_steps']!=0 or audit['neural_forwards']!=0 or audit['checks'].get('TFprimary_cond_not_supervision') is not True or audit['checks'].get('labels21773_selector9_prior6') is not True:raise ValueError('Actual independent TF/conditional/target metric audit required')
    arrays=_artifact(evaluation['arrays'],CAL/'EVALUATION_ARRAYS.npz')
    for field in ('actual_target','actual_post_targets','actualX_action','actualX_residual','conditional_mean','conditional_residual','conditional_action','conditional_post_targets'):_header(arrays,field,(1920,12,42))
    _header(arrays,'valid',(1920,12));_header(arrays,'v1',(12,));_header(arrays,'target_selector',(12,));_header(arrays,'halfspan',(42,));proofs.append(arrays)
    # Direct producer's STD parent proof preserves all four41fa body FAIL; it
    # explicitly does not require or infer parent behavior PASS.
    parent_proofs=_cal_call(parent_proofs=True)
    if any(b['data_hashes'].get(path)!=value for path,value in parent_proofs.items()):raise ValueError('Actual41fa numeric qualification/fourbodyFAIL proof must be bound in cal data')
    for value in (tr,identity,evaluation):verify(value['source_hashes']);verify(value['data_hashes'])
    source=_merge(b['source_hashes'],tr['source_hashes'],identity['source_hashes'],evaluation['source_hashes'],code['source_hashes'],own)
    data=_merge(b['data_hashes'],tr['data_hashes'],identity['data_hashes'],evaluation['data_hashes']);proof=_merge(parent_proofs,{str(path):sha(path) for path in proofs})
    verify(source);verify(data);verify(proof)
    return dict(kind=FORMAT,checkpoint=str(target),checkpoint_sha256=digest,motor_format=MOTOR_FORMAT,final_step=900,state_dims=[128,64,64,96],state_total=352,control_source_candidate=3,source_hashes=source,data_hashes=data,proof_hashes=proof,total_updates=900,new_updates=300,ancestor_updates=600,resume_checkpoint=b['resume_checkpoint'],resume_checkpoint_sha256=PARENT_SHA,resource_wall_replan=b['resource_wall_replan'],resource_plan_receipt_sha256=sha(CAL/'RESOURCE_REPLAN.json'),proof_layout='FRESH600_750_900_ACTUAL20_TF12_CONDITIONAL12',action_supervision='actualXTFonly',conditional_action_labels_supervised=False,parent_four_body_FAIL_preserved=True,eight_Adam900_metadata_step_bound_proof=True,PT_payload_original_anchor_RNG_validation='DEFERRED_TO_STRICT_COMMON_LOAD_CHECKPOINT_FINAL_TRUE_900',unknownSHA_admitted=False,body_behavior_pass=None,new_neck_deployed=False,scope='BodyANNcomponent_NOTnewbrainneckdeployment')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bind-completed-final900',action='store_true');parser.add_argument('--sha256',required=True);args=parser.parse_args()
    if not args.bind_completed_final900:parser.error('Explicit actual completed final900 binding required')
    if (ROOT/'INPUT_BINDINGS.json').exists():raise FileExistsError('No overwrite/rebinding')
    binding=qualification(args.sha256)
    for key in ('source_hashes','data_hashes','proof_hashes'):verify(binding[key])
    save(ROOT/'INPUT_BINDINGS.json',binding)
    print(json.dumps(dict(binding=str(ROOT/'INPUT_BINDINGS.json'),checkpoint_sha256=args.sha256,total_updates=900,new_updates=300,body_behavior_pass=None)))


if __name__=='__main__':main()
