"""Final900 producer contracts and import context; standard library only."""
from pathlib import Path
from contextlib import contextmanager
import hashlib,importlib,importlib.util,json,math,os,sys,tempfile
ROOT=Path(__file__).resolve().parent
PRODUCER=ROOT.parent/'state_aligned_action_calibration/wall_replan_9000'
ORIGINAL=PRODUCER.parent
PYTHON=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python')
FORMAT='INDEPENDENT_MOTOR_STATE_ALIGNED_ACTION_V1'
FINAL_STEP=900
SAVE_STEPS=(600,750,900)
STAGES=('RESUME601_TO900_AND_FRESH600_750_900','FINAL900_STATE_ALIGNED_EVALUATION','INDEPENDENT_SAVED_METRICS')
ALIASES=('adapter','sources','common','model','learning','scan','data','utils')
MODULES=('sources','common','model','learning','scan','data','utils')
COND_NAMES=('mean','residual','h96','pretarget','posttarget','action')
TF_NAMES=('mean','residual','h96','action')
GATE=1e-12
WALL_S=600
RSS_MIB=1800

def require(condition,message):
    if not condition:raise ValueError(message)
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
def replace_failed_output(directory,name,value):
    """Only fail-closed correction inside this invocation's exclusively new RUN."""
    require(directory.parent==ROOT and directory.name.startswith('RUN_') and name in ('VERIFY.json','RESOURCE_COMPLETION.json'),'Failure correction must stay in owned new audit RUN')
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf8',dir=directory,prefix='.failure_commit_',suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name);json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,directory/name)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()
def ledger(paths):return {str(path):sha(path) for path in dict.fromkeys(paths)}
def merge(*maps):
    result={}
    for values in maps:
        for path,digest in values.items():
            require(path not in result or result[path]==digest,'Conflicting declared SHA: '+str(path));result[path]=digest
    return result
def verify(values):
    require(isinstance(values,dict) and bool(values),'Nonempty SHA ledger required')
    for path,digest in values.items():require(sha(path)==digest,'Declared source/evidence drift: '+str(path))
def sha_argument(digest):return isinstance(digest,str) and len(digest)==64 and all(char in '0123456789abcdef' for char in digest)
def all_true(value):return isinstance(value,dict) and bool(value) and all(item is True for item in value.values())
def completion_success(result,completion):
    return result.get('status')=='PASS' and result.get('qualified_final900') is True and completion.get('pass') is True and completion.get('wall_s',float('inf'))<=WALL_S and completion.get('RSS_MiB',float('inf'))<=RSS_MIB

def accept_run(directory):
    """Final stdlib readback gate: flags alone never authorize acceptance."""
    require(directory.parent==ROOT and directory.name.startswith('RUN_'),'Audit output scope differs')
    require(not (directory/'FINALIZATION_FAILURE.json').exists(),'Finalization failure marker invalidates this run')
    result=read(directory/'VERIFY.json');completion=read(directory/'RESOURCE_COMPLETION.json')
    require(completion_success(result,completion) and all_true(result['checks']),'Final900 result/resource status must both pass')
    require(result['format_version']==FORMAT and result['final_step']==FINAL_STEP and sha_argument(result['checkpoint_sha256']) and directory.name=='RUN_'+result['checkpoint_sha256'],'Final900 result/run identity differs')
    require(completion['wall_limit_s']==WALL_S and completion['RSS_limit_MiB']==RSS_MIB and completion['science_and_success_outputs_included'] is True,'Resource receipt must use the predeclared600s/1800MiB budget')
    expected={str(directory/name) for name in ('VERIFY.json','ROWS.json','SOURCE_LEDGER.json','REPORT.zh-CN.md')}
    require(set(completion['artifacts'])==expected,'Completion must bind every exact owned final artifact')
    verify(completion['artifacts'])
    return True

def equivalence_now():
    path=PRODUCER/'equivalence.py';spec=importlib.util.spec_from_file_location('_final900_current_resource_science_equivalence',path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    require(module.ROOT==PRODUCER and module.ORIGINAL==ORIGINAL,'Wrong resource equivalence scope')
    return module.verify_equivalence()

def reused_preflight_qualification(binding,preflight):
    require(preflight['all_passed'] is True and preflight['timing_status']=='READY_STATE_ALIGNED600_TO900','New900 resource-qualified readiness required')
    require(preflight['source_hashes']==binding['source_hashes'] and preflight['data_hashes']==binding['data_hashes'],'New preflight/binding provenance differs')
    require(len(preflight['checks'])==96 and all_true(preflight['checks']),'All96 actual correctness checks required')
    require(preflight['resource_replan_pretraining'] is True and preflight['corrected_readiness_against9000'] is True and preflight['original_all_passed_false_preserved'] is True,'Preserve original timing-only failure')
    require(preflight['new_numeric_preflight_runs']==0 and preflight['new_NN_forwards']==0 and preflight['new_optimizer_probe_updates']==0,'This producer reuses actual96; a new numeric preflight is not required')
    require(preflight['wall_limit_s']==9000 and preflight['training_RSS_limit_MiB']==1800 and preflight['preflight_RSS_limit_MiB']==3072,'Exact producer resource amendment required')
    reference=preflight['reused_actual_numeric_preflight'];old_path=ORIGINAL/'PREFLIGHT.json'
    require(reference['path']==str(old_path) and reference['sha256']==sha(old_path) and reference['original_all_passed'] is False and reference['original_timing_status']=='BLOCKED_BEFORE_TRAINING_TIMING' and reference['actual_correctness_checks']==96,'Original132 actual correctness provenance required')
    require(preflight['numeric_metrics_origin']==reference,'Reused numeric metric origin differs')
    old=read(old_path);old_binding=read(ORIGINAL/'INPUT_BINDINGS.json');receipt=read(PRODUCER/'RESOURCE_REPLAN.json')
    require(old['all_passed'] is False and old['timing_status']=='BLOCKED_BEFORE_TRAINING_TIMING' and old['wall_limit_s']==7200,'Original7200 timing failure must remain false')
    require(old['checks']==preflight['checks'] and old['source_hashes']==old_binding['source_hashes'] and old['data_hashes']==old_binding['data_hashes'],'Actual96 correctness/source identity changed')
    require(preflight['qualification_basis']=='REUSE_ACTUAL96_CORRECTNESS_AND_EXACT_SCIENCE_SOURCE_EQUIVALENCE_AND_PRETRAINING9000_WALL','Qualification basis must declare correctness reuse')
    estimate=old['predicted300total_withverification_s'];recomputed=300*old['actual_two_branch_micro4x3_update_s']*1.3+old['three_fresh_save_allowance_s']+old['evaluation_allowance_s']
    require(math.isfinite(estimate) and 7200<estimate<=9000 and abs(estimate-recomputed)<=1e-8,'Measured original complete estimate must fit predeclared9000 only')
    require(receipt['kind']=='PRETRAINING_MEASURED_WALL_REPLAN_7200_TO9000' and receipt['old_numeric_preflight']==reference and receipt['old_estimate_s']==estimate and receipt['new_wall_s']==9000 and receipt['NN_forwards']==receipt['model_loads']==receipt['Core_constructions']==receipt['physical_steps']==0,'Resource-only receipt must not claim a numeric rerun')
    actual_equivalence=equivalence_now()
    require(actual_equivalence['scientific_equivalence'] is True and actual_equivalence==preflight['science_equivalence']==receipt['science_equivalence']==binding['resource_wall_replan']['science_equivalence'],'Current science must equal the qualified original apart from registered budget metadata')
    require(binding['resource_wall_replan']['old_wall_s']==7200 and binding['resource_wall_replan']['new_wall_s']==9000 and binding['resource_wall_replan']['no_numeric_rerun'] is True,'Declared pretraining resource amendment differs')
    for path in (old_path,ORIGINAL/'INPUT_BINDINGS.json',ORIGINAL/'PREREG.json'):
        require(binding['source_hashes'].get(str(path))==sha(path) and binding['data_hashes'].get(str(path))==sha(path),'Original false proof must be pinned in new source/data ledgers')
    return ledger((PRODUCER/'PREFLIGHT.json',PRODUCER/'INPUT_BINDINGS.json',PRODUCER/'RESOURCE_REPLAN.json',PRODUCER/'equivalence.py',old_path,ORIGINAL/'INPUT_BINDINGS.json',ORIGINAL/'PREREG.json'))

def final_qualification(checkpoint,digest):
    require(sha_argument(digest),'Actual lowercase64 final SHA argument required; no inferred SHA')
    require(checkpoint==PRODUCER/'TRAIN/checkpoint_000900.pt','Only exact new producer final900 admitted')
    require(sha(checkpoint)==digest,'Actual final900 checkpoint SHA differs')
    binding=read(PRODUCER/'INPUT_BINDINGS.json');preflight=read(PRODUCER/'PREFLIGHT.json');proof=reused_preflight_qualification(binding,preflight)
    jobs=read(PRODUCER/'JOBS.json');stage=read(PRODUCER/'STAGE_VERIFICATION.json');tr=read(PRODUCER/'TRAIN/RESULT.json')
    require(len(jobs)==3 and tuple(row['stage'] for row in jobs)==STAGES,'Exactly three real final900 main stages required')
    for row,script in zip(jobs,('train.py','evaluate.py','audit_eval.py')):
        require(row['status']=='COMPLETED' and row['exit_code']==0 and str(PRODUCER/script) in row['command'],'Wait for all three producer main stages exit0')
    require(stage['kind']=='STATE_ALIGNED_ACTION_RESUME600_TO900' and stage['all_stages_exit0'] is True and stage['train_updates']==900 and stage['new_updates']==300,'Final900 stage identity required')
    require(tr['updates']==tr['total_updates']==900 and tr['new_updates']==300 and tr['already_completed_updates']==600,'Exactly original600 plus new300 updates required')
    for record,key in ((stage,'checkpoint_sha256'),(tr,'sha256')):
        require(record['checkpoint']==str(checkpoint) and record[key]==digest,'Final stage/TRAIN SHA or path mismatch')
    paths=[PRODUCER/'JOBS.json',PRODUCER/'STAGE_VERIFICATION.json',PRODUCER/'TRAIN/RESULT.json'];saved_rows=[]
    for step in SAVE_STEPS:
        path=PRODUCER/'TRAIN'/f'VERIFY_{step:06d}.json';v=read(path);saved=PRODUCER/'TRAIN'/f'checkpoint_{step:06d}.pt'
        require(v['step']==step and v['integrity_all_pass'] is True and all_true(v['checks']),'All new600/750/900 fresh saved proofs must PASS')
        require(v['checkpoint']==str(saved) and v['checkpoint_sha256']==sha(saved),'Saved checkpoint path/SHA mismatch')
        if step==FINAL_STEP:require(v['checkpoint_sha256']==digest,'Saved final900 differs from supplied actual SHA')
        require(v['source_hashes']==tr['source_hashes'] and v['data_hashes']==tr['data_hashes'],'Saved proof/TRAIN provenance differs')
        require(len(v['rows'])==20 and sum(row['training'] is True for row in v['rows'])==12,'Fresh12TRAIN8DEV proof required')
        for suffix,count in (('_actualX_cachedproduction20',20),('_TFproduction',12),('_conditional_fullproduction',12)):
            require(sum(key.endswith(suffix) for key in v['checks'])==count,'Actual/TF/conditional storage proof category missing: '+suffix)
        require(v['checks'].get('canonical8Adamstates') is True,'Strict saved eight Adam states required without running an optimizer')
        for row in v['rows'][:12]:
            require(row['actual_rows']==1920 and len(row['conditional_output_errors'])==6 and all(math.isfinite(value) and 0.<=value<=GATE for value in row['conditional_output_errors']),'All six stored conditional errors must retain strict precision')
        item=v['conditional_arrays'];require(item['path']==str(PRODUCER/'TRAIN'/f'CONDITIONAL_{step:06d}.npz') and sha(item['path'])==item['sha256'],'Saved actual/conditional arrays SHA mismatch')
        paths.extend((path,saved,Path(item['path'])));saved_rows.append(v['rows'])
    e=read(PRODUCER/'EVALUATION.json');ev=read(PRODUCER/'EVALUATION_VERIFY.json')
    require(e['kind']=='STATE_ALIGNED_ACTION_FINAL900_EVALUATION_V1' and e['step']==900 and e['checkpoint']==str(checkpoint) and e['checkpoint_sha256']==digest,'Final state-aligned evaluation identity required')
    require(ev['integrity_all_pass'] is True and all_true(ev['checks']) and ev['checkpoint_sha256']==digest,'Independent final900 evaluation audit must PASS')
    require(ev['evaluation_sha256']==stage['evaluation_sha256']==sha(PRODUCER/'EVALUATION.json'),'Final evaluation proof SHA mismatch')
    require(e['arrays']['path']==str(PRODUCER/'EVALUATION_ARRAYS.npz') and sha(e['arrays']['path'])==e['arrays']['sha256'],'Evaluation array SHA differs')
    require(e['source_hashes']==tr['source_hashes'] and e['data_hashes']==tr['data_hashes'],'Evaluation/TRAIN provenance differs')
    require(e['primary_supervision_metric']=='actualX_actionfit' and e['conditional_action_label_distance']=='COUNTERFACTUAL_DISTANCE_NOT_SUPERVISION','Actual TF primary/conditional non-supervision semantics must remain declared')
    require(len(e['actualX_rows'])==len(e['conditional_rows'])==12 and [row['id'] for row in e['actualX_rows']]==[row['id'] for row in e['conditional_rows']],'All12 actual/conditional evaluation identities required')
    require(all(row['conditional_action_label_loss_used'] is False and row['distance_semantics']=='COUNTERFACTUAL_DISTANCE_NOT_SUPERVISION' for row in e['conditional_rows']),'Conditional distance must not become action-label supervision')
    require(all([row['id'] for row in rows[:12]]==[row['id'] for row in e['actualX_rows']] for rows in saved_rows),'All saved checkpoints must retain the same12 cases')
    require(len(e['DEV'])==8 and all(row['labels_available'] is False and row['teacher_MSE'] is None for row in e['DEV']) and e['behavior_pass'] is None and e['new_neck_deployed'] is False,'No inferred DEV labels/body/neck acceptance')
    paths.extend((PRODUCER/'EVALUATION.json',PRODUCER/'EVALUATION_VERIFY.json',Path(e['arrays']['path'])))
    return merge(proof,ledger(paths)),tr

@contextmanager
def producer_context():
    """Producer has no adapter.py; save/remove it for safe ancestor adapter contexts."""
    sentinel=object();old={name:sys.modules.get(name,sentinel) for name in ALIASES};oldpath=list(sys.path)
    try:
        for name in ALIASES:sys.modules.pop(name,None)
        sys.path.insert(0,str(PRODUCER));modules={}
        for name in MODULES:
            value=importlib.import_module(name);require(Path(value.__file__).resolve()==PRODUCER/(name+'.py'),'Wrong final900 public module: '+name);modules[name]=value
        yield modules
    finally:
        sys.path[:]=oldpath
        for name,value in old.items():
            if value is sentinel:sys.modules.pop(name,None)
            else:sys.modules[name]=value
        require(sys.path==oldpath and all(sys.modules.get(name,sentinel) is value for name,value in old.items()),'Public producer/ancestor aliases or path restoration failed')
