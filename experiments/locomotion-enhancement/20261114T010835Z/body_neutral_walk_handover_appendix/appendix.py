"""Single neutralwalk saved-data appendix. Stdlib metadata; NumPy only in main."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import resource
import signal
import sys
import time

ROOT=Path(__file__).resolve().parent
SCOPE=ROOT.parent
RECOVERY=SCOPE/'body_training_handover_parity_recovery_v1'
ORIGINAL=SCOPE/'body_training_handover_parity'
FILES=('PREREG.json','appendix.py','static_check.py','README.zh-CN.md')
MODEL_SHA='41fa1e2232f5143fec0a3ebb5d506c819af0ba084b3a05041e829fd1477b906b'
FORBIDDEN=('torch','mujoco','physics','flygym','student_wave','static_teacher')


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def read(path):return json.loads(Path(path).read_text())

def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')


def load_frozen_helpers():
    """Explicit immutable helper provenance; sentinel restoration of metadata alias."""
    path=RECOVERY/'metadata.py'
    if sha(path)!='cdc6d0c537912fcd0d0c14c5737f1420222b9e177fed6b41aa5a21b2a405c0f1':raise ValueError('Frozen recovery metadata drift')
    spec=importlib.util.spec_from_file_location('_neutral_actual_recovery_metadata',path);metadata=importlib.util.module_from_spec(spec);sys.modules[spec.name]=metadata;spec.loader.exec_module(metadata)
    source=ORIGINAL/'analyze.py'
    if sha(source)!='b8b3cac78bf43465fb260c93fdca6059e9ab1474d9d72d79dfdb8e2cebb0c1d8':raise ValueError('Frozen generic case_report source drift')
    sentinel=object();prior=sys.modules.get('metadata',sentinel)
    try:
        sys.modules['metadata']=metadata
        spec=importlib.util.spec_from_file_location('_neutral_original_case_report_exact',source);analysis=importlib.util.module_from_spec(spec);sys.modules[spec.name]=analysis;spec.loader.exec_module(analysis)
    finally:
        if prior is sentinel:sys.modules.pop('metadata',None)
        else:sys.modules['metadata']=prior
    if sys.modules.get('metadata',sentinel) is not prior:raise ValueError('Metadata alias restoration failed')
    # Original case_report closes over original ordinary-module globals; no
    # function cloning, AST editing, contextmanager rebinding or main call.
    return metadata,analysis


def qualify():
    """Actual STD admission; no NumPy or arrays, including the index8 contract."""
    code=read(ROOT/'CODECHECK.json')
    if not code.get('checks') or not all(v is True for v in code['checks'].values()):raise ValueError('Own STD source audit required')
    metadata,analysis=load_frozen_helpers();metadata.verify(code['source_hashes'])
    if any(code['source_hashes'].get(str(ROOT/n))!=sha(ROOT/n) for n in FILES):raise ValueError('Frozen own source identities required')
    meta=metadata.qualify();binding=meta['cal_binding'];entry=binding['entries'][8]
    if meta['checkpoint_sha256']!=MODEL_SHA or entry['id']!='old_walk_control' or entry['source_kind']!='collection_old_walk_control' or entry['collection'] is not True or entry['teacher'] is not False or entry['command']!=[1.,0.] or entry['start']!=1160 or entry['warm_start']!=200 or entry['warm_length']!=960 or entry['length']!=1920:raise ValueError('Actual neutral command collection source index8 required')
    folder=Path(entry['folder']);r=read(folder/'RESULT.json');v=read(folder/'NN_VERIFY.json');hand=read(folder/'HANDOVER.json');initial=read(entry['initial_state_path'])
    if r['kind']!='TEACHER_ASSISTED_COLLECTION_FIXED4_V1' or r['case']!='old_walk_control' or r['condition']!='walk' or r['teacher_assisted'] is not False or r['student_shadow_only'] is not True or r['oldcontrol_weight']!=1. or r['student_weight']!=0. or r['teacher_weight']!=0. or r['source_candidate']!=8 or r['teacher_calls_after_handover']!=0 or r['teaching_label_target']!='direct_frozen_oldmean_tanh' or r['actual_ms']!=15400 or r['neural_records']!=3080 or r['state_dims']!=[128,64,64,96] or r['state_total']!=352:raise ValueError('Actual pure old256 drive and fullshadow352 source flags required')
    if v['integrity_all_pass'] is not True or len(v['checks'])!=59 or not all(x is True for x in v['checks'].values()) or v['result_sha256']!=sha(folder/'RESULT.json') or entry['result_sha256']!=sha(folder/'RESULT.json') or entry['audit_sha256']!=sha(folder/'NN_VERIFY.json'):raise ValueError('Actual59saved source proof and calbinding identity required')
    if not r['original_five']['all_gates'] or not r['candidate_metrics']['task_pass_candidate']:raise ValueError('Original oldcontrol positive qualification required')
    if any(key not in initial for key in ('qpos','qvel','ctrl','xfrc_applied','qfrc_applied','target','buffer')) or len(initial['buffer'])!=40 or any(len(row)!=42 for row in initial['buffer']):raise ValueError('Actual fullsource initial target/history required')
    bodycase=next(row for row in meta['cases'] if row['condition']=='walk')
    paths=dict(samples=entry['path'],trace=entry['trace_path'],identity=entry['identity_path'],initial_state=entry['initial_state_path'],result=str(folder/'RESULT.json'),verify=str(folder/'NN_VERIFY.json'),handover=str(folder/'HANDOVER.json'))
    headers={key:metadata.header(entry['path'],key) for key in ('X238','command','hidden_before','hidden_after','oldmean_direct','supervision_target','actually_applied_action','time','physical_step_ms')}
    if headers['hidden_before']['shape']!=[3080,352] or headers['hidden_after']['shape']!=[3080,352] or headers['command']['shape']!=[3080,2] or headers['X238']['shape']!=[3080,238]:raise ValueError('Actual source352 and command/X header contract')
    tag='old_walk_control8';conditional_headers={field:metadata.header(meta['conditional_arrays']['path'],tag+'_'+field) for field in ('mean','action','pretarget','posttarget','new96')}
    if any(record['shape']!=[1920,96 if field=='new96' else 42] for field,record in conditional_headers.items()):raise ValueError('Actual new600 index8 conditional outputs required')
    proofs=dict(meta['proof_hashes']);source=dict(meta['source_hashes']);source.update(code['source_hashes'])
    for key in ('trace','neural_decisions','initial_state','handover','warm_boundaries'):
        item=r[key]
        if sha(item['path'])!=item['sha256']:raise ValueError('Neutral source artifact drift')
        proofs[item['path']]=item['sha256']
    for path in list(paths.values())+[ROOT/'CODECHECK.json',RECOVERY/'CODECHECK.json',ORIGINAL/'analyze.py',RECOVERY/'metadata.py']:proofs[str(path)]=sha(path)
    case=dict(condition='walk',entry=entry,index_in_training=8,sourcefolder=str(folder),sourcepaths=paths,bodyfolder=bodycase['bodyfolder'],bodypaths=bodycase['bodypaths'],conditional_tag=tag)
    for values in (source,meta['data_hashes'],proofs):metadata.verify(values)
    return dict(case=case,metadata=metadata,analysis=analysis,normalizer_mapping=meta['normalizer_mapping'],evaluation_arrays=meta['evaluation_arrays'],conditional_arrays=meta['conditional_arrays'],source_hashes=source,data_hashes=meta['data_hashes'],proof_hashes=proofs,source_headers=headers,conditional_headers=conditional_headers,source_recorded_handover=hand,checkpoint_sha256=MODEL_SHA,neutral_training_coverage=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute-parent-neutral-walk-appendix',action='store_true');args=p.parse_args()
    if not args.execute_parent_neutral_walk_appendix:p.error('Explicit parent neutralwalk saved-array appendix required')
    if any(name in sys.modules for name in FORBIDDEN):raise ValueError('Fresh noNN/noCore process required')
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(key)!='1':raise ValueError(key+'=1 required')
    if (ROOT/'RESULT.json').exists():raise FileExistsError('No overwrite/repeated appendix')
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});started=time.monotonic()
    def deadline(*unused):raise RuntimeError('Neutral appendix90second cap exceeded')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(90)
    try:
        meta=qualify();analysis=meta['analysis'];metadata=meta['metadata'];analysis.resource_gate(started)
        evaluation=analysis.take_arrays(meta['evaluation_arrays']['path'],('conditional_mean','conditional_action','conditional_post_targets'))
        tag='old_walk_control8';conditional=analysis.take_arrays(meta['conditional_arrays']['path'],tuple(tag+'_'+field for field in ('mean','action','pretarget','posttarget','new96')))
        row=analysis.case_report(meta['case'],evaluation,conditional,meta['normalizer_mapping'],started)
        for key in ('source_hashes','data_hashes','proof_hashes'):metadata.verify(meta[key])
        if any(name in sys.modules for name in FORBIDDEN):raise ValueError('Forbidden neural/physical module loaded')
        rss=analysis.resource_gate(started)
        save(ROOT/'RESULT.json',dict(kind='NEUTRAL_WALK_INDEX8_HANDOVER_APPENDIX_V1',neutral_command_has_actual_training_coverage=True,training_index=8,conditional_tag=tag,checkpoint_sha256=MODEL_SHA,row=row,source_headers=meta['source_headers'],conditional_headers=meta['conditional_headers'],source_hashes=meta['source_hashes'],data_hashes=meta['data_hashes'],proof_hashes=meta['proof_hashes'],wall_s=time.monotonic()-started,peak_RSS_MiB=rss,NN_forwards=0,Core_constructions=0,physical_steps=0,training_updates=0,scope='Saved neutralwalk initial parity only; BodyANNcomponent_NOTnewbrainneckdeployment'))
        with (ROOT/'REPORT.zh-CN.md').open('x') as stream:stream.write('# 中性walk匹配来源附录\n\n中性命令[1,0]确有actual index8 old_walk_control训练覆盖。原source是冻结789f直接驱动／007d仅shadow；本次new41fa预测参考仅EVAL[0,8]与CONDITIONAL600 old_walk_control8输出。\n\n保存初输入匹配：'+str(row['initial_saved_inputs_match'])+'；首输出测量≤1e−12：'+str(row['first_newNN_output_matches_training_saved_reference_le1e12'])+'；受控状态：'+row['first_action_expectation_status']+'。\n\n仅初始一致不证明后续闭环或行为成功，四源主报告和95行为FAIL均保持。Source后96属于007d不能当新600状态参考；此genericcase_report只比较source前256及new600 conditional输出。\n')
        checks=dict(neutral_metadata_coverage_verified=True,all_inputs_unchanged=True,original_case_report_unmodified=True,noNN_Core_physics=True,resources=rss<=1500 and time.monotonic()-started<=90)
        save(ROOT/'VERIFY.json',dict(integrity_all_pass=all(checks.values()),checks=checks,result_sha256=sha(ROOT/'RESULT.json'),report_sha256=sha(ROOT/'REPORT.zh-CN.md'),NN_forwards=0,Core_constructions=0,physical_steps=0,parity_difference_is_not_integrity_failure=True))
        if not all(checks.values()):raise ValueError('Appendix integrity/resource gate failed; preserveoutputs,noretry')
        print(json.dumps(dict(status='NEUTRAL_WALK_SAVED_APPENDIX_COMPLETED',result=str(ROOT/'RESULT.json'),NN_forwards=0,physical_steps=0)),flush=True)
    finally:signal.alarm(0)


if __name__=='__main__':main()
