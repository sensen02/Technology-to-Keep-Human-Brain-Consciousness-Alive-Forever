"""STD-only conditional post-deadline saved900 admission; never train or load arrays.

qualify is admissible only after the ORIGINAL pipeline terminal FAILED/124, with
all300 recorded updates and an explicitly supplied opaque original900 SHA.
Preview does not read the live TRAINING journal. No original files are written.
"""
from pathlib import Path
import ast
import datetime
import hashlib
import json
import math
import struct
import zipfile

ROOT=Path(__file__).resolve().parent
ORIGINAL=ROOT.parent/'state_aligned_action_calibration/wall_replan_9000'
PYTHON='/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python'
FILES=('metadata.py','runtime.py','run.py','PREREG.json','README.zh-CN.md','staticcheck.py')
FORMAT='INDEPENDENT_MOTOR_STATE_ALIGNED_ACTION_V1'
PARENT_SHA='41fa1e2232f5143fec0a3ebb5d506c819af0ba084b3a05041e829fd1477b906b'
LINEAGE='STATE_ALIGNED_ACTION_RESUME600_TO900'
STAGES=(('RESUME601_TO900_AND_FRESH600_750_900','train.py','--execute-parent-resume600-to900'),('FINAL900_STATE_ALIGNED_EVALUATION','evaluate.py','--execute-parent-evaluation'),('INDEPENDENT_SAVED_METRICS','audit_eval.py','--execute-parent-saved-audit'))


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def _digest(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def verify(hashes):
    if not isinstance(hashes,dict) or not hashes:raise ValueError('Nonempty immutable identity graph required')
    for path,digest in hashes.items():
        if not isinstance(path,str) or not Path(path).is_absolute() or not _digest(digest) or sha(path)!=digest:raise ValueError('Source/artifact identity drift: '+str(path))
    return True


def header(path,key):
    """ZIP/NPY header only: never decode or read an array payload."""
    with zipfile.ZipFile(path) as archive:
        members=[info for info in archive.infolist() if info.filename==key+'.npy']
        if len(members)!=1:raise ValueError('Unique saved field required: '+key)
        with archive.open(members[0]) as stream:
            if stream.read(6)!=b'\x93NUMPY':raise ValueError('NPY magic')
            version=tuple(stream.read(2))
            if version not in ((1,0),(2,0),(3,0)):raise ValueError('NPY version')
            width=2 if version==(1,0) else 4;raw=stream.read(width)
            if len(raw)!=width:raise ValueError('Truncated header length')
            length=struct.unpack('<H' if width==2 else '<I',raw)[0]
            if length>65536:raise ValueError('NPY header cap')
            text=stream.read(length)
            if len(text)!=length:raise ValueError('Truncated header')
            value=ast.literal_eval(text.decode('utf8' if version==(3,0) else 'latin1'))
    if not isinstance(value,dict) or set(value)!={'descr','fortran_order','shape'} or value['fortran_order'] is not False or not isinstance(value['descr'],str) or 'O' in value['descr'] or not isinstance(value['shape'],tuple) or any(type(n) is not int or n<0 for n in value['shape']):raise ValueError('Row-major nonobject header required')
    return dict(shape=list(value['shape']),descr=value['descr'],fortran_order=False)


def _merge(*maps):
    out={}
    for values in maps:
        for path,digest in values.items():
            if path in out and out[path]!=digest:raise ValueError('Conflicting immutable identity: '+path)
            out[path]=digest
    return out


def _jobs(terminal=False):
    path=ORIGINAL/'JOBS.json';before=sha(path);jobs=read(path)
    if not isinstance(jobs,list) or not 1<=len(jobs)<=3:raise ValueError('Original finite pipeline stage-prefix required')
    previous=0.
    for i,row in enumerate(jobs):
        name,script,flag=STAGES[i]
        if row['stage']!=name or row['command']!=[PYTHON,'-B',str(ORIGINAL/script),flag] or row['log']!=str(ORIGINAL/(name+'.log')):raise ValueError('Actual original stage/command/log identity required')
        if i<len(jobs)-1 and (row.get('status')!='COMPLETED' or row.get('exit_code')!=0):raise ValueError('Only completed exit0 prefix may precede terminal stage')
        if row.get('status')=='COMPLETED' and row.get('exit_code')!=0:raise ValueError('Inconsistent completed stage')
        if 'pipeline_elapsed_s' in row:
            elapsed=row['pipeline_elapsed_s']
            if not isinstance(elapsed,(int,float)) or not math.isfinite(elapsed) or elapsed<previous:raise ValueError('Original cumulative stage clock invalid')
            previous=elapsed
    if terminal:
        if any(row.get('status')=='RUNNING' for row in jobs):raise ValueError('Original stage still RUNNING; no conditional release')
        last=jobs[-1]
        if last.get('status')!='FAILED' or last.get('exit_code')!=124:raise ValueError('Only original terminal deadline FAILED exit124 admitted; all0/otherfail rejected')
        if last.get('reason')=='pipelinewall9000expired':
            if len(jobs)<2 or 'pipeline_elapsed_s' in last:raise ValueError('Actual producer pre-stage exhausted deadline schema required')
        elif not isinstance(last.get('pipeline_elapsed_s'),(int,float)) or last['pipeline_elapsed_s']<9000 or 'end_utc' not in last:raise ValueError('Actual timeout terminal clock/end marker required')
        if 'end_utc' in last and datetime.datetime.fromisoformat(last['end_utc'])<datetime.datetime.fromisoformat(last['start_utc']):raise ValueError('Original terminal time precedes start')
    if sha(path)!=before:raise ValueError('Original JOBS changed during metadata snapshot')
    return jobs,before


def _saved_proof(step,binding):
    path=ORIGINAL/'TRAIN'/f'VERIFY_{step:06d}.json';v=read(path);checks=v.get('checks');checkpoint=ORIGINAL/'TRAIN'/f'checkpoint_{step:06d}.pt'
    if v.get('integrity_all_pass') is not True or not isinstance(checks,dict) or len(checks)!=81 or not all(x is True for x in checks.values()) or v['step']!=step or v['checkpoint']!=str(checkpoint) or sha(checkpoint)!=v['checkpoint_sha256'] or v['physical_steps']!=0 or v['training_RSS_limit_MiB']!=1800 or v['conditional_notphysicalcounterfactual'] is not True:raise ValueError('Actual original600/750 fresh81 proof required')
    for key in ('strictloader_RNGunchanged','oldweights_frozen','canonical8Adamstates','all_model_weights_unchanged','no_Core_physics_teacherbackend'):
        if checks.get(key) is not True:raise ValueError('Actual frozen/Adam/RNG saved qualification required')
    entries=binding['entries']+binding['dev_entries']
    if len(entries)!=20 or len(v['rows'])!=20:raise ValueError('Actual12TRAIN8DEV saved proof required')
    for i,(row,entry) in enumerate(zip(v['rows'],entries)):
        tag=entry['id']+str(i)
        if row['id']!=entry['id'] or row['training'] is not (i<12) or row['labels_available'] is not (i<12) or row['actual_rows']!=entry['length'] or any(checks.get(tag+s) is not True for s in ('_production_finite','_actualX_cachedproduction20')):raise ValueError('Actual saved20 production identities mismatch')
        if i<12:
            errors=row['conditional_output_errors']
            if len(errors)!=6 or any(not isinstance(x,(int,float)) or not math.isfinite(x) or not 0<=x<=1e-12 for x in errors) or any(checks.get(tag+s) is not True for s in ('_TFproduction','_conditional_fullproduction','_conditional_finite')):raise ValueError('Actual TF12/conditional12 six-output qualification required')
    artifact=v['conditional_arrays'];array_path=ORIGINAL/'TRAIN'/f'CONDITIONAL_{step:06d}.npz'
    if artifact['path']!=str(array_path) or sha(array_path)!=artifact['sha256']:raise ValueError('Original completed conditional artifact SHA required')
    verify(v['source_hashes']);verify(v['data_hashes'])
    for i,entry in enumerate(binding['entries']):
        for field in ('mean','residual','new96','pretarget','posttarget','action','TFmean','TFaction'):
            h=header(array_path,entry['id']+str(i)+'_'+field)
            if h['shape']!=[1920,96 if field=='new96' else 42] or h['descr']!='<f8':raise ValueError('Actual saved conditional/TF header mismatch')
    return v,{str(path):sha(path),str(checkpoint):v['checkpoint_sha256'],str(array_path):artifact['sha256']}


def qualification_preview():
    """Safe STD observation only; never qualifies a RUNNING or unknown900 run."""
    try:
        jobs,jobs_sha=_jobs();binding=read(ORIGINAL/'INPUT_BINDINGS.json');completed=[]
        for step in (600,750):
            p=ORIGINAL/'TRAIN'/f'VERIFY_{step:06d}.json'
            if p.is_file():
                proof,hashes=_saved_proof(step,binding);completed.append(dict(step=step,checks=len(proof['checks']),checkpoint=proof['checkpoint'],checkpoint_sha256=proof['checkpoint_sha256'],proof_hashes=hashes))
        running=any(row.get('status')=='RUNNING' for row in jobs)
        if running:status='PENDING_ORIGINAL_RUNNING'
        elif jobs[-1].get('status')=='FAILED' and jobs[-1].get('exit_code')==124:status='PENDING_EXPLICIT_900_SHA_AND_TERMINAL_QUALIFICATION'
        else:status='NOT_ADMITTED_ORIGINAL_NOT_DEADLINE_FAILED'
        return dict(status=status,qualified=False,original_jobs=str(ORIGINAL/'JOBS.json'),original_jobs_sha256=jobs_sha,original_stage_statuses=[dict(stage=r['stage'],status=r.get('status'),exit_code=r.get('exit_code')) for r in jobs],completed_saved_proofs=completed,saved900_exists=(ORIGINAL/'TRAIN/checkpoint_000900.pt').is_file(),explicit900_digest_required=True,live_training_journal_read=False,NN_forwards=0,Core_constructions=0,physical_steps=0)
    except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError,zipfile.BadZipFile) as exc:return dict(status='PENDING_OR_INVALID_METADATA',qualified=False,error_type=type(exc).__name__,error=str(exc),NN_forwards=0,Core_constructions=0,physical_steps=0)

inspect_pending=qualification_preview


def qualify(digest):
    if not _digest(digest):raise ValueError('Explicit actual original900 lowercase SHA256 required; unknown refused')
    jobs,jobs_sha=_jobs(terminal=True)
    code=read(ROOT/'CODECHECK.json')
    if not code.get('checks') or not all(value is True for value in code['checks'].values()):raise ValueError('Completed own STD source qualification required')
    own={str(ROOT/n):sha(ROOT/n) for n in FILES}
    if any(code['source_hashes'].get(p)!=h for p,h in own.items()):raise ValueError('All six frozen own source identities required')
    verify(code['source_hashes'])
    binding=read(ORIGINAL/'INPUT_BINDINGS.json');identity=read(ORIGINAL/'TRAIN/IDENTITY.json');pre=read(ORIGINAL/'PREFLIGHT.json')
    verify(binding['source_hashes']);verify(binding['data_hashes']);verify(identity['source_hashes']);verify(identity['data_hashes'])
    if pre['all_passed'] is not True or pre['timing_status']!='READY_STATE_ALIGNED600_TO900' or pre['source_hashes']!=binding['source_hashes'] or pre['data_hashes']!=binding['data_hashes']:raise ValueError('Original fixed readiness/source graphs required')
    expected=dict(lineage=LINEAGE,resume_step=600,total_updates=900,new_updates=300,parent_checkpoint_sha256=PARENT_SHA,original_anchor_preserved=True,originalAdamRNGrestored=True,action_supervision='actualXTFonly',conditionalactionlabels_supervised=False)
    if any(identity.get(k)!=v or (isinstance(v,bool) and identity.get(k) is not v) for k,v in expected.items()):raise ValueError('Original restored600Adam/RNG/anchor identity required')
    journal=ORIGINAL/'TRAIN/TRAINING.jsonl';journal_sha=sha(journal)
    with journal.open() as stream:lines=stream.readlines()
    if len(lines)!=300 or any(not line.strip() for line in lines):raise ValueError('Exactly300 original updates, no extra/retry journal rows required')
    rows=[json.loads(line) for line in lines]
    if [r.get('step') for r in rows]!=list(range(601,901)) or any(r.get('lineage')!=LINEAGE for r in rows):raise ValueError('Original update sequence must be601..900 exactly')
    for r in rows:
        if not isinstance(r.get('loss'),(int,float)) or not math.isfinite(r['loss']) or not isinstance(r.get('wall_s'),(int,float)) or not math.isfinite(r['wall_s']) or r['wall_s']<0 or not isinstance(r.get('peak_RSS_MiB'),(int,float)) or not math.isfinite(r['peak_RSS_MiB']) or not 0<=r['peak_RSS_MiB']<=1800:raise ValueError('Actual original finite update/resource journal metadata required')
    if any(a['wall_s']>b['wall_s'] for a,b in zip(rows,rows[1:])) or sha(journal)!=journal_sha:raise ValueError('Original training journal changed or nonmonotonic')
    target=ORIGINAL/'TRAIN/checkpoint_000900.pt'
    if not target.is_file() or sha(target)!=digest:raise ValueError('Actual original900 opaque checkpoint required;750/unknown not admitted')
    source=_merge(binding['source_hashes'],identity['source_hashes'],own);data=_merge(binding['data_hashes'],identity['data_hashes']);proofs={};saved={}
    for step in (600,750):
        v,hashes=_saved_proof(step,binding);saved[str(step)]=v;proofs=_merge(proofs,hashes);source=_merge(source,v['source_hashes']);data=_merge(data,v['data_hashes'])
    resultpath=ORIGINAL/'TRAIN/RESULT.json';result=None
    if resultpath.is_file():
        result=read(resultpath)
        if result.get('kind')!='STATE_ALIGNED_ACTION900_TRAINED_NOT_BODY_VERIFIED' or result.get('updates')!=900 or result.get('total_updates')!=900 or result.get('new_updates')!=300 or result.get('already_completed_updates')!=600 or result.get('checkpoint')!=str(target) or result.get('sha256')!=digest or result.get('final',{}).get('step')!=900 or result.get('physical_steps')!=0 or result.get('new_neck_deployment') is not False:raise ValueError('Existing original TRAIN RESULT must match actual900; absence is allowed')
        verify(result['source_hashes']);verify(result['data_hashes']);source=_merge(source,result['source_hashes']);data=_merge(data,result['data_hashes']);proofs[str(resultpath)]=sha(resultpath)
    # Snapshot all original failure bookkeeping plus existing saved900 partial
    # proof files by opaque SHA. A partial/failed900 VERIFY is never promoted.
    paths=[ORIGINAL/'JOBS.json',ORIGINAL/'INPUT_BINDINGS.json',ORIGINAL/'PREFLIGHT.json',ORIGINAL/'RESOURCE_REPLAN.json',ORIGINAL/'TRAIN/IDENTITY.json',journal,target,ROOT/'CODECHECK.json']
    paths += list(ORIGINAL.glob('*.log'))+list(ORIGINAL.glob('*FAILURE*.json'))
    paths += [p for p in (ORIGINAL/'TRAIN/VERIFY_000900.json',ORIGINAL/'TRAIN/CONDITIONAL_000900.npz',ORIGINAL/'EVALUATION.json',ORIGINAL/'EVALUATION_VERIFY.json',ORIGINAL/'STAGE_VERIFICATION.json') if p.is_file()]
    for path in paths:proofs[str(path)]=sha(path)
    if proofs[str(ORIGINAL/'JOBS.json')]!=jobs_sha or proofs[str(journal)]!=journal_sha:raise ValueError('Original terminal failure/journal mutated during admission')
    verify(source);verify(data);verify(proofs);_jobs(terminal=True)
    if sha(ORIGINAL/'JOBS.json')!=jobs_sha:raise ValueError('Original pipeline resumed or changed during qualification')
    return dict(kind='POSTRUN_SAVED900_DEADLINE_CONDITIONAL_ADMISSION_V1',checkpoint=str(target),checkpoint_sha256=digest,motor_format=FORMAT,step=900,original_completed_training_updates=300,original_total_updates=900,postrun_training_updates=0,postrun_optimizer_operations=0,postrun_gradient_operations=0,original_deadline_failure=dict(path=str(ORIGINAL/'JOBS.json'),sha256=jobs_sha,status='FAILED',exit_code=124,last_stage=jobs[-1]['stage'],original_jobs=jobs,pipeline_not_running=True),original_RESULT_status='RECORDED_AND_MATCHED' if result is not None else 'NOT_RECORDED_DEADLINE_MAY_PRECEDE_EXPORT',training_rows=rows,training_journal_sha256=journal_sha,original_saved_proofs=saved,source_hashes=source,data_hashes=data,proof_hashes=proofs,original_binding_sha256=sha(ORIGINAL/'INPUT_BINDINGS.json'),original900_saved_numeric_qualification='NOT_ADMITTED_BY_METADATA_REQUIRES_POSTRUN_STRICT_FRESH_VERIFY',original_failure_preserved=True,new_training_authorized=False,body_behavior_pass=None,new_neck_deployed=False)


def _copy_exact(source,target):
    source=Path(source);target=Path(target);before=sha(source);payload=source.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=before:raise ValueError('Source changed while staging exact bytes')
    with target.open('xb') as stream:stream.write(payload)
    if sha(target)!=before or sha(source)!=before:raise ValueError('Exact staging SHA mismatch')
    return dict(source=str(source),target=str(target),sha256=before,exact_bytes=True)


def prepare(digest):
    """Parent-only exclusive metadata staging; no large arrays/checkpoints copied."""
    if any((ROOT/n).exists() for n in ('INPUT_BINDINGS.json','SOURCE_LEDGER.json','TRAIN')):raise FileExistsError('No staging overwrite/retry')
    meta=qualify(digest);(ROOT/'TRAIN').mkdir(exist_ok=False)
    copies=[_copy_exact(ORIGINAL/'INPUT_BINDINGS.json',ROOT/'INPUT_BINDINGS.json')]
    for step in (600,750):copies.append(_copy_exact(ORIGINAL/'TRAIN'/f'VERIFY_{step:06d}.json',ROOT/'TRAIN'/f'VERIFY_{step:06d}.json'))
    for key in ('source_hashes','data_hashes','proof_hashes'):verify(meta[key])
    save(ROOT/'SOURCE_LEDGER.json',dict(kind=meta['kind'],checkpoint=meta['checkpoint'],checkpoint_sha256=digest,original_deadline_failure=meta['original_deadline_failure'],original_completed_training_updates=300,original_total_updates=900,postrun_training_updates=0,postrun_optimizer_operations=0,postrun_gradient_operations=0,source_hashes=meta['source_hashes'],data_hashes=meta['data_hashes'],proof_hashes=meta['proof_hashes'],exact_metadata_copies=copies,large_array_or_checkpoint_copies=0,original_paths_inside_copies_retained=True,original900_numeric_admission='REQUIRES_NEW_STRICT_FRESH_SAVED_VERIFY',original_RESULT_status=meta['original_RESULT_status'],no_original_writes=True))
    return meta
