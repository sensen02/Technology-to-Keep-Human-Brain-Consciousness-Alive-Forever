"""Parent-only original saved-model science, relocated outputs and readonly observer."""
from pathlib import Path
import argparse
import ast
import importlib.util
import json
import os
import resource
import signal
import sys
import time
from metadata import ROOT,ORIGINAL,PYTHON,read,save,sha,verify,qualify

STAGES=('FRESH900_SAVED_AUDIT','SAVED900_EVALUATION','INDEPENDENT_SAVED_METRICS')
PHYSICAL=('mujoco','physics','flygym','student_wave','static_teacher')


def stage_source(name):
    filename={'FRESH900_SAVED_AUDIT':'verify_saved.py','SAVED900_EVALUATION':'evaluate.py','INDEPENDENT_SAVED_METRICS':'audit_eval.py'}[name]
    tree=ast.parse((ORIGINAL/filename).read_text());changes=0
    if name=='FRESH900_SAVED_AUDIT':
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='common' and node.func.attr=='load_checkpoint':
                for keyword in node.keywords:
                    if keyword.arg=='final' and isinstance(keyword.value,ast.Constant) and keyword.value.value is False:keyword.value=ast.Constant(True);changes+=1
        if changes!=1:raise ValueError('Exactly one original finalFalse->True admission strengthening required')
    return filename,ast.fix_missing_locations(tree),changes


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


def resources(end):
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
    if rss>1800:raise RuntimeError('Postrun qualification RSS1800MiB cap exceeded')
    if time.monotonic()>=end:raise RuntimeError('Postrun shared3600second deadline exceeded')
    return rss


def observe_payload(model,payload,meta):
    """Observe already strict-loaded objects, return nothing, no second load/updates."""
    import torch
    if payload['step']!=900 or payload['format_version']!='INDEPENDENT_MOTOR_STATE_ALIGNED_ACTION_V1':raise ValueError('Original strict final900 format required')
    slots=payload['optimizer_state_dict']['state']
    if len(slots)!=8 or any(float(slot['step'])!=900 for slot in slots.values()):raise ValueError('Actual eight savedAdam counters900 required')
    if payload['identity']!=model.identity() or not model.verify_old() or any(p.grad is not None for p in model.parameters()):raise ValueError('Actual strict identity/frozenold/no accumulated gradients required')
    curve=payload['curve'];saved=[row for row in curve if 601<=row['step']<=900]
    actual=meta['training_rows']
    if len(saved)!=300 or [row['step'] for row in saved]!=list(range(601,901)):raise ValueError('Saved curve does not contain exactly original300 updates')
    for logged,checkpoint_row in zip(actual,saved):
        if any(key not in checkpoint_row or checkpoint_row[key]!=value for key,value in logged.items()):raise ValueError('Saved900 curve differs from original logged update row')
        if set(checkpoint_row)-set(logged)-{'metrics_after_update'}:raise ValueError('Undeclared saved-curve metadata differs from actual original log')
    if payload['config']['resume_parent_sha256']!='41fa1e2232f5143fec0a3ebb5d506c819af0ba084b3a05041e829fd1477b906b' or payload['config']['lineage']!='STATE_ALIGNED_ACTION_RESUME600_TO900' or 'initial_anchor' not in payload['config']:raise ValueError('Original600 lineage/anchor required')
    # Anchor tensor equality, fullRNG and all moment shapes are validated by the
    # unchanged original strict loader before this readonly observation.
    if not isinstance(payload['config']['initial_anchor'],dict) or len(payload['config']['initial_anchor'])!=8:raise ValueError('Original eight-tensor anchor required')
    return dict(step=900,optimizer_states=8,all_Adam_counters900=True,original_logged_update_rows=300,curve_logged_fields_exact=True,extra_checkpoint_metrics_only=True,strict_payload_identity_and_oldweights=True,original_anchor_RNG_validation='UNCHANGED_ORIGINAL_STRICT_LOADER',observer_training_updates=0,observer_optimizer_operations=0)


def execute(stage,digest,end):
    if sys.executable!=str(PYTHON):raise ValueError('Qualified original125712 Python required')
    if any(name in sys.modules for name in PHYSICAL):raise ValueError('Fresh noCore process required')
    meta=qualify(digest);resources(end)
    # Data/common/model/scan/learning/sources all retain ORIGINAL modules and ROOT.
    names=('sources','common','model','scan','data','learning','utils')
    sentinel=object();prior={name:sys.modules.get(name,sentinel) for name in names};paths=list(sys.path);argv=list(sys.argv);patches=[];hook_state={};touched={}
    try:
        sys.path.insert(0,str(ORIGINAL))
        sources=load_module('sources',ORIGINAL/'sources.py')
        filename,tree,changes=stage_source(stage)
        spec=importlib.util.spec_from_file_location('_postrun_original_'+stage,ORIGINAL/filename);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
        exec(compile(tree,str(ORIGINAL/filename)+'::postrun_output_ROOT_only','exec'),module.__dict__)
        module.ROOT=ROOT
        if sources.ROOT!=ORIGINAL or module.main.__globals__ is not module.__dict__:raise ValueError('Original module/function globals provenance')
        if stage!='INDEPENDENT_SAVED_METRICS':
            common=load_module('common',ORIGINAL/'common.py')
            import torch
            # No optimizer operation or backward is permitted in qualification.
            def forbidden(*args,**kwargs):raise RuntimeError('Postrun qualification forbids optimizer/gradient operations')
            for obj,name in ((torch.optim.Adam,'step'),(torch.optim.Adam,'zero_grad'),(torch.optim.Optimizer,'step'),(torch.optim.Optimizer,'zero_grad'),(torch.Tensor,'backward'),(torch.autograd,'backward')):
                patches.append((obj,name,getattr(obj,name)));setattr(obj,name,forbidden)
            if stage=='FRESH900_SAVED_AUDIT':
                original_load=common.load_checkpoint
                def observed_load(path,checksum,final=True):
                    if Path(path).resolve()!=Path(meta['checkpoint']).resolve() or checksum!=digest or final is not True:raise ValueError('Observed loader must receive exact original900 and strictfinalTrue')
                    model,payload=original_load(path,checksum,final=True)
                    if hook_state:raise ValueError('Unexpected duplicate final900 load in fresh audit')
                    hook_state.update(summary=observe_payload(model,payload,meta),model=model)
                    numeric_model=sys.modules[type(model).__module__]
                    hook_state['digest_function']=numeric_model.state_digest
                    hook_state['weights_before']=numeric_model.state_digest(model.state_dict())
                    return model,payload
                patches.append((common,'load_checkpoint',original_load));common.load_checkpoint=observed_load
        flag={'FRESH900_SAVED_AUDIT':'--execute-parent-saved-verify','SAVED900_EVALUATION':'--execute-parent-evaluation','INDEPENDENT_SAVED_METRICS':'--execute-parent-saved-audit'}[stage]
        sys.argv=[str(ROOT/'runtime.py'),flag]
        if stage=='FRESH900_SAVED_AUDIT':sys.argv+=['--checkpoint',meta['checkpoint'],'--sha256',digest]
        module.main();rss=resources(end)
        if stage=='FRESH900_SAVED_AUDIT':
            proof=read(ROOT/'TRAIN/VERIFY_000900.json')
            if not proof['integrity_all_pass'] or len(proof['checks'])!=81 or not all(value is True for value in proof['checks'].values()) or proof['checkpoint_sha256']!=digest or not hook_state:raise ValueError('Fresh original81 saved900 checks required')
            if hook_state['digest_function'](hook_state['model'].state_dict())!=hook_state['weights_before'] or any(p.grad is not None for p in hook_state['model'].parameters()):raise ValueError('Weights or gradients changed during saved qualification')
            save(ROOT/'TRAIN/RESULT.json',dict(kind='SAVED900_QUALIFIED_NO_NEW_UPDATES',qualification_mode='POSTRUN_SAVED_CHECKPOINT_ONLY',updates=900,total_updates=900,new_updates=300,already_completed_updates=600,qualification_new_updates=0,checkpoint=meta['checkpoint'],sha256=digest,original_training_rows=300,original_training_log=str(ORIGINAL/'TRAIN/TRAINING.jsonl'),original_training_log_sha256=sha(ORIGINAL/'TRAIN/TRAINING.jsonl'),original_deadline_failure=meta['original_deadline_failure'],strict_payload_observation=hook_state['summary'],source_hashes=meta['source_hashes'],data_hashes=meta['data_hashes'],proof_hashes=meta['proof_hashes'],physical_steps=0,new_neck_deployment=False,original_pipeline_success_claim=False,wall_s=None))
        for key in ('source_hashes','data_hashes','proof_hashes'):verify(meta[key])
        if any(name in sys.modules for name in PHYSICAL):raise ValueError('Physical backend imported during saved qualification')
        save(ROOT/(stage+'_COMPLETION.json'),dict(stage=stage,passed=True,original_module=str(ORIGINAL/filename),original_source_sha256=sha(ORIGINAL/filename),output_ROOT=str(ROOT),AST_changes='ONE_FINAL_FALSE_TO_TRUE' if changes else 'NONE',strict_payload_observer_readonly=stage=='FRESH900_SAVED_AUDIT',NN_execution=stage!='INDEPENDENT_SAVED_METRICS',optimizer_operations=0,training_updates=0,Core_constructions=0,physical_steps=0,peak_RSS_MiB=rss,deadline_remaining_s=end-time.monotonic(),source_and_input_unchanged=True))
    finally:
        for obj,name,value in reversed(patches):setattr(obj,name,value)
        sys.argv[:]=argv;sys.path[:]=paths
        for name,value in prior.items():
            if value is sentinel:sys.modules.pop(name,None)
            else:sys.modules[name]=value


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=STAGES,required=True);p.add_argument('--sha256',required=True);p.add_argument('--deadline-monotonic',type=float,required=True);p.add_argument('--execute-parent-postrun-stage',action='store_true');a=p.parse_args()
    if not a.execute_parent_postrun_stage:p.error('Explicit parent postrun stage required')
    def cap(*unused):resources(a.deadline_monotonic)
    signal.signal(signal.SIGALRM,cap);signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGALRM});signal.setitimer(signal.ITIMER_REAL,1.,1.)
    try:execute(a.stage,a.sha256,a.deadline_monotonic)
    finally:signal.setitimer(signal.ITIMER_REAL,0.)


if __name__=='__main__':main()
