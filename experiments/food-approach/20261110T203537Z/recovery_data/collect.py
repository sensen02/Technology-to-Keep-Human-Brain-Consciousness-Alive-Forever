"""Prepared actual student-state collection. Main requires explicit diagnostic guard.
No execution/import of physical backend occurs on module import.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import numpy as np
from contract import source_at,metadata,admit
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
BASE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
TEACHER=ROOT.parent/'20261110T191449Z/teacher/turning_teacher.py'


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def jsonable(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,Path):return str(v)
    if isinstance(v,dict):return {str(k):jsonable(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [jsonable(x) for x in v]
    return v


def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--condition',choices=['left','right'],required=True);p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--run-diagnostic',action='store_true')
    p.add_argument('--run-plan',required=True,help='future explicit registered plan, not this draft')
    return p


def validate_plan(args):
    if not args.run_diagnostic:raise ValueError('code preparation only; explicit --run-diagnostic required')
    if args.seed<0:raise ValueError('nonnegative seed required')
    for name in ('checkpoint','out','run_plan'):
        if not Path(getattr(args,name)).is_absolute():raise ValueError('absolute paths required: '+name)
    plan=json.loads(Path(args.run_plan).read_text())
    if plan.get('status')!='REGISTERED_FOR_ACTUAL_DIAGNOSTIC':raise ValueError('explicit future fixed-run registration required')
    expected=dict(checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_sha256=args.sha256,condition=args.condition,seed=args.seed)
    if expected not in plan.get('runs',[]):raise ValueError('run not registered')
    if plan.get('draft_sha256')!=digest(HERE/'PREREG_DRAFT.json'):raise ValueError('draft identity changed')
    if not Path(args.out).resolve().is_relative_to(HERE) or Path(args.out).exists():raise ValueError('new output strictly inside recovery_data required')
    return plan


def collect(args):
    plan=validate_plan(args)  # Guard before any neural or physics import.
    checks=json.loads((HERE/'OFFLINE_CHECKS.json').read_text())
    if not checks['all_passed']:raise ValueError('offline preparation checks required')
    for filename,expected in checks['source_hashes'].items():
        if digest(filename)!=expected:raise ValueError('offline checked source drift: '+filename)
    import torch
    torch.set_num_threads(1);torch.manual_seed(args.seed);np.random.seed(args.seed)
    for name in ('physics','student_wave','models','identity','static_teacher','wave_reference','cycle_reward'):
        if name in sys.modules and Path(sys.modules[name].__file__).resolve()!=BASE/(name+'.py'):raise ValueError('wrong baseline module '+name)
    sys.path.insert(0,str(BASE));sys.path.insert(0,str(ROOT))
    import fast_motor as fm
    import student_wave as sw
    import identity
    from physics import W
    model,payload=fm.load(args.checkpoint,args.sha256)
    if payload['curve'][-1]['step']!=3000:raise ValueError('fixed3000 checkpoint required')
    spec=importlib.util.spec_from_file_location('_recovery_teacher_frozen',TEACHER)
    tm=importlib.util.module_from_spec(spec);spec.loader.exec_module(tm);tm.pin_baseline()
    sw.teacher_preflight();sw.student_preflight()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    paths=[HERE/n for n in ('collect.py','contract.py','PREREG_DRAFT.json')]+[TEACHER,Path(args.run_plan),Path(args.checkpoint),ROOT/'fast_motor.py']
    hashes={str(p.resolve()):digest(p) for p in paths}
    core=sw.Core(adhesion=False)
    sw.load_student(fm.CHECKPOINT,core)  # Verify actual original runtime/assets.
    ident=identity.capture(core,dict(kind='training_only_student_state_mixed_collection',cli=vars(args),plan=plan,draft=json.loads((HERE/'PREREG_DRAFT.json').read_text()),source_hashes=hashes),extra_paths=paths)
    def dump(name,value):identity.atomic_json(out/name,jsonable(value))
    dump('IDENTITY.json',ident)
    trace=sw.Trace(full=True);rows=[];boundaries=[];stop=None;failure=None;result=None;unsafe=0
    observer=None;events=[];teacher_queries=0;teacher_applied=0;student_applied=0;recovery=[]
    state=model.initial_state(1);previous=np.zeros(42)
    def tick(action,source):
        nonlocal unsafe
        measured=trace.tick(core,action,source)
        unsafe=unsafe+1 if measured['com']<.8 or measured['nonfoot']>.5*W else 0
        return measured
    try:
        for _ in range(1000):
            tick(np.zeros(42),0)
            if unsafe>=100:
                stop=dict(reason='100ms COM<.8 or nonfoot>.5W during settle',time=float(core.d.time))
                raise RuntimeError(stop['reason'])
        teacher=tm.make_teacher(core,speed_scale=1.,yaw_rate=.008 if args.condition=='left' else -.008,prime_s=4.8)
        observer=sw.tracker(core)
        for k in range(1920):
            meta=metadata(k);source=meta['source'];elapsed=meta['teacher_elapsed_s']
            if k==960:observer=sw.tracker(core);events=[]
            if meta['burst_boundary']:
                before=sw.continuity_snapshot(core,previous,state)
                after=sw.continuity_snapshot(core,previous,state)
                exact={key:bool(np.array_equal(before[key],after[key])) for key in before}
                if not all(exact.values()):raise ValueError('boundary reset/discontinuity')
                boundaries.append(dict(decision=k,time=float(core.d.time),source=source,all_exact=True,before=sw.snapshot_hashes(before),after=sw.snapshot_hashes(after)))
                dump('BOUNDARIES.json',boundaries)
            x,_=sw.observations(core,previous,(1920-k)*.005)
            phase,scale,curvature=teacher.reference.command(elapsed)
            command=np.array([scale-1.,.075*scale*curvature/.008])
            hidden_before=state.clone()
            with torch.no_grad():mean,state=model.step(torch.tensor(x[None],dtype=torch.float64),torch.tensor(command[None],dtype=torch.float64),state)
            student=sw.bounded_action(mean[0].tanh().numpy())
            label,diag=teacher.action(core,elapsed);teacher_queries+=1
            label=sw.bounded_action(label)
            applied=student if source==3 else label
            teacher_applied+=int(source!=3);student_applied+=int(source==3)
            rows.append(dict(time=float(core.d.time),X238=x,command2=command,teacher_action=label,student_mean=mean[0].numpy(),student_action=student,
                             actually_applied_action=applied.copy(),hidden_before=hidden_before[0].numpy(),hidden_after=state[0].numpy(),source=source,
                             teacher_phase_provenance_only=float(phase),root_balance_residual=float(diag['root_balance_residual_norm_over_W'])))
            start_com=core.d.subtree_com[core.th].copy()
            for _ in range(5):
                m=tick(applied,source)
                if k>=960:events.extend(observer.measure(core)['events'])
                if unsafe>=100:
                    stop=dict(reason='100ms COM<.8 or nonfoot>.5W',time=float(core.d.time),decision=k);break
            if k>=960 and (k-960)%40 in (4,39):
                recovery.append(dict(decision=k,time=float(core.d.time),source=source,com=core.d.subtree_com[core.th].copy(),foot_load_over_W=float(m['loads'].sum()/W),nonfoot=float(m['nonfoot']),interval_com_delta=core.d.subtree_com[core.th]-start_com,
                                     interpretation='raw segment endpoint, not proof recovery improved'))
            previous=applied.copy()
            if stop is not None:break
        # Include all collection sources; never summarize only25ms bursts as walking pass.
        collection_rec=[r for r,s in zip(trace.rec,trace.sources) if s in (3,5)]
        summary=sw.summarise(collection_rec,len(collection_rec)*80,initial_heading=core.initial_heading) if len(collection_rec)>=2 else {}
        strat={}
        for source in (3,5):
            subset=[r for r,s in zip(trace.rec,trace.sources) if s==source]
            strat[str(source)]=dict(endpoints=len(subset),summary=sw.summarise(subset,len(subset)*80,initial_heading=core.initial_heading) if len(subset)>1 else None,
                                    interpretation='disjoint source-stratified diagnostic; gaps included in elapsed denominator; NOT continuous autonomy')
        complete=len(trace.rec)==10600 and len(rows)==1920 and stop is None
        result=dict(status='TRAINING_ONLY_MIXED_CONTROL_NOT_AUTONOMOUS_PASS',dataset_safety_admitted=admit(summary,complete,stop,None),
                    complete=complete,early_stop=stop,collection=summary,source_stratified_diagnostic=strat,
                    cycles_diagnostic=sw.cycle_report(observer,events) if observer is not None else None,
                    teacher_queries=teacher_queries,teacher_applied_decisions=teacher_applied,student_applied_decisions=student_applied,
                    source_decision_counts={str(s):sum(r['source']==s for r in rows) for s in (2,3,5)},
                    source_endpoint_counts={str(s):trace.sources.count(s) for s in (0,2,3,5)},
                    checkpoint_sha256=args.sha256,identity_sha256=ident['identity_sha256'],recovery_effect='NOT_ESTABLISHED; raw endpoints recorded separately',
                    physical_steps=len(trace.rec)*80,autonomous_walking_pass=False,food_success=False)
        identity.verify(ident);identity.verify_runtime(ident,core);fm.verify_hashes(hashes);fm.load(args.checkpoint,args.sha256)
    except BaseException as exc:failure=exc
    finally:
        artifacts={}
        try:
            if trace.rows:artifacts['trace']=trace.save(out/'trace_1ms.npz')
            if rows:
                p=out/'SAMPLES_5MS.npz';np.savez_compressed(p,**{key:np.asarray([row[key] for row in rows]) for key in rows[0]});artifacts['samples']=dict(path=str(p),sha256=digest(p),decisions=len(rows))
            dump('BOUNDARIES.json',boundaries);dump('RECOVERY_ENDPOINTS.json',recovery)
            if failure is not None:dump('FAILURE.json',dict(error=repr(failure),early_stop=stop,completed_endpoints=len(trace.rec),decisions=len(rows),artifacts=artifacts,dataset_safety_admitted=False,autonomous_walking_pass=False))
            elif result is not None:result['artifacts']=artifacts;dump('RESULT.json',result)
        except BaseException as saving:
            if failure is None:failure=saving
            else:print('Additional saving failure: '+repr(saving),file=sys.stderr)
    if failure is not None:raise failure
    print(json.dumps(dict(out=str(out),dataset_safety_admitted=result['dataset_safety_admitted'],complete=result['complete'],autonomous_walking_pass=False)))
    return result

if __name__=='__main__':collect(parser().parse_args())
