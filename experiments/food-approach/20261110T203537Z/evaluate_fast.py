"""Fixed1000 fast motor evaluation; no physics on import, robust partial recording."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
ORIGINAL=BASE.parent/'20261110T152754Z/NATIVE_ROLE_2500'
COMMANDS={'neutral':[0.,0.],'left':[0.,1.],'right':[0.,-1.],'speed085':[-.15,0.],'speed090':[-.10,0.]}


def jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [jsonable(v) for v in value]
    return value


def pin():
    for name in ('physics','static_teacher','student_wave','models','identity','cycle_reward','wave_reference'):
        if name in sys.modules and Path(sys.modules[name].__file__).resolve()!=BASE/(name+'.py'):
            raise ValueError('conflicting baseline import '+name)
    sys.path.insert(0,str(BASE));sys.path.insert(0,str(HERE))
    import student_wave as sw
    import identity
    import fast_motor as fm
    if Path(fm.__file__).resolve()!=HERE/'fast_motor.py':raise ValueError('wrong fast implementation')
    return sw,identity,fm


def condition_score(condition,auto,cycles,yaw,neutral_speed=None):
    strict=bool(auto['nonfoot_force_max_native']==0.)
    direction=True if condition not in ('left','right') else bool(yaw*(1 if condition=='left' else -1)>=.01)
    slowed=True
    if condition.startswith('speed'):
        if neutral_speed is None or not np.isfinite(neutral_speed) or neutral_speed<=0:raise ValueError('valid matched neutral speed required')
        slowed=bool(auto['forward_signed_native_per_s']<=.95*neutral_speed)
    return dict(original_five=bool(auto['all_gates']),all_six=bool(cycles['all_legs_completed']),strict_nonfoot_zero=strict,
                turn_sign_magnitude=direction,speed_reduction=slowed,
                eligible=bool(auto['all_gates'] and cycles['all_legs_completed'] and strict and direction and slowed))


def checked_load(fm,path,digest):
    model,payload=fm.load(path,digest)
    if not payload.get('curve') or payload['curve'][-1]['step']!=1000:raise ValueError('physical evaluation only fixed1000')
    return model,payload


def neutral_reference(path,args,identity,fm,current_ident):
    if path is None:raise ValueError('--neutral-reference required for speed')
    path=Path(path).resolve();r=json.loads(path.read_text())
    if r.get('condition')!='neutral' or not r.get('engineering_pass') or r.get('checkpoint_sha256')!=args.sha256:
        raise ValueError('neutral must pass first at same checkpoint')
    if Path(r['checkpoint']).resolve()!=Path(args.checkpoint).resolve():raise ValueError('neutral checkpoint path mismatch')
    manifest=json.loads((path.parent/'IDENTITY.json').read_text());identity.verify(manifest)
    if manifest['compiled_arrays']!=current_ident['compiled_arrays']:raise ValueError('neutral physical identity mismatch')
    if r['source_hashes']!=current_ident['config']['evaluation_source_hashes']:raise ValueError('neutral evaluation source mismatch')
    for name,key in [('trace_1ms.npz','trace'),('neural_decisions.npz','neural_decisions')]:
        if fm.sha256(path.parent/name)!=r[key]['sha256']:raise ValueError('neutral saved evidence changed')
    return dict(path=str(path),sha256=fm.sha256(path),forward_signed_native_per_s=r['autonomous']['forward_signed_native_per_s'])


def evaluate(args):
    import torch
    torch.set_num_threads(1);torch.manual_seed(20261110)
    if not Path(args.checkpoint).is_absolute():raise ValueError('--checkpoint must be absolute')
    if args.neutral_reference is not None and not Path(args.neutral_reference).is_absolute():raise ValueError('--neutral-reference must be absolute')
    sw,identity,fm=pin();model,payload=checked_load(fm,args.checkpoint,args.sha256)
    sw.teacher_preflight();sw.student_preflight()
    out=Path(args.out).resolve()
    if not Path(args.out).is_absolute() or not out.is_relative_to(HERE):raise ValueError('fresh absolute output within new root required')
    if out.exists():raise FileExistsError(out)
    c=None;recorder=None;tr=None;rows=[];yaws=[];ident=None;error=None;result=None
    source_paths=[HERE/n for n in ('evaluate_fast.py','verify_fast_actual.py','EVAL_PREREG.json','PREREG.json','fast_motor.py')]
    sources={str(p):fm.sha256(p) for p in source_paths}
    prereg=json.loads((HERE/'EVAL_PREREG.json').read_text())
    if prereg['conditions']!=COMMANDS or prereg['checkpoint_step']!=1000:raise ValueError('evaluation prereg drift')
    out.mkdir(parents=True,exist_ok=False)
    def dump(name,data):identity.atomic_json(out/name,jsonable(data))
    try:
        recorder=sw.NativeRecorder(out/'native') if args.native else None
        c=sw.Core(adhesion=False,recorder=recorder)
        original,_=sw.load_student(fm.CHECKPOINT,c)
        if not all(torch.equal(v,model.base.state_dict()[k]) for k,v in original.actor.state_dict().items()):raise ValueError('fast base differs from verified runtime')
        import mujoco as mj
        mesh=c.plant.geom_type==mj.mjtGeom.mjGEOM_MESH
        if mesh.sum()!=71 or not np.all(c.plant.geom_rgba[mesh,3]==1.):raise ValueError('require71opaque meshes')
        ident=identity.capture(c,dict(kind='fixed1000 fast neural motor actual evaluation',cli=vars(args),condition=args.condition,command=COMMANDS[args.condition],evaluation_source_hashes=sources,timing=prereg['timing']),extra_paths=source_paths+[Path(args.checkpoint).resolve()]+[Path(p) for p in payload['config']['sources']])
        dump('IDENTITY.json',ident)
        runtime_path=out/'runtime_geometry.npz'
        np.savez_compressed(runtime_path,geom_type=c.plant.geom_type,geom_rgba=c.plant.geom_rgba)
        ref=neutral_reference(args.neutral_reference,args,identity,fm,ident) if args.condition.startswith('speed') else None
        tr=sw.Trace(full=True)
        def tick(action,source):
            tr.tick(c,action,source)
            rot=c.d.xmat[c.th].reshape(3,3);yaws.append(float(np.arctan2(rot[1,0],rot[0,0])))
        for _ in range(1000):tick(np.zeros(42),sw.SOURCE['settle'])
        teacher=sw.StaticTeacher(c,sw.teacher_config(3))
        previous=np.zeros(42);state=model.initial_state(1);observer=sw.tracker(c);events=[];transition=None;teacher_calls=0
        for k in range(2880):
            warm=k<960
            if k==960:
                before=sw.continuity_snapshot(c,previous,state)
                observer=sw.tracker(c);events=[]
                after=sw.continuity_snapshot(c,previous,state)
                exact={key:bool(np.array_equal(before[key],after[key])) for key in before}
                if not all(exact.values()):raise ValueError('withdrawal state discontinuity')
                transition=dict(time_s=float(c.d.time),all_exact=True,exact=exact,before_sha256=sw.snapshot_hashes(before),after_sha256=sw.snapshot_hashes(after),no_reset=True)
                # Persist before ANY autonomous step or final summary serialization.
                dump('HANDOVER.json',transition)
            x,_=sw.observations(c,previous,(2880-k)*.005)
            command=np.zeros(2) if warm else np.asarray(COMMANDS[args.condition],dtype=np.float64)
            hb=state.clone()
            with torch.no_grad():mean,state=model.step(torch.tensor(x[None],dtype=torch.float64),torch.tensor(command[None],dtype=torch.float64),state)
            policy=mean[0].tanh().numpy()
            if warm:
                action,_=teacher.action(c,k*.005);teacher_calls+=1;source=sw.SOURCE['teacher']
            else:action=policy;source=sw.SOURCE['student']
            rows.append(dict(time=float(c.d.time),X238=x,command=command.copy(),hidden_before=hb[0].numpy(),hidden_after=state[0].numpy(),mean=mean[0].numpy(),policy_action=policy,actually_applied_action=action.copy(),control_source=int(source)))
            for _ in range(5):
                tick(action,source)
                if not warm:events.extend(observer.measure(c)['events'])
            previous=action.copy()
        auto=tr.summary(c,sw.SOURCE['student']);cycles=sw.cycle_report(observer,events)
        yaw=np.unwrap(np.asarray(yaws));delta=float(yaw[-1]-yaw[5799])
        score=condition_score(args.condition,auto,cycles,delta,None if ref is None else ref['forward_signed_native_per_s'])
        result=dict(status='ENGINEERING_ONLY_NOT_FULL_NATIVE_CERTIFICATION',engineering_pass=score['eligible'],criteria=score,condition=args.condition,checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_sha256=args.sha256,
                    command=COMMANDS[args.condition],autonomous=auto,autonomous_cycles=cycles,yaw_change_rad=delta,neutral_reference=ref,
                    transition=transition,physical_steps=len(tr.rec)*80,teacher_calls=teacher_calls,teacher_calls_after_handover=0,
                    mesh_count=71,all_mesh_opaque=True,extra_forces=False,source_hashes=sources,identity_sha256=ident['identity_sha256'],
                    native_audit='NOT_PERFORMED',food_success=False)
        if args.condition=='neutral':
            with np.load(ORIGINAL/'neural_decisions.npz',allow_pickle=False) as z:old={k:z[k].copy() for k in z.files}
            parity={key:bool(np.array_equal(np.asarray([r[key] for r in rows]),old[key])) for key in ('X238','hidden_before','hidden_after','mean','policy_action','actually_applied_action','control_source')}
            result['neutral_baseline_neural_parity']=parity
            if not all(parity.values()):raise ValueError('neutral actual baseline neural mismatch')
        identity.verify(ident);identity.verify_runtime(ident,c);fm.verify_hashes(sources);checked_load(fm,args.checkpoint,args.sha256)
    except BaseException as exc:
        error=exc
    finally:
        # Preserve every completed endpoint and all attempted decisions, even on failure.
        artifacts={}
        try:
            if tr is not None and tr.rows:artifacts['trace']=tr.save(out/'trace_1ms.npz')
            if rows:
                path=out/'neural_decisions.npz'
                np.savez_compressed(path,**{key:np.asarray([r[key] for r in rows]) for key in rows[0]},yaw_after_each_1ms=np.asarray(yaws))
                artifacts['neural_decisions']=dict(path=str(path),sha256=fm.sha256(path),rows=len(rows))
            if recorder is not None:artifacts['native']=recorder.close()
            if (out/'runtime_geometry.npz').exists():artifacts['runtime_geometry']=dict(path=str(out/'runtime_geometry.npz'),sha256=fm.sha256(out/'runtime_geometry.npz'))
            if error is not None:
                dump('FAILURE.json',dict(status='INCOMPLETE_OR_FAILED',error_type=type(error).__name__,error=str(error),engineering_pass=False,completed_endpoints=0 if tr is None else len(tr.rec),attempted_decisions=len(rows),artifacts=artifacts,source_hashes=sources))
            elif result is not None:
                result.update(artifacts);dump('RESULT.json',result)
        except BaseException as save_error:
            if error is None:error=save_error
            else:print('Additional evidence save failure: '+repr(save_error),file=sys.stderr,flush=True)
    if error is not None:raise error
    print(json.dumps(jsonable(dict(out=str(out),condition=args.condition,engineering_pass=result['engineering_pass'],gates=result['autonomous']['gates'],slip=result['autonomous']['slip_ratio_max'],yaw=result['yaw_change_rad']))),flush=True)
    return result


def parser():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--sha256',required=True);p.add_argument('--condition',choices=list(COMMANDS),required=True);p.add_argument('--out',required=True);p.add_argument('--neutral-reference');p.add_argument('--native',action='store_true');return p

if __name__=='__main__':evaluate(parser().parse_args())
