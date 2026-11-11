"""Draft closed-loop food evaluator. No physics on import; launch only after parent qualification."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import traceback
import numpy as np
from contract import digest,layout_from,source_center,qualification,Arrival

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
BASE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')


def pin():
    for name in ('physics','static_teacher','wave_reference','models','student_wave','identity','cycle_reward'):
        if name in sys.modules and Path(sys.modules[name].__file__).resolve()!=BASE/(name+'.py'):raise RuntimeError('conflicting baseline import '+name)
    sys.path.insert(0,str(BASE));sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'residual_candidate'))
    import evaluate_candidate
    if Path(evaluate_candidate.__file__).resolve()!=ROOT/'residual_candidate/evaluate_candidate.py':raise RuntimeError('wrong residual checked loader')
    import odor_loop
    import steering_behavior
    import student_wave as sw
    import identity
    for module,path in ((odor_loop,ROOT/'odor_loop.py'),(steering_behavior,ROOT/'steering_behavior.py'),(sw,BASE/'student_wave.py'),(identity,BASE/'identity.py')):
        if Path(module.__file__).resolve()!=path:raise RuntimeError('wrong module '+str(path))
    return evaluate_candidate,odor_loop,steering_behavior,sw,identity


def load_behavior(path,expected_sha,module):
    import torch
    if digest(path)!=expected_sha:raise ValueError('behavior checkpoint digest mismatch')
    p=torch.load(path,map_location='cpu',weights_only=False)
    model=module.SteeringBehavior()
    if p.get('format_version')!=1 or p.get('kind')!='food_steering_behavior' or p.get('identity')!=model.identity():raise ValueError('behavior schema/identity mismatch')
    for group in ('source_hashes','data_hashes'):
        entries=p['config'].get(group)
        if not entries:raise ValueError('behavior '+group+' must be nonempty')
        for filename,sha in entries.items():
            if not Path(filename).is_absolute() or digest(filename)!=sha:raise ValueError('behavior '+group+' drift '+filename)
    source_path=str(ROOT/'steering_behavior.py')
    if p['config']['source_hashes'].get(source_path)!=digest(source_path):raise ValueError('behavior actual module source not bound')
    model.load_state_dict(p['state_dict'],strict=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('nonfinite behavior')
    return model.eval(),p


def flat(state):
    import torch
    return (torch.cat(state,dim=-1) if isinstance(state,(tuple,list)) else state)[0].detach().numpy().copy()


def prefix_digest(trace):
    if len(trace.rows)<5800:raise ValueError('complete5800ms prefix required')
    h=hashlib.sha256()
    for key in ('time','qpos','qvel','target','actualctrl','action','control_source'):
        a=np.asarray([row[key] for row in trace.rows[:5800]])
        h.update(key.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def absolute_inputs(args):
    for key in ('motor_checkpoint','behavior_checkpoint','qualification','layout','out'):
        if not Path(getattr(args,key)).is_absolute():raise ValueError(key+' must be absolute')
    for key in ('motor_sha256','behavior_sha256'):
        value=getattr(args,key)
        if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('exact lowercase digest required')
    if args.control=='replay' and (not args.replay_file or not args.replay_sha256):raise ValueError('replay file and digest required')
    if args.control!='replay' and (args.replay_file or args.replay_sha256):raise ValueError('replay arguments only for replay control')
    if args.replay_file and not Path(args.replay_file).is_absolute():raise ValueError('replay file must be absolute')


def evaluate(args):
    absolute_inputs(args)
    out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    layout=layout_from(args.layout)
    q=json.loads(Path(args.qualification).read_text())
    qualified=qualification(q,args.motor_checkpoint,args.motor_sha256)
    if not qualified and not args.diagnostic:raise ValueError('motor qualification absent/mismatched; diagnostic explicit flag required')
    out.mkdir(parents=True,exist_ok=False)
    c=tr=ident=None;rows=[];geometry=[];identity_api=None;result=None;failure=None;verify_error=None
    try:
        import torch
        torch.set_num_threads(1);torch.manual_seed(layout['seed'])
        candidate,odor,behavior_module,sw,identity_api=pin()
        motor,motor_payload=candidate.load_checked(args.motor_checkpoint,args.motor_sha256)
        behavior,behavior_payload=load_behavior(args.behavior_checkpoint,args.behavior_sha256,behavior_module)
        replay=None
        if args.control=='replay':
            if digest(args.replay_file)!=args.replay_sha256:raise ValueError('replay digest mismatch')
            with np.load(args.replay_file,allow_pickle=False) as z:
                required={'X238':(2880,238),'actually_applied_action':(2880,42),'motor_hidden_before':(2880,192),'motor_hidden_after':(2880,192)}
                replay={key:z[key].copy() for key in required}
                replay['physical_prefix_sha256']=str(z['physical_prefix_sha256'].item())
            if any(replay[key].shape!=shape or not np.isfinite(replay[key]).all() for key,shape in required.items()):raise ValueError('replay count/schema/nonfinite')
            if np.max(np.abs(replay['actually_applied_action']))>1:raise ValueError('replay unbounded action')
        sw.teacher_preflight();sw.student_preflight()
        c=sw.Core(adhesion=False);original,_=sw.load_student(candidate.CHECKPOINT,c)
        if not all(torch.equal(v,motor.base.state_dict()[k]) for k,v in original.actor.state_dict().items()):raise ValueError('motor frozen base differs from original runtime')
        del original
        import mujoco
        mesh=np.asarray(c.plant.geom_type)==int(mujoco.mjtGeom.mjGEOM_MESH)
        visible=int(np.sum(mesh & (np.asarray(c.plant.geom_rgba)[:,3]==1.)))
        if int(mesh.sum())!=71 or visible!=71:raise ValueError('all71 opaque fly meshes required')
        paths=[Path(__file__),HERE/'contract.py',HERE/'PREREG_DRAFT.json',Path(args.motor_checkpoint),Path(args.behavior_checkpoint),Path(args.qualification),Path(args.layout),ROOT/'odor/HEAD_GEOMETRY_AUDIT.json']
        if args.replay_file:paths.append(Path(args.replay_file))
        for payload in (motor_payload,behavior_payload):
            paths += [Path(p) for group in ('source_hashes','data_hashes') for p in payload['config'][group]]
        ident=identity_api.capture(c,dict(kind='draft actual closedloop food evaluation',cli=vars(args),layout=layout,motor_qualified=qualified,
            prereg=json.loads((HERE/'PREREG_DRAFT.json').read_text()),actor_inputs='receptors2 andactualphysical238 only',source_units='engineering NOT_ESTABLISHED SI'),extra_paths=paths)
        identity_api.atomic_json(out/'IDENTITY.json',ident)
        policy=odor.FoodNeuralPolicy(motor,behavior) # Initialized at physical reset, held during settle and behavior held during prime.
        tr=sw.Trace(full=True);sw.settle(c,tr)
        teacher=sw.StaticTeacher(c,sw.teacher_config(3))
        previous=np.zeros(42);observer=sw.tracker(c);events=[];sensor=None;source=None;handover=None;withdrawal=None;arrival=Arrival(layout['arrival_dwell_s'])
        teacher_calls=0;odor_calls=0
        replay_source=max(sw.SOURCE.values())+1
        auto_source=replay_source if args.control=='replay' else sw.SOURCE['student']
        source_codes={**sw.SOURCE,'food_replay':replay_source}
        for k in range(2880):
            warm=k<960
            if k==960:
                if replay is not None and prefix_digest(tr)!=replay['physical_prefix_sha256']:raise ValueError('replay actual physical prefix differs')
                before=sw.continuity_snapshot(c,previous,torch.as_tensor(np.r_[flat(policy.motor_state),flat(policy.behavior_state)][None]))
                withdrawal=float(c.d.time)
                # Geometry only: obtain actual handover sensor midpoint before creating the source.
                dummy=odor.FoodSource(center_xy=(0.,0.),patch_radius=layout['patch_radius'],amplitude=layout['amplitude'],sigma=layout['sigma'],onset_s=withdrawal,rise_s=layout['rise_s'])
                temporary=odor.FoodObservation(c,dummy,seed=layout['seed'])
                position=odor.bilateral_positions(c.d.xpos[temporary.head],c.d.xmat[temporary.head].reshape(3,3),temporary.forward,temporary.lateral)
                center=source_center(position.mean(axis=0),c.d.xmat[temporary.head].reshape(3,3),layout)
                source=odor.FoodSource(center_xy=tuple(center),patch_radius=layout['patch_radius'],amplitude=layout['amplitude'],sigma=layout['sigma'],onset_s=withdrawal+layout['onset_elapsed_s'],rise_s=layout['rise_s'])
                sensor=odor.FoodObservation(c,source,noise_std=layout['noise_std'],seed=layout['seed'])
                observer=sw.tracker(c);events=[]
                after=sw.continuity_snapshot(c,previous,torch.as_tensor(np.r_[flat(policy.motor_state),flat(policy.behavior_state)][None]))
                exact={key:bool(np.array_equal(before[key],after[key])) for key in before}
                if not all(exact.values()):raise ValueError('handover discontinuity')
                handover=dict(time_s=withdrawal,no_reset=True,exact=exact,before_sha256=sw.snapshot_hashes(before),after_sha256=sw.snapshot_hashes(after),actual_source_center_xy=center.tolist())
            x,_=sw.observations(c,previous,(2880-k)*.005)
            mb=flat(policy.motor_state);bb=flat(policy.behavior_state)
            receptors=np.zeros(2);raw=np.zeros(2)
            if warm:
                predicted=policy.prime(x);command=np.zeros(2)
                action,_=teacher.action(c,k*.005);code=sw.SOURCE['teacher'];teacher_calls+=1
            else:
                raw=sensor.receptors(c);odor_calls+=1
                receptors=np.zeros(2) if args.control=='noodor' else raw[::-1].copy() if args.control=='swap' else raw.copy()
                prediction=policy.step(x,receptors);predicted=prediction['action'];command=prediction['command']
                action=replay['actually_applied_action'][k].copy() if replay is not None else predicted.copy();code=auto_source
                if command[0]!=0 or not np.isfinite(command).all():raise ValueError('structural zero forward/finite command required')
            ma=flat(policy.motor_state);ba=flat(policy.behavior_state)
            if rows and (not np.array_equal(mb,rows[-1]['motor_hidden_after']) or not np.array_equal(bb,rows[-1]['behavior_hidden_after'])):raise ValueError('actual hidden chain broken')
            if warm and not np.array_equal(ba,np.zeros(32)):raise ValueError('behavior memory advanced during prime')
            if replay is not None and warm:
                for key,value in (('X238',x),('actually_applied_action',action),('motor_hidden_before',mb),('motor_hidden_after',ma)):
                    if not np.array_equal(value,replay[key][k]):raise ValueError('replay actual prefix mismatch '+key)
            if not warm and replay is None and not np.array_equal(action,predicted):raise ValueError('autonomous applied output mismatch')
            rows.append(dict(time=float(c.d.time),X238=x,raw_receptors=raw,receptors=receptors,command=command.copy(),motor_hidden_before=mb,motor_hidden_after=ma,
                behavior_hidden_before=bb,behavior_hidden_after=ba,policy_action=predicted.copy(),actually_applied_action=action.copy(),control_source=code))
            for _ in range(5):
                tr.tick(c,action,code)
                if not warm:
                    events.extend(observer.measure(c)['events'])
                    diag=sensor.diagnostics(c);elapsed=float(c.d.time)-withdrawal
                    arrival.tick(elapsed,diag['inside_patch'],layout['onset_elapsed_s'])
                    geometry.append(dict(time=float(c.d.time),elapsed=elapsed,sensor_xy=np.asarray(diag['sensor_xy']),sampling_midpoint_xy=np.asarray(diag['mouth_proxy_xy']),
                        edge_distance=diag['edge_distance'],inside_patch=diag['inside_patch'],source_center_xy=np.asarray(source.center_xy),control_source=code))
            previous=action.copy()
        counts={str(code):tr.sources.count(code) for code in set(tr.sources)}
        expected={str(sw.SOURCE['settle']):1000,str(sw.SOURCE['teacher']):4800,str(auto_source):9600}
        if counts!=expected or len(rows)!=2880 or len(geometry)!=9600 or teacher_calls!=960 or odor_calls!=1920:raise ValueError('finite actual phase/source counts mismatch')
        auto=tr.summary(c,auto_source);cycles=sw.cycle_report(observer,events);arrival_report=arrival.report()
        passed=bool(auto['all_gates'] and cycles['all_legs_completed'] and arrival_report['arrived'])
        result=dict(status='MEASURED_DRAFT_ENGINEERING_NOT_FULL_NATIVE_CERTIFICATION',acceptance=bool(passed and qualified and args.control=='normal' and not args.diagnostic),
            measured_gates_cycles_arrival=passed,control=args.control,diagnostic=args.diagnostic,motor_qualified=qualified,neural_autonomy=args.control!='replay',
            autonomous=auto,autonomous_cycles=cycles,arrival=arrival_report,ingestion=False,handover=handover,source_codes=source_codes,actual_source_counts_1ms=counts,
            teacher_calls=teacher_calls,teacher_calls_after_handover=0,actual_odor_calls=odor_calls,physical_steps=len(tr.rec)*80,all71_visible_meshes=visible,
            source_non_depleting=True,source_count=1,actor_source_lookups=0,extra_forces=False,hidden_chain_exact=True,layout=layout,
            motor_checkpoint=str(Path(args.motor_checkpoint)),motor_checkpoint_sha256=args.motor_sha256,behavior_checkpoint_sha256=args.behavior_sha256,
            replay_neural_autonomy_forbidden=args.control=='replay',replay_prefix_exact=None if replay is None else True,identity_sha256=ident['identity_sha256'])
    except BaseException as exc:
        failure=dict(error=repr(exc),traceback=traceback.format_exc(),acceptance=False,status='FAILED_PARTIAL',actual_decisions=len(rows),actual_geometry_rows=len(geometry))
    finally:
        try:
            if tr is not None and tr.rows:tr.save(out/('trace_1ms.npz' if failure is None else 'partial_trace_1ms.npz'))
            for name,data in (('neural_decisions',rows),('geometry_1ms',geometry)):
                if data:
                    arrays={key:np.asarray([r[key] for r in data]) for key in data[0]}
                    if name=='neural_decisions' and tr is not None and len(tr.rows)>=5800:arrays['physical_prefix_sha256']=np.asarray(prefix_digest(tr))
                    np.savez_compressed(out/(name+'.npz'),**arrays)
        except BaseException as exc:verify_error='partial evidence save: '+repr(exc)
        try:
            if ident is not None:identity_api.verify_runtime(ident,c)
            if digest(args.motor_checkpoint)!=args.motor_sha256 or digest(args.behavior_checkpoint)!=args.behavior_sha256:raise ValueError('checkpoint drift during run')
        except BaseException as exc:verify_error=(verify_error+'; ' if verify_error else '')+'source/runtime verification: '+repr(exc)
        if failure is not None or verify_error is not None:
            report=failure or dict(acceptance=False,status='FAILED_FINALIZATION')
            report['finalization_error']=verify_error
            (out/'FAILURE.json').write_text(json.dumps(report,indent=2)+'\n')
        elif result is not None:
            result['files']={str(p):digest(p) for p in out.iterdir() if p.is_file()}
            identity_api.atomic_json(out/'RESULT.json',result)
    if failure is not None or verify_error is not None:raise RuntimeError('food evaluation failed; inspect '+str(out/'FAILURE.json'))
    print(json.dumps(dict(out=str(out),control=args.control,acceptance=result['acceptance'],arrival=result['arrival'])))
    return result


def main():
    p=argparse.ArgumentParser()
    for option in ('motor-checkpoint','motor-sha256','behavior-checkpoint','behavior-sha256','qualification','layout','out'):p.add_argument('--'+option,required=True)
    p.add_argument('--control',choices=['normal','noodor','swap','replay'],required=True);p.add_argument('--diagnostic',action='store_true')
    p.add_argument('--replay-file');p.add_argument('--replay-sha256')
    evaluate(p.parse_args())

if __name__=='__main__':main()
