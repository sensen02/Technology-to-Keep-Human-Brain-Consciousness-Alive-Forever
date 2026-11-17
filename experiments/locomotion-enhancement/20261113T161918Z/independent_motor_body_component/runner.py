"""Parent-only final600 motor/body component diagnostic; no sensory/brain/neck."""
import argparse
import hashlib
import random
import sys
import time
from common import *

TASKS={'walk':[1.,0.],'stop':[0.,0.],'left_pivot':[0.,1.],'right_pivot':[0.,-1.]}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--condition',choices=tuple(TASKS),required=True);parser.add_argument('--execute-parent-physical',action='store_true');args=parser.parse_args()
    if not args.execute_parent_physical:parser.error('Explicit parent component physical release required')
    if args.condition not in CONDITIONS or CONDITIONS[args.condition]!=TASKS[args.condition]:raise ValueError('Fixed four independent task commands required')
    cpu1();sources=frozen();verify(sources);pre=read(ROOT/'PREFLIGHT.json');binding=read(ROOT/'INPUT_BINDINGS.json')
    if not pre['all_passed'] or pre['source_hashes']!=sources:raise ValueError('Actual same-source component preflight required')
    # Strict checkpoint validation precedes every physical backend import.
    import torch
    import numpy as np
    torch.manual_seed(20261113);np.random.seed(20261113);random.seed(20261113)
    motor,payload=load_motor();binding=read(ROOT/'INPUT_BINDINGS.json')
    if pre['actual_checkpoint_sha256']!=binding['checkpoint_sha256'] or sha(binding['checkpoint'])!=binding['checkpoint_sha256']:raise ValueError('Actual preflight/checkpoint SHA must match current binding')
    if payload['step']!=600 or payload['format_version']!=binding['motor_format'] or motor.identity()['state_dims']!=[128,64,64,96]:raise ValueError('Actual final600 independent352 ABI required')
    if any(k in sys.modules for k in ('mujoco','physics','flygym','student_wave')):raise ValueError('Motor must load before physics')
    sw,physics,identity,static,Peak,ast_ids=physical_stack()
    import mujoco as mj
    out=ROOT/'trials'/args.condition;out.mkdir(parents=True,exist_ok=False)
    c=None;trace=None;recorder=Peak();neural=[];geometry=[];result=None;error=None;early_stop=None;unsafe=0;teacher_calls=0;started=time.monotonic();boundary=None;body=None;handover=None
    try:
        c=physics.Core(adhesion=False,recorder=recorder)
        mesh=c.plant.geom_type==mj.mjtGeom.mjGEOM_MESH
        if int(mesh.sum())!=71 or not np.all(c.plant.geom_rgba[mesh,3]==1.):raise ValueError('All71 complete body meshes must be opaque')
        ident=identity.capture(c,dict(kind=FORMAT,condition=args.condition,component_only=True,seed=20261113,motor_checkpoint=binding['checkpoint'],motor_sha256=binding['checkpoint_sha256'],state_dims=[128,64,64,96],source_hashes=sources,selected_legacy_ast=ast_ids),extra_paths=list(sources)+[ROOT/'PREFLIGHT.json'])
        save(out/'IDENTITY.json',ident)
        save(out/'INITIAL_STATE.json',dict(time_s=float(c.d.time),**snapshot(c)))
        save(out/'RNG_INITIAL.json',dict(seed=20261113,python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state().tolist()))
        np.savez_compressed(out/'runtime_geometry.npz',geom_type=c.plant.geom_type.copy(),geom_rgba=c.plant.geom_rgba.copy())
        trace=sw.Trace(full=True);state=motor.initial_state(1);previous=np.zeros(42);teacher=None;head=c.plant.body('Head').id
        for k in range(3080):
            ms=k*5
            if abs(float(c.d.time)-ms*.001)>1e-8:raise ValueError('Contiguous5ms body/NN clock required')
            settle=ms<1000;warm=1000<=ms<5800
            if ms==1000:
                body=body_length(c);save(out/'BODY_LENGTH.json',body)
                teacher=static.StaticTeacher(c,sw.teacher_config(3))
            if ms==5800:
                before=snapshot(c);hidden_before=tuple(v.detach().clone() for v in state);previous_before=previous.copy()
                boundary=dict(time=float(c.d.time),com=c.d.subtree_com[c.th].copy(),yaw=yaw(c),no_reset=True)
                after=snapshot(c);assert_same(before,after)
                hidden_exact=all(torch.equal(a,b) for a,b in zip(hidden_before,state))
                if not hidden_exact or not np.array_equal(previous_before,previous) or bool(state[3].abs().max()):raise ValueError('352handover reset or new96advanced duringwarm')
                handover=dict(**boundary,actual_state_exact=True,neural_state_exact=hidden_exact,new96_zero=True,state_dims=[128,64,64,96],no_state_reset=True,physical_before_sha256={key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in before.items()},physical_after_sha256={key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in after.items()},hidden_before_sha256=hashlib.sha256(torch.cat(hidden_before,-1).numpy().tobytes()).hexdigest(),hidden_after_sha256=hashlib.sha256(torch.cat(state,-1).numpy().tobytes()).hexdigest(),scope='Old256 teacherwarm -> ANN352 task, no brain/sensory/neck model or ports')
                save(out/'HANDOVER.json',handover)
            x=c.obs238(previous,target_speed=.075).copy();hb=torch.cat(state,-1)[0].detach().numpy().copy()
            command=np.zeros(2) if settle else (np.array([1.,0.]) if warm else np.asarray(TASKS[args.condition],dtype=float))
            xt=torch.tensor(x[None],dtype=torch.float64);ut=torch.tensor(command[None],dtype=torch.float64)
            with torch.no_grad():
                if settle:
                    mean=torch.zeros((1,42),dtype=torch.float64);residual=torch.zeros_like(mean)
                elif warm:
                    mean,state=motor.prime_old(xt,ut,state);residual=torch.zeros_like(mean)
                    if bool(state[3].abs().max()):raise ValueError('New96must remain zero in explicit oldwarm')
                else:mean,state,residual=motor.step_details(xt,ut,state)
                oldmean=mean-residual;policy=mean.tanh()[0].numpy().copy()
            if settle:action=np.zeros(42);source=sw.SOURCE['settle']
            elif warm:action,_=teacher.action(c,(ms-1000)*.001);action=sw.bounded_action(action);source=sw.SOURCE['teacher'];teacher_calls+=1
            else:action=sw.bounded_action(policy);source=SOURCE_CANDIDATE
            ha=torch.cat(state,-1)[0].detach().numpy().copy()
            neural.append(dict(time=float(c.d.time),physical_step_ms=ms,X238=x,command=command.copy(),mean=mean[0].numpy().copy(),rawmean=mean[0].numpy().copy(),oldmean=oldmean[0].numpy().copy(),residual=residual[0].numpy().copy(),hidden_before=hb,hidden_after=ha,policy_action=policy,actually_applied_action=action.copy(),control_source=source))
            for _ in range(5):
                measured=trace.tick(c,action,source)
                if ms>=1000:
                    rot=c.d.xmat[c.th].reshape(3,3).copy()
                    geometry.append(dict(time=float(c.d.time),thorax_yaw=yaw(c),thorax_rotation=rot,thorax_xyz=c.d.xpos[c.th].copy(),head_xyz=c.d.xpos[head].copy(),head_rotation=c.d.xmat[head].reshape(3,3).copy(),foot_world=measured['foot']@rot.T+c.d.xpos[c.th],control_source=source))
                # Preserve the preregistered common warm/autonomous unsafe run.
                unsafe=unsafe+1 if measured['com']<.8 or measured['nonfoot']>.5*physics.W else 0
                if unsafe>=100:
                    early_stop=dict(time_s=float(c.d.time),reason='100ms COM<.8 or nonfoot>.5W',during_candidate=source==SOURCE_CANDIDATE);break
            previous=action.copy()
            if k%240==0:print(json.dumps(dict(condition=args.condition,decision=k,actual_s=float(c.d.time),wall_s=time.monotonic()-started,label='INDEPENDENT_MOTOR_BODY_COMPONENT_NO_NECK')),flush=True)
            if early_stop is not None:break
        native=recorder.close()
        if native['post_native_steps']==0:native['post_COM_min']=None
        arrays={key:np.asarray([row[key] for row in trace.rows]) for key in trace.rows[0]}
        garrays={key:np.asarray([row[key] for row in geometry]) for key in geometry[0]} if geometry else {}
        metrics=score_from_arrays(arrays,garrays,boundary,body['length_native'],physics.W,args.condition,native) if boundary is not None else None
        legacy=trace.summary(c,SOURCE_CANDIDATE)
        identity.verify_runtime(ident,c);verify(sources);motor.verify_old()
        result=dict(kind=FORMAT,condition=args.condition,command=TASKS[args.condition],checkpoint=binding['checkpoint'],checkpoint_sha256=binding['checkpoint_sha256'],motor_format=payload['format_version'],final_step=payload['step'],state_dims=[128,64,64,96],state_total=352,neural_records=len(neural),actual_ms=len(trace.rows),actual_native_steps=len(trace.rows)*80,native=native,teacher_calls=teacher_calls,teacher_calls_after_handover=0,candidate_metrics=metrics,original_five=legacy,body_length=body,candidate_start=boundary,transition=handover,early_stop=early_stop,full_duration=early_stop is None and len(trace.rows)==15400,weight_native=physics.W,source_hashes=sources,identity_sha256=ident['identity_sha256'],selected_legacy_ast=ast_ids,all71opaque=True,physical_component_only=True,brain_or_sensory_models_loaded=False,neck_ports_present=False,no_root_or_extra_force_control=True,training_updates=0,wall_s=time.monotonic()-started,scope='Explicit body command diagnostic of ANN352 only; no neckdeployment or nativebiology inference. OldworldX/slip gates preserved separately.')
    except BaseException as exc:error=exc
    finally:
        artifacts={}
        if trace is not None and trace.rows:artifacts['trace']=trace.save(out/'trace_1ms.npz')
        for filename,rows,key in (('neural_decisions.npz',neural,'neural_decisions'),('geometry_1ms.npz',geometry,'geometry')):
            if rows:
                path=out/filename;np.savez_compressed(path,**{name:np.asarray([row[name] for row in rows]) for name in rows[0]});artifacts[key]=dict(path=str(path),sha256=sha(path),rows=len(rows))
        for filename,key in (('INITIAL_STATE.json','initial_state'),('BODY_LENGTH.json','body_length_artifact'),('HANDOVER.json','handover'),('runtime_geometry.npz','runtime_geometry'),('RNG_INITIAL.json','rng_initial')):
            path=out/filename
            if path.exists():artifacts[key]=dict(path=str(path),sha256=sha(path))
        if error is not None:
            native=recorder.close() if recorder.count else dict(actual_native_steps=0)
            if native.get('post_native_steps',0)==0 and 'post_COM_min' in native:native['post_COM_min']=None
            save(out/'FAILURE.json',dict(error=repr(error),artifacts=artifacts,native=native,actual_ms=0 if trace is None else len(trace.rows),neural_records=len(neural),no_overwrite_retry=True))
        elif result is not None:result.update(artifacts);save(out/'RESULT.json',result)
    if error is not None:raise error
    print(json.dumps(dict(condition=args.condition,result=str(out/'RESULT.json'),candidate_metrics=result['candidate_metrics'],early_stop=early_stop)),flush=True)


if __name__=='__main__':main()
