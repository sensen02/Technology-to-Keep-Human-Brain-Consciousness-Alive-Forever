"""Parent finite assisted collection. Never neural behavior acceptance."""
import argparse
import random
import time
from common import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=tuple(CASES),required=True);parser.add_argument('--execute-parent-physical',action='store_true');args=parser.parse_args()
    if not args.execute_parent_physical:parser.error('Explicit parent assisted collection release required')
    case=args.case;condition,command=CASES[case];assisted=case!='old_walk_control';source_candidate=7 if assisted else 8
    cpu1();sources=frozen();verify(sources);pre=read(ROOT/'PREFLIGHT.json')
    if not pre['all_passed'] or pre['source_hashes']!=sources:raise ValueError('Same-source actual preflight required')
    import torch
    import numpy as np
    torch.manual_seed(20261113);np.random.seed(20261113);random.seed(20261113)
    motor,payload=load_motor();binding=read(ROOT/'INPUT_BINDINGS.json')
    if payload['step']!=600 or payload['format_version']!=MOTOR_FORMAT or pre['actual_checkpoint_sha256']!=ACTUAL_SHA:raise ValueError('Actual007d preflight identity required')
    sw,physics,identity,static,Peak,ast_ids=physical_stack()
    import mujoco as mj
    out=ROOT/'trials'/case;out.mkdir(parents=True,exist_ok=False)
    recorder=Peak();c=None;trace=None;neural=[];geometry=[];warm_boundaries=[];result=None;error=None;early_stop=None;boundary=None;body=None;handover=None
    unsafe=0;warm_teacher_calls=0;candidate_teacher_calls=0;qualified_count=0;diagnostic_rows=0;started=time.monotonic()
    diagnostic_stream=(out/'teacher_diagnostics.jsonl').open('x')
    try:
        c=physics.Core(adhesion=False,recorder=recorder)
        mesh=c.plant.geom_type==mj.mjtGeom.mjGEOM_MESH
        if int(mesh.sum())!=71 or not np.all(c.plant.geom_rgba[mesh,3]==1.) or c.plant.nu!=42:raise ValueError('Complete71opaque42 body required')
        ident=identity.capture(c,dict(kind=FORMAT,label=LABEL,case=case,condition=condition,component_only=True,seed=20261113,motor_checkpoint=binding['checkpoint'],motor_sha256=ACTUAL_SHA,state_dims=[128,64,64,96],source_hashes=sources,selected_legacy_ast=ast_ids,teacher_assisted=assisted,student_weight=.25 if assisted else 0.),extra_paths=list(sources)+[ROOT/'PREFLIGHT.json'])
        save(out/'IDENTITY.json',ident);save(out/'INITIAL_STATE.json',dict(time_s=float(c.d.time),**snapshot(c)))
        save(out/'RNG_INITIAL.json',dict(seed=20261113,python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state().tolist()))
        np.savez_compressed(out/'runtime_geometry.npz',geom_type=c.plant.geom_type.copy(),geom_rgba=c.plant.geom_rgba.copy())
        trace=sw.Trace(full=True);state=motor.initial_state(1);previous=np.zeros(42);warm_teacher=None;teacher=None;head=c.plant.body('Head').id
        for k in range(3080):
            ms=k*5
            if abs(float(c.d.time)-ms*.001)>1e-8:raise ValueError('Contiguous5ms physical/NN clock required')
            settle=ms<1000;warm=1000<=ms<5800
            if ms==1000:
                body=body_length(c);save(out/'BODY_LENGTH.json',body)
                factory_before=snapshot(c);factory_hash=state_hashes(c);warm_teacher=static.StaticTeacher(c,sw.teacher_config(3));assert_same(factory_before,snapshot(c))
                warm_boundaries.append(dict(event='first_warm_factory',physical_ms=ms,before_sha256=factory_hash,after_sha256=state_hashes(c),target=c.actuator.target_rad.copy(),buffer=c.buf.copy(),hidden=torch.cat(state,-1)[0].detach().numpy().copy()))
            if ms==5800:
                before=snapshot(c);before_hash=state_hashes(c);hidden_before=tuple(v.detach().clone() for v in state);previous_before=previous.copy()
                boundary=dict(time=float(c.d.time),com=c.d.subtree_com[c.th].copy(),yaw=yaw(c),no_reset=True)
                if assisted:teacher=make_teacher(c,condition)
                after_hash=state_hashes(c);assert_same(before,snapshot(c))
                if not all(torch.equal(a,b) for a,b in zip(hidden_before,state)) or not np.array_equal(previous,previous_before) or bool(state[3].abs().max()):raise ValueError('Handover factory altered352/previous/new96zero')
                handover=dict(**boundary,actual_state_exact=True,neural_state_exact=True,new96_zero=True,state_dims=[128,64,64,96],no_state_reset=True,physical_before_sha256=before_hash,physical_after_sha256=after_hash,teacher_factory_before_sha256=before_hash,teacher_factory_after_sha256=after_hash,teacher_factory_created=assisted,hidden_before_sha256=hashlib.sha256(torch.cat(hidden_before,-1).numpy().tobytes()).hexdigest(),hidden_after_sha256=hashlib.sha256(torch.cat(state,-1).numpy().tobytes()).hexdigest(),scope=SCOPE)
                save(out/'HANDOVER.json',handover)
            x=c.obs238(previous,target_speed=.075).copy();hb=torch.cat(state,-1)[0].detach().numpy().copy()
            task=np.zeros(2) if settle else (np.array([1.,0.]) if warm else np.asarray(command,dtype=float))
            xt=torch.tensor(x[None],dtype=torch.float64);ut=torch.tensor(task[None],dtype=torch.float64)
            with torch.no_grad():
                if settle:mean=torch.zeros((1,42),dtype=torch.float64);residual=torch.zeros_like(mean);oldmean_direct=mean
                elif warm:mean,state=motor.prime_old(xt,ut,state);residual=torch.zeros_like(mean);oldmean_direct=mean
                else:mean,state,residual,oldmean_direct=shadow_details(motor,xt,ut,state)
                oldmean_compat=mean-residual;student_action=mean.tanh()[0].numpy().copy();direct_action=oldmean_direct.tanh()[0].numpy().copy()
            if warm and bool(state[3].abs().max()):raise ValueError('Warm new96 must remain zero')
            teacher_action=np.zeros(42);teacher_available=False;mask={name:False for name in MASK_NAMES};qualified=False
            if settle:action=np.zeros(42);source=sw.SOURCE['settle'];target=action.copy()
            elif warm:
                before=snapshot(c);action,_=warm_teacher.action(c,(ms-1000)*.001);assert_same(before,snapshot(c))
                action=sw.bounded_action(action);teacher_action=action.copy();teacher_available=True;target=action.copy();source=sw.SOURCE['teacher'];warm_teacher_calls+=1
            elif not assisted:
                action=sw.bounded_action(direct_action);target=direct_action.copy();source=source_candidate
            else:
                before=snapshot(c);pre_query_hashes=state_hashes(c);teacher_action,diag=teacher.action(c,(ms-5800)*.001);post_query_hashes=state_hashes(c);assert_same(before,snapshot(c))
                teacher_available=True;candidate_teacher_calls+=1
                if teacher_action.shape!=(42,) or not np.isfinite(teacher_action).all() or np.max(np.abs(teacher_action))>1.+1e-12:raise ValueError('Teacher finite bounded42 required; preserve partial failure')
                mask,unbacked=teacher_mask(c,teacher_action,diag,source_exact=True,actual_com=c.d.subtree_com[c.th],actual_nf=trace.rows[-1]['nonfoot'])
                qualified=bool(all(mask.values()));qualified_count+=int(qualified)
                action=sw.bounded_action(.75*teacher_action+.25*student_action);target=teacher_action.copy();source=source_candidate
                row=dict(time_s=float(c.d.time),physical_ms=ms,teacher_elapsed_s=(ms-5800)*.001,candidate_phase=True,diagnostic=diag,teacher_action=teacher_action,label_mask=mask,label_qualified=qualified,unbacked_planned_load_over_W=unbacked,pre_query_hashes=pre_query_hashes,post_query_hashes=post_query_hashes)
                diagnostic_stream.write(json.dumps(jsonable(row),ensure_ascii=False,allow_nan=False)+'\n');diagnostic_rows+=1
            ha=torch.cat(state,-1)[0].detach().numpy().copy()
            if ms in (1000,5795,5800):warm_boundaries.append(dict(event='neural_and_actual_boundary',physical_ms=ms,physical_sha256=state_hashes(c),target=c.actuator.target_rad.copy(),buffer=c.buf.copy(),hidden_before=hb.copy(),hidden_after=ha.copy(),actually_applied_action=action.copy()))
            neural.append(dict(time=float(c.d.time),physical_step_ms=ms,X238=x,command=task.copy(),mean=mean[0].numpy().copy(),rawmean=mean[0].numpy().copy(),oldmean=oldmean_compat[0].numpy().copy(),oldmean_compat=oldmean_compat[0].numpy().copy(),oldmean_direct=oldmean_direct[0].numpy().copy(),residual=residual[0].numpy().copy(),hidden_before=hb,hidden_after=ha,policy_action=student_action,student_action=student_action.copy(),actually_applied_action=action.copy(),mixed_applied_action=action.copy(),teacher_action=teacher_action.copy(),teacher_available=teacher_available,supervision_target=target,label_mask=np.asarray(list(mask.values()),dtype=bool),label_qualified=qualified,control_source=source))
            for _ in range(5):
                measured=trace.tick(c,action,source)
                if ms>=1000:
                    rot=c.d.xmat[c.th].reshape(3,3).copy();geometry.append(dict(time=float(c.d.time),thorax_yaw=yaw(c),thorax_rotation=rot,thorax_xyz=c.d.xpos[c.th].copy(),head_xyz=c.d.xpos[head].copy(),head_rotation=c.d.xmat[head].reshape(3,3).copy(),foot_world=measured['foot']@rot.T+c.d.xpos[c.th],control_source=source))
                unsafe=unsafe+1 if measured['com']<.8 or measured['nonfoot']>.5*physics.W else 0
                if unsafe>=100:early_stop=dict(time_s=float(c.d.time),reason='100ms COM<.8 or nonfoot>.5W',during_collection=source==source_candidate);break
            previous=action.copy()
            if k%240==0:print(json.dumps(dict(case=case,decision=k,actual_s=float(c.d.time),wall_s=time.monotonic()-started,label=LABEL)),flush=True)
            if early_stop is not None:break
        native=recorder.close()
        if native['post_native_steps']==0:native['post_COM_min']=None
        arrays={key:np.asarray([row[key] for row in trace.rows]) for key in trace.rows[0]};garrays={key:np.asarray([row[key] for row in geometry]) for key in geometry[0]} if geometry else {}
        metrics=score(arrays,garrays,boundary,body['length_native'],physics.W,condition,native,source_candidate) if boundary is not None else None
        legacy=trace.summary(c,source_candidate);identity.verify_runtime(ident,c);verify(sources);motor.verify_old()
        result=dict(kind=FORMAT,label=LABEL,scope=SCOPE,case=case,condition=condition,command=command,checkpoint=binding['checkpoint'],checkpoint_sha256=ACTUAL_SHA,motor_format=payload['format_version'],final_step=600,state_dims=[128,64,64,96],state_total=352,neural_records=len(neural),actual_ms=len(trace.rows),actual_native_steps=len(trace.rows)*80,native=native,teacher_calls=warm_teacher_calls+candidate_teacher_calls,warm_teacher_calls=warm_teacher_calls,teacher_calls_after_handover=candidate_teacher_calls,teacher_diagnostic_rows=diagnostic_rows,qualified_teacher_labels=qualified_count,label_mask_names=list(MASK_NAMES),teacher_weight=.75 if assisted else 0.,student_weight=.25 if assisted else 0.,oldcontrol_weight=0. if assisted else 1.,teaching_label_target='unmixed_full_teacher_action' if assisted else 'direct_frozen_oldmean_tanh',oldmean_source='direct m.cached_features frozenold.step once; never mean-residual',student_shadow_only=True if not assisted else False,source_candidate=source_candidate,teacher_assisted=assisted,candidate_metrics=metrics,original_five=legacy,body_length=body,candidate_start=boundary,transition=handover,early_stop=early_stop,full_duration=early_stop is None and len(trace.rows)==15400,weight_native=physics.W,source_hashes=sources,identity_sha256=ident['identity_sha256'],selected_legacy_ast=ast_ids,all71opaque=True,physical_component_only=True,brain_or_sensory_models_loaded=False,neck_ports_present=False,no_root_or_extra_force_control=True,training_updates=0,neural_acceptance=False,data_qualified_for_training=False,teacher_label_replay_required_before_training=assisted,wall_s=time.monotonic()-started,label_kind='CURRENT_STATE_FULL_TEACHER_ACTION_WITH_CONSERVATIVE_MASK' if assisted else 'FROZEN_OLD_MATCHED_CONTROL_NO_TEACHER_MASK')
    except BaseException as exc:error=exc
    finally:
        diagnostic_stream.flush();diagnostic_stream.close();artifacts={}
        save(out/'WARM_BOUNDARIES.json',dict(rows=warm_boundaries,scope='Actual first/last warm and handover physical target/buffer and neural snapshots; no reset'))
        artifacts['warm_boundaries']=dict(path=str(out/'WARM_BOUNDARIES.json'),sha256=sha(out/'WARM_BOUNDARIES.json'))
        if trace is not None and trace.rows:artifacts['trace']=trace.save(out/'trace_1ms.npz')
        for filename,rows,key in (('neural_decisions.npz',neural,'neural_decisions'),('geometry_1ms.npz',geometry,'geometry')):
            if rows:
                path=out/filename;np.savez_compressed(path,**{name:np.asarray([row[name] for row in rows]) for name in rows[0]});artifacts[key]=dict(path=str(path),sha256=sha(path),rows=len(rows))
        for filename,key in (('INITIAL_STATE.json','initial_state'),('BODY_LENGTH.json','body_length_artifact'),('HANDOVER.json','handover'),('runtime_geometry.npz','runtime_geometry'),('RNG_INITIAL.json','rng_initial')):
            path=out/filename
            if path.exists():artifacts[key]=dict(path=str(path),sha256=sha(path))
        artifacts['teacher_diagnostics']=dict(path=str(out/'teacher_diagnostics.jsonl'),sha256=sha(out/'teacher_diagnostics.jsonl'),rows=diagnostic_rows)
        if error is not None:
            partial_native=recorder.close() if recorder.count else dict(actual_native_steps=0)
            if partial_native.get('post_native_steps',0)==0 and 'post_COM_min' in partial_native:partial_native['post_COM_min']=None
            save(out/'FAILURE.json',dict(label=LABEL,error=repr(error),artifacts=artifacts,native=partial_native,actual_ms=0 if trace is None else len(trace.rows),neural_records=len(neural),no_overwrite_retry=True,neural_acceptance=False))
        elif result is not None:result.update(artifacts);save(out/'RESULT.json',result)
    if error is not None:raise error
    print(json.dumps(dict(case=case,result=str(out/'RESULT.json'),label=LABEL,early_stop=early_stop,qualified_teacher_labels=qualified_count,neural_acceptance=False)),flush=True)


if __name__=='__main__':main()
