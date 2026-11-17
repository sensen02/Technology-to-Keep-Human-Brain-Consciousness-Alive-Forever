"""Parent-only oneCore zero-integration numerical interface fixture, not true warm."""
import argparse
import json
import sys
from common import ROOT,FORMAT,ACTUAL_SHA,CASES,MASK_NAMES,read,save,sha,verify,cpu1,load_motor,frozen,physical_stack,snapshot,assert_same,body_length,make_teacher,shadow_details,teacher_mask

DIAG_FIELDS=('per_leg_fz_over_W','unbacked_support_indices','ik_error_norm_per_leg','unsupported','qdot_des_clip_count','rate_clip_count','motor_target_bound_indices','tau_ff_native_clip_indices','actual_contact_counts','q_des_act','desired_motor_target')


def probe(counts):
    cpu1();sources=frozen();verify(sources)
    # Every strict checkpoint check happens before any physical backend import.
    m,payload=load_motor();counts['checkpoint_loads']+=1
    if any(name in sys.modules for name in ('physics','mujoco','flygym','student_wave')):raise ValueError('Strict007d loader must precede physical backend')
    import torch
    import numpy as np
    sw,physics,identity,static,Peak,ast_ids=physical_stack()
    import mujoco as mj
    recorder=Peak();c=physics.Core(adhesion=False,recorder=recorder);counts['Core_constructions']+=1
    initial=snapshot(c);checks={};cases={};teachers=[];hooks=[]
    def hook(name):
        def count(*unused):counts['adapter_module_forward_calls'][name]+=1
        return count
    for name in ('encoder','gru','head'):hooks.append(getattr(m,name).register_forward_hook(hook(name)))
    try:
        binding=read(ROOT/'INPUT_BINDINGS.json')
        checks['fixed_four_cases_commands']=CASES=={'old_walk_control':('walk',[1.,0.]),'assisted_stop':('stop',[0.,0.]),'assisted_left_pivot':('left_pivot',[0.,1.]),'assisted_right_pivot':('right_pivot',[0.,-1.])}
        checks['strict_actual007d_final600']=binding['checkpoint_sha256']==ACTUAL_SHA and sha(binding['checkpoint'])==ACTUAL_SHA and payload['format_version']=='INDEPENDENT_SPEED_YAW_MOTOR_FASTPATH_REPAIR_V1' and payload['step']==600
        checks['old789f_frozen']=all(not p.requires_grad for p in m.old.parameters()) and bool(m.verify_old())
        mesh=c.plant.geom_type==mj.mjtGeom.mjGEOM_MESH
        checks['full71opaque42finitePD']=bool(mesh.sum()==71 and np.all(c.plant.geom_rgba[mesh,3]==1.) and c.plant.nu==42 and np.isfinite(c.actuator.kp).all() and np.all(c.actuator.kp>0.) and np.isfinite(c.actuator.torque_limit).all() and np.all(c.actuator.torque_limit>0.) and np.isfinite(c.actuator.target_bounds_rad).all())
        length=body_length(c);checks['compiled_sevenbody_extent']=len(length['segments'])==7 and length['length_native']>0.
        state=m.initial_state(1);state_before=tuple(s.detach().clone() for s in state)
        x=torch.tensor(c.obs238(np.zeros(42),target_speed=.075)[None],dtype=torch.float64);x_before=x.clone()
        checks['initial352zero']=all(s.shape==(1,n) and not bool(s.abs().max()) for s,n in zip(state,(128,64,64,96))) and len(state)==4
        with torch.no_grad():
            counts['NN_API_calls']['prime_old']+=1
            prime_mean,primed=m.prime_old(x,x.new_tensor([[1.,0.]]),state)
            checks['fixture_prime_old_new96_zero']=prime_mean.shape==(1,42) and not bool(primed[3].abs().max())
            checks['prime_input_state_immutable']=all(torch.equal(a,b) for a,b in zip(state,state_before))
            primed_before=tuple(s.clone() for s in primed)
            for case,(condition,command) in CASES.items():
                u=x.new_tensor([command]);u_before=u.clone()
                counts['NN_API_calls']['shadow_details']+=1
                mean,after,residual,direct=shadow_details(m,x,u,primed)
                counts['NN_API_calls']['independent_old_step']+=1
                independent,oldafter=m.old.step(x,m.legacy_projection(u),primed[:3])
                counts['NN_API_calls']['step_details']+=1
                production,production_after,production_residual=m.step_details(x,u,primed)
                mean_error=float((mean-production).abs().max());res_error=float((residual-production_residual).abs().max());state_error=max(float((a-b).abs().max()) for a,b in zip(after,production_after))
                checks[case+'_directold_exact']=torch.equal(direct,independent) and all(torch.equal(a,b) for a,b in zip(after[:3],oldafter))
                checks[case+'_fullfastpath_production_parity']=mean_error<=1e-12 and res_error<=1e-12 and state_error<=1e-12
                checks[case+'_finite42_state352']=mean.shape==residual.shape==direct.shape==(1,42) and all(s.shape==(1,n) and bool(torch.isfinite(s).all()) for s,n in zip(after,(128,64,64,96))) and bool(torch.isfinite(mean).all()) and bool(torch.isfinite(residual).all()) and bool(torch.isfinite(direct).all()) and float(residual.abs().max())<=6.
                checks[case+'_input_immutable']=torch.equal(x,x_before) and torch.equal(u,u_before) and all(torch.equal(a,b) for a,b in zip(primed,primed_before))
                cases[case]=dict(condition=condition,command=command,mean_max_error=mean_error,residual_max_error=res_error,state_max_error=state_error,directold_exact=bool(checks[case+'_directold_exact']),old_policy=direct[0].tanh().numpy().copy(),new_policy=mean[0].tanh().numpy().copy())
        for condition in ('stop','left_pivot','right_pivot'):
            before=snapshot(c);teacher=make_teacher(c,condition);counts['teacher_factories']+=1;assert_same(before,snapshot(c))
            action,diag=teacher.action(c,0.);counts['teacher_queries']+=1;assert_same(before,snapshot(c))
            twin=make_teacher(c,condition);counts['teacher_factories']+=1;assert_same(before,snapshot(c))
            action2,diag2=twin.action(c,0.);counts['teacher_queries']+=1;assert_same(before,snapshot(c))
            checks[condition+'_diagnostic_schema']=all(name in diag and name in diag2 for name in DIAG_FIELDS) and np.asarray(diag['q_des_act']).shape==(42,) and np.asarray(diag['desired_motor_target']).shape==(42,) and np.asarray(diag['ik_error_norm_per_leg']).shape==(6,) and np.asarray(diag['per_leg_fz_over_W']).shape==(6,) and np.asarray(diag['actual_contact_counts']).shape==(6,)
            checks[condition+'_teacher_finite_bounded42']=action.shape==action2.shape==(42,) and bool(np.isfinite(action).all()) and bool(np.isfinite(action2).all()) and float(np.max(np.abs(action)))<=1.+1e-12
            checks[condition+'_fresh_samefixture_query_exact']=np.array_equal(action,action2) and all(np.array_equal(np.asarray(diag[k]),np.asarray(diag2[k])) for k in DIAG_FIELDS)
            mask,unbacked=teacher_mask(c,action,diag,source_exact=True);assert_same(before,snapshot(c))
            independently_unbacked=sum(float(diag['per_leg_fz_over_W'][i]) for i in diag['unbacked_support_indices'])
            measured=c.measure();expected=dict(actual_COM=float(c.d.subtree_com[c.th,2])>=1.2,actual_NF_1ms=float(measured['nonfoot'])==0.,IK=float(max(diag['ik_error_norm_per_leg']))<=.01,unbacked_plannedload=independently_unbacked<=.01,teacher_supported=not diag['unsupported'],qdot_clip=diag['qdot_des_clip_count']==0,rate_clip=diag['rate_clip_count']==0,target_bounds=len(diag['motor_target_bound_indices'])==0,tau_ff_clip=len(diag['tau_ff_native_clip_indices'])==0,finite42=action.shape==(42,) and bool(np.isfinite(action).all()),actual_restore=True)
            checks[condition+'_11mask_schema_and_formula']=tuple(mask)==tuple(MASK_NAMES)==tuple(expected) and len(mask)==11 and mask==expected and abs(float(unbacked)-independently_unbacked)<=1e-12
            case='assisted_'+condition;student_action=cases[case]['new_policy']
            mixed=.75*action+.25*student_action
            checks[condition+'_convex75teacher25full007dstudent_bounded']=mixed.shape==(42,) and bool(np.isfinite(mixed).all()) and float(np.max(np.abs(mixed)))<=1.+1e-12 and bool(np.all(mixed>=np.minimum(action,student_action)-1e-12)) and bool(np.all(mixed<=np.maximum(action,student_action)+1e-12))
            teachers.append(dict(condition=condition,mask=mask,unbacked_planned_load_over_W=float(unbacked),label_qualified_at_raw_fixture=bool(all(mask.values())),fixture_only=True,actual_teacher_elapsed_s=0.,query_action_exact_between_fresh_factories=bool(np.array_equal(action,action2))))
        assert_same(initial,snapshot(c));checks['all_seven_mechanical_arrays_unchanged']=True
        checks['zero_actual_root_or_extra_force_write']=bool(np.max(np.abs(c.d.xfrc_applied))==0. and np.max(np.abs(c.d.qfrc_applied))==0.)
        checks['oneCore_zero_integration']=counts['Core_constructions']==1 and float(c.d.time)==0. and c.t==0 and recorder.count==0
        checks['nonzero_NN_calls_recorded']=sum(counts['NN_API_calls'].values())==13 and all(counts['adapter_module_forward_calls'][name]==8 for name in ('encoder','gru','head'))
        ident=identity.capture(c,dict(kind=FORMAT+'_RAWSTATE_ZERO_INTEGRATION_PREFLIGHT',source_hashes=sources,fixture_only=True),extra_paths=list(sources));identity.verify_runtime(ident,c)
        verify(sources);m.verify_old()
        for row in cases.values():row.pop('old_policy');row.pop('new_policy')
        return dict(all_passed=bool(all(checks.values())),checks=checks,source_hashes=sources,actual_checkpoint_sha256=ACTUAL_SHA,identity=ident,body_length=length,selected_legacy_ast=ast_ids,cases=cases,teacher_fixtures=teachers,Core_constructions=counts['Core_constructions'],teacher_queries=counts['teacher_queries'],physical_native_steps=0,physical_integration_steps=0,NN_forwards=sum(counts['NN_API_calls'].values()),NN_forward_count_semantics='Explicit top-level neural API invocations; nested frozen-old work not added twice. Adapter module hooks separately record actual encoder/GRU/head forward calls.',counts=counts,fixture_only=True,true_warm_handover_verified=False,true_run_invariants_deferred_to_collector=True,training_updates=0,label='TEACHER_ASSISTED_DATA_NOT_NEURAL_ACCEPTANCE',scope='BodyANNcomponent_NOTnewbrainneckdeployment')
    finally:
        for handle in hooks:handle.remove()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute-parent-preflight',action='store_true');args=parser.parse_args()
    if not args.execute_parent_preflight:parser.error('Explicit parent numerical fixture release required')
    out=ROOT/'PREFLIGHT.json'
    if out.exists():raise FileExistsError(out)
    counts=dict(Core_constructions=0,checkpoint_loads=0,teacher_factories=0,teacher_queries=0,NN_API_calls=dict(prime_old=0,shadow_details=0,independent_old_step=0,step_details=0),adapter_module_forward_calls=dict(encoder=0,gru=0,head=0))
    try:result=probe(counts)
    except Exception as exc:
        save(out,dict(all_passed=False,error_type=type(exc).__name__,error=repr(exc),counts=counts,NN_forwards=sum(counts['NN_API_calls'].values()),fixture_only=True,true_warm_handover_verified=False,scope='BodyANNcomponent_NOTnewbrainneckdeployment'))
        raise
    save(out,result)
    if not result['all_passed']:raise ValueError('Assisted collection zero-integration fixture failed; no overwrite/retry')
    print(json.dumps(dict(status='TEACHER_ASSISTED_COLLECTION_RAWSTATE_PREFLIGHT_PASS',fixture_only=True,NN_forwards=result['NN_forwards'],teacher_queries=counts['teacher_queries'],physical_steps=0)))


if __name__=='__main__':main()
