"""Independent current selected native evidence. No Core construction or integration."""
from pathlib import Path
import argparse
import ast
import importlib.util
import json
import types
import numpy as np
from contract import HERE, ROOT, BANC, PINS, admitted, identity, require, sha, load, save_new


def module_at(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def logical_audit(run):
    # Execute only the exact pinned array/actual-192-neuron verifier functions.
    import execute_native
    parent=execute_native.parent_module()
    path=ROOT.parent/'20261110T235830Z/evaluation/verify_actual.py'
    require(sha(path)==PINS[path], 'logical auditor method changed')
    tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('arrays','verify')]
    require(len(nodes)==2,'logical verifier function contract changed')
    namespace=dict(Path=Path,np=np,json=json,hashlib=__import__('hashlib'),HERE=ROOT,
                   BASE=parent.BASE,ORIGINAL=parent.ORIGINAL,COMMANDS=parent.COMMANDS,
                   pin=parent.pin,checked_load=parent.checked_load,flatten=parent.flatten,
                   command_at=parent.command_at,score=parent.score,jsonable=parent.jsonable,
                   sha256=sha,ROOT_HEAD_AUDIT=lambda:parent.HEAD_AUDIT)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path)+'::pinned-array-functions','exec'),namespace)
    return namespace['verify'](run)


def compare_archives(left,right):
    with np.load(left,allow_pickle=False) as a,np.load(right,allow_pickle=False) as b:
        require(set(a.files)==set(b.files),'selected/native replay field mismatch')
        fields={key:bool(a[key].dtype==b[key].dtype and np.array_equal(a[key],b[key])) for key in a.files}
    require(all(fields.values()),'selected engineering/native logical replay differs')
    return dict(all_exact=True,fields=fields,left=str(left),right=str(right),left_sha256=sha(left),right_sha256=sha(right))


def tagged_native(manifest,neural,geometry,auto_start,auto_end):
    count=0;phase_counts=dict(settle=0,auxiliary=0,autonomous=0);yaw_increment=0.;schema=None
    body_forward_integral=body_lateral_integral=duration=0.
    for chunk in manifest['chunks']:
        require(sha(chunk['path'])==chunk['sha256'],'native chunk drift before tagged scan')
        with np.load(chunk['path'],allow_pickle=False) as z:
            if schema is None:schema={k:dict(tail_shape=list(z[k].shape[1:]),dtype=str(z[k].dtype)) for k in z.files if k!='contact_offsets'}
            before=z['time_before_s'];after=z['time_after_s'];n=len(before)
            logical=(np.arange(count,count+n)//80).astype(np.int64)
            source=np.where(logical<1000,0,np.where(logical<5800,2,3))
            decision=np.where(logical<1000,-1,(logical-1000)//5)
            require(np.array_equal(z['logical_ms_index'],logical) and np.array_equal(z['native_control_source'],source),'native physical source/tag timeline differs')
            require(np.array_equal(z['neural_decision_index'],decision),'native decision alignment differs')
            expected=np.zeros((n,2));valid=decision>=0;expected[valid]=neural['command'][decision[valid]]
            require(np.array_equal(z['neural_command'],expected),'native held neural test command differs')
            require(np.array_equal(z['action'][valid],neural['actually_applied_action'][decision[valid]]),'native held actuator action differs')
            require(np.all(z['action'][~valid]==0),'native settle actuator nonzero')
            rotation=z['thorax_rotation_world_before'];q=z['qpos_before'][:,3:7]
            actual=np.arctan2(rotation[:,1,0],rotation[:,0,0]);expected_yaw=np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
            require(np.max(np.abs(np.angle(np.exp(1j*(actual-expected_yaw)))))<1e-10,'native body heading disagrees with actual quaternion')
            require(np.max(np.abs(np.einsum('tji,tjk->tik',rotation,rotation)-np.eye(3)))<1e-10,'invalid native body rotation')
            qa=z['qpos_after'][:,3:7];ya=np.arctan2(2*(qa[:,0]*qa[:,3]+qa[:,1]*qa[:,2]),1-2*(qa[:,2]**2+qa[:,3]**2))
            keep=(before>=auto_start-1e-10)&(after<=auto_end+1e-10)
            yaw_increment+=float(np.sum(np.angle(np.exp(1j*(ya[keep]-expected_yaw[keep])))))
            current_heading=rotation[keep,:2,0];heading_norm=np.linalg.norm(current_heading,axis=1)
            require(np.all(heading_norm>1e-12),'native body heading projection degenerate')
            current_heading=current_heading/heading_norm[:,None]
            current_velocity=z['com_velocity_before'][keep,:2];dt=(after-before)[keep]
            body_forward_integral+=float(np.sum(np.sum(current_heading*current_velocity,axis=1)*dt))
            body_lateral_integral+=float(np.sum((current_heading[:,0]*current_velocity[:,1]-current_heading[:,1]*current_velocity[:,0])*dt))
            duration+=float(dt.sum())
            for code,key in ((0,'settle'),(2,'auxiliary'),(3,'autonomous')):phase_counts[key]+=int(np.count_nonzero(source==code))
            count+=n
        require(sha(chunk['path'])==chunk['sha256'],'native chunk drift during tagged scan')
    require(count==1232000 and phase_counts==dict(settle=80000,auxiliary=384000,autonomous=768000),'full actual native counts missing')
    endpoint_yaw=np.unwrap(geometry['thorax_yaw']);delta=float(endpoint_yaw[-1]-endpoint_yaw[5799])
    require(abs(yaw_increment-delta)<=1e-9,'native all-step turn disagrees with endpoint yaw')
    return dict(samples=count,phase_counts=phase_counts,all_actual_sources_and_commands_exact=True,
                current_body_heading_verified_every_native_step=True,whole_native_yaw_rad=yaw_increment,
                endpoint_yaw_rad=delta,signed_left_yaw_pass=yaw_increment>=.01,native_schema=schema,
                current_body_heading_diagnostic=dict(forward_velocity_time_mean=body_forward_integral/duration,
                lateral_velocity_time_mean=body_lateral_integral/duration,duration_s=duration,
                scope='actual native before-step COM velocity projected on current body XY heading; diagnostic only, original signed initial-heading and XY gates unchanged'))


def audit(run,plan_path,checkpoint,digest):
    plan,proof=admitted(plan_path,checkpoint,digest);run=Path(run).resolve()
    require(run==HERE/'selected_native/EVAL_left','scoped selected native run required')
    result=load(run/'RESULT.json');ident=load(run/'IDENTITY.json');identity(ident)
    require(result['checkpoint_sha256']==digest and result['condition']=='left','native selected candidate differs')
    require(result['transition']['all_exact'] and result['transition']['no_reset'],'native handover not continuous')
    static_path=run/'NATIVE_STATIC.json';static=load(static_path)
    require(str(static_path) in ident['files'] and sha(static_path)==ident['files'][str(static_path)]['sha256'],'compiled static contract not captured in execution identity')
    require(static['checkpoint_sha256']==digest and static['condition']=='left' and static['native_dt_s']==12.5e-6,'static run identity differs')
    require([static[k] for k in ('nq','nv','nu','mesh_count')]==[49,48,42,71] and static['all_mesh_opaque'] is True and static['adhesion_enabled'] is False,'full-body runtime contract differs')
    require(static['gravity_native']==[0.,0.,-9801.] and abs(static['mass_native']*9801.-static['weight_native'])<1e-10,'native mass/gravity/weight inconsistent')
    require(sha(static['xml'])==static['xml_sha256'],'actual XML changed')
    baseline=Path(proof['cases']['left']['result']).parent
    baseline_identity=load(baseline/'IDENTITY.json')
    require(ident['compiled_arrays']==baseline_identity['compiled_arrays'] and ident['physics']==baseline_identity['physics'] and ident['actuator']==baseline_identity['actuator'],'native compiled physical/actuator arrays differ from selected engineering run')
    logical=logical_audit(run)
    require(logical['integrity_all_pass'] and logical['engineering_pass'],'native logical/current-neural audit did not pass')
    replay={name:compare_archives(run/name,baseline/name) for name in ('trace_1ms.npz','neural_decisions.npz','geometry_1ms.npz','runtime_geometry.npz')}
    require(load(run/'HANDOVER.json')==load(baseline/'HANDOVER.json'),'native engineering HANDOVER differs')
    with np.load(run/'neural_decisions.npz',allow_pickle=False) as z:neural={k:z[k].copy() for k in z.files}
    with np.load(run/'trace_1ms.npz',allow_pickle=False) as z:clock=z['time'].copy()
    with np.load(run/'geometry_1ms.npz',allow_pickle=False) as z:geometry={k:z[k].copy() for k in z.files}
    require(neural['time'][0]==clock[999] and neural['time'][960]==clock[5799],'actual source boundary mismatch')
    windows={'auxiliary':(float(neural['time'][0]),float(neural['time'][960])), 'autonomous':(float(neural['time'][960]),float(clock[-1]))}
    require(all(np.allclose(windows[k],v,atol=1e-8,rtol=0) for k,v in {'auxiliary':(1.,5.8),'autonomous':(5.8,15.4)}.items()),'declared full window mismatch')
    manifest=result['native'];require(manifest['native_samples']==1232000 and manifest['pending_rows']==0,'native recording incomplete')
    for c in manifest['chunks']:require(Path(c['path']).resolve().parent==run/'native','native chunk outside same-run directory')
    frozen=module_at('_selected_frozen_native_acceptance',BANC/'20261110T054510Z/native_acceptance.py')
    legacy=frozen.evaluate_directory(run/'native',autonomous_window=windows['autonomous'],auxiliary_window=windows['auxiliary'],
        mass_native=static['mass_native'],gravity_native=9801.,heading_world_xy=static['initial_heading_world_xy'],source_manifest=manifest,
        full_model=static['xml'],metadata={'run_id':str(run),'evidence':[{'path':str(run/'RESULT.json'),'sha256':sha(run/'RESULT.json'),'claims':['auxiliary_control_valid','continuous_handover']}]},max_scalar_samples=2000000)
    require(legacy['status']=='OK','native physical metric method ERROR: '+repr(legacy.get('errors')))
    extra_module=module_at('_selected_pinned_force_diagnostics',BANC/'20261110T152725Z/final_native_audit.py')
    extra=extra_module.extra_native(manifest,static['weight_native'],run/'trace_1ms.npz',windows=windows)
    tags=tagged_native(manifest,neural,geometry,*windows['autonomous'])
    require(legacy['integrity']['samples']==1232000 and legacy['windows']['autonomous']['samples']==768000 and legacy['windows']['auxiliary']['samples']==384000,'native physical method omitted steps')
    for key in windows:
        primary=next(r for r in extra['windows'][key]['D6_sensitivity'] if r['tau_ms']==1.)
        require(abs(primary['signed_p95_W']-legacy['windows'][key]['original_D6_signed_p95_W'])<1e-10,'independent force filter differs')
        require(extra['windows'][key]['D7_sensitivity'][-1]['median']==legacy['windows'][key]['loaded_leg_count_median'],'D7 native count differs')
    criteria={k:dict(v) for k,v in legacy['criteria'].items()}
    median=extra['windows']['autonomous']['D7_sensitivity'][-1]['median']
    criteria['D7']=dict(status='PASS' if median>=3 else 'UNKNOWN',value=median,threshold=3,
                        reason='CURRENT actual native >=.1W count median supplies conservative sufficient proof for all lower reported carrying thresholds; insufficient proof remains UNKNOWN')
    engineering={k:v for k,v in legacy['engineering_only'].items() if isinstance(v,dict) and 'status' in v}
    failures=[key for key,value in criteria.items() if value['status']!='PASS']
    failures += ['native_engineering.'+key for key,value in engineering.items() if value['status']!='PASS']
    if not extra['all_actual_extra_forces_zero']:failures.append('actual_applied_or_adhesion_force')
    if not tags['signed_left_yaw_pass']:failures.append('signed_left_yaw_.01')
    if not result['autonomous_cycles']['all_legs_completed']:failures.append('all_six_cycles')
    # Revalidate immutable selection, all current source/artifacts and all native chunks at end.
    admitted(plan_path,checkpoint,digest);identity(ident)
    return dict(status='PASS' if not failures else 'FAIL',scope='current selected LEFT native physical/neural audit; full food objective NOT_ESTABLISHED',
                checkpoint=str(checkpoint),checkpoint_sha256=digest,run=str(run),run_plan_sha256=sha(plan_path),
                failures_or_unknowns=failures,physical_criteria=criteria,frozen_legacy_report=legacy,
                legacy_D7_preserved=legacy['criteria']['D7'],engineering_native=engineering,
                engineering_1ms=result['autonomous'],six_cycles=result['autonomous_cycles'],logical_192_neural_audit=logical,
                selected_engineering_native_bitexact_replay=replay,whole_native_tags_heading=tags,native_force_and_D7_diagnostics=extra,
                static_contract=static,methods_sha256=sha(HERE/'METHODS_DRAFT.json'),
                units='native; independent SI/biology/connectome NOT_ESTABLISHED',physical_steps_in_audit=0,
                constructor_calls_in_audit=0,full_goal_status='NOT_ESTABLISHED',same_run_video_review='NOT_PERFORMED_BY_AUDIT')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--run-plan',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--sha256',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out).resolve();require(out.is_relative_to(HERE) and not out.exists(),'fresh scoped audit output required')
    try:r=audit(a.run,a.run_plan,a.checkpoint,a.sha256)
    except Exception as exc:
        save_new(out,dict(status='ERROR',error=repr(exc),full_goal_status='NOT_ESTABLISHED',physical_steps_in_audit=0));raise
    save_new(out,r);print(json.dumps(dict(status=r['status'],failures=r['failures_or_unknowns'],physical_steps=0)))
    if r['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
