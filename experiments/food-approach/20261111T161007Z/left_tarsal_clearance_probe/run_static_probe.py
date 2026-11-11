"""Default CODEONLY. Only parent --run-static enables bounded private static FK.

No execution of this flag is authorized during code preparation. Runtime creates
no new trajectory and never integrates; all outputs are new exclusive files.
"""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from probe_identity import HERE,FREEZE,verify_sources,sha256,exclusive_result,cpu1


def run(prereg):
    # All numerical/model imports are deliberately below explicit runtime flag.
    import hashlib
    import importlib.metadata
    import numpy as np
    import mujoco as mj
    baseline=Path(prereg['baseline']);sys.path.insert(0,str(baseline))
    import physics
    import identity
    if Path(physics.__file__).resolve()!=baseline/'physics.py':raise RuntimeError('Wrong baseline physics')
    from probe_geometry import StaticGeometry
    folder=Path(prereg['input_run'])
    inherited=json.loads((folder/'IDENTITY.json').read_text());identity.verify(inherited)
    c=physics.Core(adhesion=False)
    identity.verify_runtime(inherited,c)
    if float(c.plant.opt.timestep)!=inherited['physics']['timestep'] or not np.array_equal(c.plant.opt.gravity,inherited['physics']['gravity']):
        raise RuntimeError('Compiled opt timestep/gravity mismatch')
    def live_digest():
        h=hashlib.sha256()
        for name in ('qpos','qvel','ctrl','xfrc_applied','qfrc_applied'):
            h.update(np.asarray(getattr(c.d,name)).tobytes())
        h.update(np.asarray([c.d.time]).tobytes());h.update(c.actuator.target_rad.tobytes())
        return h.hexdigest()
    live_before=live_digest()
    with np.load(folder/'TRACE_1MS.npz',allow_pickle=False) as z:
        source={k:z[k] for k in ('time','qpos','target')}
    if source['qpos'].shape!=(15400,49) or source['target'].shape!=(15400,42):raise RuntimeError('Unexpected saved trace contract')
    engine=StaticGeometry(c,source['qpos'][999])
    inputs=[];observed_margins=[]
    for requested in prereg['sample_times_s']:
        ix=int(np.argmin(abs(source['time']-requested)));actual=float(source['time'][ix])
        if abs(actual-requested)>.000501:raise RuntimeError('Selected endpoint too far from preregistered sample')
        for variant in prereg['pose_variants']:
            q=source['qpos'][ix].copy()
            if variant=='actual_saved_qpos_with_RH_motor_target_substitution':q[engine.rqa]=source['target'][ix,engine.rh]
            elif variant!='actual_saved_qpos':raise RuntimeError('Unknown pose variant')
            b=engine.pose(q)
            observed_margins.extend(ct['includemargin'] for ct in b['collision']['contacts'])
            inputs.append(dict(requested_time=requested,time=actual,index=ix,variant=variant,q=q,baseline=b))
    margins=sorted(set(float(x) for x in observed_margins))
    margin_ok=bool(margins and max(margins)-min(margins)<1e-12)
    effective_margin=float(margins[0]) if margin_ok else None
    explicit=[]
    for i in range(c.plant.npair):
        if {int(c.plant.pair_geom1[i]),int(c.plant.pair_geom2[i])}=={engine.geom,engine.floor}:
            explicit.append(dict(pair_id=i,margin=float(c.plant.pair_margin[i]),gap=float(c.plant.pair_gap[i])))
    rows=[]
    for item in inputs:
        row=dict(requested_time_s=item['requested_time'],actual_saved_time_s=item['time'],saved_index=item['index'],pose_variant=item['variant'],
            saved_root_qpos=item['q'][:7].tolist(),saved_RH_motor_target=source['target'][item['index'],engine.rh].tolist(),
            note='Motor-target substitution is geometry only; never teacher q_des. No integration or force prediction.')
        try:
            b=engine.pose(item['q']);x=item['q'][engine.rqa].copy()
            Z,s,rank,tol=engine.nullspace(b['Jpad']);fd=engine.fd(x,b)
            pad_fd_status='NONDIFFERENTIABLE' if fd['pad_nondifferentiability_possible'] else ('PASS' if fd['pad_jacobian_max_abs_error']<1e-5 else 'FAIL')
            mincontact=min((ct['distance'] for ct in b['collision']['contacts']),default=None)
            row.update(status='STATIC_BASELINE_CHECKED',RH_angles=x.tolist(),original_joint_bounds=engine.bounds.tolist(),
                original_joint_bounds_respected=bool(np.all(x>=engine.bounds[:,0]-1e-12) and np.all(x<=engine.bounds[:,1]+1e-12)),
                clearance=b['h'],active_mesh_vertex=b['vertex'],active_point_world=b['point'].tolist(),mesh_active_gap=b['mesh_active_gap'],
                pad_primary=b['pad'].tolist(),pad_bottoms_actual_world=b['pad_bottoms_actual_world'].tolist(),pad_bottoms_initial_frame=b['pad_bottoms_initial_frame'].tolist(),pad_active_gap=b['pad_active_gap'],
                Jpad=b['Jpad'].tolist(),pad_singular_values=s.tolist(),svd_tolerance=tol,pad_rank=rank,null_dimension=int(Z.shape[1]),null_basis=Z.tolist(),
                J_times_null_max_abs=float(np.max(np.abs(b['Jpad']@Z))),clearance_gradient=b['g'].tolist(),clearance_gradient_times_null=(b['g']@Z).tolist(),gZ_norm=float(np.linalg.norm(b['g']@Z)),
                finite_difference=fd,attachedpoint_fd_pass=bool(fd['attachedpoint_jacobian_max_abs_error']<1e-6),pad_fd_status=pad_fd_status,
                geometry_crosscheck=b['collision'],geometry_crosscheck_pass=bool(b['collision']['support_distance_error']<1e-6),
                collision_min_distance_vs_support_error=None if mincontact is None else abs(mincontact-b['h']),
                RH_nonfoot_minimum_signed_plane_heights=b['RH_nonfoot_minimum_signed_plane_heights'],
                root_unchanged=True,other_angles_unchanged=True)
            if not margin_ok:
                row['secondary']=dict(status='NO_UNAMBIGUOUS_OBSERVED_PAIR_MARGIN',local_improvement=False,safe_margin_reached=False)
            elif not row['attachedpoint_fd_pass'] or not row['geometry_crosscheck_pass'] or pad_fd_status=='FAIL':
                row['secondary']=dict(status='BASELINE_NUMERICAL_CHECK_FAILED_NOT_OPTIMIZED',local_improvement=False,safe_margin_reached=False)
            else:
                engine.pose(item['q'])
                row['secondary']=engine.bounded_secondary(x,b,effective_margin+prereg['bounded_secondary']['reserve_native'],prereg['bounded_secondary'])
                engine.pose(item['q'])
                row['kinematic_sequence']=engine.kinematic_sequence(x,b,effective_margin+prereg['bounded_secondary']['reserve_native'],prereg['bounded_secondary'])
        except Exception as exc:
            row.update(status='ROW_FAILURE_RETAINED',error=str(exc),traceback=traceback.format_exc())
        rows.append(row)
    verify_sources();identity.verify_runtime(inherited,c)
    live_after=live_digest()
    if live_after!=live_before:raise RuntimeError('Live Core data changed during private static probe')
    mapping=dict(RH_actuator_indices=engine.rh.tolist(),RH_qpos_indices=engine.rqa.tolist(),RH_dof_indices=engine.rva.tolist(),RH_joint_names=engine.names,
        geom_id=engine.geom,geom_name=c.plant.geom(engine.geom).name,body_id=engine.body,body_name=c.plant.body(engine.body).name,
        floor_id=engine.floor,floor_name=c.plant.geom(engine.floor).name,pad_geom_ids=engine.pads.tolist(),
        geom_classification=int(c.geom_leg[engine.geom]),geom_contype=int(c.plant.geom_contype[engine.geom]),geom_conaffinity=int(c.plant.geom_conaffinity[engine.geom]),
        mesh_id=int(c.plant.geom_dataid[engine.geom]),compiled_mesh_vertices=len(engine.mesh_cache[engine.geom]))
    report=dict(status='STATIC_PROBE_COMPLETE',scope='Local saved-pose static feasibility only; no integration, force prediction, new teacher or behavior qualification',
        sourcehash_sha256=sha256(FREEZE),input_trace_sha256=sha256(folder/'TRACE_1MS.npz'),input_identity_sha256=sha256(folder/'IDENTITY.json'),
        compiled_identity_verified_before_after=True,live_core_data_unchanged=True,live_core_digest_before=live_before,live_core_digest_after=live_after,
        model_and_collision_classification_unchanged=True,geometry_mapping=mapping,
        primary_convention='mean pad body XY and lowest sphere bottom Z in original settled initial-root frame; actual saved root unchanged in private data',
        plane_normal=engine.plane_n.tolist(),plane_point=engine.plane_p.tolist(),
        margins=dict(geom_margin=[float(c.plant.geom_margin[g]) for g in (engine.floor,engine.geom)],geom_gap=[float(c.plant.geom_gap[g]) for g in (engine.floor,engine.geom)],explicit_pairs=explicit,
            all_observed_contact_includemargins=margins,effective_margin_verified=margin_ok,effective_includemargin=effective_margin,
            safe_height=None if effective_margin is None else effective_margin+.002,note='Verified by geometric shadow contacts only, not a force prediction'),
        rows=rows,row_count=len(rows),calls=engine.calls,physical_integration_steps=0,
        summary=dict(rows_with_local_improvement=sum(bool(r.get('secondary',{}).get('local_improvement')) for r in rows),rows_reaching_safe_margin=sum(bool(r.get('secondary',{}).get('safe_margin_reached')) for r in rows),
            rows_sequences_with_improvement=sum(bool(r.get('kinematic_sequence',{}).get('local_improvement')) for r in rows),rows_sequences_reaching_safe_margin=sum(bool(r.get('kinematic_sequence',{}).get('safe_margin_reached')) for r in rows),
            row_failures=sum(r['status']=='ROW_FAILURE_RETAINED' for r in rows)), 
        environment=dict(python=sys.executable,cpu_affinity=list(os.sched_getaffinity(0)),versions={k:importlib.metadata.version(k) for k in ('numpy','mujoco','scipy')}))
    out=exclusive_result('RESULT_STATIC_PROBE',report)
    print(json.dumps(dict(result=str(out),sha256=sha256(out),status=report['status'],rows=len(rows),summary=report['summary'],integration_steps=0)))
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-static',action='store_true',help='Parent launches only after explicitly releasing unique heavy resource')
    args=parser.parse_args();frozen,prereg=verify_sources()
    if not args.run_static:
        print(json.dumps(dict(status='CODEONLY_READY_STATIC_NOT_RUN',root=str(HERE),frozen_files=len(frozen['files']),heavy_imports=False)))
        return
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(name)!='1':raise RuntimeError(f'{name}=1 required before numerical imports')
    cpu1()
    if list(HERE.glob('RESULT_STATIC_PROBE_*.json')):raise RuntimeError('Static result already exists; preserve original attempt, no automatic rerun')
    try:run(prereg)
    except Exception as exc:
        out=exclusive_result('FAILED_STATIC_PROBE',dict(status='FAILED_STATIC_PROBE',error=str(exc),traceback=traceback.format_exc(),sourcehash_sha256=sha256(FREEZE),physical_integration_steps=0))
        print(str(out));raise


if __name__=='__main__':main()
