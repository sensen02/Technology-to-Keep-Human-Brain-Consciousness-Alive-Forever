"""Deferred parent-only saved-array diagnostics. No simulator, network or model imports."""
from pathlib import Path
import argparse
import hashlib
import json
import resource
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
LEGS = ('LF','LH','LM','RF','RH','RM')
W = 24.446354755878005
DT = .001
EDGES = (5800,15400,25000,34600,44200)

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def require(ok, message):
    if not ok:
        raise ValueError(message)

def expired(*_):
    raise TimeoutError('Saved-array90s parent budget exceeded')

def load(np, case, name, keys):
    path=Path(case['root'])/name; spec=case['archives'][name]
    require(sha(path)==spec['recorded_sha256'],'Saved archive hash drift: '+str(path))
    out={}
    with np.load(path,allow_pickle=False) as z:
        require(set(z.files)==set(spec['headers']),'Archive schema keys')
        for k,h in spec['headers'].items():
            a=z[k]
            require(list(a.shape)==h['shape'] and a.dtype.str==h['descr'],'Saved shape/dtype '+k)
            require(a.dtype.kind in 'fiub' and bool(np.isfinite(a).all()),'Nonfinite saved field '+k)
            if k in keys:
                out[k]=a
    require(set(out)==set(keys),'Required recorded fields missing')
    return out

def stats(np,a):
    a=np.asarray(a)
    if not a.size:
        return dict(status='NO_SAMPLES')
    return dict(mean=np.mean(a,axis=0).tolist(),rms=np.sqrt(np.mean(a*a,axis=0)).tolist(),minimum=np.min(a,axis=0).tolist(),maximum=np.max(a,axis=0).tolist(),p95_abs=np.quantile(np.abs(a),.95,axis=0).tolist())

def runs(np,mask):
    x=np.r_[False,np.asarray(mask,dtype=bool),False].astype(int)
    starts=np.flatnonzero(np.diff(x)==1); ends=np.flatnonzero(np.diff(x)==-1)
    return list(zip(map(int,starts),map(int,ends)))

def longest(np,mask):
    return max((b-a for a,b in runs(np,mask)),default=0)*DT

def point_progress(np,p,yaw,L,lo,hi,times):
    # Include exact preceding boundary, including index5799 at5.800s.
    points=p[lo-1:hi,:3]; d=np.diff(points[:,:2],axis=0)
    boundary_duration=float(times[hi-1]-times[lo-1]); sampled_duration=float(times[hi-1]-times[lo])
    path=float(np.linalg.norm(d,axis=1).sum()); net=float(np.linalg.norm(points[-1,:2]-points[0,:2])); step_dt=np.diff(times[lo-1:hi])
    heading=yaw[lo-1:hi-1]
    f=np.c_[np.cos(heading),np.sin(heading)]; left=np.c_[-np.sin(heading),np.cos(heading)]
    delta=points[-1]-points[0]; fixed=np.array([np.cos(yaw[lo-1]),np.sin(yaw[lo-1])])
    return dict(boundary_duration_actual_s=boundary_duration,postsample_endpoint_duration_actual_s=sampled_duration,nominal_interval_duration_s=(hi-lo)*DT,start_xyz_native=points[0].tolist(),end_xyz_native=points[-1].tolist(),delta_xyz_native=delta.tolist(),path_xy_native=path,net_xy_native=net,net_xy_speed_native_per_s=net/boundary_duration,path_xy_speed_native_per_s=path/boundary_duration,net_over_path=net/path if path>1e-12 else None,net_forward_window_heading_bodylength=float(delta[:2]@fixed/L),net_lateral_window_heading_bodylength=float(delta[:2]@np.array([-fixed[1],fixed[0]])/L),integrated_forward_bodylength=float((d*f).sum()/L),integrated_lateral_bodylength=float((d*left).sum()/L),forward_velocity_bodylength_per_s=stats(np,(d*f).sum(1)/(step_dt*L)),lateral_velocity_bodylength_per_s=stats(np,(d*left).sum(1)/(step_dt*L)),world_x_net_native_per_s=float(delta[0]/boundary_duration),negative_forward_sample_fraction=float(np.mean((d*f).sum(1)<0)))

def cycles(np,t,load,clearance):
    # Descriptive unload-clearance-reload, independent of legacy acceptance cycles.
    # All stages require observed samples; unknown initial load is never assumed.
    state='UNARMED'; high=low=reload=0; armed_start=None; onset=None; clear=False; events=[]
    for i in range(5799,len(t)):
        grounded=load[i]>=.10*W and clearance[i]<=.004
        unloaded=load[i]<=.05*W
        if state=='UNARMED':
            high=high+1 if grounded else 0
            if high>=10:
                state='ARMED';armed_start=i-9
        elif state=='ARMED':
            low=low+1 if unloaded else 0
            if low>=5:
                onset=i-4;clear=bool(np.any((load[onset:i+1]<=.05*W)&(clearance[onset:i+1]>.015)));state='UNLOADED';reload=0
        else:
            clear=clear or bool(unloaded and clearance[i]>.015)
            reload=reload+1 if grounded else 0
            if reload>=10:
                events.append(dict(loaded_index=armed_start,unload_index=onset,reload_index=i-9,complete_index=i,unload_time_s=float(t[onset]),complete_time_s=float(t[i]),unload_clearance_reload=clear,peak_clearance_native=float(clearance[onset:i+1].max()),duration_s=float(t[i]-t[onset])))
                state='ARMED';armed_start=i-9;low=reload=0
    return dict(events=events,right_censored_state=state,right_censored_unload_index=onset if state=='UNLOADED' else None,definition='load>=.10W & padz<=.004 for10ms -> load<=.05W for5ms -> padz>.015 while unloaded -> grounded load>=.10W for10ms; component diagnostic omits original .03advance and4stimeout; original strict counts/events preserved separately')

def case_diagnostic(np,case):
    root=Path(case['root']); result=read(root/'RESULT.json');xml=case['xml_schema']; L=case['body_length_reference_native']
    t=load(np,case,'trace_1ms.npz',('time','qpos','com','target','actualctrl','actuator_force','control_source','external_body_force_max','external_generalized_force_max'))
    g=load(np,case,'geometry_1ms.npz',('time','head_xyz','thorax_yaw','control_source'))
    a=load(np,case,'terrain_1ms.npz',('time','com_world_xyz','loads_normal','loads_world_z_positive','tangent_speed','padz','pad_center_world','pad_ground_height','registered_contact_count','legacy_plane_contact_count','nonfoot_normal','nonfoot_world_z_positive','control_source'))
    d=load(np,case,'neural_decisions.npz',('time','physical_step_ms','raw_turn','visual_raw_turn','visual_brain_input','odor_brain_input','command','DN_held','actually_applied_action','control_source'))
    load(np,case,'runtime_geometry.npz',())
    n=len(t['time']);require(n in (25000,44200),'Full19.2/38.4 saved duration')
    require(abs(xml['native_dt_s']-12.5e-6)<1e-15,'Native dt')
    expected=np.arange(1,n+1)*DT
    checks={}
    for name,z in (('trace',t),('geometry',g),('terrain',a)):
        checks[name+'_clock']=bool(np.max(np.abs(z['time']-expected))<1e-8)
        checks[name+'_source3_allann']=bool(np.all(z['control_source'][5800:]==3))
    checks['decision_clock']=bool(np.max(np.abs(d['time']-np.arange(n//5)*.005))<1e-8)
    checks['decision_physical_ms']=bool(np.array_equal(d['physical_step_ms'],np.arange(n//5)*5))
    checks['COM_capture_equal']=bool(np.array_equal(t['com'],a['com_world_xyz']))
    checks['no_extra_force']=bool(np.all(t['external_body_force_max']==0) and np.all(t['external_generalized_force_max']==0))
    checks['preaction5799_time']=abs(float(t['time'][5799])-5.8)<1e-8
    checks['nonnegative_load_speed']=bool(np.all(a['loads_normal']>=0) and np.all(a['loads_world_z_positive']>=0) and np.all(a['tangent_speed']>=0))
    checks['weight_positive']=W>0
    require(all(checks.values()),'Clock/source/finite/shape/mass integrity '+str(checks))
    yaw=np.unwrap(g['thorax_yaw']); qadr=[x['qpos_address'] for x in xml['actuators']];bounds=np.array([x['bounds_rad'] for x in xml['actuators']]);ctrlbounds=np.array([x['ctrlrange_native'] for x in xml['actuators']])
    errors=t['target']-t['qpos'][:,qadr]
    cyc={leg:cycles(np,t['time'],a['loads_world_z_positive'][:,j],a['padz'][:,j]) for j,leg in enumerate(LEGS)}
    windows=[]
    for lo,hi in zip(EDGES[:-1],EDGES[1:]):
        if hi>n:
            windows.append(dict(requested_edges_s=[lo/1000,hi/1000],status='NOT_RECORDED_FULL_WINDOW',samples=0));continue
        s=slice(lo,hi); neural=(d['physical_step_ms']>=lo)&(d['physical_step_ms']<hi)
        require(hi-lo==9600 and int(neural.sum())==1920,'Equal window mass')
        com_window=a['com_world_xyz'][s,:2]; sampled_duration=float(t['time'][hi-1]-t['time'][lo]); net_speed=float(np.linalg.norm(com_window[-1]-com_window[0])/sampled_duration)
        sampled_path=float(np.linalg.norm(np.diff(com_window,axis=0),axis=1).sum());path_speed=sampled_path/sampled_duration
        x=t['target'][s];lower=np.abs(x-bounds[:,0])<=1e-8;upper=np.abs(x-bounds[:,1])<=1e-8
        require(bool(np.all(x>=bounds[:,0]-1e-8) and np.all(x<=bounds[:,1]+1e-8)),'Targets inside source ranges')
        foot={}
        for j,leg in enumerate(LEGS):
            normal=a['loads_normal'][s,j];vertical=a['loads_world_z_positive'][s,j];speed=a['tangent_speed'][s,j];z=a['padz'][s,j]
            loaded=vertical>=.10*W; denominator=float(normal.sum());events=cyc[leg]['events']
            edge=[];centers=a['pad_center_world'][s,j]
            for b in xml['blocks']:
                # Geometry-only side-wall proximity of any original pad sphere.
                gap=b['front_x']-(centers[:,:,0]+.008)  # signed wall minus approaching sphere's +X surface
                near=(np.abs(gap)<=.02)&(centers[:,:,1]>=b['y_min']-.008)&(centers[:,:,1]<=b['y_max']+.008)&(centers[:,:,2]-.008<=b['top_z']+.004)&(centers[:,:,2]+.008>=0)
                near=near.any(axis=1)
                edge.append(dict(block=b['name'],geometry_only_not_contact=True,front_x_native=b['front_x'],sphere_front_gap_min_native=float(gap.min()),sphere_front_gap_max_native=float(gap.max()),near_front_loaded_duration_s=float(np.count_nonzero(near&loaded)*DT),near_front_loaded_slow_duration_s=float(np.count_nonzero(near&loaded&(speed<=.01))*DT),longest_near_front_loaded_s=float(longest(np,near&loaded))))
            numerator=float((normal*speed).sum()/denominator) if denominator>1e-12 else None
            ratio=numerator/net_speed if numerator is not None and net_speed>1e-9 else None
            original_events=[e for e in result['autonomous_cycles']['events'] if e['leg']==leg and lo < round(e['time']*1000) <= hi]
            foot[leg]=dict(original_strict_cycle_events=original_events,original_strict_cycle_complete_count=sum(e['type']=='complete' for e in original_events),original_strict_cycle_fully_contained_count=sum(e['type']=='complete' and round(e['start_time']*1000)>=lo for e in original_events),original_strict_cycle_boundary_assignment='completion time in(left,right], start may precede left; fullycontained count also supplied',registered_slip_diagnostic=dict(numerator_loadweighted_tangent_speed_native_per_s=numerator,denominator_net_COM_xy_speed_native_per_s=net_speed,denominator_path_COM_xy_speed_native_per_s=path_speed,ratio=ratio,ratio_status='DEFINED' if ratio is not None else 'UNDEFINED_ZERO_LOAD_OR_NEARZERO_NET_SPEED',nearzero_net_speed_threshold_native_per_s=1e-9,clamping_applied=False,postsample_endpoint_duration_actual_s=sampled_duration),normal_load_native=stats(np,normal),world_z_load_native=stats(np,vertical),normal_load_integral_native_force_s=float(normal.sum()*DT),world_z_load_integral_native_force_s=float(vertical.sum()*DT),tangent_speed_native_per_s=stats(np,speed),normal_load_weighted_tangent_speed_native_per_s=float((normal*speed).sum()/denominator) if denominator>1e-12 else None,loaded_fraction=float(loaded.mean()),longest_loaded_no_unload_s=longest(np,vertical>.05*W),unloaded_fraction=float(np.mean(vertical<=.05*W)),clearance_native=stats(np,z),full_unload_clearance_reload_inside_window=sum(bool(e['unload_clearance_reload'] and e['unload_index']>=lo and e['complete_index']<hi) for e in events),completed_reload_in_window_including_cross_boundary=sum(bool(e['unload_clearance_reload'] and lo<=e['complete_index']<hi) for e in events),front_edge_geometry=edge)
        command={k:stats(np,d[k][neural]) for k in ('raw_turn','visual_raw_turn','visual_brain_input','odor_brain_input','command','DN_held')}
        cc=t['actualctrl'][s];af=t['actuator_force'][s]
        control=dict(actuator_names=[x['name'] for x in xml['actuators']],target_bound_low_fraction=lower.mean(0).tolist(),target_bound_high_fraction=upper.mean(0).tolist(),target_bound_longest_dwell_s=[longest(np,lower[:,j]|upper[:,j]) for j in range(42)],target_minus_q_rad=stats(np,errors[s]),actualctrl_native_1ms_endpoints=stats(np,cc),actuator_force_native_1ms_endpoints=stats(np,af),ctrl_xml_bound_endpoint_fraction=np.mean((np.abs(cc-ctrlbounds[:,0])<=1e-8)|(np.abs(cc-ctrlbounds[:,1])<=1e-8),axis=0).tolist(),native_substep_tau_peak='NOT_RECORDED',cap_causal_claim='NOT_ESTABLISHED; endpoint ctrl-bound residency is not proof of causal torque insufficiency')
        mass=dict(samples=9600,interval_duration_s=9.6,decision_samples=1920,normal_support_mean_over_W=float(a['loads_normal'][s].sum(1).mean()/W),world_z_support_mean_over_W=float(a['loads_world_z_positive'][s].sum(1).mean()/W),total_normal_load_integral_native_force_s=float(a['loads_normal'][s].sum()*DT))
        windows.append(dict(status='MEASURED_FROM_SAVED_ARRAYS',edges_s=[lo/1000,hi/1000],post_slice=[lo,hi],preaction_boundary_index=lo-1,first_post_time_s=float(t['time'][lo]),last_post_time_s=float(t['time'][hi-1]),window_mass=mass,points=dict(root_freejoint=point_progress(np,t['qpos'][:,:3],yaw,L,lo,hi,t['time']),COM_thorax_subtree=point_progress(np,a['com_world_xyz'],yaw,L,lo,hi,t['time']),head_body_origin=point_progress(np,g['head_xyz'],yaw,L,lo,hi,t['time']),thorax_body_origin_xyz='SOURCE_PROVEN_SAME_POINT_AS_root_freejoint: direct world Thorax with sole freejoint; COM is distinct subtree COM'),yaw=dict(start_rad=float(yaw[lo-1]),end_rad=float(yaw[hi-1]),delta_rad=float(yaw[hi-1]-yaw[lo-1]),max_excursion_rad=float(np.max(np.abs(yaw[lo-1:hi]-yaw[lo-1])))),feet=foot,control=control,saved_neural_signals=command,contacts_1ms=dict(registered_contact_sample_sum=int(a['registered_contact_count'][s].sum()),plane_contact_sample_sum=int(a['legacy_plane_contact_count'][s].sum()),registered_count=stats(np,a['registered_contact_count'][s]),per_block_window_contact='NOT_RECORDED; geometric proximity is not contact',nonfoot_normal_peak_sampled=float(a['nonfoot_normal'][s].max()),nonfoot_world_z_peak_sampled=float(a['nonfoot_world_z_positive'][s].max()))))
    measured=[z for z in windows if z['status']=='MEASURED_FROM_SAVED_ARRAYS']
    checks['window_mass_exact']=sum(z['window_mass']['samples'] for z in measured)==n-5800
    require(checks['window_mass_exact'],'All ANN sample coverage exactly once')
    return dict(root=str(root),checks=checks,preaction_index=5799,native_dt_s=xml['native_dt_s'],sample_dt_s=.001,weight_native=W,absolute_SI='NOT_ESTABLISHED',body_length_native=L,body_length_provenance=case['body_length_meaning'],root_COM_distinct=True,bodypoints='root freejoint translation, subtree COM, head origin; no reconstruction or FK',windows=windows,per_leg_component_cycle_events=cyc,original_strict_cycle_config=result['autonomous_cycles']['config'],original_strict_cycles_saved=result['autonomous_cycles'],actual_original_source_bindings=case['source_hashes'],source_drift_from_original_record=[p for p,h in case['source_hashes'].items() if h['current_sha256']!=h['recorded_sha256']],per_block_native_contacts_whole_saved_summary=result['native'].get('native_contacts_by_terrain'),per_block_native_contacts_allANN_saved_summary=result['native'].get('post_native_contacts_by_terrain'),per_block_native_window_split='NOT_RECORDED',legacy_engineering_reference=dict(orig_autonomous=result['autonomous'],reference_registered=result['terrain_summary'],engineering_pass=result['engineering_pass'],functional_stable_body=result['functional_stable_body'],old_FAIL_preserved=True),interpretation='Descriptive joint evidence for progress/yaw, persistent loading/edge proximity, target saturation/tracking. Visual causal contribution and mechanical blocking remain hypotheses; slip alone is not root cause. Current18spheres without adhesion are not implemented hooks.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute-parent-saved-arrays',action='store_true')
    p.add_argument('--parent-resource-released',action='store_true')
    p.add_argument('--output',type=Path,default=HERE/'RESULT_SAVED_DIAGNOSTIC.json')
    args=p.parse_args()
    if not (args.execute_parent_saved_arrays and args.parent_resource_released):
        print('ONLYCODE: no payload loaded; parent exclusive resource release and both flags required');return
    require(sys.executable=='/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python','Exact125712 environment required')
    require(not args.output.exists() and args.output.parent.resolve()==HERE,'New output inside diagnostic directory only')
    started=time.monotonic();signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,90)
    resource.setrlimit(resource.RLIMIT_AS,(1500*1024**2,1500*1024**2))
    binding=read(HERE/'SOURCE_SCHEMA_BINDINGS.json')
    for path,digest in binding['actual_sources_sha256'].items():
        require(sha(path)==digest,'Bound source/metadata drift '+path)
    # Deliberately deferred. Never import project modules or instantiate model/Core.
    import numpy as np
    output=dict(kind='ROUGH_STALL_SAVED_ARRAYS_DIAGNOSTIC_V1',cases=[case_diagnostic(np,c) for c in binding['cases']],actual_sources_hash=binding['actual_sources_sha256'],binding_sha256=sha(HERE/'SOURCE_SCHEMA_BINDINGS.json'),optimizer_updates=0,Core_constructions=0,NN_forwards=0,physical_integration_steps=0,render_calls=0,native_dt_s=12.5e-6,preaction_index=5799)
    forbidden=[n for n in sys.modules if n.split('.')[0] in ('torch','mujoco','flygym','physics','runtime')]
    require(not forbidden,'Forbidden numerical model/backend import')
    elapsed=time.monotonic()-started;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    require(elapsed<90 and rss<=1500,'Parent90s/RSS1500 budget')
    output.update(wall_s=elapsed,peak_RSS_MiB=rss,resource_budget=dict(seconds=90,RSS_MiB=1500),code_only_preparation_did_not_execute_numerical_script=True)
    with args.output.open('x') as f:
        json.dump(output,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(output=str(args.output),wall_s=elapsed,peak_RSS_MiB=rss,updates=0,Core=0,NN=0,integration=0)))

if __name__=='__main__':
    main()
