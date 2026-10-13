"""Manual authorized timestep-only refinement: exactly two new 200ms normal cases."""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[_k]='1'
import json, re, signal, sys, time, shutil
from pathlib import Path
import numpy as np
import run_fine_clock_branch as fine
zero=fine.zero; closure=fine.closure; branch=fine.branch; sha=fine.sha
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/model_branches/clock_refinement'
NEW_US=(12.5,6.25)
WINDOWS_MS=(25,50,100,200)
# Defined before either new episode; descriptive reference, never a fitted criterion.
COM_REFERENCE_MM=.01

def dump(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def label(us): return f'{us:g}us_normal'

def validate_history():
    doc=json.loads((fine.OUT/'fine_clock_results.json').read_text())
    assert doc['execution_status']=='completed' and all(doc['checks'].values())
    assert sha(fine.OUT/'fine_clock_traces.npz')==doc['provenance']['npz_sha256']
    assert doc['provenance']['source_hashes_before']==doc['provenance']['source_hashes_after']
    for p,h in doc['provenance']['source_hashes_after'].items(): assert sha(p)==h,p
    for item in doc['model_manifest'].values(): assert sha(item['xml'])==item['sha256']
    return doc

def geometry(model,qpos):
    import mujoco
    d=mujoco.MjData(model); com=[]
    for q in qpos:
        d.qpos[:]=q; mujoco.mj_kinematics(model,d); mujoco.mj_comPos(model,d)
        com.append(d.subtree_com[0].copy())
    return np.asarray(com)

def event_set(summary):
    out=set()
    for k,r in enumerate(summary['records']):
        for step,col in r['probe_emitted_spike_events_step_column']:
            out.add((k*10+step+1,summary['probe_local_indices'][col]))
    return out

def compare(a,b,sa,sb,joint_names,joint_qadr):
    assert np.array_equal(a['body_time_s'],b['body_time_s'])
    ea,eb=event_set(sa),event_set(sb)
    root=np.linalg.norm(a['body_qpos'][:,:3]-b['body_qpos'][:,:3],axis=1)
    com=np.linalg.norm(a['body_com_xyz_mm']-b['body_com_xyz_mm'],axis=1)
    qa=a['body_qpos'][:,3:7]; qb=b['body_qpos'][:,3:7]
    dots=np.abs(np.sum(qa*qb,axis=1)/(np.linalg.norm(qa,axis=1)*np.linalg.norm(qb,axis=1)))
    orient=2*np.arccos(np.clip(dots,0,1))
    angles=np.abs(a['body_qpos'][:,joint_qadr]-b['body_qpos'][:,joint_qadr])
    windows={}
    for ms in WINDOWS_MS:
        n=ms*10+1; nc=ms//5; ticks=ms*2
        per_joint=np.max(angles[:n],axis=0)
        diff=np.abs(a['controls'][:nc]-b['controls'][:nc])
        aset={x for x in ea if x[0]<=ticks}; bset={x for x in eb if x[0]<=ticks}
        windows[str(ms)]=dict(max_com_distance_mm=float(com[:n].max()),max_root_translation_distance_mm=float(root[:n].max()),
            max_root_orientation_angle_rad=float(orient[:n].max()),max_individual_joint_angle_error_rad=float(per_joint.max()),
            joint_angle_max_error_rad=dict(zip(joint_names,per_joint.tolist())),
            com_below_descriptive_0p01mm=bool(np.all(com[:n]<COM_REFERENCE_MM)),
            max_applied_control_difference=float(diff.max()),different_control_elements=int(np.count_nonzero(diff)),
            max_spike_count_difference=int(np.max(np.abs(a['spike_counts'][:nc].astype(int)-b['spike_counts'][:nc].astype(int)))),
            probe_events_only_a=len(aset-bset),probe_events_only_b=len(bset-aset),probe_events_common=len(aset&bset))
    def first_body(x):
        ii=np.flatnonzero(x); return float(a['body_time_s'][ii[0]]) if len(ii) else None
    ix=np.flatnonzero(np.any(a['controls']!=b['controls'],axis=1))
    delta=ea^eb
    return dict(windows_ms=windows,first_com_nonzero_difference_s=first_body(com!=0),
        first_control_difference_applied_start_s=float(ix[0]*.005) if len(ix) else None,
        first_probe_event_symmetric_difference_s=min(t for t,_ in delta)*.0005 if delta else None,
        probe_event_timing_resolution_s=.0005,
        event_comparison='exact (neural tick, local neuron) sets; no event matching or smooth-function assumption')

def main(report):
    import mujoco
    start=time.monotonic(); OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'clock_refinement_results.json').exists(),'Do not repeat the two-case experiment'
    history=validate_history(); provenance=branch.verify_source_copy()
    protocol=dict(new_cases=[label(x) for x in NEW_US],duration_s=.2,neural_dt_s=.0005,command_and_sensory_dt_s=.005,
        sample_dt_s=.0001,sample_integer_strides={'12.5':8,'6.25':16},windows_ms=list(WINDOWS_MS),
        window_definition='body samples inclusive [0,T]; command intervals [0,T); emitted neural ticks (0,T]',
        com_resolution_reference_mm=COM_REFERENCE_MM,threshold_role='upfront descriptive only; not acceptance or biological validation',
        no_gain_control_activation_changes=True,no_parameter_search=True,replay_at_6p25us_tested=False,
        convergence_claim=False,comparison_units='model length convention mm; translation/COM Euclidean mm, joint angles rad separately')
    dump(OUT/'protocol_before_run.json',protocol)
    frozen=dict(history['provenance']['source_hashes_after'])
    for mod in list(sys.modules.values()):
        p=getattr(mod,'__file__',None)
        if p and Path(p).suffix=='.py' and Path(p).resolve().is_relative_to(HERE) and not any(t in Path(p).parts for t in ('venv','venv_body')):
            frozen[str(Path(p).resolve())]=sha(p)
    for p in [HERE/'test_clock_refinement.py',fine.OUT/'fine_clock_results.json',fine.OUT/'fine_clock_traces.npz',OUT/'protocol_before_run.json',zero.OUT/'model_manifest.json']:
        frozen[str(p)]=sha(p)
    assets=json.loads((zero.OUT/'model_manifest.json').read_text())['declared_asset_path_changes']
    for item in assets:
        assert sha(item['branch'])==item['sha256']; frozen[item['branch']]=item['sha256']
    for item in history['model_manifest'].values(): frozen[item['xml']]=item['sha256']
    archive=OUT/'source_archive'
    for p,h in frozen.items():
        path=Path(p)
        if path.suffix=='.py':
            target=archive/path.relative_to(HERE); target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target);assert sha(target)==h
    dump(OUT/'source_hashes_before.json',frozen)
    base=mujoco.MjModel.from_xml_path(str(zero.XML)); models={}; model_reports={}
    text=zero.XML.read_text()
    for us in NEW_US:
        updated,n=re.subn(r'(<option\b[^>]*\btimestep=")[^"]+(\")',lambda m:m[1]+str(us*1e-6)+m[2],text);assert n==1
        p=OUT/f'fruitfly_zero_floor_{us:g}us.xml';p.write_text(updated)
        model=mujoco.MjModel.from_xml_path(str(p));models[us]=model
        comparisons={'zero_floor_100us':fine.compiled_diff(base,model)}
        for oldus in (50,25):
            old=mujoco.MjModel.from_xml_path(history['model_manifest'][str(oldus)]['xml'])
            comparisons[f'old_{oldus}us']=fine.compiled_diff(old,model)
        model_reports[str(us)]=dict(xml=str(p),sha256=sha(p),compiled_comparisons=comparisons)
    _,pose,_,joints,geom_leg=closure.prepare_body()
    built=closure.build_neural_tier(closure.TierConfig(background_mean_nA=0,background_sd_nA=0)); assert built['n']==12000
    names=[mujoco.mj_id2name(base,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(base.nu)]
    manifest=closure.build_motor_manifest(json.loads(closure.POOLS.read_text()),built['tier']['root_ids'],names)
    groups,_=closure.channel_index_map(int(built['n']),built['tier']['root_ids'])
    results={};episodes={};arrays={'seated_qpos':pose,'tier_root_ids':np.asarray(built['tier']['root_ids'])}
    with np.load(fine.OUT/'fine_clock_traces.npz') as old:
        assert np.array_equal(pose,old['seated_qpos']) and np.array_equal(arrays['tier_root_ids'],old['tier_root_ids'])
        for us in (50,25):
            key=label(us); results[key]={k[len(key)+1:]:old[k].copy() for k in old.files if k.startswith(key+'_')}
            episodes[key]=history['episodes'][key]
    for us in NEW_US:
        key=label(us)
        episodes[key],results[key]=fine.run_arm(models[us],pose,joints,geom_leg,built,manifest,groups,us,'normal')
        episodes[key]['sample_integer_stride']=round(.0001/(us*1e-6))
        print(key,'steps',episodes[key]['completed_body_steps'],'activation',episodes[key]['activation'],'wall',time.monotonic()-start,flush=True)
    for key,r in results.items():
        assert np.array_equal(r['body_time_s'],np.arange(2001)*.0001)
        r['body_com_xyz_mm']=geometry(base,r['body_qpos'])
        assert np.array_equal(r['body_com_xyz_mm'][:,2],r['heights_mm'][:,2])
        arrays.update({key+'_'+k:v for k,v in r.items()})
    hinge=[i for i in range(base.njnt) if base.jnt_type[i]==mujoco.mjtJoint.mjJNT_HINGE]
    joint_names=[mujoco.mj_id2name(base,mujoco.mjtObj.mjOBJ_JOINT,i) for i in hinge];joint_qadr=base.jnt_qposadr[hinge]
    comparisons={}
    for a,b in ((50,25),(25,12.5),(12.5,6.25)):
        ka,kb=label(a),label(b);comparisons[ka+'_vs_'+kb]=compare(results[ka],results[kb],episodes[ka],episodes[kb],joint_names,joint_qadr)
    after={p:sha(p) for p in frozen};assert after==frozen
    dump(OUT/'source_hashes_after.json',after)
    for us in NEW_US: assert fine.compiled_diff(base,models[us])==model_reports[str(us)]['compiled_comparisons']['zero_floor_100us']
    npz=OUT/'clock_refinement_traces.npz';np.savez_compressed(npz,**arrays)
    guard=report('completed')
    if not np.isfinite(guard['previous_soft_limit_mb']):guard['previous_soft_limit_mb']='unlimited'
    doc=dict(schema_version=1,status='PASS_REFINEMENT_INVARIANTS_NO_CONVERGENCE_CLAIM',execution_status='completed',protocol=protocol,
        episodes={label(us):episodes[label(us)] for us in NEW_US},historical_episode_keys=[label(50),label(25)],comparisons=comparisons,
        model_manifest=model_reports,joint_names=joint_names,joint_qpos_addresses=joint_qadr.tolist(),
        provenance=dict(episode_source=provenance,source_hashes_before=frozen,source_hashes_after=after,npz_sha256=sha(npz),mujoco_version=mujoco.__version__,
            historical_results=str(fine.OUT/'fine_clock_results.json'),historical_npz_sha256=history['provenance']['npz_sha256']),
        checks=dict(historical_hashes_valid=True,frozen_sources_unchanged=True,compiled_arrays_unchanged=True,all_common_times_exact=True,com_z_reconstruction_exact=True),
        limitations=['Manual experimental implementation, not a finished controller.','Only two new NORMAL cases; no new silence or replay, specifically no 6.25us replay.',
            'Sampled-window maxima at 0.1ms are not bounds between samples. Per-step actual-state finite/warning/activation extrema are checked.',
            'Deterministic threshold spike events can change discretely with body feedback; do not assume a smooth control function or uniform long-time convergence.',
            'Probe event timing covers mapped motor and stimulated sensory cells; full-tier counts are at 5ms, not full-tier subinterval raster.',
            'No biological validation, no standing claim, no tuned post-hoc threshold.'],memory_guard=guard,wall_s=time.monotonic()-start)
    dump(OUT/'clock_refinement_results.json',doc)
    print('COMPLETED',doc['status'],'wall_s',doc['wall_s'],flush=True)
    print(json.dumps({k:{w:{x:y for x,y in v.items() if x!='joint_angle_max_error_rad'} for w,v in c['windows_ms'].items()} for k,c in comparisons.items()},indent=2),flush=True)
    return doc

if __name__=='__main__':
    def deadline(*_): raise TimeoutError('60 second clock-refinement process deadline')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(60)
    with closure.mg.guarded_run('clock_refinement',**closure.mg.BUDGETS['neural_muscle_closure']) as report: main(report)
    signal.alarm(0)
