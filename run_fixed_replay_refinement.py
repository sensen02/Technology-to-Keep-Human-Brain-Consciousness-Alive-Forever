"""Explicit diagnostic prescribed-command replay, NOT autonomous or a controller cure.
Exactly two 200ms episodes; unchanged fine.run_arm and dt-only branch episode.
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[_k]='1'
import hashlib, json, shutil, signal, sys, time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import run_clock_refinement as refinement
fine=refinement.fine; closure=fine.closure; branch=fine.branch; zero=fine.zero; sha=fine.sha
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/model_branches/fixed_replay_refinement'
NEW_US=(12.5,6.25)
SOURCE_KEY='6.25us_normal'
METRICS=('max_com_distance_mm','max_root_translation_distance_mm','max_root_orientation_angle_rad','max_individual_joint_angle_error_rad')
def label(us): return f'{us:g}us_replay'
def dump(p,x): refinement.dump(p,x)
def array_hash(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def load_history():
    refinement.validate_history()
    doc=json.loads((refinement.OUT/'clock_refinement_results.json').read_text())
    assert doc['execution_status']=='completed' and all(doc['checks'].values())
    assert sha(refinement.OUT/'clock_refinement_traces.npz')==doc['provenance']['npz_sha256']
    assert doc['provenance']['source_hashes_before']==doc['provenance']['source_hashes_after']
    for p,h in doc['provenance']['source_hashes_after'].items(): assert sha(p)==h,p
    for item in doc['model_manifest'].values(): assert sha(item['xml'])==item['sha256']
    return doc

def observed_replay(model,pose,joints,geom_leg,built,manifest,groups,us,commands):
    """Observe every applied step outside unchanged fine.run_arm; never alter state."""
    import mujoco
    step=mujoco.mj_step; count=0; first={}; steps_per_command=round(.005/(us*1e-6))
    def audit(m,d):
        nonlocal count
        assert np.array_equal(d.ctrl,commands[count//steps_per_command])
        if count==0:
            assert np.array_equal(d.qpos,pose)
            assert all(np.all(getattr(d,k)==0) for k in ('qvel','act','ctrl'))
            first.update(time_before_s=float(d.time),declared_model_dt_s=float(m.opt.timestep),initial_qpos_exact=True,initial_qvel_act_ctrl_zero=True)
        step(m,d)
        if count==0:
            first['time_after_s']=float(d.time)
            assert d.time==m.opt.timestep
        count+=1
    with patch.object(mujoco,'mj_step',audit):
        summary,result=fine.run_arm(model,pose,joints,geom_leg,built,manifest,groups,us,'replay',replay=commands)
    assert count==round(.2/(us*1e-6))
    assert summary['activation']['act_min']>=0 and summary['activation']['act_max']<=1
    summary.update(sample_integer_stride=round(.0001/(us*1e-6)),applied_command_exact_checked_body_steps=count,first_step=first)
    return summary,result

def main(report):
    import mujoco
    start=time.monotonic(); OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'protocol_before_run.json').exists(),'Do not repeat the two-episode diagnostic'
    history=load_history(); provenance=branch.verify_source_copy()
    source_npz=refinement.OUT/'clock_refinement_traces.npz'
    with np.load(source_npz) as saved:
        source={k[len(SOURCE_KEY)+1:]:saved[k].copy() for k in saved.files if k.startswith(SOURCE_KEY+'_')}
        source_pose=saved['seated_qpos'].copy(); source_ids=saved['tier_root_ids'].copy()
    commands=source['controls'].copy(); commands.setflags(write=False)
    assert commands.shape==(40,90) and np.isfinite(commands).all()
    protocol=dict(new_cases=[label(x) for x in NEW_US],maximum_new_episodes=2,duration_s=.2,neural_dt_s=.0005,command_and_sensory_dt_s=.005,
        sample_dt_s=.0001,windows_ms=list(refinement.WINDOWS_MS),window_definition=history['protocol']['window_definition'],
        source_commands_npz=str(source_npz),source_commands_key=SOURCE_KEY+'_controls',source_commands_npz_sha256=sha(source_npz),
        source_commands_array_sha256=array_hash(commands),array_hash_encoding='C-contiguous raw bytes; dtype and shape recorded separately',source_commands_dtype=str(commands.dtype),source_commands_shape=list(commands.shape),
        explicitly_diagnostic=True,autonomous=False,actual_neural_integration=True,neural_decoder_output_applied=False,
        controller_cure=False,convergence_claim=False,entire_causal_attribution=False,
        inference_scope='Conditional mechanical timestep sensitivity under the one saved 6.25us NORMAL command schedule, not complete attribution of closed-loop error.',
        com_resolution_reference_mm=refinement.COM_REFERENCE_MM,threshold_role='unchanged upfront descriptive threshold, not an acceptance criterion',
        no_gain_control_activation_changes=True,no_activation_clamp=True,no_new_controller=True,no_parameter_search=True,
        reuse_prior_xml_and_metadata_unchanged=True,threads=1,process_deadline_s=60)
    dump(OUT/'protocol_before_run.json',protocol)
    frozen=dict(history['provenance']['source_hashes_after'])
    for mod in list(sys.modules.values()):
        p=getattr(mod,'__file__',None)
        if p and Path(p).suffix=='.py' and Path(p).resolve().is_relative_to(HERE) and not any(t in Path(p).parts for t in ('venv','venv_body')): frozen[str(Path(p).resolve())]=sha(p)
    for p in (HERE/'test_fixed_replay_refinement.py',refinement.OUT/'clock_refinement_results.json',source_npz,OUT/'protocol_before_run.json',refinement.OUT/'source_hashes_before.json',refinement.OUT/'source_hashes_after.json'):
        frozen[str(p)]=sha(p)
    for item in history['model_manifest'].values(): frozen[item['xml']]=item['sha256']
    for p,h in frozen.items():
        path=Path(p)
        if path.suffix=='.py':
            dest=OUT/'source_archive'/path.relative_to(HERE);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest);assert sha(dest)==h
    dump(OUT/'source_hashes_before.json',frozen)
    base=mujoco.MjModel.from_xml_path(str(zero.XML));models={}
    for us in NEW_US:
        item=history['model_manifest'][str(us)]; model=mujoco.MjModel.from_xml_path(item['xml']);models[us]=model
        assert fine.compiled_diff(base,model)==item['compiled_comparisons']['zero_floor_100us']
    _,pose,_,joints,geom_leg=closure.prepare_body()
    assert np.array_equal(pose,source_pose)
    built=closure.build_neural_tier(closure.TierConfig(background_mean_nA=0,background_sd_nA=0));assert built['n']==12000
    assert np.array_equal(np.asarray(built['tier']['root_ids']),source_ids)
    names=[mujoco.mj_id2name(base,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(base.nu)]
    manifest=closure.build_motor_manifest(json.loads(closure.POOLS.read_text()),built['tier']['root_ids'],names)
    groups,_=closure.channel_index_map(int(built['n']),built['tier']['root_ids'])
    episodes={};results={};checks={};arrays={'seated_qpos':pose,'tier_root_ids':source_ids,'reference_commands':commands}
    for us in NEW_US:
        key=label(us)
        episodes[key],results[key]=observed_replay(models[us],pose,joints,geom_leg,built,manifest,groups,us,commands)
        r=results[key];r['body_com_xyz_mm']=refinement.geometry(base,r['body_qpos'])
        assert np.array_equal(r['body_time_s'],np.arange(2001)*.0001)
        checks[key+'_commands_array_exact']=bool(np.array_equal(r['controls'],commands))
        checks[key+'_command_hash_exact']=array_hash(r['controls'])==protocol['source_commands_array_sha256']
        checks[key+'_initial_pose_exact']=bool(np.array_equal(r['body_qpos'][0],pose))
        arrays.update({key+'_'+k:v for k,v in r.items()})
        print(key,'steps',episodes[key]['completed_body_steps'],'wall_s',time.monotonic()-start,flush=True)
    for name,value in source.items():
        checks['6.25us_saved_normal_exact_'+name]=bool(np.array_equal(results[label(6.25)][name],value))
    checks['6.25us_saved_normal_records_exact']=episodes[label(6.25)]['records']==[{**r,'command_origin':'offline_normal_replay'} for r in history['episodes'][SOURCE_KEY]['records']]
    joint_names=history['joint_names'];qadr=np.asarray(history['joint_qpos_addresses'])
    comparison=refinement.compare(results[label(12.5)],results[label(6.25)],episodes[label(12.5)],episodes[label(6.25)],joint_names,qadr)
    closed=history['comparisons']['12.5us_normal_vs_6.25us_normal'];relative={}
    for ms in refinement.WINDOWS_MS:
        w=str(ms);relative[w]={}
        for metric in METRICS:
            a=comparison['windows_ms'][w][metric];b=closed['windows_ms'][w][metric]
            relative[w][metric]=dict(fixed_replay=a,closed_feedback=b,decreased=a<b,ratio_fixed_over_closed=a/b if b else None)
    after={p:sha(p) for p in frozen};checks['frozen_sources_and_prior_artifacts_unchanged']=after==frozen
    checks['model_metadata_reused_exact']=history['model_manifest']==load_history()['model_manifest']
    assert all(checks.values()),{k:v for k,v in checks.items() if not v}
    dump(OUT/'source_hashes_after.json',after)
    npz=OUT/'fixed_replay_refinement_traces.npz';np.savez_compressed(npz,**arrays)
    guard=report('completed')
    if not np.isfinite(guard['previous_soft_limit_mb']):guard['previous_soft_limit_mb']='unlimited'
    doc=dict(schema_version=1,status='PASS_FIXED_REPLAY_DIAGNOSTIC_NOT_AUTONOMOUS_NOT_CURE',execution_status='completed',protocol=protocol,episodes=episodes,
        comparisons={'12.5us_replay_vs_6.25us_replay':comparison},closed_feedback_comparison=closed,relative_to_closed_feedback=relative,
        checks=checks,model_manifest=history['model_manifest'],joint_names=joint_names,joint_qpos_addresses=qadr.tolist(),
        provenance=dict(episode_source=provenance,source_hashes_before=frozen,source_hashes_after=after,npz_sha256=sha(npz),mujoco_version=mujoco.__version__),
        limitations=['Manual experimental implementation, not a finished controller.',
            'Prescribed offline replay is explicitly diagnostic, NOT autonomous control and NOT a cure for instability.',
            'Conditional fixed-command mechanical sensitivity only; does not provide entire causal attribution of closed-feedback divergence.',
            'One command schedule and two timesteps cannot establish global or near convergence.',
            'Raw integrated neural records/events may differ because body feedback differs, while applied commands are exactly identical.',
            '0.1ms sampled-window errors are not bounds between samples; every actual body step has finite/warning/activation and command checks.',
            'No standing or biological validation claim; descriptive 0.01mm threshold unchanged.'],memory_guard=guard,wall_s=time.monotonic()-start)
    dump(OUT/'fixed_replay_refinement_results.json',doc)
    print('COMPLETED',doc['status'],'wall_s',doc['wall_s'],flush=True)
    print(json.dumps(relative,indent=2),flush=True)
    return doc
if __name__=='__main__':
    def deadline(*_): raise TimeoutError('60 second fixed-replay refinement process deadline')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(60)
    with closure.mg.guarded_run('fixed_replay_refinement',**closure.mg.BUDGETS['neural_muscle_closure']) as report: main(report)
    signal.alarm(0)
