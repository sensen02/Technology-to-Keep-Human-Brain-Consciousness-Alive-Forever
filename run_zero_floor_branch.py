"""Manual floor-only alternative branch; frozen canonical inputs never edited."""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import json
import re
import signal
import sys
import time
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import run_neural_duration_probe as duration
import run_native_floor_audit as native
from engine.embodied.closure_bundle import validate_bundle
closure = duration.closure
HERE = Path(__file__).resolve().parent
OUT = HERE/'outputs/model_branches/zero_floor'
XML = OUT/'fruitfly_zero_floor.xml'
SHA = native.EXPECTED_XML_SHA
sha = native.sha

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def compiled_diff(a,b):
    equal, changed, strings = [], [], []
    for prefix in ('','opt.','vis.','stat.'):
        aa,bb = (a,b) if not prefix else (getattr(a,prefix[:-1]),getattr(b,prefix[:-1]))
        for name in dir(aa):
            if name.startswith('_'): continue
            x,y = getattr(aa,name),getattr(bb,name)
            key=prefix+name
            if isinstance(x,np.ndarray):
                (equal if np.array_equal(x,y) else changed).append(key)
            elif isinstance(x,(str,bytes,int,float,bool)) and x!=y:
                strings.append(key)
    assert changed == ['actuator_ctrlrange'], changed
    assert not strings or set(strings)<= {'paths','npaths'}, strings
    assert np.array_equal(a.actuator_ctrlrange[:,1],b.actuator_ctrlrange[:,1])
    assert np.all(a.actuator_ctrlrange[:,0]==.0001) and np.all(b.actuator_ctrlrange[:,0]==0)
    return dict(equal_arrays=equal,changed_arrays=changed,changed_scalar_or_string_fields=strings,changed_control_lower_count=90)

def build_branch():
    import mujoco
    source=closure.cp.CANONICAL_XML
    assert sha(source)==SHA
    OUT.mkdir(parents=True,exist_ok=True)
    text=source.read_text()
    assert text.count('ctrlrange="0.0001 1"')==90
    text=text.replace('ctrlrange="0.0001 1"','ctrlrange="0 1"')
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    spec=_load_mjcf(str(source))
    resolved={Path(mesh.file).name:Path(mesh.file) for mesh in spec.meshes}
    paths=[]
    def asset(match):
        old=match[1]; target=resolved[Path(old).name].resolve()
        assert target.is_file()
        paths.append(dict(original=old,branch=str(target),sha256=sha(target)))
        return 'file="'+str(target)+'"'
    text=re.sub(r'file="([^"]+)"',asset,text)
    XML.write_text(text)
    a=closure.cp.load_model(); b=mujoco.MjModel.from_xml_path(str(XML))
    assert b.nu==b.na==90
    report=compiled_diff(a,b)
    report.update(canonical_xml=str(source),canonical_sha256=SHA,branch_xml=str(XML),branch_sha256=sha(XML),declared_asset_path_changes=paths,geometry_content_changes=False)
    dump(OUT/'model_manifest.json',report)
    return a,b,report

def run_arm(model,pose,joints,geom_leg,built,manifest,groups,arm,replay=None):
    import mujoco
    original_episode=closure.episode
    forward=mujoco.mj_forward
    live={k:[] for k in ('act','ctrl','act_dot','actuator_force','qfrc_actuator','qfrc_passive')}
    def observer(m,d):
        forward(m,d)
        for k in live: live[k].append(getattr(d,k).copy())
    def episode(*args,**kwargs):
        args=list(args);args[7]=arm
        return original_episode(*args,**kwargs,replay=replay)
    with patch.object(closure,'episode',episode),patch.object(mujoco,'mj_forward',observer):
        summary,result=duration.run_arm(model,pose,joints,geom_leg,built,manifest,groups)
    # Observer also sees no scratch mj_forward calls in duration.run_arm.
    result.update({'body_'+k:np.asarray(v) for k,v in live.items()})
    assert len(result['body_act'])==2001
    # Same-state zero-activation scratch forces, never an integrated activation reset.
    scratch=mujoco.MjData(model)
    passive=[]
    for q,v in zip(result['body_qpos'],result['body_qvel']):
        scratch.qpos[:]=q;scratch.qvel[:]=v;scratch.act[:]=0;scratch.ctrl[:]=0
        forward(model,scratch)
        passive.append(scratch.actuator_force.copy())
    result['body_act0_actuator_force']=np.asarray(passive)
    result['body_active_delta_actuator_force']=result['body_actuator_force']-result['body_act0_actuator_force']
    summary['force_summary']=dict(max_abs_active_delta=float(abs(result['body_active_delta_actuator_force']).max()),max_abs_passive_muscle=float(abs(result['body_act0_actuator_force']).max()),max_activation=float(result['body_act'].max()))
    return summary,result

def main(report):
    import mujoco
    start=time.monotonic()
    bundle=validate_bundle(closure.OUT); source=Path(bundle['path'])
    pointer=closure.OUT/'current_run.json'; pointer_sha=sha(pointer)
    doc=json.loads((source/'closure_results.json').read_text())
    source_files=dict(doc['provenance']['code_hashes'])
    for name in ('manifest.json','closure_results.json','closure_traces.npz'):
        source_files[str(source/name)]=sha(source/name)
    baseline=closure.OUT/'neural_duration_probe.npz'
    source_files[str(baseline)]=sha(baseline)
    for module in (duration,duration.pose_runner,native): source_files[str(Path(module.__file__).resolve())]=sha(module.__file__)
    source_files[str(Path(__file__).resolve())]=sha(__file__)
    for p,h in source_files.items(): assert sha(p)==h,p
    canonical,model,model_report=build_branch()
    _,prepared,_,joints,geom_leg=closure.prepare_body() # setup only, no canonical episode
    with np.load(source/'closure_traces.npz') as old: pose=old['seated_qpos'].copy()
    assert np.array_equal(pose,prepared)
    built=closure.build_neural_tier(closure.TierConfig(background_mean_nA=0,background_sd_nA=0))
    assert int(built['n'])==12000
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(model.nu)]
    manifest=closure.build_motor_manifest(json.loads(closure.POOLS.read_text()),built['tier']['root_ids'],names)
    groups,_=closure.channel_index_map(int(built['n']),built['tier']['root_ids'])
    arrays=dict(seated_qpos=pose,tier_root_ids=np.asarray(built['tier']['root_ids']))
    episodes={}; results={}
    for arm in ('normal','spike_output_silence','replay'):
        summary,result=run_arm(model,pose,joints,geom_leg,built,manifest,groups,arm,results['normal']['controls'] if arm=='replay' else None)
        episodes[arm]=summary;results[arm]=result
        arrays.update({arm+'_'+k:v for k,v in result.items()})
        print(arm,summary['standing_status'],summary['force_summary'],flush=True)
    checks={}
    for k in ('trajectory','controls','body_qpos','body_qvel','body_act','body_ctrl','body_actuator_force','heights_mm'):
        checks['replay_exact_'+k]=bool(np.array_equal(results['normal'][k],results['replay'][k]))
    silence=results['spike_output_silence']
    for k in ('controls','body_ctrl','body_act','body_act_dot','body_active_delta_actuator_force'):
        checks['silence_exact_zero_'+k]=bool(np.all(silence[k]==0))
    for arm,r in results.items():
        checks[arm+'_zero_initial']=all(np.all(r[k][0]==0) for k in ('body_qvel','body_act','body_ctrl'))
        checks[arm+'_same_pose']=bool(np.array_equal(r['body_qpos'][0],pose))
        checks[arm+'_2001_finite_warning_free']=len(r['body_qpos'])==2001 and bool(r['body_finite'].all()) and not bool(r['body_warning_counts'].any())
    comparison={}
    with np.load(baseline) as old, np.load(source/'closure_traces.npz') as frozen:
        for k in ('trajectory','controls','spike_counts'):
            checks['saved_canonical_50ms_prefix_'+k]=bool(np.array_equal(old['original_normal_'+k][:10],frozen['normal_'+k]))
        for k in ('trajectory','controls','spike_counts','body_qpos','heights_mm'):
            x=old['original_normal_'+k];y=results['normal'][k]
            n=501 if k in ('body_qpos','heights_mm') else 10
            comparison[k]=dict(first_50ms_max_abs_difference=float(np.max(np.abs(x[:n].astype(float)-y[:n].astype(float)))),full_200ms_max_abs_difference=float(np.max(np.abs(x.astype(float)-y.astype(float)))),first_50ms_exact=bool(np.array_equal(x[:n],y[:n])))
    checks['canonical_hash_unchanged']=sha(closure.cp.CANONICAL_XML)==SHA
    checks['pointer_unchanged']=sha(pointer)==pointer_sha
    checks['source_files_unchanged']=all(sha(p)==h for p,h in source_files.items())
    checks['compiled_model_unchanged_during_episodes']=compiled_diff(canonical,model)=={k:model_report[k] for k in ('equal_arrays','changed_arrays','changed_scalar_or_string_fields','changed_control_lower_count')}
    assert all(checks.values()),checks
    output=OUT/'zero_floor_traces.npz';np.savez_compressed(output,**arrays)
    # Snapshot all loaded project Python modules plus explicit provenance sources.
    archive=OUT/'source_archive';archived={}
    for mod in list(sys.modules.values()):
        p=getattr(mod,'__file__',None)
        if p and Path(p).suffix=='.py' and Path(p).resolve().is_relative_to(HERE) and not any(t in Path(p).parts for t in ('venv','venv_body')):
            source_files.setdefault(str(Path(p).resolve()),sha(p))
    for p,h in source_files.items():
        path=Path(p)
        if path.suffix!='.py': continue
        dest=archive/path.relative_to(HERE);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        assert sha(dest)==h
        archived[str(dest)]=h
    guard=report('completed')
    if not np.isfinite(guard['previous_soft_limit_mb']):guard['previous_soft_limit_mb']='unlimited'
    result=dict(schema_version=1,status='PASS_BRANCH_INVARIANTS',standing_status='PASS' if all(e['standing_status']=='PASS' for e in episodes.values()) else 'FAIL',execution_status='completed',duration_s=.2,episodes=episodes,checks=checks,canonical_normal_comparison=comparison,model_manifest=model_report,motor_manifest=manifest,protocol=dict(neurons=12000,initialization='frozen seated_qpos; qvel/act/ctrl zero; same pose proprioceptive reference',arms=['normal','spike_output_silence','replay'],no_cpg_pd=True,no_gain_changes=True,no_stepwise_activation_reset=True,canonical_additional_episodes=0),provenance=dict(source_bundle=bundle,source_hashes=source_files,source_archive_hashes=archived,current_pointer_sha256=pointer_sha,npz_sha256=sha(output),mujoco_version=mujoco.__version__),schema=dict(body_metrics_columns=['strict_limit_count','max_limit_excess_rad','nonfoot_floor_contacts','max_abs_angular_qvel_rad_s','max_abs_translational_qvel_mm_s'],heights_mm_columns=['root','thorax','whole_body_com'],body_sampling='2001 initial + every 0.1ms step',motor_sampling='40 applied commands, 40 x 12000 emitted spike counts; raw motor probe events in episode records',force='model units; total minus same-state zero-act scratch = activation dependent; passive muscle and qfrc_passive permitted'),limitations=['Manual experimental implementation, not finished controller.','No standing guarantee; historical initial joint-limit violations preserved.','Zero floor is independently authorized model branch, not canonical repair.','Bounded 200ms experiment, not indefinite stability or biological validation.'],memory_guard=guard,wall_s=time.monotonic()-start)
    dump(OUT/'zero_floor_results.json',result)
    print('COMPLETED',result['status'],'standing',result['standing_status'],'wall_s',result['wall_s'],flush=True)
    return result

if __name__=='__main__':
    def deadline(*_): raise TimeoutError('60 second branch process deadline')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(60)
    with closure.mg.guarded_run('zero_floor_branch',**closure.mg.BUDGETS['neural_muscle_closure']) as report:main(report)
    signal.alarm(0)
