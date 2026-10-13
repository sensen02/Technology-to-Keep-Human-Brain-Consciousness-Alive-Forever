"""Manual bounded fine-clock actual-neural branch; no canonical edits or tuning."""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[_k]='1'
import json, re, signal, sys, time, shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import run_zero_floor_branch as zero
from engine.embodied import branch_episode as branch
closure=zero.closure
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/model_branches/fine_clock'
sha=zero.sha

def dump(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def compiled_diff(a,b):
    equal=[]; changed=[]
    for prefix in ('','opt.','vis.','stat.'):
        aa,bb=(a,b) if not prefix else (getattr(a,prefix[:-1]),getattr(b,prefix[:-1]))
        for name in dir(aa):
            if name.startswith('_'): continue
            x,y=getattr(aa,name),getattr(bb,name)
            if isinstance(x,np.ndarray):
                assert np.array_equal(x,y),prefix+name
                equal.append(prefix+name)
            elif isinstance(x,(str,bytes,int,float,bool)) and x!=y: changed.append(prefix+name)
    assert changed==['opt.timestep'],changed
    return dict(equal_arrays=equal,changed_scalar_fields=changed)

def build_models():
    import mujoco
    text=zero.XML.read_text(); base=mujoco.MjModel.from_xml_path(str(zero.XML))
    models={100:base}; reports={}
    for us in (50,25):
        updated,n=re.subn(r'(<option\b[^>]*\btimestep=")[^"]+(\")',lambda m:m[1]+str(us*1e-6)+m[2],text)
        assert n==1
        path=OUT/f'fruitfly_zero_floor_{us}us.xml';path.write_text(updated)
        model=mujoco.MjModel.from_xml_path(str(path)); models[us]=model
        reports[str(us)]=dict(xml=str(path),sha256=sha(path),comparison=compiled_diff(base,model))
        assert np.isclose(model.opt.timestep,us*1e-6,rtol=0,atol=1e-15)
    return models,reports

def run_arm(model,pose,joints,geom_leg,built,manifest,groups,us,arm,replay=None):
    import mujoco
    dt=us*1e-6; every=round(.0001/dt); steps=0; neural_steps=0
    rows={k:[] for k in ('body_qpos','body_qvel','body_act','body_ctrl','heights_mm','body_nonfoot_contacts','body_foot_contacts','body_warning_counts')}
    extrema=dict(act_min=0.,act_max=0.,act_above_one_values=0,act_below_zero_values=0,ctrl_saturated_values=0,act_saturated_values=0)
    warning_max=np.zeros(len(mujoco.MjData(model).warning),dtype=int); neural_finite=True; decoder_finite=True
    forward,step=mujoco.mj_forward,mujoco.mj_step
    network,decoder=branch.ConductanceNetwork,branch.MotorOutput
    floor=closure.cp.floor_geom_id(model); feet=set(closure.cp.tarsus_geom_ids(model));thorax=model.body('Thorax').id
    onset={'nonfoot':None,'foot':None}; command_valid=[]
    def finite_object(obj):return all(np.isfinite(v).all() for v in vars(obj).values() if isinstance(v,np.ndarray))
    class Network(network):
        def step(self,*a,**kw):
            nonlocal neural_steps,neural_finite
            out=super().step(*a,**kw);neural_steps+=1;neural_finite &= finite_object(self)
            assert neural_finite
            return out
    class Decoder(decoder):
        def consume(self,*a,**kw):
            nonlocal decoder_finite
            out=super().consume(*a,**kw);decoder_finite &= finite_object(self);assert decoder_finite
            return out
    def observed_step(m,d):
        nonlocal steps,warning_max
        if steps % round(.005/dt)==0:
            command_valid.append(bool(np.isfinite(d.ctrl).all() and np.all(d.ctrl>=0) and np.all(d.ctrl<=1)))
        step(m,d);steps+=1
        assert all(np.isfinite(getattr(d,k)).all() for k in zero.duration.BODY_FIELDS)
        warnings=np.array([w.number for w in d.warning]);warning_max=np.maximum(warning_max,warnings);assert not warnings.any()
        extrema['act_min']=min(extrema['act_min'],float(d.act.min()));extrema['act_max']=max(extrema['act_max'],float(d.act.max()))
        extrema['act_above_one_values']+=int((d.act>1+1e-12).sum());extrema['act_below_zero_values']+=int((d.act< -1e-12).sum())
        extrema['act_saturated_values']+=int((d.act>=1-1e-12).sum());extrema['ctrl_saturated_values']+=int((d.ctrl>=1-1e-12).sum())
    def observed_forward(m,d):
        forward(m,d)
        if steps % every: return
        nonfoot=0; foot=0
        for c in d.contact:
            if floor in (int(c.geom1),int(c.geom2)):
                other=int(c.geom2) if int(c.geom1)==floor else int(c.geom1)
                if other in feet: foot+=1
                else: nonfoot+=1
        for label,count in (('foot',foot),('nonfoot',nonfoot)):
            if count and onset[label] is None:onset[label]=steps*dt
        for k in ('qpos','qvel','act','ctrl'):rows['body_'+k].append(getattr(d,k).copy())
        rows['heights_mm'].append([d.qpos[2],d.xpos[thorax,2],d.subtree_com[0,2]])
        rows['body_nonfoot_contacts'].append(nonfoot);rows['body_foot_contacts'].append(foot)
        rows['body_warning_counts'].append([int(w.number) for w in d.warning])
    with patch.object(mujoco,'mj_step',observed_step),patch.object(mujoco,'mj_forward',observed_forward),patch.object(branch,'ConductanceNetwork',Network),patch.object(branch,'MotorOutput',Decoder):
        summary,traj,controls,spikes=branch.episode(model,pose,joints,geom_leg,built,manifest,groups,arm,.2,replay=replay,body_dt=dt)
    result={k:np.asarray(v) for k,v in rows.items()}
    result.update(trajectory=traj,controls=controls,spike_counts=spikes,body_time_s=np.arange(2001)*.0001,command_valid=np.array(command_valid))
    assert steps==round(.2/dt) and neural_steps==400 and len(rows['body_act'])==2001 and len(command_valid)==40 and all(command_valid)
    assert all(np.all(result[k][0]==0) for k in ('body_act','body_ctrl','body_qvel'))
    summary.update(body_dt_s=dt,completed_body_steps=steps,actual_neural_steps=neural_steps,body_samples=2001,command_valid_per_5ms=command_valid,
                   activation=extrema,health=dict(all_fine_body_steps_finite=True,all_neural_steps_finite=neural_finite,decoder_finite=decoder_finite,warning_max=warning_max.tolist()),
                   contact_onset_s=onset,contact_onset_resolution_s=.0001,endpoint_heights_mm=result['heights_mm'][-1].tolist(),
                   com_z_min_mm=float(result['heights_mm'][:,2].min()),com_z_displacement_mm=float(result['heights_mm'][-1,2]-result['heights_mm'][0,2]))
    return summary,result

def main(report):
    import mujoco
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    provenance=branch.verify_source_copy();(OUT/'episode_source.diff').write_text(provenance['diff'])
    source_files={}
    for mod in list(sys.modules.values()):
        p=getattr(mod,'__file__',None)
        if p and Path(p).suffix=='.py' and Path(p).resolve().is_relative_to(HERE) and not any(t in Path(p).parts for t in ('venv','venv_body')):source_files[str(Path(p).resolve())]=sha(p)
    source_files[str(HERE/'test_fine_clock_branch.py')]=sha(HERE/'test_fine_clock_branch.py')
    for p in (closure.cp.CANONICAL_XML,zero.XML,zero.OUT/'zero_floor_results.json',zero.OUT/'zero_floor_traces.npz',closure.OUT/'current_run.json'):source_files[str(p)]=sha(p)
    bundle=zero.validate_bundle(closure.OUT)
    for p in Path(bundle['path']).iterdir():
        if p.is_file():source_files[str(p)]=sha(p)
    archive=OUT/'source_archive'
    for p,h in source_files.items():
        path=Path(p)
        if path.suffix=='.py':
            dest=archive/path.relative_to(HERE);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest);assert sha(dest)==h
    dump(OUT/'source_hashes_before.json',source_files)
    models,model_reports=build_models()
    _,pose,_,joints,geom_leg=closure.prepare_body()
    with np.load(zero.OUT/'zero_floor_traces.npz') as old:assert np.array_equal(pose,old['seated_qpos'])
    built=closure.build_neural_tier(closure.TierConfig(background_mean_nA=0,background_sd_nA=0));assert built['n']==12000
    names=[mujoco.mj_id2name(models[100],mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(models[100].nu)]
    manifest=closure.build_motor_manifest(json.loads(closure.POOLS.read_text()),built['tier']['root_ids'],names)
    groups,_=closure.channel_index_map(int(built['n']),built['tier']['root_ids'])
    episodes={};results={};checks={};arrays={'seated_qpos':pose,'tier_root_ids':np.asarray(built['tier']['root_ids'])}
    for us,arm in ((100,'normal'),(50,'normal'),(25,'normal'),(25,'spike_output_silence'),(25,'replay')):
        key=f'{us}us_{arm}'
        summary,result=run_arm(models[us],pose,joints,geom_leg,built,manifest,groups,us,arm,results['25us_normal']['controls'] if arm=='replay' else None)
        episodes[key]=summary;results[key]=result;arrays.update({key+'_'+k:v for k,v in result.items()})
        if us==100:
            with np.load(zero.OUT/'zero_floor_traces.npz') as old:
                for k in ('trajectory','controls','spike_counts','body_qpos','body_qvel','body_act','body_ctrl'):
                    checks['100us_saved_zero_floor_exact_'+k]=bool(np.array_equal(result[k],old['normal_'+k]))
            assert all(checks.values()),checks
        print(key,'steps',summary['completed_body_steps'],'activation',summary['activation'],'wall',time.monotonic()-start,flush=True)
    for k in ('trajectory','controls','spike_counts','body_qpos','body_qvel','body_act','body_ctrl','heights_mm'):
        checks['25us_replay_exact_'+k]=bool(np.array_equal(results['25us_normal'][k],results['25us_replay'][k]))
    for k in ('controls','body_ctrl','body_act','spike_counts'):checks['25us_silence_zero_'+k]=bool(np.all(results['25us_spike_output_silence'][k]==0))
    comparisons={}
    for a,b in (('100us_normal','50us_normal'),('50us_normal','25us_normal'),('25us_normal','25us_spike_output_silence')):
        comparisons[a+'_vs_'+b]={k:dict(max_abs_difference=float(np.max(abs(results[a][k].astype(float)-results[b][k].astype(float)))),exact=bool(np.array_equal(results[a][k],results[b][k]))) for k in ('trajectory','controls','spike_counts','body_qpos','body_act','heights_mm')}
    after={p:sha(p) for p in source_files};dump(OUT/'source_hashes_after.json',after);checks['frozen_sources_unchanged']=after==source_files
    for us in (50,25):checks[f'{us}us_model_unchanged']=compiled_diff(models[100],models[us])==model_reports[str(us)]['comparison']
    assert all(checks.values()),checks
    npz=OUT/'fine_clock_traces.npz';np.savez_compressed(npz,**arrays)
    guard=report('completed')
    if not np.isfinite(guard['previous_soft_limit_mb']):guard['previous_soft_limit_mb']='unlimited'
    doc=dict(schema_version=1,status='PASS_BRANCH_INVARIANTS',execution_status='completed',episodes=episodes,checks=checks,comparisons=comparisons,model_manifest=model_reports,
             protocol=dict(duration_s=.2,neural_dt_s=.0005,command_and_sensory_dt_s=.005,neurons=12000,no_cpg_pd=True,no_gain_changes=True,no_activation_clamp=True,no_parameter_search=True,zero_initial_qvel_act_ctrl=True,replay_neural_integration=True),
             provenance=dict(episode_source=provenance,source_hashes_before=source_files,source_hashes_after=after,npz_sha256=sha(npz),mujoco_version=mujoco.__version__),
             schema=dict(body_sampling='2001 rows at 0.1ms including initial, all arms; activation extrema and finite/warning checks every fine step',heights_mm_columns=['root','thorax','whole_body_com'],trajectory='40 post-command endpoints: qpos,qvel,act',controls='40x90 applied commands',spike_counts='40x12000 emitted counts; silence zeros output only',saturation='act/ctrl >=1-1e-12 element-step counts, denominator completed_body_steps*90',contact_onset='sampled at 0.1ms; escape_time_s checks every fine step'),
             limitations=['Manual experimental implementation, not a finished controller.','Bounded 200ms only; no standing or biological validation claim.','25us versus 50us differences are not evidence of numerical convergence.','Zero-floor and fine-timestep XMLs are authorized alternative branches, canonical remains frozen.'],memory_guard=guard,wall_s=time.monotonic()-start)
    dump(OUT/'fine_clock_results.json',doc);print('COMPLETED',doc['status'],doc['wall_s'],flush=True)
    return doc

if __name__=='__main__':
    def deadline(*_):raise TimeoutError('60 second fine-clock process deadline')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(60)
    with closure.mg.guarded_run('fine_clock_branch',**closure.mg.BUDGETS['neural_muscle_closure']) as report:main(report)
    signal.alarm(0)
