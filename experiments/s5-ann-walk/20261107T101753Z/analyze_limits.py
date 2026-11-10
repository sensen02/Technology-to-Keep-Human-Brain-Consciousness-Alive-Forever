"""Read-only matched finite-PD tracking/cap diagnosis from saved traces.

No mj_step. BODY is sampled every5ms; transient native-step saturation between
samples cannot be excluded. Raw endpoint PD demand is not the last applied torque.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import mujoco as mj
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE=ROOT/'experiments/s5-ann-incremental/20261107T010000Z'
sys.path.insert(0,str(BASE))
from plant_probe import Plant
from incremental_actuator import DEFAULT_MODEL_PATH
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T101753Z')
RUNS=dict(gain32=OUT.parent/'20261107T101114Z/neural/teacher_lift_4s',
          gain128=OUT/'teacher_gain128_4s')
LEGS=('LF','LM','LH','RF','RM','RH')
WEIGHT=24.446354755878005


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def max_run(mask,dt):
    lengths=[];n=0
    for value in mask:
        if value:n+=1
        elif n:lengths.append(n);n=0
    if n:lengths.append(n)
    return float(max(lengths,default=0)*dt)


def summary(error,weights):
    return dict(RMS_rad=float(np.sqrt(np.mean(error*error))),max_abs_rad=float(abs(error).max()),
        normal_force_weighted_RMS_rad=float(np.sqrt(np.sum(weights[:,None]*error*error)/max(weights.sum()*error.shape[1],1e-30))))


def analyze(folder,p,offset,duty):
    B=load(folder/'BODY.npz');C=load(folder/'COMMANDS.npz');N=load(folder/'NATIVE.npz')['rows']
    rows=C['rows'];assert rows.shape[1]==127
    lo,hi=p.m.jnt_range[p.jids].T
    target=rows[0,1:43].copy();alltargets=[];pre_error=0.
    for row in rows:
        pre_error=max(pre_error,float(abs(target-row[1:43]).max()))
        target=np.clip(target+.001*np.clip(row[43:85]+row[85:127],-30.,30.),lo,hi)
        alltargets.append(target.copy())
    alltargets=np.asarray(alltargets)
    assert pre_error<1e-12
    end_index=np.rint(B['time']/.001).astype(int)-1
    native_index=np.rint(B['time']/.0000125).astype(int)-1
    end=alltargets[end_index]
    dt=float(np.median(np.diff(B['time'])))
    alignment_time=float(abs(B['time']-(rows[end_index,0]+.001)).max())
    load_alignment=float(abs(B['time']-N[native_index,0]).max())
    assert alignment_time<1e-9 and load_alignment<1e-12
    q=B['qpos'][:,p.qa];v=B['qvel'][:,p.va];error=end-q
    limits=C['limits'];assert np.array_equal(limits,p.limits)
    assert np.max(abs(B['ctrl'])-limits[None,:])<1e-9
    cap=np.isclose(abs(B['ctrl']),limits[None,:],rtol=1e-7,atol=1e-9)
    raw=C['kp'][None,:]*error-C['kd'][None,:]*v
    rawcap=abs(raw)>=limits[None,:]
    release=B['time']>=1.-1e-9
    local=((B['time'][:,None]-.25)+offset[None,:])%1.
    stance=local<duty
    forces=np.maximum(N[native_index,1:7],0.)
    names=[p.m.joint(int(j)).name for j in p.jids]
    perleg={}
    for li,leg in enumerate(LEGS):
        ix=np.array([i for i,name in enumerate(names) if name.startswith('joint_'+leg)])
        assert len(ix)==7
        mask=release&stance[:,li];loaded=mask&(forces[:,li]>.001*WEIGHT)
        weights=forces[mask,li];e=error[mask][:,ix]
        perjoint=[]
        for j in ix:
            cmask=cap[:,j]&mask;cload=cap[:,j]&loaded
            perjoint.append(dict(joint=names[j],RMS_rad=float(np.sqrt(np.mean(error[mask,j]**2))),
                max_abs_error_rad=float(abs(error[mask,j]).max()),
                signed_error_at_max=float(error[np.flatnonzero(mask)[np.argmax(abs(error[mask,j]))],j]),
                applied_native_torque_limit=float(limits[j]),
                sampled_stance_cap_fraction=float(cap[mask,j].mean()),
                sampled_loaded_stance_cap_fraction=float(cap[loaded,j].mean()) if loaded.any() else None,
                normal_force_weighted_stance_cap_fraction=float(np.sum(weights*cap[mask,j])/max(weights.sum(),1e-30)),
                sampled_max_continuous_cap_duration_s=max_run(cmask,dt),
                sampled_max_continuous_loaded_cap_duration_s=max_run(cload,dt),
                mean_absolute_torque_to_limit=float(np.mean(abs(B['ctrl'][mask,j])/limits[j])),
                p95_absolute_torque_to_limit=float(np.percentile(abs(B['ctrl'][mask,j])/limits[j],95)),
                raw_endpoint_PD_over_cap_fraction=float(rawcap[mask,j].mean()),
                raw_endpoint_PD_to_limit_max=float((abs(raw[mask,j])/limits[j]).max())))
        worst=max(perjoint,key=lambda x:x['max_abs_error_rad'])
        worst_rms=max(perjoint,key=lambda x:x['RMS_rad'])
        anycap=cap[:,ix].any(1)
        middle=mask&(local[:,li]>=.1*duty)&(local[:,li]<=.9*duty)
        loaded_middle=middle&(forces[:,li]>.001*WEIGHT)
        perleg[leg]=dict(stance=summary(e,weights),
            loaded_stance=summary(error[loaded][:,ix],forces[loaded,li]) if loaded.any() else None,
            stance_samples=int(mask.sum()),loaded_stance_samples=int(loaded.sum()),
            loaded_fraction_above_001W=float(loaded.sum()/mask.sum()),
            mean_stance_foot_force_W=float(forces[mask,li].mean()/WEIGHT),
            sampled_stance_normal_impulse=float(forces[mask,li].sum()*dt),
            worst_max_joint=worst['joint'],worst_RMS_joint=worst_rms['joint'],
            sampled_stance_any_joint_cap_fraction=float(anycap[mask].mean()),
            sampled_loaded_stance_any_joint_cap_fraction=float(anycap[loaded].mean()) if loaded.any() else None,
            sampled_stance_joint_sample_cap_fraction=float(cap[mask][:,ix].mean()),
            normal_force_weighted_any_joint_cap_fraction=float(np.sum(weights*anycap[mask])/max(weights.sum(),1e-30)),
            middle80percent_stance_any_joint_cap_fraction=float(anycap[middle].mean()),
            middle80percent_loaded_stance_any_joint_cap_fraction=float(anycap[loaded_middle].mean()) if loaded_middle.any() else None,
            sampled_max_stance_any_joint_cap_duration_s=max_run(anycap&mask,dt),
            raw_endpoint_stance_any_joint_over_cap_fraction=float(rawcap[mask][:,ix].any(1).mean()),
            per_joint=perjoint)
    result=dict(folder=str(folder),physics_steps=0,policy='original finitePD zeroDtarget; kp/kd/limits read fromsavedCOMMANDS',
        gain_scale=float(np.median(C['kp']*.2/limits)),kp=C['kp'].tolist(),kd=C['kd'].tolist(),
        target_reconstruction=dict(pre_command_max_error=pre_error,body_endpoint_time_error=alignment_time,
            native_load_time_error=load_alignment,command_rows=int(len(rows)),body_samples=int(len(B['time']))),
        endpoint_raw_clipped_PD_vs_saved_applied_ctrl_max=float(abs(np.clip(raw,-limits,limits)-B['ctrl']).max()),
        per_leg=perleg,hashes={str(folder/f):sha(folder/f) for f in ('BODY.npz','COMMANDS.npz','NATIVE.npz','RESULT.json')})
    return result,dict(target=end,rows=rows,time=B['time'],limits=limits)


def main():
    def forbidden(*a,**kw):raise AssertionError('No mj_step in saved tracking diagnosis')
    mj.mj_step=forbidden
    output=OUT/'TRACKING_LIMITS.json'
    if output.exists():raise FileExistsError('Preserve prior diagnosis '+str(output))
    wavepath=OUT.parent/'20261107T081009Z/METADATA.json'
    wave=json.loads(wavepath.read_text());offset=np.array(wave['phase_offsets']);duty=wave['parameters']['duty']
    p=Plant();runs={};arrays={}
    for name,folder in RUNS.items():runs[name],arrays[name]=analyze(folder,p,offset,duty)
    comparison={}
    for leg in LEGS:
        a=runs['gain32']['per_leg'][leg];b=runs['gain128']['per_leg'][leg]
        comparison[leg]=dict(stance_RMS_gain32=a['stance']['RMS_rad'],stance_RMS_gain128=b['stance']['RMS_rad'],
            RMS_error_reduction_factor=a['stance']['RMS_rad']/max(b['stance']['RMS_rad'],1e-30),
            force_weighted_RMS_reduction_factor=a['stance']['normal_force_weighted_RMS_rad']/max(b['stance']['normal_force_weighted_RMS_rad'],1e-30),
            loaded_stance_any_joint_cap_fraction_gain32=a['sampled_loaded_stance_any_joint_cap_fraction'],
            loaded_stance_any_joint_cap_fraction_gain128=b['sampled_loaded_stance_any_joint_cap_fraction'],
            force_weighted_any_joint_cap_fraction_gain128=b['normal_force_weighted_any_joint_cap_fraction'],
            worst_RMS_joint_gain128=b['worst_RMS_joint'],worst_max_joint_gain128=b['worst_max_joint'])
    matched=dict(times_bitwise=bool(np.array_equal(arrays['gain32']['time'],arrays['gain128']['time'])),
        commands_bitwise=bool(np.array_equal(arrays['gain32']['rows'],arrays['gain128']['rows'])),
        integrated_endpoint_targets_bitwise=bool(np.array_equal(arrays['gain32']['target'],arrays['gain128']['target'])),
        native_limits_bitwise=bool(np.array_equal(arrays['gain32']['limits'],arrays['gain128']['limits'])))
    report=dict(stage='S5 matchedgain tracking/native-torque ceiling diagnostic only',physics_steps=0,
        source=str(Path(__file__).resolve()),source_sha256=sha(__file__),matched=matched,
        alignment='BODYat5msframeEND comparedwithtargetAFTERlast1msintegratorupdate, nativefootforcefromsamepost-forwardtimestamp. Targetrateincludesrecordedperturbation; original30rad/s/nativeanglecaps.',
        phase='Nominalglobalcycle=BODY.time-.25at1Hz, local=(cycle+081009offset)%1, stance<5/6; release[1,end]with1e-9timeboundarytolerance. No fitted phase shift.',
        cap_definition='ActualBODY.ctrl withinrtol1e-7/atol1e-9 of native torque limit. Loadedthreshold.001bodyweight. Middle80percent is descriptive10%-90% ofnominalstance, nottrimmedacceptance.',
        limitations='BODYctrl sampled5ms andrepresentslastnative torque appliedBEFOREthelast12.5usstep; rawPDcomputedfrompost-stepq/qvel isaseparatecounterfactualdemand. Samplingcannotexclude briefcapsbetweensamples. NATIVEforceweightsareactualforces, nooriginvelocityproxy/material-slip estimate. Runsdifferphysicalstates; fourfoldkp isnotpurefixed-state scaling. Thisreportdoesnotauthorizeanothergainorclaimneuralwalking.',
        model_sources={str(f):sha(f) for f in (DEFAULT_MODEL_PATH,BASE/'incremental_actuator.py',BASE/'plant_probe.py',wavepath)},
        comparison=comparison,runs=runs)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output),matched=matched,comparison=comparison),indent=2),flush=True)


if __name__=='__main__':main()
