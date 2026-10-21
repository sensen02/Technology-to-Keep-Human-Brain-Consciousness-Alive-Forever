"""Gate B: ONE bounded fixed-command body replay at the qualified 25 us branch.

Commands come from the frozen P1 control archive; the timestep is the already-qualified
25 us. No pose search, no force-scale tuning, no canonical XML change, exactly one run.
The report keeps three things SEPARATE: (1) output-layer replay, (2) native body per-step
result, (3) contact/displacement facts. No merged "success" sentence.
"""
from pathlib import Path
import json,time,hashlib,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parent
P1=ROOT/'outputs/experiment_platform/20261013_233237_7be93a1d'
POSE=ROOT/'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz'
DT=0.000025; CMD_DT=0.005
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    import mujoco
    from engine import canonical_physics as cp
    out=Path('/home/sensen/Desktop/engine修复验收')/('B-fixed-command-replay-25us-'+time.strftime('%Y%m%dT%H%M%S'));out.mkdir(exist_ok=False)
    r=dict(status='RUNNING',scope='ONE_BOUNDED_FIXED_COMMAND_BODY_REPLAY_AT_25US',
           manual_research_not_product=True,standing=False,
           declared=dict(commands='frozen P1 archive normal_controls (10x90, held 5 ms)',timestep_s=DT,integrator='RK4',
                        model='canonical XML, native activation bounds [0.0001,1] (no zero-floor, no XML change)',
                        initial_pose='saved projected_seated_qpos (declared engineering state, NOT an animal record)',
                        not_done='no pose search, no force-scale tuning, no canonical XML change, no repeat/hook-off arms, no neural simulation'),
           sections=dict(output_layer={},native_body_steps={},contact_and_displacement={}))
    paths=[Path(__file__),P1/'traces.npz',P1/'result.json',POSE,cp.CANONICAL_XML]
    hashes={str(p):sha(p) for p in paths};r['input_hashes']=hashes
    try:
        old=json.loads((P1/'result.json').read_text())
        assert sha(P1/'traces.npz')==old['traces_sha256']
        with np.load(P1/'traces.npz') as z:commands=z['normal_controls'].copy()
        assert commands.shape==(10,90)
        with np.load(POSE) as z:pose=z['projected_seated_qpos'].copy()
        m=cp.load_model();assert (m.nq,m.nv,m.na,m.nu)==(49,48,90,90)
        assert np.all(m.actuator_dyntype==int(mujoco.mjtDyn.mjDYN_MUSCLE)) and not m.actuator_actlimited.any()
        m.opt.integrator=mujoco.mjtIntegrator.mjINT_RK4;m.opt.timestep=DT
        d=mujoco.MjData(m);d.qpos[:]=pose;d.qvel[:]=0;d.act[:]=0;d.ctrl[:]=0;mujoco.mj_forward(m,d)
        steps=int(round(.05/DT));per_round=int(round(CMD_DT/DT));assert steps==2000 and per_round==200
        root0=d.qpos[:3].copy()
        qpos=[];qvel=[];act=[];ctrl=[];qfrc=[];ncon=[];warn=[];times=[];nonfoot=[];tarsus=[]
        geom_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or '' for g in range(m.ngeom)]
        for i in range(steps):
            row=i//per_round;d.ctrl[:]=commands[row];d.qfrc_applied[:]=0;d.xfrc_applied[:]=0
            mujoco.mj_step(m,d)
            if d.warning.number.any():raise RuntimeError('native warning at step %d'%i)
            if not all(np.isfinite(x).all() for x in (d.qpos,d.qvel,d.act)):raise FloatingPointError('nonfinite at step %d'%i)
            qpos.append(d.qpos.copy());qvel.append(d.qvel.copy());act.append(d.act.copy());ctrl.append(d.ctrl.copy())
            qfrc.append(d.qfrc_actuator.copy());ncon.append(d.ncon);warn.append(d.warning.number.copy());times.append(float(d.time))
            nonfoot.append(sum(1 for c in d.contact if not all('tarsus' in geom_names[int(g)].lower() for g in (c.geom1,c.geom2))))
            tarsus.append(min(float(d.xpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f'{leg}Tarsus5')][2]) for leg in ('LF','LM','LH','RF','RM','RH')))
        qpos=np.array(qpos);qvel=np.array(qvel);act=np.array(act);ctrl=np.array(ctrl);qfrc=np.array(qfrc);ncon=np.array(ncon);warn=np.array(warn);tarsus=np.array(tarsus)
        # ---- 1) output layer: commands applied faithfully + independent activation reference
        applied_ok=all(np.array_equal(ctrl[i*per_round:(i+1)*per_round],np.tile(commands[i],(per_round,1))) for i in range(10))
        from scipy.integrate import solve_ivp
        def rhs(a,u):
            s=.5+1.5*np.clip(a,0,1);delta=np.clip(u,.0001,1)-a
            return delta/np.where(delta>0,.0001*s,.0004/s)
        ref=[];v=np.zeros(90)
        for k in range(10):
            u=commands[k];sol=solve_ivp(lambda t,y:rhs(y,u),(k*CMD_DT,(k+1)*CMD_DT),v,method='DOP853',rtol=1e-12,atol=1e-14,max_step=1e-6,dense_output=True)
            assert sol.success;ref.append(sol.sol((k*per_round+np.arange(1,per_round+1))*DT).T);v=sol.y[:,-1]
        ref=np.concatenate(ref);err=act-ref
        r['sections']['output_layer']=dict(commands_sha256=hashlib.sha256(commands.tobytes()).hexdigest(),commands_applied_bitwise=bool(applied_ok),
            nonzero_command_scalars=int(np.count_nonzero(commands)),native_ctrlrange=m.actuator_ctrlrange[0].tolist(),
            activation_reference='independent DOP853 on the native muscle ODE with u_eff=clip(ctrl,0.0001,1)',
            activation_vs_reference_max_error=float(abs(err).max()),activation_vs_reference_rms_error=float(np.sqrt(np.mean(err**2))),
            body_activation_min=float(act.min()),body_activation_max=float(act.max()),activation_above_one_samples=int((act>1).sum()))
        # ---- 2) native body per-step
        r['sections']['native_body_steps']=dict(steps=steps,samples=len(times),all_finite=True,warnings_total=int(warn.sum()),
            wall_time_s=None,native_warning_free=True,qfrc_actuator_finite=True,
            root_height_model_units=dict(start=float(qpos[0,2]),end=float(qpos[-1,2]),min=float(qpos[:,2].min()),max=float(qpos[:,2].max())),
            qvel_abs_max=float(np.abs(qvel).max()),note='per-step qpos/qvel/act/ctrl/qfrc_actuator archived in replay.npz')
        # ---- 3) contact and displacement facts (facts only)
        r['sections']['contact_and_displacement']=dict(
            contact_samples=dict(min=int(ncon.min()),max=int(ncon.max()),mean=float(ncon.mean()),steps_with_zero_contact=int((ncon==0).sum()),total=len(ncon)),
            non_tarsus_contact_steps=int(sum(1 for x in nonfoot if x)),
            root_horizontal_displacement_model_units=float(np.linalg.norm(qpos[-1,:2]-root0[:2])),
            root_vertical_change_model_units=float(qpos[-1,2]-root0[2]),
            lowest_tarsus_height_model_units=dict(start=float(tarsus[0]),end=float(tarsus[-1]),min=float(tarsus.min()),max=float(tarsus.max()),floor_plane_z=0.0),
            any_non_tarsus_contact=bool(sum(1 for x in nonfoot if x)>0),note='facts only; no standing/support interpretation')
        np.savez_compressed(out/'replay.npz',lowest_tarsus_z=tarsus,qpos=qpos,qvel=qvel,act=act,ctrl=ctrl,qfrc_actuator=qfrc,ncon=ncon,time_s=np.array(times),activation_reference=ref,commands=commands)
        r['status']='PASS_ONE_BOUNDED_FIXED_COMMAND_REPLAY_RECORDED'
    except Exception as e:r.update(status='FAIL_STOP',error=type(e).__name__+': '+str(e))
    finally:
        r['hashes_after']={p:sha(p) for p in hashes};r['hashes_unchanged']=r['input_hashes']==r['hashes_after']
        with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:z.write(Path(__file__),Path(__file__).name)
        (out/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(out);print(r['status']);print(json.dumps(r['sections']['output_layer'],indent=1)[:600]);print(json.dumps(r['sections']['contact_and_displacement'],indent=1));print(r.get('error',''));return 0 if r['status'].startswith('PASS') else 1
if __name__=='__main__':raise SystemExit(main())
