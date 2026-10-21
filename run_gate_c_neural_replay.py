"""Gate C minimal implementation: bounded native replay driven by archived neural output.

RESEARCH SCRIPT, NOT A PRODUCT. Starts from the DECLARED keyframe pose, qvel = 0, with NO
xfrc_applied and no hidden support. Commands are the archived neural-loop output (frozen P1
control archive). No XML change, no pose search, no external CPG, no per-step pose correction,
no force-scale tuning.

Records per body step: qacc, thorax/COM height, contact count and total contact normal force,
so the achieved support can be compared with the Gate-A relaxation margin (~4.5 percent).
"""
from pathlib import Path
import json,time,hashlib,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parent
P1=ROOT/'outputs/experiment_platform/20261013_233237_7be93a1d'
DT=0.000025;CMD_DT=0.005;DURATION=0.05
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    import mujoco
    from engine import canonical_physics as cp
    out=Path('/home/sensen/Desktop/engine修复验收')/('C-gate-neural-replay-'+time.strftime('%Y%m%dT%H%M%S'));out.mkdir(exist_ok=False)
    r=dict(status='RUNNING',scope='GATE_C_MINIMAL_BOUNDED_NATIVE_REPLAY_FROM_DECLARED_KEYFRAME',
        research_script_not_product=True,standing=False,
        declared=dict(initial_pose='canonical model keyframe qpos (declared), qvel=0',
                      external_force='none: qfrc_applied and xfrc_applied are zeroed every step',
                      hidden_support='none',commands='archived neural-loop output (frozen P1 normal_controls, 10x90 at 5 ms)',
                      timestep_s=DT,integrator='RK4',duration_s=DURATION,
                      not_done='no canonical XML change, no pose search, no external CPG, no per-step pose correction, no force-scale tuning'),
        sections=dict(initial_state={},trajectory={},contact={},margin_comparison={}))
    paths=[Path(__file__),P1/'traces.npz',P1/'result.json',cp.CANONICAL_XML]
    hashes={str(p):sha(p) for p in paths};r['input_hashes']=hashes
    try:
        old=json.loads((P1/'result.json').read_text());assert sha(P1/'traces.npz')==old['traces_sha256']
        with np.load(P1/'traces.npz') as z:commands=z['normal_controls'].copy()
        assert commands.shape==(10,90)
        m=cp.load_model();assert (m.nq,m.nv,m.na,m.nu)==(49,48,90,90)
        assert not m.actuator_actlimited.any() and not m.actuator_actearly.any()
        m.opt.integrator=mujoco.mjtIntegrator.mjINT_RK4;m.opt.timestep=DT
        d=mujoco.MjData(m)
        key=m.key_qpos[0].copy()
        d.qpos[:]=key;d.qvel[:]=0;d.act[:]=0;d.ctrl[:]=0;mujoco.mj_forward(m,d)
        weight=float(m.body_mass.sum()*abs(m.opt.gravity[2]))
        com0=float(np.average(d.xipos[:,2],weights=m.body_mass))
        thorax=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'Thorax')
        range_viol=[dict(joint=m.joint(j).name,value=float(key[m.jnt_qposadr[j]]),range=m.jnt_range[j].tolist())
                    for j in range(m.njnt) if m.jnt_limited[j] and (key[m.jnt_qposadr[j]]<m.jnt_range[j,0]-1e-10 or key[m.jnt_qposadr[j]]>m.jnt_range[j,1]+1e-10)]
        r['sections']['initial_state']=dict(qpos_sha256=hashlib.sha256(key.tobytes()).hexdigest(),thorax_z=float(d.xpos[thorax,2]),
            com_z=com0,ncon=d.ncon,body_weight_model_units=weight,declared_keyframe_joint_range_violations=range_viol,
            note='the declared keyframe itself violates these scalar ranges; recorded as a fact, not corrected (no pose search)')
        steps=int(round(DURATION/DT));per_round=int(round(CMD_DT/DT));assert steps==2000 and per_round==200
        qacc=[];th=[];com=[];ncon=[];cfn=[];t=[]
        for i in range(steps):
            d.ctrl[:]=commands[i//per_round];d.qfrc_applied[:]=0;d.xfrc_applied[:]=0
            mujoco.mj_step(m,d)
            if d.warning.number.any():raise RuntimeError('native warning at step %d'%i)
            if not all(np.isfinite(x).all() for x in (d.qpos,d.qvel,d.qacc,d.act)):raise FloatingPointError('nonfinite at %d'%i)
            total=0.0
            for ci in range(d.ncon):
                f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f);total+=float(np.linalg.norm(f[:3]))
            qacc.append(d.qacc.copy());th.append(float(d.xpos[thorax,2]));com.append(float(np.average(d.xipos[:,2],weights=m.body_mass)))
            ncon.append(int(d.ncon));cfn.append(total);t.append(float(d.time))
        qacc=np.array(qacc);th=np.array(th);com=np.array(com);ncon=np.array(ncon);cfn=np.array(cfn);t=np.array(t)
        r['sections']['trajectory']=dict(samples=len(t),all_finite=True,warnings=0,
            thorax_z=dict(start=float(th[0]),end=float(th[-1]),min=float(th.min()),drop=float(th[0]-th[-1])),
            com_z=dict(start=float(com[0]),end=float(com[-1]),min=float(com.min()),drop=float(com[0]-com[-1])),
            qacc_rms=float(np.sqrt(np.mean(qacc**2))),qacc_max_abs=float(np.abs(qacc).max()))
        r['sections']['contact']=dict(ncon=dict(min=int(ncon.min()),max=int(ncon.max()),mean=float(ncon.mean()),steps_with_zero=int((ncon==0).sum())),
            contact_normal_force_over_weight=dict(start=float(cfn[0]/weight),end=float(cfn[-1]/weight),min=float(cfn.min()/weight),max=float(cfn.max()/weight),mean=float(cfn.mean()/weight)))
        r['sections']['margin_comparison']=dict(
            gate_a_relaxation_margin='about 4.5 percent force-scale margin in the DECLARED-MODEL STATIC LP RELAXATION (s marginally above 1 would make the relaxation feasible); that is a relaxation margin, NOT a standing proof',
            observed='see contact.contact_normal_force_over_weight: the achieved contact support fraction over the 50 ms window',
            statement='facts placed side by side; no claim that closing a 4.5 percent margin yields standing')
        np.savez_compressed(out/'trajectory.npz',qacc=qacc,thorax_z=th,com_z=com,ncon=ncon,contact_force=cfn,time_s=t,commands=commands)
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(14,4))
        axes[0].plot(t*1000,th,label='thorax z');axes[0].plot(t*1000,com,'--',label='COM z')
        axes[0].set(xlabel='time (ms)',ylabel='height (model units)',title='sinking (declared keyframe start)');axes[0].legend(fontsize=7)
        axes[1].plot(t*1000,cfn/weight,label='contact normal force / body weight');axes[1].axhline(1.0,color='k',lw=.8,label='1 = body weight')
        axes[1].set(xlabel='time (ms)',ylabel='force / body weight',title='achieved contact support');axes[1].legend(fontsize=7)
        axes[2].plot(t*1000,np.sqrt((qacc**2).mean(axis=1)),label='qacc RMS')
        axes[2].plot(t*1000,ncon/float(ncon.max()),'--',label='ncon / max ncon')
        axes[2].set(xlabel='time (ms)',ylabel='qacc RMS / normalised ncon',title='qacc and contact count');axes[2].legend(fontsize=7)
        fig.suptitle('Gate C minimal bounded neural replay (RK4 25 us, no external force) - NOT standing',fontsize=9)
        fig.tight_layout();fig.savefig(out/'gate-c-replay.png');plt.close(fig)
        r['status']='PASS_GATE_C_BOUNDED_REPLAY_RECORDED'
    except Exception as e:r.update(status='FAIL_STOP',error=type(e).__name__+': '+str(e))
    finally:
        r['hashes_after']={p:sha(p) for p in hashes};r['hashes_unchanged']=r['input_hashes']==r['hashes_after']
        with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z2:z2.write(Path(__file__),Path(__file__).name)
        (out/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(out);print(r['status']);print(json.dumps({k:r['sections'].get(k) for k in ('initial_state','trajectory','contact')},indent=1)[:1400]);print(r.get('error',''))
    return 0 if r['status'].startswith('PASS') else 1
if __name__=='__main__':raise SystemExit(main())
