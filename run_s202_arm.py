"""S202 one arm of the tactile contrast. Engine-derived tactile mapping by BODY name.
arms: A no tactile | B raw force | C force/weight_per_leg | D sign-flipped force.
No CPG/PD/per-step pose correction; inside the loop only d.ctrl[:] is written."""
from pathlib import Path
import json, importlib, sys, hashlib, time, argparse
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype'); DESK=Path('/home/sensen/Desktop/engine修复验收')
S154=DESK/'s154-activation-rate/20261024T215751775489Z'
DT=2.5e-5; CMD_EVERY=200; NT_MS=0.5; NT_STEPS=CMD_EVERY//20; LEGS=('LF','LM','LH','RF','RM','RH')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(out,arm,seconds):
    out=Path(out); assert not out.exists(); out.mkdir(parents=True)
    steps=int(round(seconds/DT))
    import mujoco
    sys.path.insert(0,str(ROOT)); L=importlib.import_module('run_flybody_loop')
    from engine.embodied import adapters as ad
    from engine.embodied.motor_output import MotorOutput, MotorOutputConfig, build_motor_manifest
    from engine.embodied.sensory_channels import ChannelSensoryEncoder, ChannelConfig, ChannelQuantities, channel_index_map, LEGS as SENS_LEGS
    assert tuple(SENS_LEGS)==tuple(l.lower() for l in LEGS), (SENS_LEGS,LEGS)
    body=L.build_body('canonical'); model=body['model']; model.opt.timestep=DT
    WPL=float(body['weight_per_leg']); floor=int(body['floor'])
    z=np.load(S154/'init80ms/TRAJECTORY.npz'); q0=z['qpos'][0].copy(); act0=z['act'][0].copy()
    bname=lambda b: mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(b)) or ''
    gname=lambda g: mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(g)) or ''
    # ---- engine-derived tactile map: geom -> leg, by BODY name (asserted) ----
    tars={}; body_of={}
    for g in range(model.ngeom):
        b=int(model.geom_bodyid[g]); bn=bname(b); body_of[g]=bn
        for leg in LEGS:
            if bn.upper().startswith(leg) and 'TARS' in bn.upper():
                tars[g]=leg; break
    legs_with=[l for l in LEGS if any(v==l for v in tars.values())]
    assert sorted(legs_with)==sorted(LEGS), ('tarsus geom missing for some leg',{l:[g for g,v in tars.items() if v==l] for l in LEGS})
    joints={}
    for leg in LEGS:
        ids=[j for j in range(model.njnt) if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j) or '').lower().startswith(('joint_'+leg).lower())]
        tib=[j for j in ids if 'tibia' in (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j) or '').lower()]
        assert len(ids)==7 and len(tib)==1
        joints[leg]=dict(q=[int(model.jnt_qposadr[j]) for j in ids],tib_q=int(model.jnt_qposadr[tib[0]]),tib_v=int(model.jnt_dofadr[tib[0]]))
    tier_cfg=ad.TierConfig(background_mean_nA=0,background_sd_nA=0)
    built=ad.build_neural_tier(tier_cfg); ledger=ad.OwnershipLedger(); tier=ad.NeuralTier(built,tier_cfg,ledger); n=int(built['n'])
    groups,_=channel_index_map(n,built['tier']['root_ids'])
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(model.nu)]
    pools=json.loads((ROOT/'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json').read_text())
    manifest=build_motor_manifest(pools,built['tier']['root_ids'],names)
    dec=MotorOutput(manifest,MotorOutputConfig()); tier.reset()
    enc=ChannelSensoryEncoder(groups,n,ChannelConfig()); enc.set_reference_pose(q0)
    for leg in LEGS:
        j=joints[leg]; enc.assign(leg,j['tib_q'],j['tib_q'],j['q'])
    d=mujoco.MjData(model); d.qpos[:]=q0; d.qvel[:]=0.0; d.act[:]=act0; d.ctrl[:]=0.0
    mujoco.mj_forward(model,d)
    def divisor():
        return 1.0 if arm=='B' else (WPL if arm=='C' else 1.0)
    def sign():
        return -1.0 if arm=='D' else 1.0
    unmatched={}
    def observe(dv):
        force={l:0.0 for l in LEGS}; touch={l:0.0 for l in LEGS}
        for ci in range(dv.ncon):
            c=dv.contact[ci]
            if floor not in (int(c.geom1),int(c.geom2)): continue
            other=int(c.geom2) if int(c.geom1)==floor else int(c.geom1)
            leg=tars.get(other)
            if leg is None:
                nm=gname(other); unmatched[nm]=unmatched.get(nm,0)+1
                continue
            f=np.zeros(6); mujoco.mj_contactForce(model,dv,ci,f)
            mag=float(np.linalg.norm(f[:3]))
            if arm!='A':
                force[leg]+= sign()*mag/divisor(); touch[leg]=1.0
        res={}
        for leg,j in joints.items():
            res[leg]=ChannelQuantities(tibia_joint_dev_rad=float(dv.qpos[j['tib_q']]-q0[j['tib_q']]),
                tibia_joint_vel_rad_s=float(dv.qvel[j['tib_v']]),
                multi_joint_dev_rad=float(np.mean(dv.qpos[j['q']]-q0[j['q']])),
                contact_force_norm=force[leg],foot_contact=touch[leg])
        return res
    traj=dict(t=[],rootz=[],rootxy=[],nfeet=[],bodyfloor=[],ctrl_max=[],sat_frac=[],act_max=[],spikes=[],cur_max=[],fch=[],tch=[])
    spike_total=0; cur_max=0.0; sat_steps=0; ctrl_steps=0; t0=time.time()
    for k in range(steps//CMD_EVERY):
        q=observe(d)
        current,rec=enc.encode({l.lower():v for l,v in q.items()})
        cur_max=max(cur_max,float(np.max(np.abs(current))))
        sp=tier.advance(NT_STEPS,current); spike_total+=int(sp.sum())
        t_s=k*CMD_EVERY*DT; t_e=(k+1)*CMD_EVERY*DT
        pend,_=dec.consume(sp,spike_dt_s=NT_MS/1000.0,start_s=t_s,end_s=t_e)
        ctrl=np.asarray(pend,float).copy(); d.ctrl[:]=ctrl
        ctrl_steps+=1
        if float(np.max(np.abs(ctrl)))>=0.999: sat_steps+=1
        for j in range(CMD_EVERY):
            mujoco.mj_step(model,d)
            if j==CMD_EVERY-1:
                feet=set(); bf=set()
                for ci,c in enumerate(d.contact):
                    if c.efc_address<0: continue
                    if floor not in (int(c.geom1),int(c.geom2)): continue
                    other=int(c.geom2) if int(c.geom1)==floor else int(c.geom1)
                    leg=tars.get(other)
                    if leg is not None: feet.add(leg)
                    else: bf.add(gname(other))
                traj['t'].append(float(d.time)); traj['rootz'].append(float(d.qpos[2]))
                traj['rootxy'].append([float(d.qpos[0]),float(d.qpos[1])])
                traj['nfeet'].append(len(feet)); traj['bodyfloor'].append(len(bf))
                traj['ctrl_max'].append(float(np.max(np.abs(ctrl)))); traj['sat_frac'].append(sat_steps/ctrl_steps)
                traj['act_max'].append(float(np.max(np.abs(d.act)))); traj['spikes'].append(spike_total)
                traj['cur_max'].append(cur_max)
                traj['fch'].append([float(q[l].contact_force_norm) for l in LEGS])
                traj['tch'].append([float(q[l].foot_contact) for l in LEGS])
    tt=np.asarray(traj['t']); rz=np.asarray(traj['rootz']); last=rz[tt>=tt[-1]-1.0]
    fch=np.asarray(traj['fch']); tch=np.asarray(traj['tch'])
    summary=dict(arm=arm,steps=steps,duration_s=steps*DT,dt=DT,weight_per_leg=WPL,divisor=divisor(),sign=sign(),
        tactile_mapping={l:[g for g,v in tars.items() if v==l] for l in LEGS},n_tarsus_geoms=len(tars),
        unmatched_floor_geoms=dict(sorted(unmatched.items(),key=lambda kv:-kv[1])),
        first_body_floor_time=next((traj['t'][i] for i,v in enumerate(traj['bodyfloor']) if v>0),None),
        mean_bodyfloor_after_1s=float(np.mean(np.asarray(traj['bodyfloor'])[tt>1.0])),
        rootz_last1s_mean=float(np.mean(last)),rootz_last1s_std=float(np.std(last)),
        final_rootz=float(d.qpos[2]),rootz_drop=float(q0[2]-d.qpos[2]),
        min_nfeet=int(min(traj['nfeet'])),max_nfeet=int(max(traj['nfeet'])),
        horizontal_disp=float(np.hypot(d.qpos[0]-q0[0],d.qpos[1]-q0[1])),
        total_spikes=int(spike_total),current_max_nA=float(cur_max),
        ctrl_max=float(max(traj['ctrl_max'])),sat_interval_frac=float(sat_steps/ctrl_steps),
        act_max=float(max(traj['act_max'])),
        force_channel=dict(min=float(np.min(fch)),max=float(np.max(fch)),mean=float(np.mean(fch)),
                           per_leg_max={l:float(np.max(fch[:,i])) for i,l in enumerate(LEGS)}),
        touch_channel=dict(mean=float(np.mean(tch)),per_leg_mean={l:float(np.mean(tch[:,i])) for i,l in enumerate(LEGS)}),
        wall_s=float(time.time()-t0),
        no_step_note='mj_step authorised; only d.ctrl written; no training; no XML change')
    save={k:np.asarray(v) for k,v in traj.items()}
    np.savez_compressed(out/'TRACE.npz',**save)
    (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:summary[k] for k in ('arm','first_body_floor_time','mean_bodyfloor_after_1s','rootz_last1s_mean','rootz_last1s_std','min_nfeet','horizontal_disp','ctrl_max','sat_interval_frac','act_max','total_spikes','force_channel','wall_s')}))
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--arm',required=True,choices=list('ABCD')); ap.add_argument('--seconds',type=float,default=5.0)
    a=ap.parse_args(); main(a.output,a.arm,a.seconds)
