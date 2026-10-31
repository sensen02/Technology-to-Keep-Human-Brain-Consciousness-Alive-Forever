"""S198: dynamic run driven ONLY by the platform's own sensory front end.

groups <- channel_index_map(n, tier_root_ids)
current <- ChannelSensoryEncoder(declared ChannelConfig).encode(observe_quantities(...))
ctrl <- MotorOutput.consume(tier.advance(current))
No added/invented channel. Inside the loop only d.ctrl[:] is written.
"""
from pathlib import Path
import json, importlib, sys, hashlib, time
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype'); DESK=Path('/home/sensen/Desktop/engine修复验收')
S154=DESK/'s154-activation-rate/20261024T215751775489Z'
DT=2.5e-5; CMD_EVERY=200; NEURAL_TICK_MS=0.5; NEURAL_STEPS=CMD_EVERY//20
BODY_STEPS=20000; LEGS=('LF','LM','LH','RF','RM','RH')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(out,seconds=0.5):
    global BODY_STEPS
    BODY_STEPS=int(round(seconds/DT))
    out=Path(out); assert not out.exists(); out.mkdir(parents=True)
    import mujoco
    sys.path.insert(0,str(ROOT)); L=importlib.import_module('run_flybody_loop')
    from engine.embodied import adapters as ad
    from engine.embodied.motor_output import MotorOutput, MotorOutputConfig, build_motor_manifest
    from engine.embodied.sensory_channels import (ChannelSensoryEncoder, ChannelConfig, ChannelQuantities, channel_index_map)
    body=L.build_body('canonical'); model=body['model']; model.opt.timestep=DT
    z=np.load(S154/'init80ms/TRAJECTORY.npz'); q0=z['qpos'][0].copy(); act0=z['act'][0].copy()
    tier_cfg=ad.TierConfig(background_mean_nA=0,background_sd_nA=0)
    built=ad.build_neural_tier(tier_cfg); ledger=ad.OwnershipLedger(); tier=ad.NeuralTier(built,tier_cfg,ledger)
    n=int(built['n'])
    groups,stats=channel_index_map(n,built['tier']['root_ids'])
    gk={leg.lower():leg for leg in LEGS}
    nchan={leg:{ch:len(groups[leg.lower()][ch]) for ch in groups[leg.lower()]} for leg in LEGS}
    # platform's own observe_quantities, re-implemented here verbatim for the canonical body
    joints={}
    for leg in LEGS:
        ids=[j for j in range(model.njnt) if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j) or '').lower().startswith(('joint_'+leg).lower())]
        tib=[j for j in ids if 'tibia' in (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j) or '').lower()]
        assert len(ids)==7 and len(tib)==1, (leg,len(ids),len(tib))
        joints[leg]=dict(q=[int(model.jnt_qposadr[j]) for j in ids],tib_q=int(model.jnt_qposadr[tib[0]]),tib_v=int(model.jnt_dofadr[tib[0]]))
    geom_leg={}
    for g in range(model.ngeom):
        nm=(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) or '').lower()
        bd=(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or '').lower()
        m=[leg for leg in LEGS if ('tars' in nm+bd and (nm.startswith(leg) or bd.startswith(leg)))]
        if len(m)==1: geom_leg[g]=m[0]
    reference=q0.copy()
    def observe_quantities(d):
        force={leg:0.0 for leg in LEGS}; touch={leg:0.0 for leg in LEGS}
        for ci in range(d.ncon):
            c=d.contact[ci]
            for g,other in ((int(c.geom1),int(c.geom2)),(int(c.geom2),int(c.geom1))):
                if g in geom_leg and model.geom_type[other]==mujoco.mjtGeom.mjGEOM_PLANE:
                    f=np.zeros(6); mujoco.mj_contactForce(model,d,ci,f)
                    force[geom_leg[g]]+=float(np.linalg.norm(f[:3])); touch[geom_leg[g]]=1.0
        res={}
        for leg,j in joints.items():
            res[leg]=ChannelQuantities(tibia_joint_dev_rad=float(d.qpos[j['tib_q']]-reference[j['tib_q']]),
                tibia_joint_vel_rad_s=float(d.qvel[j['tib_v']]),
                multi_joint_dev_rad=float(np.mean(d.qpos[j['q']]-reference[j['q']])),
                contact_force_norm=force[leg],foot_contact=touch[leg])
        return res
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(model.nu)]
    pools=json.loads((ROOT/'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json').read_text())
    manifest=build_motor_manifest(pools,built['tier']['root_ids'],names)
    d=mujoco.MjData(model); d.qpos[:]=q0; d.qvel[:]=0.0; d.act[:]=act0; d.ctrl[:]=0.0
    mujoco.mj_forward(model,d)
    enc=ChannelSensoryEncoder(groups,n,ChannelConfig())
    enc.set_reference_pose(reference)
    for leg in LEGS:
        j=joints[leg]
        troch=[x for x in j['q'] if 'trochanter' in (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,model.jnt_qposadr.tolist().index(x)) or '').lower()]
        troch_row=int(troch[0]) if troch else int(j['tib_q'])
        enc.assign(leg,j['tib_q'],troch_row,[int(v) for v in j['q']])
    dec=MotorOutput(manifest,MotorOutputConfig()); tier.reset(); assert tier.cfg.dt_ms==NEURAL_TICK_MS
    traj=dict(t=[],rootz=[],rootxy=[],nfeet=[],nonfoot=[],ctrl_max=[],act_max=[],spikes=[],cur_max=[],ch_force=[],ch_dev=[],ncon=[])
    spike_total=0; cur_max_all=0.0; t0=time.time(); pending=np.zeros(model.nu)
    gname=lambda g: mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) or ''
    for k in range(BODY_STEPS//CMD_EVERY):
        quantities=observe_quantities(d)
        current,channels=enc.encode({leg.lower():q for leg,q in quantities.items()})
        cur_max_all=max(cur_max_all,float(np.max(np.abs(current))))
        sp=tier.advance(NEURAL_STEPS,current); spike_total+=int(sp.sum())
        t_start=k*CMD_EVERY*DT; t_end=(k+1)*CMD_EVERY*DT
        pending,_=dec.consume(sp,spike_dt_s=NEURAL_TICK_MS/1000.0,start_s=t_start,end_s=t_end)
        ctrl=np.asarray(pending,float).copy(); d.ctrl[:]=ctrl
        for j in range(CMD_EVERY):
            mujoco.mj_step(model,d)
            if j==CMD_EVERY-1:
                ncon=0;feet=set();nf=0
                for ci,c in enumerate(d.contact):
                    if c.efc_address<0: continue
                    ncon+=1; n1,n2=gname(int(c.geom1)),gname(int(c.geom2))
                    fl=(body['floor'] in (int(c.geom1),int(c.geom2)))
                    if fl and ('Tarsus' in n1 or 'Tarsus' in n2):
                        for leg in LEGS:
                            if n1.startswith(leg) or n2.startswith(leg): feet.add(leg)
                    elif not fl: nf+=1
                traj['t'].append(float(d.time)); traj['rootz'].append(float(d.qpos[2]))
                traj['rootxy'].append([float(d.qpos[0]),float(d.qpos[1])]); traj['ncon'].append(ncon)
                traj['nfeet'].append(len(feet)); traj['nonfoot'].append(nf)
                traj['ctrl_max'].append(float(np.max(np.abs(ctrl)))); traj['act_max'].append(float(np.max(np.abs(d.act))))
                traj['spikes'].append(spike_total); traj['cur_max'].append(cur_max_all)
                traj['ch_force'].append([float(quantities[leg].contact_force_norm) for leg in LEGS])
                traj['ch_dev'].append([float(quantities[leg].tibia_joint_dev_rad) for leg in LEGS])
    summary=dict(run='S198 platform-own sensory front end, no invented channel',steps=BODY_STEPS,duration_s=BODY_STEPS*DT,
        dt=DT,start_state=str(S154/'init80ms/TRAJECTORY.npz'),start_rootz=float(q0[2]),final_rootz=float(d.qpos[2]),
        final_rootxy=[float(d.qpos[0]),float(d.qpos[1])],horizontal_disp=float(np.hypot(d.qpos[0]-q0[0],d.qpos[1]-q0[1])),
        rootz_drop=float(q0[2]-d.qpos[2]),min_nfeet=int(min(traj['nfeet'])),max_nfeet=int(max(traj['nfeet'])),
        max_nonfoot=int(max(traj['nonfoot'])),first_nonfoot_time=next((traj['t'][i] for i,v in enumerate(traj['nonfoot']) if v>0),None),
        total_spikes=int(spike_total),current_max_nA=float(cur_max_all),ctrl_max=float(max(traj['ctrl_max'])),
        act_max=float(max(traj['act_max'])),wall_s=float(time.time()-t0),
        channel_cell_counts=nchan,encoder_config=dict(current_nA_per_unit=enc.cfg.current_nA_per_unit,
            reference_force=enc.cfg.reference_force,reference_angle_rad=enc.cfg.reference_angle_rad,
            provenance=enc.cfg.provenance),
        honest_limit='platform channel front end with its DECLARED normalisation; no added channel, no controller; single run from the frozen release state',
        no_step_note='mj_step authorised this round; only d.ctrl written in the loop')
    import numpy as _np
    rz=_np.asarray(traj['rootz']); tt=_np.asarray(traj['t'])
    last1=rz[tt>=tt[-1]-1.0] if tt.size else rz
    summary['last_1s_rootz_mean']=float(_np.mean(last1)); summary['last_1s_rootz_std']=float(_np.std(last1))
    nf_all=_np.asarray(traj['nfeet'])
    summary['last_1s_min_nfeet']=int(min(nf_all[tt>=tt[-1]-1.0])) if tt.size else 0
    summary['max_nfeet']=int(max(traj['nfeet'])); summary['total_nonfoot_samples']=int(sum(1 for v in traj['nonfoot'] if v>0))
    np.savez_compressed(out/'TRACE.npz',**{k:np.asarray(v) for k,v in traj.items()})
    (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
    (out/'INPUT_HASHES.json').write_text(json.dumps({str(S154/'init80ms/TRAJECTORY.npz'):sha(S154/'init80ms/TRAJECTORY.npz')},indent=2))
    print(json.dumps({k:summary[k] for k in ('total_spikes','current_max_nA','ctrl_max','rootz_drop','min_nfeet','max_nonfoot','final_rootz','wall_s')}))
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--seconds',type=float,default=0.5); a=ap.parse_args(); main(a.output,a.seconds)
