"""S197: first honest dynamic integration test with the REAL neural tier, no added channel.

observe(platform) -> zero neural current from the added tarsal channel (gain=0, i.e. channel absent)
-> NeuralTier.advance -> MotorOutput.consume -> d.ctrl[:] -> mj_step.

Inside the loop only d.ctrl[:] is written. No CPG, no PD, no per-step pose correction, no scripted motion.
Start = the frozen S154 release state used by S169-S196. Bounded run.
"""
from pathlib import Path
import json, importlib, sys, hashlib, time
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype'); DESK=Path('/home/sensen/Desktop/engine修复验收')
S154=DESK/'s154-activation-rate/20261024T215751775489Z'
DT=2.5e-5; CMD_EVERY=200; NEURAL_TICK_MS=0.5; NEURAL_EVERY=20; NEURAL_STEPS=CMD_EVERY//NEURAL_EVERY
BODY_STEPS=20000   # 0.5 s
LEGS=('LF','LM','LH','RF','RM','RH')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(out):
    out=Path(out); assert not out.exists(); out.mkdir(parents=True)
    import mujoco
    sys.path.insert(0,str(ROOT)); L=importlib.import_module('run_flybody_loop')
    from engine.embodied import adapters as ad
    from engine.embodied.motor_output import MotorOutput, MotorOutputConfig, build_motor_manifest
    body=L.build_body('canonical'); model=body['model']
    model.opt.timestep=DT
    z=np.load(S154/'init80ms/TRAJECTORY.npz')
    q0=z['qpos'][0].copy(); act0=z['act'][0].copy()
    assert model.nq==q0.size and model.nu==act0.size,(model.nq,model.nu,q0.size,act0.size)
    gname=lambda g: mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) or ''
    tier_cfg=ad.TierConfig(background_mean_nA=0,background_sd_nA=0)
    built=ad.build_neural_tier(tier_cfg); ledger=ad.OwnershipLedger()
    tier=ad.NeuralTier(built,tier_cfg,ledger); n=int(built['n'])
    body['reference']=q0.copy()
    obs_keys=tuple(sorted(L.observe(body,mujoco.MjData(model),body['weight_per_leg']).keys()))
    assert len(obs_keys)==6, obs_keys
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(model.nu)]
    pools=json.loads((ROOT/'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json').read_text())
    manifest=build_motor_manifest(pools,built['tier']['root_ids'],names)
    d=mujoco.MjData(model)
    d.qpos[:]=q0; d.qvel[:]=0.0; d.act[:]=act0; d.ctrl[:]=0.0
    mujoco.mj_forward(model,d)
    decoder=MotorOutput(manifest,MotorOutputConfig()); tier.reset()
    assert tier.cfg.dt_ms==NEURAL_TICK_MS
    f6=np.zeros(6)
    def census():
        ncon=0; feet=set(); nonfoot=[]; 
        for ci,c in enumerate(d.contact):
            if c.efc_address<0: continue
            ncon+=1
            n1,n2=gname(int(c.geom1)),gname(int(c.geom2))
            pair=(n1,n2)
            is_floor=(body['floor'] in (int(c.geom1),int(c.geom2)))
            is_tarsus=('Tarsus' in n1) or ('Tarsus' in n2)
            if is_floor and is_tarsus:
                for leg in LEGS:
                    if n1.startswith(leg) or n2.startswith(leg): feet.add(leg)
            elif not (is_floor):
                nonfoot.append(pair)
        return ncon,len(feet),nonfoot
    traj=dict(t=[],rootz=[],rootxy=[],ncon=[],nfeet=[],nonfoot=[],ctrl_max=[],ctrl_absmean=[],act_max=[],act_mean=[],spikes=[])
    spike_total=0; t0=time.time()
    pending=np.zeros(model.nu)
    for k in range(BODY_STEPS//CMD_EVERY):
        quantities=L.observe(body,d,body['weight_per_leg'])
        current=np.zeros(n)   # NO added channel: zero external current, only the tier's own dynamics
        sp=tier.advance(NEURAL_STEPS,current)
        spike_total+=int(sp.sum())
        t_start=k*CMD_EVERY*DT; t_end=(k+1)*CMD_EVERY*DT
        pending,_=decoder.consume(sp,spike_dt_s=NEURAL_TICK_MS/1000.0,start_s=t_start,end_s=t_end)
        ctrl=np.asarray(pending,float).copy()
        d.ctrl[:]=ctrl
        for j in range(CMD_EVERY):
            mujoco.mj_step(model,d)
            i=k*CMD_EVERY+j+1
            if j==CMD_EVERY-1:
                ncon,nfeet,nonfoot=census()
                traj['t'].append(float(d.time)); traj['rootz'].append(float(d.qpos[2]))
                traj['rootxy'].append([float(d.qpos[0]),float(d.qpos[1])])
                traj['ncon'].append(ncon); traj['nfeet'].append(nfeet); traj['nonfoot'].append(len(nonfoot))
                traj['ctrl_max'].append(float(np.max(np.abs(ctrl))))
                traj['ctrl_absmean'].append(float(np.mean(np.abs(ctrl))))
                traj['act_max'].append(float(np.max(np.abs(d.act))))
                traj['act_mean'].append(float(np.mean(np.abs(d.act))))
                traj['spikes'].append(spike_total)
    summary=dict(run='S197 first dynamic integration test, no added sensory channel',
        steps=BODY_STEPS,dt=DT,duration_s=BODY_STEPS*DT,
        start_state=str(S154/'init80ms/TRAJECTORY.npz'),start_rootz=float(q0[2]),
        final_rootz=float(d.qpos[2]),min_rootz=float(min(traj['rootz'])),max_rootz=float(max(traj['rootz'])),
        rootz_drop=float(q0[2]-d.qpos[2]),
        final_rootxy=[float(d.qpos[0]),float(d.qpos[1])],
        final_ncon=int(traj['ncon'][-1]),min_nfeet=int(min(traj['nfeet'])),max_nfeet=int(max(traj['nfeet'])),
        max_nonfoot_contacts=int(max(traj['nonfoot'])),first_nonfoot_time=next((traj['t'][i] for i,v in enumerate(traj['nonfoot']) if v>0),None),
        total_spikes=int(spike_total),ctrl_max_all=float(max(traj['ctrl_max'])),ctrl_mean_all=float(np.mean(traj['ctrl_absmean'])),
        act_max_all=float(max(traj['act_max'])),
        wall_s=float(time.time()-t0),
        honest_limit='this is a 0.5 s unscripted run from the frozen release state; it is NOT a standing/walking result and the tier had no external current',
        no_step_note='mj_step WAS used here (authorised for this round); no neural retraining, no controller, only d.ctrl written in the loop')
    np.savez_compressed(out/'TRACE.npz',t=np.array(traj['t']),rootz=np.array(traj['rootz']),rootxy=np.array(traj['rootxy']),
        ncon=np.array(traj['ncon']),nfeet=np.array(traj['nfeet']),nonfoot=np.array(traj['nonfoot']),
        ctrl_max=np.array(traj['ctrl_max']),act_max=np.array(traj['act_max']),spikes=np.array(traj['spikes']))
    (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
    (out/'INPUT_HASHES.json').write_text(json.dumps({str(S154/'init80ms/TRAJECTORY.npz'):sha(S154/'init80ms/TRAJECTORY.npz')},indent=2))
    print(json.dumps({k:summary[k] for k in ('steps','duration_s','final_rootz','rootz_drop','min_nfeet','max_nonfoot_contacts','first_nonfoot_time','total_spikes','ctrl_max_all','wall_s')}))
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); main(ap.parse_args().output)
