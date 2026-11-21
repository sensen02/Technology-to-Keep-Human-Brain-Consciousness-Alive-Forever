"""Settle on six feet with muscle feedback, warm real-time afferents, then hand off to BANC."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from solve import HERE, PREVIOUS, BodyObservations, body, load_pose, mj
from teacher import MuscleTeacher
from assay import ConductanceNetwork, DT_MS, Interfaces, sha, MODEL

SETTLE, WARM, POST = 2500, 2500, 2500  # native 0.1 ms ticks
HANDOFF=SETTLE+WARM
MODES=('warm_release','cold_release','motor_cut','teacher_hold')


def support(z, depth, height, all_six=False):
    feet=z['ground_load_w'][:-1]
    lower,upper=(.98,1.02) if all_six else (.8,1.2)
    return bool((feet>.01).sum()>=(6 if all_six else 3) and lower<=feet.sum()<=upper
                and z['ground_load_w'][-1]==0 and z['torso'][0]>=.9*height
                and z['torso'][1]>=.95 and depth>=-.012)


def run(out, mode, checkpoint, vx=0., pose=HERE/'stance.json'):
    stance=json.loads(pose.read_text())
    model=body(stance['floor_contact_time_ms'])
    assert stance['body_xml_sha256']==sha(MODEL)
    assert stance['joint_order']==[model.joint(j).name for j in range(model.njnt)]
    assert stance['actuator_order']==[model.actuator(j).name for j in range(model.nu)]
    q=np.array(stance['qpos'])
    u=np.array(stance['tonic_activation'])
    obs=BodyObservations(model)
    io=Interfaces(model)
    io.restore(checkpoint)
    net=ConductanceNetwork(len(io.roots),dt_ms=DT_MS)
    net.set_connectivity(*io.connectivity(io.fractions))
    teacher=MuscleTeacher(model,q,u)
    d=load_pose(model,q)
    d.qvel[0]=vx
    d.act[:]=u
    d.ctrl[:]=u
    mj.mj_forward(model,d)
    dt=model.opt.timestep
    assert np.isclose(dt,.0001) and np.isclose(dt*5000,DT_MS)
    initial_angle=d.qpos[obs.tibia_qpos].copy()
    height=d.subtree_com[0,2]
    drive=np.zeros(net.n)
    trace=np.zeros(len(io.motor))
    neural=np.zeros(model.nu)
    records={k:[] for k in ('qpos','qvel','activation','loads','torso','depth_mm','stand','six_foot','command','teacher_on')}
    events={k:[] for k in ('input_nA','receptor_spikes','motor_spikes','neural_command')}
    contact_points=np.array([c.pos[:2].copy() for c in d.contact if obs.floor in c.geom])
    hull=ConvexHull(contact_points)
    margin=-np.max(hull.equations[:,:2]@d.subtree_com[0,:2]+hull.equations[:,2])
    for tick in range(HANDOFF+POST+1):
        z=obs.measure(d)
        depth=min((c.dist for c in d.contact if obs.floor in c.geom),default=0.)
        for key,value in [('qpos',d.qpos),('qvel',d.qvel),('activation',d.act),('loads',z['ground_load_w']),
                          ('torso',z['torso']),('depth_mm',depth),('stand',support(z,depth,height)),
                          ('six_foot',support(z,depth,height,True))]:
            records[key].append(np.copy(value))
        if tick in (SETTLE,HANDOFF):
            if not np.all(records['six_foot'][-1001:]):
                raise RuntimeError(f'No continuous six-foot support in last 100 ms at tick {tick}')
        if tick==HANDOFF+POST:
            break
        use_teacher=tick<HANDOFF or mode=='teacher_hold'
        if tick%5==0:
            if use_teacher:
                command=teacher.command(d)
            else:
                command=np.zeros_like(neural) if mode=='motor_cut' else neural
            d.ctrl[:]=command
            # Completed interval's spikes may set the NEXT interval's command.
            drive.fill(0.)
            if tick>=HANDOFF or (tick>=SETTLE and mode!='cold_release'):
                drive[io.sensors]=np.clip(z['ground_load_w'][io.feet]*io.load_gain,0,.12)
                signal=np.where(io.angle,(z['tibia_angle_rad']-initial_angle)[io.feco_feet],
                                z['tibia_velocity_rad_s'][io.feco_feet])
                drive[io.feco]=np.clip(signal*io.feco_gain,0,.12)
            spiked=net.step(drive)
            trace*=np.exp(-DT_MS/10)
            trace+=spiked[io.motor]
            neural=np.clip(io.motor_gain@trace,0,1)
            for name,value in [('input_nA',drive[io.sensors]),('receptor_spikes',spiked[io.sensors]),
                               ('motor_spikes',spiked[io.motor].sum()),('neural_command',neural)]:
                events[name].append(np.copy(value))
        records['teacher_on'].append(use_teacher)
        records['command'].append(d.ctrl.copy())
        assert not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)
        mj.mj_step(model,d)
        mj.mj_forward(model,d)
    assert net.steps*5==HANDOFF+POST and not any(d.warning.number)
    data={k:np.asarray(v) for k,v in (records|events).items()}
    assert all(np.isfinite(v).all() for v in data.values())
    warm=slice(SETTLE//5,HANDOFF//5)
    post=slice(HANDOFF,None)
    def first(mask):
        indexes=np.flatnonzero(mask)
        return round(float(indexes[0]*dt*1000),4) if indexes.size else None
    result=dict(mode=mode,contact_time_ms=stance['floor_contact_time_ms'],dt_s=dt,settle_ms=250,warm_ms=250,post_ms=250,
        native_steps=HANDOFF+POST,neural_ticks=net.steps,initial_vx_mm_s=vx,
        body_xml_sha256=sha(MODEL),pose_sha256=sha(pose),checkpoint_sha256=sha(checkpoint),
        hull_margin_mm=float(margin),closed_loop_linear_radius=teacher.radius,
        warm_all_six=bool(data['six_foot'][SETTLE:HANDOFF+1].all()),
        warm_spikes=int(data['receptor_spikes'][warm].sum()),
        warm_active_receptors=int(data['receptor_spikes'][warm].any(axis=0).sum()),
        warm_motor_spikes=int(data['motor_spikes'][warm].sum()),
        pre_foot_fraction=data['loads'][HANDOFF,:6].tolist(),
        pre_max_penetration_mm=float(-data['depth_mm'][:HANDOFF+1].min()),
        pre_drift_mm=float(np.linalg.norm(data['qpos'][:HANDOFF+1,:3]-q[:3],axis=1).max()),
        post_teacher_ticks=int(data['teacher_on'][post].sum()),
        first_stance_violation_ms=first(~data['stand'][post]),
        first_nonfoot_ms=first(data['loads'][post,-1]>0),
        first_com_drop_10pct_ms=first(data['torso'][post,0]<.9*height),
        post_entire_stand=bool(data['stand'][post].all()),
        post_tail_stand=bool(data['stand'][-1001:].all()),
        post_tail_feet=float(data['loads'][-1001:,:6].sum(axis=1).mean()))
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/(mode+'.npz'),**data)
    (out/(mode+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(mode,result['warm_spikes'],result['first_stance_violation_ms'],result['first_nonfoot_ms'],
          result['post_tail_feet'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--mode',choices=MODES+('all',),default='warm_release')
    parser.add_argument('--parameters',type=Path,default=PREVIOUS/'verified/PARAMETERS.npz')
    parser.add_argument('--initial-vx',type=float,default=0.)
    parser.add_argument('--pose',type=Path,default=HERE/'stance.json')
    args=parser.parse_args()
    for mode in (MODES if args.mode=='all' else (args.mode,)):
        run(args.output,mode,args.parameters,args.initial_vx,args.pose)


if __name__=='__main__':
    main()
