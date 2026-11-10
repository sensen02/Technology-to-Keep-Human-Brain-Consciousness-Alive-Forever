"""Retargeted reference forcing in original current body; no neural model claimed."""
from pathlib import Path
import sys,argparse,json,time,hashlib
import numpy as np
import mujoco as mj
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype');BASE=ROOT/'experiments/s5-ann-incremental/20261107T010000Z'
sys.path[:0]=[str(BASE),str(ROOT/'tools')]
from plant_probe import Plant
from incremental_actuator import IncrementalActuator,DEFAULT_MODEL_PATH
import dynamic_support_metrics as dsm
HERE=Path(__file__).resolve().parent;OUT=Path('/home/sensen/Desktop/engine修复验收/flybody-retarget/20261107T182940Z')
LEGS=['LF','LM','LH','RF','RM','RH']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,o):
    with Path(p).open('x') as f:json.dump(o,f,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--targets',type=Path,required=True);ap.add_argument('--tag',default='teacherderived_physics');ap.add_argument('--time-scale',type=float,default=1.);a=ap.parse_args();assert a.time_scale in [1.,4.]
    folder=OUT/a.tag;folder.mkdir(parents=True,exist_ok=False)
    with np.load(a.targets) as z:t=z['time_s'];q=z['targets_rad'];names=z['ordered_joint_names'];source=str(z['source_id'])
    p=Plant();m=p.m;d=p.fresh()
    assert mj.__version__=='3.9.0' and np.isfinite(q).all() and np.allclose(np.diff(t),.002,atol=1e-12)
    assert np.array_equal(names,[m.joint(int(j)).name for j in p.jids])
    assert np.max(np.abs(np.diff(q,axis=0))/.002)<=30+1e-7
    assert np.all(q>=m.jnt_range[p.jids,0]-1e-9) and np.all(q<=m.jnt_range[p.jids,1]+1e-9)
    d.qpos[p.qa]=q[0];d.qvel[:]=0;mj.mj_forward(m,d)
    low=np.inf
    for g in np.flatnonzero(p.feet>=0):
        mid=int(m.geom_dataid[g]);assert m.geom_type[g]==mj.mjtGeom.mjGEOM_MESH
        vert=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
        world=vert@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];low=min(low,float(world[:,2].min()))
    dz=.005-low;d.qpos[2]+=dz;mj.mj_forward(m,d)
    actuator=IncrementalActuator(m,d,kp=p.limits/.2*128,kd=p.limits*.005*np.sqrt(128),max_rate_rad_s=30)
    spec=mj.mjtState.mjSTATE_INTEGRATION;initial=np.zeros(mj.mj_stateSize(m,spec));mj.mj_getState(m,d,initial,spec)
    np.savez_compressed(folder/'INITIAL.npz',state=initial,target=actuator.target_rad,kp=actuator.kp,kd=actuator.kd,limits=actuator.torque_limit,qa=p.qa,va=p.va,joint_bounds=m.jnt_range[p.jids])
    mj.mj_saveModel(m,str(folder/'MODEL.mjb'),None)
    dump(folder/'PROTOCOL.json',dict(target_file=str(a.targets),target_sha256=sha(a.targets),source=source,mode='reference_forcing_NO_NEURAL_MODEL',hold_s=.2,source_duration_s=float(t[-1]),time_scale=a.time_scale,playback_duration_s=float(t[-1]*a.time_scale),source_frames=len(t),initial_lowest_foot=low,initial_root_z_shift=dz,
        physics_dt=m.opt.timestep,command_dt=.001,body_source=str(DEFAULT_MODEL_PATH),body_source_sha256=sha(DEFAULT_MODEL_PATH),runner_sha256=sha(__file__),geometry_acceptance='not implied by simulation completion',root_writes='initialization only; no root reference copied',force_phase='retained current baseline mj_step followed by mj_forward'))
    initial_height=float(d.subtree_com[p.thorax,2]);nt=round((.2+t[-1]*a.time_scale)/.001)
    native=np.empty((nt*80,20));commands=[];body=[];slip=[];saturations=0;start=time.monotonic()
    def state():return dict(time=float(d.time),qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),qacc_warmstart=d.qacc_warmstart.copy(),com=d.subtree_com[p.thorax].copy())
    body.append(state())
    for k in range(nt):
        dest_time=np.clip(((k+1)*.001-.2)/a.time_scale,0,t[-1]);index=min(int(dest_time/.002),len(t)-2);alpha=np.clip((dest_time-t[index])/.002,0,1)
        desired=(1-alpha)*q[index]+alpha*q[index+1]
        rate=(desired-actuator.target_rad)/.001
        assert np.max(np.abs(rate))<=30+2e-6
        actuator.update(rate,.001);commands.append(np.r_[k*.001,rate,actuator.target_rad])
        for sub in range(80):
            d.ctrl[:]=actuator.torque(d);saturations+=int(np.count_nonzero(np.abs(d.ctrl)>=actuator.torque_limit-1e-12))
            assert not d.qfrc_applied.any() and not d.xfrc_applied.any()
            mj.mj_step(m,d);mj.mj_forward(m,d);native[k*80+sub]=p.contact(d)
            if sub%40==39:
                load=np.zeros(6);vel=np.zeros(6)
                for ci,c in enumerate(d.contact):
                    ga,gb=int(c.geom1),int(c.geom2)
                    if not(p.planes[ga] or p.planes[gb]):continue
                    geom=gb if p.planes[ga] else ga;leg=p.feet[geom]
                    if leg<0:continue
                    force=np.zeros(6);mj.mj_contactForce(m,d,ci,force)
                    fz=max(0.,float((c.frame.reshape(3,3).T@force[:3])[2])*(1 if p.planes[ga] else -1))
                    jac=np.zeros((3,m.nv));jr=np.zeros_like(jac);mj.mj_jac(m,d,jac,jr,c.pos,int(m.geom_bodyid[geom]))
                    load[leg]+=fz;vel[leg]+=fz*np.linalg.norm((jac@d.qvel)[:2])
                slip.append(np.r_[d.time,load,vel])
        body.append(state())
        assert np.isfinite(d.qpos).all() and not d.warning.number.any()
        if (k+1)%500==0:print(json.dumps(dict(sim_s=float(d.time),wall_s=time.monotonic()-start)),flush=True)
    for name,values in [('BODY',{key:np.asarray([b[key] for b in body]) for key in body[0]}),('NATIVE',dict(rows=native)),('COMMANDS',dict(rows=np.asarray(commands))),('SLIP',dict(rows=np.asarray(slip)))]:np.savez_compressed(folder/(name+'.npz'),**values)
    release=native[native[:,0]>=.2];gate=dsm.evaluate_native(dict(foot_support_mean=1.,com_z_mean=initial_height),release);gate['checks']['D1_no_nonfoot_ground_force']=bool(release[:,7].max()==0);gate['passed']=all(gate['checks'].values())
    bd=np.array([b['qpos'] for b in body]);com=np.array([b['com'] for b in body]);length=np.linalg.norm(np.diff(com[200:,:2],axis=0),axis=1).sum();ss=np.asarray(slip);mask=ss[:,0]>=.2
    speed=length/float(t[-1]*a.time_scale);slips=ss[mask,7:13].sum(0)/np.maximum(ss[mask,1:7].sum(0),1e-15)
    result=dict(status='RECORDED_DIAGNOSTIC_NOT_NEURAL_SUCCESS',native_steps=nt*80,commands=nt,sim_s=float(d.time),reference_source=source,physical_gate=gate,
        nonfoot_force_max=float(native[:,7].max()),root_displacement=(bd[-1,:3]-bd[200,:3]).tolist(),posthold_COM_path_speed=float(speed),material_slip_weighted_speed=slips.tolist(),slip_to_pathspeed_ratio=(slips/max(speed,1e-12)).tolist(),
        initial_COM_height=initial_height,COM_height_min=float(com[:,2].min()),torque_saturated_entries=saturations,torque_total_entries=nt*80*42,warnings=d.warning.number.tolist(),wall_s=time.monotonic()-start,
        files={p.name:sha(p) for p in folder.iterdir() if p.is_file()},limitations=['Open-loop reference forcing; no brain/VNC/policy learning.','Turning reference: no straight-walking acceptance inferred.','Initial geometry placement is not sustained assistance.'])
    dump(folder/'RESULT.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
