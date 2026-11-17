"""Native physical range diagnostics; oracle projections are explicitly privileged."""
import argparse
import copy
import sys
import time
import torch
from context import *
from flygym.compose.world import FlatGroundWorld
from flygym_demo.complex_terrain.hybrid_controller import HybridControllerObservation

MODES=('teacher','oracle015','oracle15','student')
SCENES={'flat':0.,'block015':.15,'block030':.30}
SEEDS=(119,227)
TRACE=BASE/'20261117T170820Z-body-readaptation/TEACHER/tr_cruise_0/TRACE.npz'


class ProbeWorld(FlatGroundWorld):
    def __init__(self,height):
        super().__init__()
        if height:
            geom=self.mjcf_root.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,name='reach_block',
                 size=(.20,3.,height/2),pos=(2.45,0,height/2),rgba=(.3,.48,.7,1),contype=0,conaffinity=0)
            self.ground_geoms.append(geom)


def make_body(scene,seed):
    saved=body_run.FlatGroundWorld
    body_run.FlatGroundWorld=lambda:ProbeWorld(SCENES[scene])
    try:
        body=body_run.Body()
    finally:
        body_run.FlatGroundWorld=saved
    body.at_start(seed)
    body.ground_geoms=set(int(i) for i in body.sim._internal_ground_geom_ids)
    body.geom_leg={}
    for segment,geoms in body.sim._internal_geomid_by_bodyseg_by_fly[body.name].items():
        if segment.pos in body.motor.legs:
            for geom in np.atleast_1d(geoms):body.geom_leg[int(geom)]=body.motor.legs.index(segment.pos)
    body.foot_ids=[body.sim._internal_bodyids_by_fly[body.name][body.fly.get_bodysegs_order().index(BodySegment(f'{leg}_tarsus5'))]
                   for leg in body.motor.legs]
    body.block_ids={i for i in body.ground_geoms if 'reach_block' in (mujoco.mj_id2name(body.m,mujoco.mjtObj.mjOBJ_GEOM,i) or '')}
    return body


def contacts(body):
    force=np.zeros((6,3));found=np.zeros(6);wrench=np.zeros(6);nonfoot=0.;block=0.
    for i,c in enumerate(body.d.contact[:body.d.ncon]):
        if c.exclude:continue
        if c.geom1 in body.ground_geoms:g,other,sign=int(c.geom1),int(c.geom2),1.
        elif c.geom2 in body.ground_geoms:g,other,sign=int(c.geom2),int(c.geom1),-1.
        else:continue
        mujoco.mj_contactForce(body.m,body.d,i,wrench)
        magnitude=float(np.linalg.norm(wrench[:3]))
        if g in body.block_ids:block+=magnitude
        leg=body.geom_leg.get(other)
        if leg is not None:
            found[leg]+=1.
            force[leg]+=sign*c.frame.reshape(3,3).T@wrench[:3]
        elif body.m.geom_bodyid[other] in body.axis_bodies:nonfoot+=magnitude
    return found,force,nonfoot,block


def canonical_template(body):
    result={}
    from flygym_demo.complex_terrain.common import dof_spec_to_jointdof
    for i,leg in enumerate(body.motor.legs):
        q=body.motor.preprogrammed_steps.get_joint_angles(leg,body.motor.cpg_network.curr_phases[i],
                                                        body.motor.cpg_network.curr_magnitudes[i])
        for j,spec in enumerate(body.motor.preprogrammed_steps.dofs_per_leg):
            result[dof_spec_to_jointdof(leg,spec)]=q[j]
    return np.array([result[d] for d in body.dofs])


def metrics(arrays,body,scene):
    p=arrays['position'];q=arrays['joint_position'];target=arrays['control']
    jid=body.m.actuator_trnid[body.actuators,0];limited=body.m.jnt_limited[jid].astype(bool)
    lo,hi=body.m.jnt_range[jid].T
    exceed=lambda a:np.where(limited,np.maximum(np.maximum(lo-a,a-hi),0.),0.)
    command=arrays['command'];active=np.max(np.abs(command),axis=1)>.3
    residual=arrays['teacher']-arrays['rhythmic'];gain=np.clip(np.max(np.abs(command),axis=1)/.3,0,1)
    force_ranges=body.m.actuator_forcerange[body.actuators]
    saturation=(arrays['actuator_force']<=force_ranges[:,0]*.99)|(arrays['actuator_force']>=force_ranges[:,1]*.99)
    last=2000
    stop_rms=float(np.sqrt(np.mean(np.sum((np.diff(p[-last:,:2],axis=0)/body_run.DT)**2,axis=1))))
    gates=dict(completed=len(p)==17000,finite=bool(all(np.isfinite(a).all() for a in arrays.values())),
       upright=bool(arrays['upright'].min()>=.8),height=bool(p[:,2].min()>=.7*arrays['initial_position'][2]),
       nonfoot=bool(arrays['nonfoot_force'].max()/body.weight<=.2),joint_limit=bool(exceed(q).max()<=.02),
       stand=stop_rms<=.30)
    if scene=='flat':gates['advance']=bool(p[-1,0]-arrays['initial_position'][0]>=2.)
    else:
        gates['block_contact']=bool(arrays['block_force'].max()>1e-6)
        gates['cross']=bool(p[-1,0]>=3.15)
    return dict(pass_=all(gates.values()),gates=gates,steps=len(p),dx_native=float(p[-1,0]-arrays['initial_position'][0]),
        final_x_native=float(p[-1,0]),final_stand_rms_native_s=stop_rms,min_upright=float(arrays['upright'].min()),
        min_height_native=float(p[:,2].min()),nonfoot_peak_bodyweight=float(arrays['nonfoot_force'].max()/body.weight),
        block_contact_ticks=int(np.sum(arrays['block_force']>1e-6)),actual_joint_excess_max_rad=float(exceed(q).max()),
        target_joint_excess_max_rad=float(exceed(target).max()),actuator_saturation_fraction=float(saturation.mean()),
        tracking_rms_rad=float(np.sqrt(np.mean((target-q)**2))),
        required_residual_abs_quantiles_rad=np.quantile(np.abs(residual[active]),[.5,.95,.99,1]).tolist(),
        pure_teacher_correction_abs_quantiles_rad=np.quantile(np.abs(arrays['teacher_correction'][active]),[.5,.95,.99,1]).tolist(),
        residual015_exceed_fraction=float((np.abs(residual[active])>.015*gain[active,None]+1e-8).mean()),
        residual15_exceed_fraction=float((np.abs(residual[active])>.15*gain[active,None]+1e-8).mean()))


@torch.inference_mode()
def run(scene,mode,seed,out):
    start=time.monotonic();torch.set_num_threads(1)
    body=make_body(scene,seed);actor=phase.read_policy(Path(CURRENT['checkpoint']))
    oscillator=phase.Oscillator(body)
    # The diagnostic teacher must not advance the student's actual CPG.
    body.motor=copy.deepcopy(body.motor)
    commands=np.load(TRACE,allow_pickle=False)['trace'][:,6:8].copy()
    assert len(commands)==17000
    initial=body.d.xpos[body.thorax].copy()
    assert not any(c.geom1 in body.block_ids or c.geom2 in body.block_ids for c in body.d.contact[:body.d.ncon])
    jid=body.m.actuator_trnid[body.actuators,0];qadr=body.m.jnt_qposadr[jid];vadr=body.m.jnt_dofadr[jid]
    columns=('position','upright','command','joint_position','joint_velocity','control','teacher','rhythmic',
             'teacher_correction','actuator_force','nonfoot_force','block_force','contact_force','phase')
    rows={name:[] for name in columns};poses=[];warning=None
    for tick,command in enumerate(commands):
        found,forces,_,_=contacts(body)
        if scene=='flat':found,forces=body.contact()
        label=body.clip_action(body.teacher(command))
        if mode=='student':
            oscillator.step(command);state=oscillator.features()
        else:
            cpg=body.motor.cpg_network
            state=np.r_[cpg.curr_phases%(2*np.pi),cpg.curr_magnitudes].astype(np.float32)
        rhythmic=body.neutral+actor.rhythmic_batch(state[None])[0]
        correction=label-canonical_template(body)
        if tick%body_run.HOLD==0:
            if mode=='teacher':control=label.copy()
            elif mode=='student':control=body.clip_action(body.neutral+actor.act(body.observation(command,(found,forces)),state,True))
            else:
                bound=(.015 if mode=='oracle015' else .15)*np.clip(np.max(np.abs(command))/.3,0,1)
                control=body.clip_action(rhythmic+np.clip(label-rhythmic,-bound,bound))
        body.apply(control)
        _,forces,nf,bf=contacts(body)
        row=dict(position=body.d.xpos[body.thorax].copy(),upright=body.d.xmat[body.thorax,8],command=command.copy(),
             joint_position=body.d.qpos[qadr].copy(),joint_velocity=body.d.qvel[vadr].copy(),control=control.copy(),teacher=label.copy(),
             rhythmic=rhythmic.copy(),teacher_correction=correction.copy(),actuator_force=body.d.actuator_force[body.actuators].copy(),
             nonfoot_force=nf,block_force=bf,contact_force=forces.copy(),phase=state.copy())
        for name in columns:rows[name].append(row[name])
        if tick%10==0:poses.append(body.d.qpos.copy())
        if not np.isfinite(body.d.qpos).all() or body.d.warning.number.any():
            warning=body.d.warning.number.tolist();break
    arrays={k:np.asarray(v) for k,v in rows.items()};arrays['qpos']=np.asarray(poses);arrays['initial_position']=initial
    out.mkdir(parents=True,exist_ok=False)
    archive=out/'TRACE.npz';np.savez_compressed(archive,**arrays)
    mujoco.mj_saveModel(body.m,str(out/'BODY.mjb'),None)
    result=metrics(arrays,body,scene);result.update(scene=scene,mode=mode,seed=seed,warning=warning,
        wall_seconds=time.monotonic()-start,weight_sha256=CURRENT['sha256'],trace_sha256=sha(archive),
        input_command_sha256=sha(TRACE),model=model_identity(body),privileged_oracle=mode.startswith('oracle'))
    if warning:result['pass_']=False
    write(out/'RESULT.json',result)
    print(scene,mode,seed,result['pass_'],round(result['dx_native'],3),
          'needed',np.round(result['required_residual_abs_quantiles_rad'],4).tolist(),flush=True)
    return result


def freeze_inputs():
    paths=[HERE/'PROTOCOL.zh-CN.md',HERE/'context.py',HERE/'drive.sh',HERE/'reach.py',Path(CURRENT['checkpoint']),TRACE,
           BASE/'BODY_CURRENT.json',body_run.MODEL_FILE]
    paths += [Path(mod.__file__).resolve() for mod in list(sys.modules.values()) if getattr(mod,'__file__',None)
              and str(ROOT) in str(mod.__file__) and str(mod.__file__).endswith('.py')]
    write(HERE/'INPUTS.json',dict(source_sha256={str(p):sha(p) for p in dict.fromkeys(paths)},
         scenes=SCENES,seeds=SEEDS,modes=MODES,expected_steps=408000))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('batch','single'))
    parser.add_argument('--scene',choices=SCENES);parser.add_argument('--mode',choices=MODES);parser.add_argument('--seed',type=int)
    parser.add_argument('--out',type=Path);args=parser.parse_args()
    if args.command=='single':run(args.scene,args.mode,args.seed,args.out)
    else:
        freeze_inputs();started=time.monotonic();results=[]
        for scene in SCENES:
            for seed in SEEDS:
                for mode in MODES:
                    if time.monotonic()-started>900:raise TimeoutError('predeclared 900 second body budget')
                    results.append(run(scene,mode,seed,HERE/'RUNS'/f'{scene}_{mode}_{seed}'))
                    write(HERE/'PROGRESS.json',dict(completed=len(results),steps=sum(r['steps'] for r in results),last=results[-1]))
        inputs=json.loads((HERE/'INPUTS.json').read_text())
        assert all(sha(p)==value for p,value in inputs['source_sha256'].items())
        write(HERE/'SUMMARY.json',dict(results=results,physical_steps=sum(r['steps'] for r in results),
             training_updates=0,neural_ticks=0,inputs_unchanged=True,wall_seconds=time.monotonic()-started))
