"""Frozen official policy following a predetermined measured-data reference.
Only reference initialization differs from previous synthetic experiment.
"""
from pathlib import Path
import sys,argparse,json,hashlib,time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from flybody.fly_envs import walk_imitation
PREV=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-teacher/20261107T174640Z')
sys.path.insert(0,str(PREV))
from run_teacher import action,observation,real_action,dump,sha,LEGS
from run_teacher_audited import AuditedRecorder
ROOT=Path(__file__).resolve().parent
OUT=Path('/home/sensen/Desktop/engine修复验收/flybody-teacher/20261107T180844Z')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--trajectory',type=int,default=0);ap.add_argument('--tag',default='measured_small_000');args=ap.parse_args()
    folder=OUT/args.tag;folder.mkdir(parents=True,exist_ok=False)
    dump(folder/'PROTOCOL.json',dict(dataset=args.dataset,trajectory=args.trajectory,seed=41,selection='preregistered first small index0' if args.trajectory==0 else 'preregistered maximum full-dataset duration, tie lowest index; no policy scoring',reference='official processed measured walking; not synthetic root trajectory',identity='original pretrained ANN teacher',
        sources={str(p):sha(p) for p in [Path(__file__),PREV/'run_teacher.py',PREV/'run_teacher_audited.py']},teacher=str(PREV/'policies/walking'),
        physics_parameters='unchanged official walking env; retains active adhesion',interventions='none',new_training_steps=0,
        contact_phase='scratch-only forward from copied post-native-step state; live dm_control sensors never forwarded by recorder',current42_acceptance='not applicable'))
    policy=tf.saved_model.load(str(PREV/'policies/walking'))
    env=walk_imitation(ref_path=args.dataset,traj_indices=[args.trajectory],random_state=np.random.RandomState(41))
    env.task.set_next_trajectory_index(args.trajectory);start=time.monotonic()
    try:
        ts=env.reset();p=env.physics;rec=AuditedRecorder(p);rec.state()
        assert p.model.nu==59 and p.model.na==59
        p.model.save_binary(str(folder/'MODEL.mjb'))
        expected=min(5000,int(env.task._episode_steps))
        dump(folder/'MODEL_IDENTITY.json',dict(nq=p.model.nq,nv=p.model.nv,nu=p.model.nu,na=p.model.na,weight=rec.weight,gravity=p.model.opt.gravity.tolist(),
            control_dt=env.control_timestep(),physics_dt=p.timestep(),expected_controls=expected,legs=LEGS,body_names=rec.body_names,geom_names=rec.geom_names,
            mocap_joint_names=[j.full_identifier for j in env.task._mocap_joints],mocap_site_names=[s.full_identifier for s in env.task._mocap_sites],
            real_root_qpos_address=rec.root_qpos,real_root_dof_address=rec.root_dof,action_names=env.action_spec().name.split('\t')))
        np.savez_compressed(folder/'REFERENCE.npz',**env.task._snippet)
        old=env.task.after_substep
        def after(physics,random_state):old(physics,random_state);rec.native(physics,random_state)
        env.task.after_substep=after
        obsrows={k:[] for k in ts.observation};actions=[];scaled=[];rewards=[];discounts=[];tracking=[];actualjoint=[]
        for i in range(expected):
            obs=observation(ts)
            for k,v in obs.items():obsrows[k].append(v)
            a=action(policy,obs);r=real_action(a,env.action_spec());actions.append(a.copy());scaled.append(r.copy())
            before=float(p.data.time);ts=env.step(r);assert p.data.time>before
            rec.state();rewards.append(float(ts.reward));discounts.append(float(ts.discount))
            current=p.bind(env.task._mocap_joints).qpos.copy();actualjoint.append(current)
            reference=env.task._ref_qpos[i+1]
            jointdiff=(current[7:]-reference[7:]+np.pi)%(2*np.pi)-np.pi
            tracking.append(np.r_[p.data.time,np.linalg.norm(current[:3]-reference[:3]),np.sqrt(np.mean(jointdiff**2)),abs(np.dot(current[3:7],reference[3:7]))])
            if (i+1)%500==0:print(json.dumps(dict(step=i+1,sim_s=float(p.data.time),wall_s=time.monotonic()-start)),flush=True)
            if ts.last():break
        np.savez_compressed(folder/'BODY.npz',**{k:np.array([s[k] for s in rec.states]) for k in rec.states[0]})
        np.savez_compressed(folder/'OBS.npz',**{k:np.asarray(v) for k,v in obsrows.items()})
        np.savez_compressed(folder/'TERMINAL_OBS.npz',**observation(ts))
        np.savez_compressed(folder/'POLICY.npz',canonical=np.array(actions),scaled=np.array(scaled),reward=np.array(rewards),discount=np.array(discounts))
        np.savez_compressed(folder/'NATIVE.npz',rows=np.asarray(rec.rows),contacts=np.asarray(rec.contacts),environmental=np.asarray(rec.environmental))
        np.savez_compressed(folder/'TRACKING.npz',rows=np.array(tracking),actual_mocap_qpos=np.array(actualjoint))
        rows=np.array(rec.rows);en=np.array(rec.environmental);track=np.array(tracking)
        report=dict(status='RECORDED_PENDING_INDEPENDENT_REPLAY',controls=len(actions),native_steps=len(rows),sim_s=float(p.data.time),expected_controls=expected,terminal=bool(ts.last()),reference_end=bool(env.task._reached_traj_end),discount=float(ts.discount),
            root_displacement=(rec.states[-1]['qpos'][:3]-rec.states[0]['qpos'][:3]).tolist(),reward_mean=float(np.mean(rewards)),reward_min=float(np.min(rewards)),
            reference_root_error_max=float(track[:,1].max()),joint_tracking_RMSE_rad=float(np.sqrt(np.mean(track[:,2]**2))),
            net_support_over_weight=float((en[:,3]+en[:,6]).mean()/rec.weight),nonfoot_max=float(rows[:,7].max()),explicit_external_force_max=float(rows[:,24:26].max()),warnings=p.data.warning.number.tolist(),
            wall_s=time.monotonic()-start,files={x.name:sha(x) for x in folder.iterdir() if x.is_file()},not_claimed=['held-out teacher generalization','causal brain control','current42 gate pass','surgical validity'])
        dump(folder/'RESULT.json',report);print(json.dumps(report,indent=2))
    finally:
        env.close()
        env.task._traj_generator._h5.close()

if __name__=='__main__':main()
