"""Fresh independent env/policy replay. Does not import either recording driver.
No current42 support gate or original-brain acceptance is inferred.
"""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import mujoco
from flybody.fly_envs import walk_imitation
from flybody.tasks.synthetic_trajectories import constant_speed_trajectory
ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);args=ap.parse_args();p=args.folder
    prot=json.loads((p/'PROTOCOL.json').read_text());meta=json.loads((p/'MODEL_IDENTITY.json').read_text());result=json.loads((p/'RESULT.json').read_text())
    assert not (p/'VERIFICATION.json').exists()
    def arrays(name):
        with np.load(p/name) as z:return {k:z[k] for k in z.files}
    data=arrays('BODY.npz');obs=arrays('OBS.npz');pol=arrays('POLICY.npz');native=arrays('NATIVE.npz')
    for name,digest in result['files'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==digest,name
    pi=tf.saved_model.load(str(ROOT/'policies/walking'))
    env=walk_imitation(random_state=np.random.RandomState(prot['seed']))
    seconds=prot['seconds']
    if seconds is not None:
        q,v=constant_speed_trajectory(n_steps=round(seconds/.002)+65,speed=2,init_pos=(0,0,.1278),control_timestep=.002)
        env.task._traj_generator.set_next_trajectory(q,v)
    maxerr={k:0. for k in ['obs','policy','action_scaling','qpos','qvel','act','ctrl','qacc_warmstart','time','com','diagnostic_load','diagnostic_com','environmental','terminal_obs','contact_velocity_twist_vs_jacobian']}
    counters={'native':0,'contact_crosschecks':0}
    started=time.monotonic()
    def compare(k,a,b):
        assert np.shape(a)==np.shape(b),(k,np.shape(a),np.shape(b))
        err=float(np.max(np.abs(np.asarray(a)-np.asarray(b)))) if np.size(a) else 0.
        assert np.isfinite(err),k
        maxerr[k]=max(maxerr[k],err)
    try:
        ts=env.reset();phy=env.physics;m=phy.model;d=phy.data
        b=m.name2id('walker/thorax','body');rootj=m.body_jntadr[m.body_rootid[b]];r=int(m.jnt_dofadr[rootj]);s=mujoco.MjData(m.ptr)
        feet={m.name2id('walker/tarsal_claw_'+leg+'_collision','geom'):i for i,leg in enumerate(meta['legs'])}
        adhesives=[m.name2id('walker/adhere_claw_'+leg,'actuator') for leg in meta['legs']]
        old=env.task.after_substep
        def hook(physics,random_state):
            old(physics,random_state)
            n=counters['native'];counters['native']+=1
            # Independently recompute force diagnostics once per command, throughout run.
            if n%10!=9:return
            s.time=d.time
            for key in ['qpos','qvel','act','ctrl','qacc_warmstart','qfrc_applied','xfrc_applied','mocap_pos','mocap_quat','userdata']:getattr(s,key)[:]=getattr(d,key)
            mujoco.mj_forward(m.ptr,s)
            load=np.zeros(6);nonfoot=0.
            for ci,c in enumerate(s.contact):
                b1=int(m.geom_bodyid[c.geom1]);b2=int(m.geom_bodyid[c.geom2])
                if b1==0 and (m.id2name(b2,'body') or '').startswith('walker/'):body=b2;sign=1
                elif b2==0 and (m.id2name(b1,'body') or '').startswith('walker/'):body=b1;sign=-1
                else:continue
                f=np.zeros(6);mujoco.mj_contactForce(m.ptr,s,ci,f)
                world=sign*np.asarray(c.frame).reshape(3,3).T@f[:3]
                leg=feet.get(int(c.geom1),feet.get(int(c.geom2),-1))
                if leg>=0:load[leg]+=max(0.,world[2])
                else:nonfoot+=max(0.,world[2])
                if n%100==99:
                    jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jac(m.ptr,s,jp,jr,c.pos,body)
                    twist=s.cvel[body];v=twist[3:]+np.cross(twist[:3],c.pos-s.subtree_com[m.body_rootid[body]])
                    compare('contact_velocity_twist_vs_jacobian',jp@s.qvel,v);counters['contact_crosschecks']+=1
            compare('diagnostic_load',np.r_[load,nonfoot],native['rows'][n,1:8]);compare('diagnostic_com',s.subtree_com[b],native['rows'][n,8:11])
            if 'environmental' in native:
                realized=s.actuator_moment[adhesives,r:r+3]*s.actuator_force[adhesives,None]
                row=np.r_[s.time,s.qfrc_constraint[r:r+3],s.qfrc_actuator[r:r+3],realized.reshape(-1),s.actuator_force[adhesives],s.act[m.actuator_actadr[adhesives]]]
                compare('environmental',row,native['environmental'][n])
        env.task.after_substep=hook
        def state(i):
            for key in ['qpos','qvel','act','ctrl','qacc_warmstart']:compare(key,getattr(d,key),data[key][i])
            compare('time',d.time,data['time'][i]);compare('com',d.subtree_com[b],data['com'][i])
        state(0)
        spec=env.action_spec();lo=spec.minimum.astype('float32');hi=spec.maximum.astype('float32')
        for i in range(len(pol['canonical'])):
            if ts.last():raise AssertionError('Unexpected early terminal; no reset permitted')
            inputs={k:np.asarray(v,dtype=np.float32) for k,v in ts.observation.items()}
            assert set(inputs)==set(obs)
            for k,v in inputs.items():compare('obs',v,obs[k][i])
            output=pi({k:tf.convert_to_tensor(v[None]) for k,v in inputs.items()}).mean()[0].numpy()
            compare('policy',output,pol['canonical'][i])
            scaled=.5*(np.clip(output,-1.,1.)+1.);scaled*=hi-lo;scaled+=lo
            compare('action_scaling',scaled,pol['scaled'][i])
            ts=env.step(pol['scaled'][i].copy());state(i+1)
            if (i+1)%1000==0:print(json.dumps(dict(replay_s=float(d.time),wall_s=time.monotonic()-started)),flush=True)
        assert counters['native']==len(native['rows'])
        assert ts.last()==result['terminal']
        if (p/'TERMINAL_OBS.npz').exists():
            last=np.load(p/'TERMINAL_OBS.npz')
            for k,v in ts.observation.items():compare('terminal_obs',np.asarray(v,dtype=np.float32),last[k])
        weight=meta['weight'];rows=native['rows'];steady=rows[:,0]>=.5
        if not steady.any():steady=np.ones(len(rows),dtype=bool)
        speed=(data['qpos'][-1,0]-data['qpos'][0,0])/(data['time'][-1]-data['time'][0])
        slip=(rows[steady,18:24].sum(0)/np.maximum(rows[steady,1:7].sum(0),1e-15))/abs(speed)
        physics=dict(root_forward_speed_native=float(speed),diagnostic_positive_claw_load_over_weight=float(rows[steady,1:7].sum(1).mean()/weight),nonfoot_max=float(rows[:,7].max()),slip_ratios_diagnostic=slip.tolist(),statistic_window='after .5s for longer run; entire run for default')
        if 'environmental' in native:
            en=native['environmental'];physics.update(net_root_environmental_support_over_weight=float((en[steady,3]+en[steady,6]).mean()/weight),adhesive_root_z_over_weight=float(en[steady,6].mean()/weight))
        checks=dict(hashes=True,policy_replay=maxerr['policy']==0,observation_replay=maxerr['obs']==0,body_replay=max(maxerr[k] for k in ['qpos','qvel','act','ctrl','qacc_warmstart','time','com'])<1e-10,diagnostic_force_replay=maxerr['diagnostic_load']<1e-10,contact_point_jacobian_crosscheck=maxerr['contact_velocity_twist_vs_jacobian']<1e-9,environmental_replay=maxerr['environmental']<1e-10,continuous_no_reset=True)
        report=dict(status='ENGINEERING_REPLAY_VERIFIED' if all(checks.values()) else 'REPLAY_FAILED',checks=checks,max_errors=maxerr,counters=counters,physics=physics,wall_s=time.monotonic()-started,
            limitations=['Independent fresh env reset+step preserves official split stepping and observation buffers.','Diagnostic forces recomputed on scratch; not frozen live solver forces.','Reference-conditioned policy, original adhesive actuation, not original brain or current42 acceptance.'],
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        with (p/'VERIFICATION.json').open('x') as f:json.dump(report,f,indent=2)
        print(json.dumps(report,indent=2))
        if not all(checks.values()):raise SystemExit(2)
    finally:env.close()

if __name__=='__main__':main()
