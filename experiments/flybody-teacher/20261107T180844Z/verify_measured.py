"""Independent measured-reference replay; no recording-driver imports."""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import mujoco
from flybody.fly_envs import walk_imitation

def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);p=ap.parse_args().folder
    protocol=json.loads((p/'PROTOCOL.json').read_text());meta=json.loads((p/'MODEL_IDENTITY.json').read_text());record=json.loads((p/'RESULT.json').read_text())
    assert not (p/'VERIFICATION.json').exists()
    for name,digest in record['files'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==digest,name
    def load(name):
        with np.load(p/name) as f:return {k:f[k] for k in f.files}
    body=load('BODY.npz');obs=load('OBS.npz');pol=load('POLICY.npz');ref=load('REFERENCE.npz');native=load('NATIVE.npz');tracking=load('TRACKING.npz');last=load('TERMINAL_OBS.npz')
    pi=tf.saved_model.load(protocol['teacher'])
    env=walk_imitation(ref_path=protocol['dataset'],traj_indices=[protocol['trajectory']],random_state=np.random.RandomState(protocol['seed']))
    env.task.set_next_trajectory_index(protocol['trajectory'])
    errors={k:0. for k in ['reference','observations','policy','scaled_actions','reward','discount','qpos','qvel','act','ctrl','qacc_warmstart','time','com','terminal_observations','mocap_qpos','diagnostic_load','environmental']}
    count=0;t=time.monotonic()
    def compare(k,a,b):
        assert np.shape(a)==np.shape(b),(k,np.shape(a),np.shape(b))
        e=float(np.abs(np.asarray(a)-np.asarray(b)).max()) if np.size(a) else 0.
        assert np.isfinite(e);errors[k]=max(errors[k],e)
    try:
        ts=env.reset();phy=env.physics;m=phy.model;d=phy.data;thorax=m.name2id('walker/thorax','body');scratch=mujoco.MjData(m.ptr)
        for k,v in env.task._snippet.items():compare('reference',v,ref[k])
        feet={m.name2id('walker/tarsal_claw_'+leg+'_collision','geom'):i for i,leg in enumerate(meta['legs'])}
        adhesives=[m.name2id('walker/adhere_claw_'+leg,'actuator') for leg in meta['legs']]
        r=int(m.jnt_dofadr[m.body_jntadr[m.body_rootid[thorax]]]);old=env.task.after_substep
        def hook(physics,random_state):
            nonlocal count
            old(physics,random_state);index=count;count+=1
            if index%10!=9:return
            s=scratch;s.time=d.time
            for key in ['qpos','qvel','act','ctrl','qacc_warmstart','qfrc_applied','xfrc_applied','mocap_pos','mocap_quat','userdata']:getattr(s,key)[:]=getattr(d,key)
            mujoco.mj_forward(m.ptr,s);loads=np.zeros(7)
            for ci,c in enumerate(s.contact):
                b1=int(m.geom_bodyid[c.geom1]);b2=int(m.geom_bodyid[c.geom2])
                if b1==0 and (m.id2name(b2,'body') or '').startswith('walker/'):sign=1
                elif b2==0 and (m.id2name(b1,'body') or '').startswith('walker/'):sign=-1
                else:continue
                f=np.zeros(6);mujoco.mj_contactForce(m.ptr,s,ci,f);world=sign*np.asarray(c.frame).reshape(3,3).T@f[:3]
                leg=feet.get(int(c.geom1),feet.get(int(c.geom2),6));loads[leg]+=max(0.,world[2])
            compare('diagnostic_load',loads,native['rows'][index,1:8])
            realized=s.actuator_moment[adhesives,r:r+3]*s.actuator_force[adhesives,None]
            compare('environmental',np.r_[s.time,s.qfrc_constraint[r:r+3],s.qfrc_actuator[r:r+3],realized.reshape(-1),s.actuator_force[adhesives],s.act[m.actuator_actadr[adhesives]]],native['environmental'][index])
        env.task.after_substep=hook
        def state(i):
            for k in ['qpos','qvel','act','ctrl','qacc_warmstart']:compare(k,getattr(d,k),body[k][i])
            compare('time',d.time,body['time'][i]);compare('com',d.subtree_com[thorax],body['com'][i])
        state(0);spec=env.action_spec();lo=spec.minimum.astype('float32');hi=spec.maximum.astype('float32')
        for i in range(len(pol['canonical'])):
            assert not ts.last()
            inputs={k:np.asarray(v,dtype=np.float32) for k,v in ts.observation.items()}
            for k,v in inputs.items():compare('observations',v,obs[k][i])
            output=pi({k:tf.convert_to_tensor(v[None]) for k,v in inputs.items()}).mean()[0].numpy();compare('policy',output,pol['canonical'][i])
            scaled=.5*(np.clip(output,-1.,1.)+1.);scaled*=hi-lo;scaled+=lo;compare('scaled_actions',scaled,pol['scaled'][i])
            ts=env.step(pol['scaled'][i].copy());state(i+1);compare('reward',ts.reward,pol['reward'][i]);compare('discount',ts.discount,pol['discount'][i]);compare('mocap_qpos',phy.bind(env.task._mocap_joints).qpos,tracking['actual_mocap_qpos'][i])
        for k,v in ts.observation.items():compare('terminal_observations',np.asarray(v,dtype=np.float32),last[k])
        assert count==record['native_steps'] and bool(ts.last())==record['terminal']
        passed=max(errors.values())<1e-10
        report=dict(status='INDEPENDENT_MEASURED_REPLAY_PASS' if passed else 'REPLAY_FAIL',max_errors=errors,controls=len(pol['canonical']),native_steps=count,
            original_trajectory=protocol['trajectory'],completed_expected_controls=len(pol['canonical'])==meta['expected_controls'],teacher_unseen_generalization='NOT_ESTABLISHED',wall_s=time.monotonic()-t,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        with (p/'VERIFICATION.json').open('x') as f:json.dump(report,f,indent=2)
        print(json.dumps(report,indent=2))
        if not passed:raise SystemExit(2)
    finally:env.close();env.task._traj_generator._h5.close()

if __name__=='__main__':main()
