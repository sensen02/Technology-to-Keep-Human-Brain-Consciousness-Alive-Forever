"""Version and causal observation-contract check; no env.step calls."""
from pathlib import Path
import json,hashlib,importlib.metadata
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import mujoco
from flybody.fly_envs import walk_imitation
ROOT=Path(__file__).resolve().parent
OUT=Path('/home/sensen/Desktop/engine修复验收/flybody-teacher/20261107T174640Z')

def main():
    policy=tf.saved_model.load(str(ROOT/'policies/walking'))
    env=walk_imitation(random_state=np.random.RandomState(41))
    try:
        ts=env.reset()
        obs={k:tf.convert_to_tensor(v[None].astype(np.float32)) for k,v in ts.observation.items()}
        model=env.physics.model
        versions={k:importlib.metadata.version(k) for k in ['tensorflow-cpu','tensorflow-probability','numpy','dm-control','mujoco']}
        print(json.dumps(dict(versions=versions,activation=model.na,expected=59)),flush=True)
        assert model.na==59, 'Do not pad missing physical activation state'
        dist=policy(obs)
        action=dist.mean()[0].numpy()
        assert action.shape==(59,) and np.isfinite(action).all()
        result=dict(status='POLICY_LOAD_AND_INPUT_CONTRACT_PASS_NOT_PHYSICAL_ACCEPTANCE',versions=versions,
            nq=model.nq,nv=model.nv,nu=model.nu,na=model.na,nmesh=model.nmesh,neq=model.neq,
            inputs={k:list(v.shape) for k,v in obs.items()},action_shape=list(action.shape),
            physics_dt=env.physics.timestep(),control_dt=env.control_timestep(),default_controls=env.task._episode_steps,
            reset_time=float(env.physics.data.time),env_step_calls=0,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        with (OUT/'PREFLIGHT.json').open('x') as f:json.dump(result,f,indent=2)
        print(json.dumps(result,indent=2))
    finally:env.close()

if __name__=='__main__':main()
