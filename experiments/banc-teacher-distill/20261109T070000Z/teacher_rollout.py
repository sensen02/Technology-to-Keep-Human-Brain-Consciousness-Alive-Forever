"""Run the official walking policy in ITS OWN environment and record leg joint angles. Run this with the teacher venv."""
from pathlib import Path
import json, hashlib, time
import numpy as np
OUT = Path(__file__).resolve().parent
import tensorflow as tf
import tensorflow_probability as tfp   # must be imported before loading the saved model (ACTTypeSpec registration)
from flybody.fly_envs import walk_imitation
from flybody.tasks.synthetic_trajectories import constant_speed_trajectory
POLICY = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-teacher/20261107T174640Z/policies/walking')
SECONDS = 8.0
CTRL = .002
rng = np.random.RandomState(41)
env = walk_imitation(random_state=rng)
count = round(SECONDS / CTRL)
qp, qv = constant_speed_trajectory(n_steps=count + 65, speed=2, init_pos=(0, 0, .1278), control_timestep=CTRL)
env.task._traj_generator.set_next_trajectory(qp, qv)
policy = tf.saved_model.load(str(POLICY))
ts = env.reset()
model = env.physics.model
LEGS = ['T1_left', 'T2_left', 'T3_left', 'T1_right', 'T2_right', 'T3_right']
JOINTS = ['coxa_abduct', 'coxa_twist', 'coxa', 'femur_twist', 'femur', 'tibia', 'tarsus', 'tarsus2']
names = {f'{j}_{l}': j for j in JOINTS for l in LEGS}
qpos_idx = {}
for key in names:
    qpos_idx[key] = int(model.jnt_qposadr[model.joint('walker/' + key).id])
jac = np.zeros((3, model.nv))
rec = []
t0 = time.time()
for step in range(count):
    obs = {k: tf.convert_to_tensor(np.asarray(v, dtype=np.float32)[None]) for k, v in ts.observation.items()}
    act = policy(obs).mean()[0].numpy()
    ts = env.step(act)
    q = env.physics.data.qpos
    rec.append([float(q[qpos_idx[k]]) for k in names])
    if step % 250 == 0:
        print('step', step, 'of', count, 'elapsed %.2f s' % (time.time() - t0), flush=True)
rec = np.array(rec)
dt = CTRL
n_out = int(round(rec.shape[0] * dt * 200))          # resample to 200 Hz
src_t = np.arange(rec.shape[0]) * dt
dst_t = np.arange(n_out) / 200.
res = np.stack([np.interp(dst_t, src_t, rec[:, i]) for i in range(rec.shape[1])], axis=1)
np.savez_compressed(OUT / 'TEACHER_ROLLOUT.npz', times=dst_t, joint_names=np.array(list(names.keys())),
                    qpos=res.astype(np.float32), teacher_actions_hz=1 / dt, rollout_s=SECONDS)
(OUT / 'TEACHER_ROLLOUT.json').write_text(json.dumps(dict(
    policy=str(POLICY), policy_saved_model_sha256=hashlib.sha256((POLICY / 'saved_model.pb').read_bytes()).hexdigest(),
    env_nq=int(model.nq), env_nu=int(model.nu), control_timestep_s=CTRL, rollout_s=SECONDS,
    recorded_samples=int(rec.shape[0]), resampled_200hz=int(n_out), joints=list(names.keys()),
    speed_command=2, init_pos=[0, 0, .1278], wall_s=round(time.time() - t0, 1),
    note='official teacher in its own environment and body; recorded joint angles only, nothing mapped yet'), indent=2) + '\n')
print('done', rec.shape, '->', res.shape)
