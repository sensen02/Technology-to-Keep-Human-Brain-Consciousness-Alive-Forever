"""Run the pretrained walking policy on the same physical body used for flight."""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261118T234622Z-flight-fruit'))
from perching import land
from fly import teacher_action
from flybody.fly_envs import walk_imitation
from flybody.tasks.trajectory_loaders import constant_speed_trajectory


class SharedWalk:
    """Compute walking observations from the live FlyBody state in a sensor copy."""

    def __init__(self, live, duration):
        self.env = walk_imitation(joint_filter=.01)
        dt = self.env.control_timestep()
        n = round(duration / dt) + 120
        q, v = constant_speed_trajectory(n_steps=n, speed=2., init_pos=(0, 0, .1278),
                                         control_timestep=dt)
        self.env.task._traj_generator.set_next_trajectory(q, v)
        observation = self.env.reset().observation
        self.keys = tuple(observation)
        self.policy = land.load_policy(HERE / 'POLICY/walking')
        self.spec = self.env.action_spec()
        self.full = live.action_spec()
        self.commands = self.spec.name.split('\t')
        names = self.full.name.split('\t')
        self.columns = [names.index(j) for j in self.commands]
        assert np.array_equal(self.spec.minimum, self.full.minimum[self.columns])
        assert np.array_equal(self.spec.maximum, self.full.maximum[self.columns])
        self.task = self.env.task
        self.ref = self.task._ref_qpos
        self.activation = np.zeros_like(self.env.physics.data.act)
        self.step_index = 0
        self.ref_x = self.ref[0,0] + live.physics.named.data.subtree_com['walker/'][0] - self.env.physics.named.data.subtree_com['walker/'][0]
        md, ml=self.env.physics.model, live.physics.model
        self.qcopy, self.vcopy = [[],[]],[[],[]]
        for i in range(md.njnt):
            name=md.id2name(i,'joint')
            if not name or name.startswith('ghost/'):
                continue
            j=ml.name2id(name,'joint')
            width_q = 7 if md.jnt_type[i] == 0 else 1
            width_v = 6 if md.jnt_type[i] == 0 else 1
            self.qcopy[0].extend(range(md.jnt_qposadr[i], md.jnt_qposadr[i]+width_q))
            self.qcopy[1].extend(range(ml.jnt_qposadr[j], ml.jnt_qposadr[j]+width_q))
            self.vcopy[0].extend(range(md.jnt_dofadr[i], md.jnt_dofadr[i]+width_v))
            self.vcopy[1].extend(range(ml.jnt_dofadr[j], ml.jnt_dofadr[j]+width_v))
        self.last=np.zeros(self.full.shape)
        self.filtered=np.zeros(self.spec.shape)
        actuators=[md.name2id('walker/'+name,'actuator') for name in self.commands]
        self.action_for_act=list(zip(md.actuator_actadr[actuators],range(md.nu)))
        assert len(self.filtered) == md.nu

    def action(self, phy, speed, live_observation):
        # Advance the local reference using the current debiased walking command.
        ph=self.env.physics
        ph.data.qpos[self.qcopy[0]]=phy.data.qpos[self.qcopy[1]]
        ph.data.qvel[self.vcopy[0]]=phy.data.qvel[self.vcopy[1]]
        ph.forward()
        dt=self.env.control_timestep()
        self.ref_x += speed * dt
        i=self.step_index
        self.ref[i:,0]=self.ref_x+np.arange(len(self.ref)-i)*dt*speed
        self.task._step_counter=i
        updater=self.env._observation_updater
        updater.reset(self.env._physics_proxy, self.env.random_state)
        observation=updater.get_observation()
        for name in ('accelerometer','appendages_pos','force','gyro','touch','velocimeter','world_zaxis'):
            key='walker/'+name
            if key in live_observation and observation[key].shape == live_observation[key].shape:
                observation[key]=live_observation[key]
        target=teacher_action(self.policy,{k:observation[k] for k in self.keys},self.spec)
        self.filtered += -np.expm1(-dt/.01)*(target-self.filtered)
        self.last[self.columns] = self.filtered
        # Maintain the filtered state the restored walking policy was trained to observe.
        for act,g in self.action_for_act:
            if act>=0:
                ph.data.act[act]=self.filtered[g]
        self.step_index += 1
        return self.last.copy()
