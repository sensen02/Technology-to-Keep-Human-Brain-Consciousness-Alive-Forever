"""Project stock flight observations by documented joint names in the legged body."""
import argparse
import json
from pathlib import Path

import numpy as np

from fly import BASELINE, digest, flight_input, load_policy, make_env, run_one, teacher_action
from legged_fixture import build_env


class LeggedFlightTeacher:
    def __init__(self, stock, stock_observation, legged):
        self.stock_spec = stock.action_spec()
        self.legged_spec = legged.action_spec()
        self.stock_observation = stock_observation
        full_names = self.legged_spec.name.split('\t')
        stock_names = self.stock_spec.name.split('\t')
        assert len(set(full_names)) == len(full_names) == self.legged_spec.shape[0]
        self.action_columns = [full_names.index(name) for name in stock_names]
        assert np.array_equal(self.stock_spec.minimum, self.legged_spec.minimum[self.action_columns])
        assert np.array_equal(self.stock_spec.maximum, self.legged_spec.maximum[self.action_columns])
        joints = [joint.name for joint in legged.task._walker.observable_joints]
        assert len(joints) == len(set(joints))
        self.joint_columns = [joints.index(joint.name) for joint in stock.task._walker.observable_joints]
        self.other_targets = np.zeros(self.legged_spec.shape)
        for joint, angle in zip(legged.task._leg_joints, legged.task._leg_springrefs, strict=True):
            if joint.name in full_names:
                self.other_targets[full_names.index(joint.name)] = angle
        self.other_targets = np.clip(self.other_targets, self.legged_spec.minimum, self.legged_spec.maximum)
        self.contract = {'stock_action_names': stock_names,
                         'legged_action_names': full_names,
                         'stock_joint_names': [joints[i] for i in self.joint_columns],
                         'retraction_targets': {name: self.other_targets[i]
                                                for i, name in enumerate(full_names)
                                                if i not in self.action_columns},
                         'projection': 'select common joint names, omit new observation keys; retraction positions from XML'}

    def action(self, policy, observation):
        projected = {name: observation[name] for name in self.stock_observation}
        for field in ('walker/joints_pos', 'walker/joints_vel'):
            projected[field] = projected[field][self.joint_columns]
        projected['walker/actuator_activation'] = projected['walker/actuator_activation'][:0]
        assert all(value.shape == self.stock_observation[name].shape for name, value in projected.items())
        action = self.other_targets.copy()
        action[self.action_columns] = teacher_action(policy, projected, self.stock_spec)
        return action


def main(neck, output):
    names, levels, facts = flight_input(neck)
    stock = make_env()
    stock_observation = stock.reset().observation
    policy_path = BASELINE / 'flight_reference/flight'
    policy = load_policy(policy_path)
    output.mkdir()
    for case in ('sham', 'left', 'right', 'left_cut', 'right_cut', 'amp_high', 'amp_cut'):
        legged = build_env()
        legged.reset()
        extension = LeggedFlightTeacher(stock, stock_observation, legged)
        run_one(policy, levels[:, names.index(case)], facts['dt_ms'], output / case,
                seed=17, fluid=True, env=legged, extension=extension)
    (output / 'IDENTITY.json').write_text(json.dumps({
        'adapter': extension.contract, 'post_neck_sha256': facts['data_sha256'],
        'frozen_policy_sha256': digest(policy_path / 'saved_model.pb'),
        'training_updates': 0, 'seed': 17,
    }, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('neck', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    main(args.neck, args.out)
