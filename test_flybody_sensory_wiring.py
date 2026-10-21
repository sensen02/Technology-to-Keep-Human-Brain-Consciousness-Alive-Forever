"""Single-joint wiring probes. 手工研究原型，不是成品; no neural/standing claim."""
import unittest
import numpy as np
import mujoco
import run_flybody_loop as loop


class SensoryWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body = loop.build_body('flybody_servo')
        cls.body['reference'] = cls.body['seated'].copy()

    def setUp(self):
        self.data = mujoco.MjData(self.body['model'])
        self.data.qpos[:] = self.body['reference']
        mujoco.mj_forward(self.body['model'], self.data)

    def test_tibia_velocity_uses_dof_address_on_every_leg(self):
        model = self.body['model']
        for leg in loop.LEGS:
            key = leg.lower()
            jid = model.joint(self.body['role_hinge'][key]['tibia']).id
            q = int(model.jnt_qposadr[jid])
            v = int(model.jnt_dofadr[jid])
            self.assertNotEqual(q, v)  # free joint makes qpos and qvel offsets different
            self.data.qvel[:] = 0
            self.data.qvel[v] = 1.234
            old = loop.observe(self.body, self.data, self.body['weight_per_leg'])
            role = loop.observe(self.body, self.data, self.body['weight_per_leg'], {'claw': 'tibia'})
            self.assertAlmostEqual(role[key].tibia_joint_vel_rad_s, 1.234)
            self.assertAlmostEqual(role[key].tibia_joint_vel_rad_s, old[key].tibia_joint_vel_rad_s)
            # This is the formerly wrong address: its velocity must stay zero.
            self.assertEqual(float(self.data.qvel[q]), 0)

    def test_coxa_role_is_not_a_noop(self):
        model = self.body['model']
        for leg in loop.LEGS:
            key = leg.lower()
            self.data.qpos[:] = self.body['reference']
            name = self.body['role_hinge'][key]['coxa']
            q = int(model.jnt_qposadr[model.joint(name).id])
            self.data.qpos[q] += .125
            mujoco.mj_forward(model, self.data)
            role = loop.observe(self.body, self.data, self.body['weight_per_leg'], {'hairplate': 'coxa'})
            old = loop.observe(self.body, self.data, self.body['weight_per_leg'])
            self.assertAlmostEqual(role[key].multi_joint_dev_rad, .125)
            self.assertNotAlmostEqual(role[key].multi_joint_dev_rad, old[key].multi_joint_dev_rad)
            for other in role:
                if other != key:
                    self.assertAlmostEqual(role[other].multi_joint_dev_rad, 0)

    def test_current_summary_tracks_actual_vector_and_empty_groups(self):
        groups = {'lf': {'hook_direction': [1, 3], 'empty': []}}
        current = np.array([0., -.2, 0., .4])
        summary = loop.channel_current_summary(current, groups)
        self.assertAlmostEqual(summary['lf']['hook_direction']['sum_nA'], .2)
        self.assertEqual(summary['lf']['hook_direction']['min_nA'], -.2)
        self.assertEqual(summary['lf']['empty']['n_cells'], 0)
        self.assertIsNone(summary['lf']['empty']['max_nA'])
        cut = loop.channel_current_summary(np.zeros(4), groups)
        self.assertEqual(cut['lf']['hook_direction']['sum_nA'], 0)

    def test_actuator_proxy_reads_force_not_contact(self):
        for i in range(self.body['model'].nu):
            self.data.actuator_force[i] = 0
        key = 'lf'
        idx = self.body['actuators_per_leg'][key][0]
        self.data.actuator_force[idx] = -2.5
        role = loop.observe(self.body, self.data, self.body['weight_per_leg'], {'campaniform': 'actuator_torque'})
        self.assertAlmostEqual(role[key].contact_force_norm, 2.5/self.body['support_torque'][key])
        for other in role:
            if other != key:
                self.assertEqual(role[other].contact_force_norm, 0)
        self.assertTrue(np.isfinite(role[key].contact_force_norm))


if __name__ == '__main__':
    unittest.main()
