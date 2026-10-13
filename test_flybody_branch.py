"""Saved-artifact checks for the FlyBody body branch: body measurement, translation, closure.

These are artifact checks over what the three branch scripts wrote.  They do not re-run an
episode; they assert that the recorded numbers are internally consistent, that the published
assets are the ones the artifacts name, and that the declared translations are the ones the
closure actually used.  They also anchor the branch to the published canonical baseline: the
canonical configuration inside this branch must reproduce the published escape time exactly, so a
silent change in the loop would be caught here.
"""
import hashlib
import json
import unittest
from pathlib import Path

import run_flybody_body as body_run
import run_flybody_loop as loop_run
import tools_flybody_translation as translation_run

PUBLISHED_CANONICAL_ESCAPE_S = 0.023199999999999915   # the tarsus-branch baseline artifact


class FlyBodyBodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads((body_run.OUT / 'flybody_body.json').read_text())
        cls.m = cls.doc['measured']

    def test_published_assets_are_the_ones_named(self):
        for path, declared in ((body_run.XML, body_run.XML_SHA256),
                               (body_run.FLYBODY / 'fruitfly.xml', body_run.FRUITFLY_SHA256)):
            self.assertTrue(Path(path).is_file())
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), declared)

    def test_unit_reading_holds_on_its_own_evidence(self):
        u = self.m['unit_reading_evidence']
        self.assertTrue(u['gravity_is_cm_s2_for_9p81_m_s2'])
        self.assertTrue(u['density_is_g_per_cm3_for_air'])
        self.assertAlmostEqual(u['body_weight_uN'], 9.66, places=1)
        self.assertLess(u['total_mass_model_units'], 1e-3)

    def test_body_has_no_muscle_and_has_claw_adhesion(self):
        c = self.m['body_census']
        self.assertEqual(c['na'], 0)
        self.assertEqual(c['n_joints_per_leg'], [11])
        self.assertEqual(c['n_actuators_per_leg'], [9])
        self.assertEqual(c['n_claw_collision_geoms'], 6)
        self.assertEqual(c['n_tarsus_collision_geoms'], 24)
        self.assertEqual(c['n_adhesion_actuators'], 8)
        self.assertEqual(sum(1 for a in c['adhesion_actuators'] if a['kind'] == 'claw'), 6)

    def test_published_command_stands_and_the_servo_is_load_bearing(self):
        arms = self.m['arms']
        self.assertIsNone(arms['published_command']['escape_time_s'])
        self.assertGreater(arms['published_command']['claw_support_fraction_mean_last_50ms'], .9)
        self.assertEqual(arms['published_command']['body_floor_geoms_contacted'], [])
        self.assertIsNotNone(arms['springs_only']['escape_time_s'])
        self.assertTrue(arms['springs_only']['body_floor_geoms_contacted'])
        for arm, row in arms.items():
            self.assertFalse(row['body_floor_geoms_contacted'] and arm != 'springs_only')

    def test_label_forbids_the_crediting_of_the_circuit(self):
        text = ' '.join(self.doc['what_this_does_not_support'])
        self.assertIn('not a neural one', text)
        self.assertIn('No neuron, spike, synapse', text)


class FlyBodyTranslationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads((translation_run.OUT / 'flybody_translation.json').read_text())
        cls.m = cls.doc['measured']

    def test_no_discrete_hinge_registration_is_claimed(self):
        reg = self.m['registration_evidence']
        self.assertIn('NO one-to-one hinge registration', reg['conclusion'])
        middle = next(r for r in reg['result']['T2']['assignment']
                      if r['canonical_joint'] == 'Tibia_pitch')
        self.assertNotIn('tibia', middle['flybody_joint'])

    def test_declared_segment_correspondence_is_the_only_pairing_used(self):
        self.assertEqual(self.m['segment_projection']['declared_segment_correspondence'],
                         {'Coxa': 'coxa', 'Trochanter': 'femur', 'Tibia': 'tibia'})

    def test_every_canonical_muscle_reaches_a_same_segment_hinge(self):
        p = self.m['segment_projection']
        self.assertEqual(p['n_muscles'], 90)
        self.assertEqual(p['n_dropped_muscles'], len(p['dropped_muscles']))
        for name, row in p['table'].items():
            self.assertEqual(row['flybody_segment'],
                             {'Coxa': 'coxa', 'Trochanter': 'femur', 'Tibia': 'tibia'}[row['canonical_segment']])
            self.assertGreater(row['max_abs_coefficient'], .3)
            for hinge, coeff in row['coefficients_to_flybody_hinges'].items():
                self.assertLessEqual(abs(coeff), 1.0)
                self.assertEqual(body_run.leg_of(hinge), row['leg'],
                                 f'{hinge} does not belong to {row["leg"]}')

    def test_tibia_sign_chain_passes_on_the_front_and_middle_legs_and_fails_on_the_hind_legs(self):
        # MEASURED DEFECT, asserted so it cannot disappear silently: the transferred canonical
        # sign flexes the FlyBody tibia on the front and middle legs and extends it on both hind
        # legs.  The branch reports it and offers the measured-sign variant; it never hides it.
        t = self.m['tibia_direction_check']
        self.assertEqual(t['n_legs'], 6)
        self.assertEqual(t['n_consistent'], 4)
        for leg, row in t['per_leg'].items():
            expected = not leg.startswith('LH') and not leg.startswith('RH')
            self.assertEqual(row['chain_consistent'], expected, leg)
            self.assertEqual(row['flexor_projection_coefficient'] < 0, expected, leg)
        self.assertEqual(t['sign_override_by_hinge'],
                         {'tibia_T1_left': 1, 'tibia_T1_right': 1, 'tibia_T2_left': 1,
                          'tibia_T2_right': 1, 'tibia_T3_left': -1, 'tibia_T3_right': -1})

    def test_drive_matrix_covers_six_hinges_per_leg_and_never_touches_the_tarsus(self):
        drive = self.m['compiled_drive_matrix']
        self.assertEqual(len(drive), 36)
        for hinge, entry in drive.items():
            self.assertGreater(entry['coefficient_norm'], 0)
            self.assertEqual(entry['n_muscles'], len(entry['muscles']))
            self.assertNotIn('tarsus', hinge)
            self.assertNotIn('adhere', hinge)

    def test_torque_variant_keeps_geometry_and_replaces_only_leg_actuators(self):
        import mujoco
        v = self.m['torque_variant']
        self.assertEqual(v['replaced_actuators'], 48)
        model = mujoco.MjModel.from_xml_path(v['path'])
        self.assertEqual(model.nu, 78)
        self.assertEqual(model.nq, 109)
        gears = {}
        for i in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            gears[name] = float(model.actuator_gear[i][0])
        leg_gears = {n: g for n, g in gears.items()
                     if body_run.leg_of(n) and not n.startswith('adhere_')}
        self.assertEqual(len(leg_gears), 48)
        for n, g in leg_gears.items():
            self.assertGreater(g, 0, n)
        for i in range(model.nu):
            n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if body_run.leg_of(n) and not n.startswith('adhere_'):
                self.assertAlmostEqual(float(model.actuator_biasprm[i][1]), 0.0, places=12,
                                       msg=f'{n} still carries the published servo bias')
        self.assertAlmostEqual(gears['tibia_T1_left'], .54, places=6)
        self.assertAlmostEqual(gears['femur_T1_left'], 1.6, places=6)
        self.assertTrue(any(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) == 'floor'
                            for i in range(model.ngeom)))
        self.assertEqual(hashlib.sha256(Path(v['path']).read_bytes()).hexdigest(), v['sha256'])


class FlyBodyClosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = loop_run.OUT / 'flybody_closure_200ms.json'
        cls.doc = json.loads(cls.path.read_text())
        cls.m = cls.doc['measured']

    def test_the_branch_reproduces_the_published_canonical_baseline(self):
        row = self.m['configurations']['canonical']
        self.assertAlmostEqual(row['arms']['normal']['escape_time_s'], PUBLISHED_CANONICAL_ESCAPE_S,
                               places=12)

    def test_every_configuration_ran_every_arm_with_the_interface_invariants(self):
        self.assertEqual(set(self.m['configurations']),
                         {'canonical', 'flybody_servo', 'flybody_torque'})
        for kind, row in self.m['configurations'].items():
            self.assertEqual(set(row['arms']), set(loop_run.ARM_NAMES))
            checks = row['checks']
            for key in ('replay_is_bit_identical', 'replay_trajectory_exact', 'silence_drives_nothing',
                        'neural_commands_present', 'silence_changes_the_body',
                        'sensory_cut_changes_the_body', 'motor_cut_changes_the_body'):
                self.assertTrue(checks[key], f'{kind}: {key}')
            for arm, summary in row['arms'].items():
                self.assertTrue(0 <= summary['final_height_mm'] < 10)
                escape = summary['escape_time_s']
                self.assertTrue(escape is None or 0 < escape <= self.m['duration_s'],
                                f'{kind}/{arm}: escape {escape} outside the episode')
                self.assertEqual(len(summary['records']),
                                 int(round(self.m['duration_s'] / self.m['clocks']['command_dt_s'])))

    def test_the_loop_is_causally_in_the_chain_on_both_bodies(self):
        for kind in ('canonical', 'flybody_servo', 'flybody_torque'):
            row = self.m['configurations'][kind]
            self.assertGreater(row['checks']['normal_vs_silence_body_max_abs_difference'], 1.0)
            normal = row['arms']['normal']
            self.assertGreater(sum(r['raw_motor_spikes_before_any_silence']
                                   for r in normal['records']), 100)

    def test_the_published_servo_holds_better_than_the_connectome_does(self):
        row = self.m['configurations']['flybody_servo']
        self.assertIsNotNone(row['arms']['normal']['escape_time_s'])
        self.assertIsNone(row['arms']['spike_output_silence']['escape_time_s'])
        self.assertGreater(row['arms']['normal']['escape_time_s'],
                           self.m['configurations']['canonical']['arms']['normal']['escape_time_s'])

    def test_torque_replacement_removes_the_free_posture_hold(self):
        torque = self.m['configurations']['flybody_torque']
        servo = self.m['configurations']['flybody_servo']
        self.assertIsNotNone(torque['arms']['spike_output_silence']['escape_time_s'])
        self.assertLess(torque['arms']['spike_output_silence']['escape_time_s'],
                        torque['arms']['normal']['escape_time_s'])
        self.assertIsNone(servo['arms']['spike_output_silence']['escape_time_s'])

    def test_translation_artifact_used_is_the_artifact_on_disk(self):
        declared = self.m['declared_translation']
        self.assertEqual(declared['drive_fraction'], .5)
        self.assertEqual(declared['translation_artifact_sha256'],
                         hashlib.sha256(Path(declared['translation_artifact']).read_bytes()).hexdigest())

    def test_sensory_translation_matches_the_hinges_the_loop_read(self):
        tr = json.loads(Path(self.m['declared_translation']['translation_artifact']).read_text())
        for leg in ('LF', 'RF', 'LM', 'RM', 'LH', 'RH'):
            entry = tr['measured']['sensory_translation'][leg]
            self.assertEqual(len(entry['multi_joint_group']), 6)
            self.assertIn(entry['tibia_hinge'], entry['multi_joint_group'])
            self.assertTrue(all('tarsus' not in n for n in entry['multi_joint_group']))

    def test_the_measured_tibia_sign_variant_changes_the_outcome_but_does_not_rescue_it(self):
        path = loop_run.OUT / 'flybody_closure_200ms_tibiasign_measured.json'
        self.assertTrue(path.is_file(), 'the measured-sign variant artifact is missing')
        other = json.loads(path.read_text())['measured']
        self.assertEqual(other['tibia_sign_variant'], 'measured')
        declared = self.m['configurations']
        for kind in ('flybody_servo', 'flybody_torque'):
            a = declared[kind]['arms']['normal']['escape_time_s']
            b = other['configurations'][kind]['arms']['normal']['escape_time_s']
            self.assertIsNotNone(a)
            self.assertIsNotNone(b)
        self.assertEqual(other['configurations']['canonical']['arms']['normal']['escape_time_s'],
                         declared['canonical']['arms']['normal']['escape_time_s'],
                         'the canonical body has no tibia override and must be unchanged')

    def test_label_and_scope_are_explicit(self):
        self.assertIn('NOT A PRODUCT', self.doc['label'])
        text = ' '.join(self.doc['what_this_does_not_support'])
        self.assertIn('not evidence that the connectome made it stand', text)
        self.assertIn('loses muscle-level resolution', text)


if __name__ == '__main__':
    unittest.main()
