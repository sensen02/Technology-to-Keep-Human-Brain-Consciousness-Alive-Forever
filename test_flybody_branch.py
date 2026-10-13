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
JOINT_DISPLACEMENT = .10      # the joint displacement the time-scale measurement applies


def pytest_median(values):
    """The median the audit declares as its sourced value (kept here so the test re-derives it)."""
    ordered = sorted(values)
    middle = len(ordered)//2
    return ordered[middle] if len(ordered) % 2 else .5*(ordered[middle-1]+ordered[middle])


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

    def test_the_measured_tibia_sign_variant_changes_the_outcome_without_rescuing_it(self):
        path = loop_run.OUT / 'flybody_closure_200ms_tibiasign_measured.json'
        self.assertTrue(path.is_file(), 'the measured-sign variant artifact is missing')
        other = json.loads(path.read_text())['measured']
        self.assertEqual(other['tibia_sign_variant'], 'measured')
        declared = self.m['configurations']
        for kind in ('flybody_servo', 'flybody_torque'):
            self.assertIsNotNone(declared[kind]['arms']['normal']['escape_time_s'])
        # the measured sign does change the torque body's height trace a lot (it stops crossing the
        # 90% line at all), and it still does not produce a stance -- that is measured in the drive
        # audit artifact, which this module checks below.
        self.assertIsNone(other['configurations']['flybody_torque']['arms']['normal']
                          ['escape_time_s'])
        self.assertEqual(other['configurations']['canonical']['arms']['normal']['escape_time_s'],
                         declared['canonical']['arms']['normal']['escape_time_s'],
                         'the canonical body has no tibia override and must be unchanged')

    def test_label_and_scope_are_explicit(self):
        self.assertIn('NOT A PRODUCT', self.doc['label'])
        text = ' '.join(self.doc['what_this_does_not_support'])
        self.assertIn('not evidence that the connectome made it stand', text)
        self.assertIn('loses muscle-level resolution', text)


class FlyBodyDriveAuditTests(unittest.TestCase):
    """The translation-layer diagnostic: amplitude, offset, reference, bandwidth, recovery."""

    @classmethod
    def setUpClass(cls):
        cls.path = loop_run.OUT / 'flybody_drive_audit.json'
        cls.doc = json.loads(cls.path.read_text())
        cls.m = cls.doc['measured']

    def test_the_audit_is_internally_consistent(self):
        self.assertEqual(self.m['gains_scanned'][0], 0.0)
        self.assertEqual(self.m['gains_scanned'][-1], .5)
        for key, row in self.m['configurations'].items():
            self.assertTrue(row['runs']['normal_gain0']['identical_to_silence'],
                            f'{key}: zero amplitude did not reproduce the silenced arm')

    def test_the_height_metric_alone_cannot_see_a_launched_body(self):
        # MEASURED FINDING, asserted so it cannot be lost: the one run that never crosses the 90%
        # line is a body with one claw touching, 1.5 mm up and drifting -- not a stance.
        row = self.m['held_configuration']['flybody_torque|measured']['runs']['normal']
        self.assertIsNone(row['escape_time_s'])
        self.assertFalse(row['standing']['standing_under_this_projects_criteria'])
        measured = row['standing']['measured']
        self.assertLessEqual(measured['n_claw_contacts'], 2)
        self.assertGreater(measured['height_mean_last_50ms_mm'],
                           1.15*self.m['published_neutral_height_mm'])
        self.assertIn('at_height', row['standing']['failed_checks'])
        self.assertIn('feet_down', row['standing']['failed_checks'])

    def test_nothing_holds_the_torque_body_up_under_this_projects_criteria(self):
        for key, row in self.m['held_configuration'].items():
            if not key.startswith('flybody_torque'):
                continue
            for run, r in row['runs'].items():
                self.assertFalse(r['standing']['standing_under_this_projects_criteria'],
                                 f'{key}/{run} unexpectedly passed the standing checklist')

    def test_the_published_servo_body_stands_when_the_drive_is_slow_or_absent(self):
        servo = self.m['held_configuration']['flybody_servo|declared']['runs']
        self.assertTrue(servo['spike_output_silence']['standing']
                        ['standing_under_this_projects_criteria'])
        self.assertTrue(servo['static_brace_mean_only']['standing']
                        ['standing_under_this_projects_criteria'])
        self.assertFalse(servo['normal']['standing']['standing_under_this_projects_criteria'])
        runs = self.m['configurations']['flybody_servo|declared']['runs']
        self.assertTrue(runs['normal_gain0.125']['standing']
                        ['standing_under_this_projects_criteria'])
        self.assertFalse(runs['normal_gain0.5']['standing']
                         ['standing_under_this_projects_criteria'])
        self.assertFalse(runs['normal_running_mean']['standing']
                         ['standing_under_this_projects_criteria'])

    def test_the_harm_scales_with_the_drive_bandwidth(self):
        runs = self.m['bandwidth_scan']['flybody_servo|declared']['runs']
        self.assertFalse(runs['tau_5ms']['standing']['standing_under_this_projects_criteria'])
        for tau in ('tau_20ms', 'tau_40ms'):
            self.assertTrue(runs[tau]['standing']['standing_under_this_projects_criteria'],
                            f'{tau} should restore the stance')
            self.assertGreater(runs[tau]['height_mean_last_50ms_mm'],
                               runs['tau_5ms']['height_mean_last_50ms_mm'])

    def test_recovery_from_an_identical_perturbation(self):
        runs = self.m['push_recovery']['flybody_servo|declared']['runs']
        for run, row in runs.items():
            self.assertTrue(row['push']['force_model'][0] != 0)
            self.assertEqual(len(row['height_series_mm']), len(row['height_series_times_s']))
        self.assertTrue(runs['spike_output_silence']['returned_to_band'])
        self.assertTrue(runs['static_brace_mean_only']['returned_to_band'])
        self.assertFalse(runs['normal']['returned_to_band'])

    def test_the_drive_carries_no_posture_reference(self):
        servo = self.m['configurations']['flybody_servo|declared']['posture_reference']
        plain = servo['declared_translation']
        self.assertEqual(plain['n_restoring_negative'], 0)
        self.assertEqual(plain['n_reinforcing_positive'], plain['n_hinges'])
        self.assertGreater(plain['median_correlation'], .5)
        self.assertEqual(plain['median_peak_lag_ms'], 5.0)
        for key, row in self.m['configurations'].items():
            ref = row['posture_reference']['declared_translation']
            self.assertGreater(ref['median_correlation'], 0, key)

    def test_the_polarity_probe_is_recorded_as_having_no_discriminating_power(self):
        probe = self.m['polarity_probe']['flybody_servo|declared']['runs']
        plus = probe['polarity_+1']['posture_reference']['median_correlation']
        minus = probe['polarity_-1']['posture_reference']['median_correlation']
        # a global sign flip flips the drive AND the response, so the correlation is invariant;
        # the probe therefore cannot test the sign, and the artifact must say so.
        self.assertLess(abs(plus - minus), .05)
        self.assertFalse(probe['polarity_-1']['standing']['standing_under_this_projects_criteria'])
        text = ' '.join(self.doc['what_this_does_not_support'])
        self.assertIn('NO discriminating power', text)

    def test_the_audit_does_not_claim_a_fix_or_a_behaviour(self):
        text = ' '.join(self.doc['what_this_does_not_support'])
        for phrase in ('SENSITIVITY measurement', 'declared interventions',
                       'no behaviour may be claimed'):
            self.assertIn(phrase, text)
        self.assertIn('NOT A PRODUCT', self.doc['label'])


class FlyBodyTimescaleTests(unittest.TestCase):
    """The body supplies the time scale; whether the sourced value works is then measured."""

    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads((loop_run.OUT/'flybody_timescale_audit.json').read_text())
        cls.m = cls.doc['measured']

    def test_the_two_independent_measurements_of_the_hinge_time_constant_agree(self):
        s = self.m['summaries']
        crossing = s['hinge_tau_one_over_e']['median_s']
        analytic = s['hinge_damping_over_gain']['median_s']
        self.assertEqual(s['hinge_tau_one_over_e']['n'], 36)
        self.assertLess(abs(crossing - analytic)/analytic, .1,
                        'the measured 1/e crossing and damping/gain must agree within 10%')
        self.assertEqual(self.m['tau_sourced_s'],
                         pytest_median([crossing, analytic]))

    def test_the_exponential_fits_are_reported_as_unusable(self):
        # the free response includes a static droop and a slow drift, so the log-linear fit is not
        # an exponential; the artifact must keep the r2 and the empty well-fitted set visible.
        self.assertEqual(self.m['summaries']['well_fitted_only']['n'], 0)
        worst = min(r['fit_r2'] for r in self.m['joint_time_constants'].values()
                    if r['fit_r2'] is not None)
        self.assertLess(worst, .9)

    def test_the_whole_body_does_not_settle_in_three_hundred_ms(self):
        # the trace and this field are in MODEL units (cm for FlyBody); the displacement was 0.005 cm
        # and the body ends BELOW where it was pushed, i.e. it never comes back at all.
        com = self.m['com_response']
        self.assertIsNone(com['tau_fit_s'])
        self.assertGreater(abs(com['residual_at_end']), abs(com['initial_excursion']))
        self.assertAlmostEqual(com['residual_at_end']*10, com['trace_deviation_mm'][-1], places=4)

    def test_every_hinge_sags_statically_under_its_own_weight(self):
        residuals = [abs(r['residual_at_end']) for r in self.m['joint_time_constants'].values()]
        # measured range 0.00058..0.02637 rad for a 0.10 rad displacement: every hinge keeps a static
        # offset under its own weight, and none returns to the commanded neutral.
        self.assertGreater(min(residuals), 1e-4)
        self.assertLess(max(residuals), JOINT_DISPLACEMENT/3)

    def test_the_sourced_time_constant_does_not_restore_the_stance(self):
        runs = self.m['closure_at_each_tau']
        sourced = next(r for r in runs.values() if r['is_the_sourced_value'])
        self.assertFalse(sourced['standing']['standing_under_this_projects_criteria'])
        self.assertIn('feet_down', sourced['standing']['failed_checks'])
        self.assertGreater(max(r['height_mean_last_50ms_mm'] for r in runs.values()),
                           sourced['height_mean_last_50ms_mm'])
        text = ' '.join(self.doc['what_this_does_not_support'])
        self.assertIn('tuning and must be labelled as such', text)


class FlyBodyReferenceTests(unittest.TestCase):
    """Where the drive comes from, and whether a sign convention can install a reference."""

    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads((loop_run.OUT/'flybody_reference_audit.json').read_text())
        cls.m = cls.doc['measured']
        cls.att = cls.m['attribution']['runs']

    def test_only_the_contact_force_channel_keeps_the_drive_alive(self):
        self.assertEqual(self.att['ablate_campaniform_strain']['drive_fraction_of_baseline'], 0.0)
        for channel in ('claw_position', 'hook_direction', 'hairplate_angle', 'bristle_touch'):
            self.assertGreater(self.att[f'ablate_{channel}']['drive_fraction_of_baseline'], .5,
                               channel)
        self.assertEqual(self.att['ablate_all_five']['drive_fraction_of_baseline'], 0.0)

    def test_the_harm_is_carried_by_the_joint_deviation_channel(self):
        row = self.att['ablate_claw_position']
        self.assertGreater(row['drive_fraction_of_baseline'], .5)     # most of the drive survives
        self.assertTrue(row['standing']['standing_under_this_projects_criteria'])
        self.assertFalse(self.att['none_ablated']['standing']
                         ['standing_under_this_projects_criteria'])

    def test_the_declared_normalisers_make_the_contact_channel_dominant(self):
        currents = self.att['none_ablated']['channel_currents']
        means = currents['per_channel']['mean_abs_current_nA']
        shares = currents['per_channel']['share_of_total_current']
        self.assertAlmostEqual(sum(shares.values()), 1.0, places=3)
        self.assertGreater(means['campaniform_strain'], 3*means['claw_position'])
        derived = currents['derived_what_parity_would_need']
        self.assertGreater(derived['contact_over_tibia_deviation_current_ratio'], 3)
        self.assertLess(derived['reference_angle_rad_for_tibia_deviation_parity'], .15)
        self.assertIn('NOT A RECOMMENDATION', derived['note'])

    def test_no_sign_convention_installs_a_reference(self):
        runs = self.m['conventions']['runs']
        self.assertEqual(len(runs), 4)
        for name, row in runs.items():
            self.assertEqual(row['restoring_verdict']['n_restoring'], 0, name)
            self.assertFalse(row['restoring_verdict']['is_restoring'], name)
            self.assertFalse(row['standing']['standing_under_this_projects_criteria'], name)
        self.assertEqual(self.m['reference_convention'], {})

    def test_the_caveats_about_channels_are_explicit(self):
        text = ' '.join(self.doc['what_this_does_not_support'])
        self.assertIn('NO neuron-level evidence', text)
        self.assertIn('single-signed', text)
        self.assertIn('not a corrected connectome', text)


if __name__ == '__main__':
    unittest.main()
