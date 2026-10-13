"""Tests for the sourced-parameter gate: no source, no parameter.

The gate is pure (it reads a JSON file and writes another), so these tests build their own literature
files in a temporary directory and never touch the real one.  They check the two failure modes that
matter: a number without a citation must be REFUSED, and a fully cited file must be converted with the
arithmetic recorded.
"""
import json
import tempfile
import unittest
from pathlib import Path

import tools_apply_sourced_sensory_config as gate


def write(root, name, doc):
    path = Path(root)/name
    path.write_text(json.dumps(doc))
    return path


CITED = {'citation': 'Author et al. 2000, J. Exp. Biol. 203:1-10', 'url': 'https://example.org/paper'}


class SourcedParameterGateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name

    def run_gate(self, doc, name='lit.json'):
        write(self.root, name, doc)
        import tools_apply_sourced_sensory_config as mod
        mod.SOURCE = Path(self.root)/name
        mod.OUT = Path(self.root)/'out.json'
        return mod.main()

    def test_a_missing_file_is_refused(self):
        import tools_apply_sourced_sensory_config as mod
        mod.SOURCE = Path(self.root)/'absent.json'
        mod.OUT = Path(self.root)/'out.json'
        with self.assertRaises(SystemExit) as caught:
            mod.main()
        self.assertIn('REFUSED', str(caught.exception))

    def test_a_number_without_a_citation_is_refused(self):
        with self.assertRaises(SystemExit) as caught:
            self.run_gate({
                'angle_normaliser_channel': 'claw_position',
                'velocity_normaliser_channel': None,
                'force_normaliser_channel': None,
                'club_quantity': 'zero',
                'channels': {'claw_position': {
                    'quantity': 'femur-tibia joint angle',
                    'published_range': {'low': -60.0, 'high': 60.0, 'unit': 'deg'},
                    'source': {}}}})
        self.assertIn('no citation', str(caught.exception))

    def test_a_citation_without_a_url_is_refused(self):
        with self.assertRaises(SystemExit) as caught:
            self.run_gate({
                'angle_normaliser_channel': 'claw_position',
                'club_quantity': 'zero',
                'channels': {'claw_position': {
                    'quantity': 'femur-tibia joint angle',
                    'published_range': {'low': -60.0, 'high': 60.0, 'unit': 'deg'},
                    'source': {'citation': 'Author et al. 2000'}}}})
        self.assertIn('no URL', str(caught.exception))

    def test_an_undeclared_angle_source_is_refused(self):
        with self.assertRaises(SystemExit) as caught:
            self.run_gate({'club_quantity': 'zero', 'channels': {'claw_position': {
                'quantity': 'x', 'published_range': {'low': 0, 'high': 1, 'unit': 'rad'},
                'source': CITED}}})
        self.assertIn('angle_normaliser_channel is not declared', str(caught.exception))

    def test_a_wrong_unit_for_the_normaliser_is_refused(self):
        with self.assertRaises(SystemExit) as caught:
            self.run_gate({'angle_normaliser_channel': 'claw_position', 'club_quantity': 'zero',
                           'channels': {'claw_position': {
                               'quantity': 'femur-tibia joint angle',
                               'published_range': {'low': 0, 'high': 1, 'unit': 'uN'},
                               'source': CITED}}})
        self.assertIn('must be an angle', str(caught.exception))

    def test_a_fully_cited_file_is_converted_with_the_arithmetic_recorded(self):
        out = self.run_gate({
            'angle_normaliser_channel': 'claw_position',
            'velocity_normaliser_channel': None,
            'force_normaliser_channel': 'campaniform_strain',
            'club_quantity': 'tibia_joint_vel_rad_s',
            'channels': {
                'claw_position': {'quantity': 'femur-tibia joint angle',
                                  'published_range': {'low': -60.0, 'high': 60.0, 'unit': 'deg'},
                                  'source': CITED},
                'campaniform_strain': {'quantity': 'leg load',
                                       'published_range': {'low': 0.0, 'high': 5.0,
                                                           'unit': 'body_weight'},
                                       'source': CITED},
                'tibia_joint_vel_rad_s': {'quantity': 'joint angular velocity',
                                          'published_range': {'low': 0, 'high': 1, 'unit': 'rad'},
                                          'source': CITED}}})
        self.assertAlmostEqual(out['reference_angle_rad'], 1.0471975511965976, places=9)
        self.assertIsNone(out['reference_velocity_rad_s'])
        self.assertEqual(out['reference_force'], 5.0)
        self.assertEqual(out['club_quantity'], 'tibia_joint_vel_rad_s')
        self.assertIn('half of the published range', out['derivation']['reference_angle_rad']
                      ['arithmetic'])
        self.assertIn('velocity_normaliser_channel', out['refused_without_source'])
        self.assertEqual(out['derivation']['reference_force']['source'], CITED)
        self.assertTrue(Path(self.root, 'out.json').is_file())

    def test_the_loop_default_keeps_the_club_channel_at_zero(self):
        # the branch injects the club current itself, and its default must be the historical no-op so
        # no existing artifact's meaning changes.
        import inspect
        import run_flybody_loop as loop
        default = inspect.signature(loop.episode).parameters['club_quantity'].default
        self.assertEqual(default, 'zero')


class SourcedArtefactTests(unittest.TestCase):
    """The artifacts the sourced normalisers and roles actually produced."""

    @classmethod
    def setUpClass(cls):
        import run_flybody_loop as loop
        cls.out = loop.OUT
        cls.config = json.loads((cls.out/'sourced_sensory_config.json').read_text())
        cls.audit = json.loads((cls.out/'flybody_sourced_sensory_run.json').read_text())['measured']
        cls.geometry = json.loads((cls.out/'flybody_joint_geometry.json').read_text())['measured']
        cls.literature = json.loads(
            (loop.HERE/'data/literature_sensory_parameters.json').read_text())

    def test_the_angle_and_velocity_normalisers_are_half_the_cited_spans(self):
        import math
        claw = self.literature['channels']['claw_position']['published_range']
        hook = self.literature['channels']['hook_direction']['published_range']
        self.assertAlmostEqual(self.config['reference_angle_rad'],
                               math.radians((claw['high']-claw['low'])/2), places=9)
        self.assertAlmostEqual(self.config['reference_velocity_rad_s'],
                               math.radians((hook['high']-hook['low'])/2), places=9)
        self.assertIn('half of the published range', self.config['derivation']
                      ['reference_angle_rad']['arithmetic'])

    def test_the_force_normaliser_is_refused_not_invented(self):
        self.assertIsNone(self.literature['force_normaliser_channel'])
        self.assertIsNone(self.config['reference_force'])
        self.assertIn('force_normaliser_channel', self.config['refused_without_source'])
        self.assertIn('club_quantity', self.literature)
        self.assertEqual(self.config['club_quantity'], 'zero')
        self.assertNotIn('club_quantity', self.config['refused_without_source'])

    def test_the_filter_window_is_sourced_and_its_rule_stated(self):
        window = self.literature['filter_tau_window_ms']
        self.assertLess(window['low'], window['high'])
        self.assertAlmostEqual(self.config['filter_tau_s'],
                               (window['low']*window['high'])**.5/1000.0, places=12)
        note = self.config['derivation']['filter_tau_s']['note']
        self.assertIn('not a published rule', note)

    def test_the_seated_pose_sits_at_the_published_reference_angle(self):
        summary = self.geometry['seated_angle_summary']
        # the femoral chordotonal organ's published minimal-activity angle is about 80-90 deg, and the
        # model's own neutral stance is measured here, not assumed
        self.assertTrue(75.0 < summary['mean_deg'] < 100.0)
        for leg, row in self.geometry['per_leg'].items():
            lo, hi = row['femur_tibia_angle_range_deg']
            self.assertLess(lo, 30.0, leg)
            self.assertGreater(hi, 150.0, leg)
            self.assertTrue(-60.0 < row['hinge_to_anatomical_angle_deg_per_rad'] < -54.0, leg)

    def test_sourcing_the_parameters_improves_the_stance_but_not_the_reference(self):
        summary = self.audit['summary']
        published = summary['published_config_historical_roles']
        sourced = summary['sourced_normalisers_literature_roles']
        self.assertGreater(sourced['standing_measured']['n_claw_contacts'],
                           published['standing_measured']['n_claw_contacts'])
        self.assertEqual(sourced['standing_measured']['n_claw_contacts'], 6)
        self.assertEqual(sourced['failed_checks'], ['quiet'])
        self.assertGreater(sourced['height_mean_last_50ms_mm'],
                           published['height_mean_last_50ms_mm'])
        # THE KEY REGRESSION: no configuration, sourced or not, has a single restoring hinge
        for key, row in summary.items():
            self.assertEqual(row['n_restoring'], 0, key)
            self.assertGreater(row['median_correlation'], .5, key)

    def test_the_literature_roles_are_measurably_inert_because_that_channel_is_tiny(self):
        summary = self.audit['summary']
        a = summary['sourced_normalisers_historical_roles']
        b = summary['sourced_normalisers_literature_roles']
        self.assertAlmostEqual(a['mean_abs_drive'], b['mean_abs_drive'], places=12)
        self.assertLess(b['channel_current_shares']['hairplate_angle'], .02)

    def test_the_load_channel_quantity_is_recorded_as_refuted_and_a_variant_replaces_it(self):
        refutations = self.audit['refuted_quantities']
        self.assertIn('campaniform_strain', refutations)
        entry = refutations['campaniform_strain']
        self.assertIn('Saltin et al. 2025', entry['refuted_by'])
        self.assertIn('MUSCLE TENSION', entry['refuted_by'])
        self.assertIn('MODEL-INTERNAL', entry['implemented_variant'])
        # the variant ran, kept the loop alive, and moved the load channel's share upward
        summary = self.audit['summary']
        tension = summary['sourced_normalisers_literature_roles_tension_load']
        ground = summary['sourced_normalisers_literature_roles']
        self.assertGreater(tension['channel_current_shares']['campaniform_strain'],
                           ground['channel_current_shares']['campaniform_strain'])
        self.assertGreater(tension['channel_current_shares']['campaniform_strain'], .5)
        self.assertGreaterEqual(tension['standing_measured']['n_claw_contacts'], 5)
        self.assertIn('quiet', tension['failed_checks'])

    def test_the_tau_window_does_not_decide_the_filter(self):
        # with SOURCED normalisers the fastest filter gives the best stance, which reverses the
        # earlier finding that was measured with unsourced normalisers; the artifact must keep both
        window = self.audit['tau_window']
        runs = window['runs']
        best = max(runs.values(), key=lambda r: r['n_claw_contacts'])
        self.assertEqual(best['filter_tau_s'], .005)
        self.assertGreaterEqual(best['n_claw_contacts'],
                                max(r['n_claw_contacts'] for r in runs.values()))
        self.assertIn('not a published rule', window['reading'])


if __name__ == '__main__':
    unittest.main()
