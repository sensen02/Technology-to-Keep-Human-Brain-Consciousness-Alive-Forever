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


if __name__ == '__main__':
    unittest.main()
