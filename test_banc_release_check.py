"""Saved-artifact checks for the BANC release verification."""
import json
import unittest
import run_banc_release_check as run

class BancReleaseCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads((run.OUT / 'banc_release_check.json').read_text())

    def test_official_release_matches_the_local_copy(self):
        self.assertTrue(self.doc['official_matches_local'])
        self.assertEqual(self.doc['official']['efferent'], self.doc['local']['efferent'])
        self.assertEqual(self.doc['official']['targets'], self.doc['local']['targets'])

    def test_missing_targets_are_missing_in_both(self):
        for source in ('official', 'local'):
            targets = self.doc[source]['targets']
            self.assertEqual(targets['pleural_promotor_muscle']['total'], 0)
            self.assertEqual(targets['tergopleural_promotor_muscle']['total'], 8)

    def test_no_promotor_outside_the_front_leg_anywhere(self):
        c = self.doc['codex_promotor_cells']
        self.assertEqual(c['legs_of_those_cells'], ['front_leg'])
        self.assertGreater(c['annotated_rows'], 0)

    def test_user_tags_cannot_fill_pools(self):
        tags = self.doc['cave_user_tags']
        self.assertTrue(tags['Sternal adductor MN']['mixes_several_muscles'])
        self.assertTrue(tags['Pleural remotor/abductor MN']['mixes_several_muscles'])

    def test_sources_are_hashed(self):
        s = self.doc['sources']
        for key in ('official_supplemental_data_2', 'official_codex_annotations', 'official_cell_info',
                    'local_working_copy'):
            self.assertEqual(len(s[key]['sha256']), 64, key)
            self.assertGreater(s[key]['rows'], 0, key)

    def test_conclusion_says_no_token_would_help(self):
        self.assertIn('No access token', self.doc['conclusion'])

if __name__ == '__main__':
    unittest.main()
