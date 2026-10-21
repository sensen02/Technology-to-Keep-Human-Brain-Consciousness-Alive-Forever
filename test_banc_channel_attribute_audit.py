"""Identity and missingness safeguards; not biological validation."""
import json
import unittest
import pandas as pd
from tools_banc_channel_attribute_audit import OUT, column_summary, joint_terms
from run_sensory_tuning_annotation_audit import exact_groups, unique_join


class Safeguards(unittest.TestCase):
    def test_missing_not_category(self):
        r = column_summary(pd.Series([pd.NA, None, '', 'a', 'a', 'b', 'b'], dtype='str'))
        self.assertEqual(r['missing'], 3)
        self.assertEqual(r['counts'], {'a': 2, 'b': 2})
        self.assertTrue(r['two_groups_at_least_two_cells'])

    def test_join_rejects_missing_and_duplicate(self):
        for ids in (['1'], ['1', '2', '2']):
            with self.assertRaises(ValueError):
                unique_join(ids, {'1', '2'})
        self.assertEqual(unique_join(['9', '2', '1'], {'1', '2'}), {'2': 1, '1': 2})

    def test_lookup_disagreement(self):
        identity = {'tier_root_ids': ['1'], 'sensory_groups': {'lf': {'claw_position': [0]}}}
        with self.assertRaises(ValueError):
            exact_groups(identity, {'leg_channel_by_root': {'1': ['rf', 'claw_position']}})
        identity['sensory_groups']['lf']['claw_position'] = [0, 0]
        with self.assertRaises(ValueError):
            exact_groups(identity, {'leg_channel_by_root': {'1': ['lf', 'claw_position']}})

    def test_joint_terms_literal_only(self):
        self.assertEqual(joint_terms({'cell_type': 'SNpp12', 'nerve': 'left_prothoracic_leg_nerve'}), [])
        self.assertEqual(joint_terms({'body_part_sensory': 'femoral tibial'}), ['femoral', 'tibial'])

    def test_artifact_full_nonempty_coverage(self):
        r = json.loads((OUT/'channel_attribute_audit.json').read_text())
        self.assertEqual(r['join_key'], 'banc_888_id')
        self.assertEqual(r['sensory_cells'], 3788)
        self.assertEqual(r['joined_unique_roots'], 3788)
        self.assertEqual(len({c['root_id'] for c in r['cells']}), 3788)
        self.assertEqual(sum(g['n_cells'] for legs in r['per_channel'].values() for g in legs.values()), 3788)
        self.assertEqual(len(r['per_channel']), 7)
        for legs in r['per_channel'].values():
            self.assertEqual(len(legs), 6)
            self.assertTrue(all(g['n_cells'] > 0 for g in legs.values()))
        self.assertTrue(all(c['confirmed_tuning_polarity'] is None for c in r['cells']))
        self.assertTrue(all(c['confirmed_root_specific_joint'] is None for c in r['cells']))
        self.assertTrue((OUT/'runtime_sensory_coverage.png').is_file())
        self.assertEqual(len((OUT/'runtime_sensory_evidence.csv').read_text().splitlines()), 3789)


if __name__ == '__main__':
    unittest.main()
