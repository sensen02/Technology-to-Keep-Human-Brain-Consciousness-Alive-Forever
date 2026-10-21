"""Synthetic P3 audit contract tests: no project assets or simulation required."""
import unittest
from tools_p3_motor_version_crosswalk import crosswalk, project, identity, VERSIONS


def row(root, value=None):
    return {'root_id': str(root), **{c: str(value if value is not None else root) for c in VERSIONS}}


class TestMotorVersionCrosswalk(unittest.TestCase):
    def test_large_integer_no_precision_loss(self):
        r = 720575941434056663
        self.assertEqual(identity(r), str(r))
        # Frozen P1 JSON tier uses strings; int membership is a false negative.
        tier_strings = [str(r)]
        self.assertNotIn(r, tier_strings)
        self.assertIn(identity(r), {identity(v) for v in tier_strings})
        items, _ = crosswalk([row(r)], [r])
        self.assertEqual(items[0]['association_status'], 'same_value')
        self.assertEqual(items[0]['candidates'][0]['version_ids']['root_888'], str(r))
        with self.assertRaises(ValueError):
            identity(float(r))

    def test_duplicate_identical_rows_not_silently_deduplicated(self):
        items, _ = crosswalk([row(1), row(1)], [1])
        self.assertEqual(items[0]['association_status'], 'multiple_rows')
        self.assertEqual(items[0]['row_count'], 2)
        p = project([1], {'1': items[0]}, 'root_888')
        self.assertFalse(p['complete_projection'])
        self.assertEqual(p['unprojectable_declared_ids'], ['1'])

    def test_ambiguous_rows_retained(self):
        items, _ = crosswalk([row(1, 2), row(1, 3)], [1])
        self.assertEqual([x['version_ids']['root_888'] for x in items[0]['candidates']], ['2', '3'])
        self.assertEqual(items[0]['resolution'], 'unresolved')

    def test_missing_row(self):
        items, _ = crosswalk([row(2)], [1])
        self.assertEqual(items[0]['association_status'], 'missing_row')

    def test_missing_version(self):
        self.assertIsNone(identity('0'))
        x = row(1); x['root_888'] = None
        items, _ = crosswalk([x], [1])
        self.assertEqual(items[0]['association_status'], 'missing_version_id')
        self.assertFalse(project([1], {'1': items[0]}, 'root_888')['complete_projection'])

    def test_collision_different_declared_roots(self):
        items, collisions = crosswalk([row(1, 3), row(2, 3)], [1, 2])
        self.assertEqual(collisions['root_888'], {'3': ['1', '2']})
        self.assertEqual(items[0]['candidates'][0]['version_id_global_other_root_ids']['root_888'], ['2'])
        p = project([1, 2], {i['declared_root_id']: i for i in items}, 'root_888')
        self.assertEqual(p['cardinality_delta'], -1)

    def test_global_collision_outside_declared_set(self):
        items, collisions = crosswalk([row(1, 3), row(9, 3)], [1])
        self.assertEqual(collisions['root_888'], {})
        self.assertEqual(items[0]['candidates'][0]['version_id_global_row_counts']['root_888'], 2)
        self.assertEqual(items[0]['candidates'][0]['version_id_global_other_root_ids']['root_888'], ['9'])

    def test_shared_pool_and_per_column_differences(self):
        x = row(1); x['root_888'] = '2'
        items, _ = crosswalk([x], [1, 1])
        self.assertEqual(len(items), 1)
        p = project([1, 1], {'1': items[0]}, 'root_888')
        self.assertEqual(p['original_unique_count'], 1)
        self.assertEqual(p['added_ids'], ['2'])
        self.assertEqual(p['removed_ids'], ['1'])
        self.assertEqual(p['cardinality_delta'], 0)
        self.assertEqual(items[0]['association_status'], 'different_value_association')
        self.assertFalse(items[0]['root_equivalence_established'])


if __name__ == '__main__':
    unittest.main()
