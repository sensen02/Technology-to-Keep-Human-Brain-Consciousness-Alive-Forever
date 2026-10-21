import unittest
from tools_s4_interface_inventory import literal_defaults, summarize_manifest, inventory, DEFAULT_BUNDLE


class InventoryTests(unittest.TestCase):
    def test_literal_no_execution(self):
        text = "raise RuntimeError('do not execute')\nclass C:\n    x: float = 0.3\n"
        self.assertEqual(literal_defaults(text, 'C', ('x',)), {'x': .3})

    def test_missing_default(self):
        with self.assertRaises(ValueError):
            literal_defaults('class C: pass', 'C', ('x',))

    def fixture(self):
        root = '720575941458992031'
        return {'n_actuators': 1, 'shared_pools': [], 'rows': [
            {'actuator_index': 0, 'pool_root_ids': [int(root)], 'tier_root_ids': [int(root)],
             'local_indices': [0], 'status': 'complete'}]}, [root]

    def test_exact_mixed_string_int_identity(self):
        m, tier = self.fixture()
        self.assertEqual(summarize_manifest(m, tier)['present_unique_roots'], 1)

    def test_float_identity_rejected(self):
        m, tier = self.fixture()
        m['rows'][0]['tier_root_ids'] = [float(tier[0])]
        with self.assertRaises(ValueError):
            summarize_manifest(m, tier)

    def test_wrong_index_identity_rejected(self):
        m, tier = self.fixture()
        with self.assertRaises(ValueError):
            summarize_manifest(m, ['720575941622252796'])

    def test_frozen_bundle(self):
        data = inventory(DEFAULT_BUNDLE)
        self.assertEqual(data['motor_coverage']['status_counts'],
                         {'complete': 65, 'partial': 9, 'unresolved': 16})
        self.assertEqual(data['motor_coverage']['present_unique_roots'], 202)
        self.assertEqual(data['motor_coverage']['declared_unique_roots'], 208)
        self.assertEqual(data['sensory_defaults']['current_nA_per_unit'], .3)
        self.assertEqual(data['motor_defaults']['filter_tau_s'], .005)
        self.assertFalse(data['biological_mapping_verified'])


if __name__ == '__main__':
    unittest.main()
