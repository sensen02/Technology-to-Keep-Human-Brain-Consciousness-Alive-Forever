import unittest
from tools_s4_calcium_qualification import qualify


def fixture():
    return dict(asset_uri='synthetic://fixture',asset_revision='synthetic-v1',roi_identity='synthetic_roi',
                reporter='unknown',fluorescence_metric='unknown',baseline_method='unknown',
                angle_convention='unknown',time_unit='unknown',sync_evidence='unknown',side_evidence='unknown',
                modality='calcium',identity_level='roi_population',response_origin='unknown',
                asset_bytes_verified=False,banc_mapping=None)


class CalciumTests(unittest.TestCase):
    def test_unknowns_preserved_not_ready(self):
        r=qualify(fixture());self.assertTrue(r['structural_valid']);self.assertFalse(r['calibration_ready'])
        self.assertTrue(r['gaps']);self.assertFalse(r['biological_validation'])

    def test_missing_not_silent_unknown(self):
        d=fixture();d.pop('reporter');self.assertFalse(qualify(d)['structural_valid'])

    def test_no_spike_mislabel(self):
        d=fixture();d['modality']='spike';self.assertFalse(qualify(d)['structural_valid'])

    def test_predicted_not_measurement(self):
        d=fixture();d['response_origin']='predicted';self.assertIn('response is not declared measured',qualify(d)['gaps'])

    def test_optional_mapping_exact(self):
        d=fixture();d['banc_mapping']={'root_id':720575941458992031.0,'evidence':'claim'}
        self.assertFalse(qualify(d)['structural_valid'])
        d['banc_mapping']['root_id']='720575941458992031';self.assertTrue(qualify(d)['structural_valid'])
        self.assertFalse(qualify(d)['calibration_ready'])

    def test_wrong_top(self):
        self.assertFalse(qualify([])['structural_valid'])


if __name__=='__main__':unittest.main()
