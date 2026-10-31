"""Deterministic formulas vs sample-only inference; oracle never calls unwrap."""
import unittest
import numpy as np
from phase_metrology import measure_phase


def increment_path(z):
    cross = z[1:]*np.conj(z[:-1])
    return np.r_[0., np.cumsum(np.arctan2(cross.imag, cross.real))]


class PhaseMetrologyTests(unittest.TestCase):
    def check_paths(self, t, a, b):
        r = measure_phase(t, a, b, endpoint_samples=2)
        p = increment_path(a*np.conj(b))
        s = increment_path(a)-increment_path(b)
        np.testing.assert_allclose(r['product_unwrapped_rad']-r['product_unwrapped_rad'][0], p, atol=1e-12)
        np.testing.assert_allclose(r['separate_unwrapped_rad']-r['separate_unwrapped_rad'][0], s, atol=1e-12)
        self.assertFalse(r['continuous_branch_identified'])
        self.assertEqual(r['locking_status'], 'NOT_ASSESSED')
        for field, u in [('product_statistics', p), ('separate_statistics', s)]:
            advance = float(np.degrees(np.median(u[-2:])-np.median(u[:2])))
            self.assertAlmostEqual(r[field]['advance_deg'], advance, places=9)
            self.assertEqual(r[field]['halfturn_count'], int(np.floor(abs(advance)/180)))
        return r

    def test_far_from_origin_sufficient_generator_sampling(self):
        t = np.arange(101)*.01
        a = np.exp(1j*(2*np.pi*t+.4)); b = np.exp(1j*np.pi*t)
        r = self.check_paths(t, a, b)
        np.testing.assert_allclose(r['product_unwrapped_rad'], np.pi*t+.4, atol=1e-12)
        self.assertFalse(r['branch_change_intervals'])
        self.assertIn('UNIDENTIFIED', r['status'])  # generator bound is not an API observation

    def test_near_origin_without_zero_and_flag_passthrough(self):
        t = np.linspace(-1, 1, 101)
        # Continuous a(t)=t+i*0.001 never intersects zero; known from formula only.
        a = t+1j*.001; b = np.ones(len(t), complex)
        self.check_paths(t, a, b)
        low = np.abs(a)<.05  # external fixture criterion, not an interface gate
        r = measure_phase(t, a, b, endpoint_samples=2, existing_low_amplitude=low)
        np.testing.assert_array_equal(r['existing_low_amplitude'], low)
        self.assertTrue(np.isfinite(r['product_unwrapped_rad']).all())
        self.assertTrue(r['phase_defined'].all())
        self.assertFalse(r['continuous_branch_identified'])

    def test_exact_sampled_zero_is_undefined_not_bridged(self):
        t = np.arange(-4, 5, dtype=float)
        r = measure_phase(t, t.astype(complex), np.ones(9), endpoint_samples=2)
        self.assertEqual(r['sampled_zero_indices'], [4])
        self.assertTrue(np.isnan(r['product_unwrapped_rad'][4]))
        self.assertIsNone(r['product_statistics'])
        self.assertIsNone(r['separate_statistics'])
        self.assertFalse(r['legacy_statistics_defined_at_samples'])
        self.assertEqual(r['status'], 'UNDEFINED_AT_SAMPLED_ZERO')
        self.assertTrue(np.isfinite(r['product_unwrapped_rad'][5:]).all())

    def test_single_channel_alias_cannot_be_guessed(self):
        t = np.arange(12, dtype=float)
        # a_fast(t)=exp(i*1.6*pi*t) and a_alias(t)=exp(-i*.4*pi*t)
        # are indistinguishable at integer observation times, but differ continuously.
        fast = np.exp(1j*1.6*np.pi*t); alias = np.exp(-1j*.4*np.pi*t)
        np.testing.assert_allclose(fast, alias, atol=1e-13)
        r = self.check_paths(t, fast, np.ones(len(t)))
        self.assertIn('PATHS_AGREE', r['status'])
        self.assertGreater(abs((r['product_unwrapped_rad'][-1]-r['product_unwrapped_rad'][0])-1.6*np.pi*t[-1]), 10)
        self.assertFalse(r['continuous_branch_identified'])

    def test_relative_increment_alias_exposes_disagreement(self):
        t = np.arange(12, dtype=float)
        # Individual steps +/-0.75*pi, continuous relative step 1.5*pi > pi.
        r = self.check_paths(t, np.exp(1j*.75*np.pi*t), np.exp(-1j*.75*np.pi*t))
        np.testing.assert_array_equal(r['separate_minus_product_turns'], np.arange(12))
        self.assertEqual(len(r['branch_change_intervals']), 11)
        self.assertIn('PATH_DISAGREEMENT', r['status'])
        self.assertNotEqual(r['product_statistics']['halfturn_count'], r['separate_statistics']['halfturn_count'])

    def test_initial_gauge_is_not_branch_event(self):
        t = np.arange(10)*.01
        r = self.check_paths(t, np.exp(1j*(2.9+t)), np.exp(1j*(-2.9+t)))
        np.testing.assert_array_equal(r['separate_minus_product_turns'], np.zeros(10))
        self.assertEqual(r['branch_change_intervals'], [])

    def test_invalid_inputs(self):
        for t, a, b, edge in [([0,0], [1,1], [1,1],1), ([0,1], [1,np.nan],[1,1],1),
                               ([0,1], [1,1],[1,1],0), ([0,1], [1],[1,1],1)]:
            with self.assertRaises(ValueError):measure_phase(t,a,b,endpoint_samples=edge)

if __name__ == '__main__': unittest.main()
