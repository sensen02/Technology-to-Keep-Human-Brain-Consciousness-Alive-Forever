"""Finite spike-only translator contracts (P2/S4 engineering, not training).

The reference below is deliberately stateless and independent of implementation:
only the most recent 80 samples can contribute to an actuator command.  No
body state, controller, dataset, simulation, or upstream training gate is used.
"""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from engine.embodied.motor_translation import FiniteMotorTranslator


DT = 0.0005
BINS = ((0, 5), (5, 10), (10, 20), (20, 40), (40, 80))
DEFAULT_WEIGHTS = np.array([0.4, 0.25, 0.2, 0.1, 0.05])


def manifest():
    # First two rows are the SAME pool, despite opposite index order.  Neurons
    # 3 and 4 are outside the motor whitelist; the final actuator is unmapped.
    return dict(n_neurons=5, n_actuators=4, rows=[
        dict(actuator_index=0, local_indices=[2, 0]),
        dict(actuator_index=1, local_indices=[0, 2]),
        dict(actuator_index=2, local_indices=[1]),
        dict(actuator_index=3, local_indices=[])])


def block(n, dtype=float):
    return np.zeros((n, 5), dtype=dtype)


def reference(history, m, weights=None):
    """Zero-pad absent past samples, mean over neurons and full bin duration."""
    padded = np.zeros((80, m['n_neurons']))
    tail = np.asarray(history)[-80:]
    if len(tail):
        padded[-len(tail):] = tail
    # Canonical first-appearance pool ordering; shared rows use one weight row.
    pools = list(dict.fromkeys(tuple(sorted(r['local_indices']))
                               for r in m['rows'] if r['local_indices']))
    w = np.tile(DEFAULT_WEIGHTS, (len(pools), 1)) if weights is None else weights
    ctrl = np.zeros(m['n_actuators'])
    for i, row in enumerate(m['rows']):
        pool = tuple(sorted(row['local_indices']))
        if not pool:
            continue
        rates = np.array([padded[80-hi:80-lo, list(pool)].mean() / DT / 100
                          for lo, hi in BINS])
        ctrl[i] = np.clip(np.dot(w[pools.index(pool)], rates), 0, 1)
    return ctrl


class FiniteMotorTranslatorTests(unittest.TestCase):
    def consume(self, translator, spikes, start=0., **kwargs):
        return translator.consume(spikes, spike_dt_s=DT, start_s=start,
                                  end_s=start + len(spikes)*DT, **kwargs)

    def test_zero_input_is_exact_zero_including_unmapped(self):
        out = FiniteMotorTranslator(manifest())
        for n, start in ((1, 0.), (80, DT), (160, 81*DT)):
            ctrl, trace = self.consume(out, block(n), start)
            np.testing.assert_array_equal(ctrl, np.zeros(4))
            self.assertIsInstance(trace, dict)
            self.assertTrue(np.isfinite(ctrl).all())

    def test_each_default_age_bin_matches_independent_rate_formula(self):
        # A single spike in a two-neuron pool stays below clipping in every bin.
        for age in (0, 4, 5, 9, 10, 19, 20, 39, 40, 79, 80):
            with self.subTest(age=age):
                spikes = block(81)
                spikes[-1-age, 0] = 1
                ctrl, _ = self.consume(FiniteMotorTranslator(manifest()), spikes)
                np.testing.assert_allclose(ctrl, reference(spikes, manifest()),
                                           rtol=1e-13, atol=0)

    def test_finite_cutoff_exactly_after_80_zero_samples(self):
        out = FiniteMotorTranslator(manifest())
        pulse = block(1)
        pulse[0, 0] = 1
        self.assertGreater(self.consume(out, pulse)[0][0], 0)
        ctrl, _ = self.consume(out, block(79), DT)
        self.assertGreater(ctrl[0], 0, 'age 79 must still contribute')
        ctrl, _ = self.consume(out, block(1), 80*DT)
        np.testing.assert_array_equal(ctrl, np.zeros(4))
        ctrl, _ = self.consume(out, block(800), 81*DT)
        np.testing.assert_array_equal(ctrl, np.zeros(4))

    def test_shared_pool_sorted_indices_and_unmapped_zero(self):
        weights = np.array([[.11, .22, .33, .44, .55],
                            [.51, .41, .31, .21, .11]])
        spikes = block(80)
        spikes[-1, 0] = 1
        spikes[-7, 2] = 1
        spikes[-31, 1] = 1
        ctrl, _ = self.consume(FiniteMotorTranslator(manifest(), weights), spikes)
        np.testing.assert_allclose(ctrl, reference(spikes, manifest(), weights))
        self.assertEqual(ctrl[0], ctrl[1])
        self.assertEqual(ctrl[3], 0)

    def test_pool_rate_is_mean_not_sum(self):
        spikes = block(80)
        spikes[-11, 0] = 1
        spikes[-11, 1] = 1
        ctrl, _ = self.consume(FiniteMotorTranslator(manifest()), spikes)
        self.assertAlmostEqual(ctrl[2], 2*ctrl[0])
        self.assertAlmostEqual(ctrl[0], .2 * (1 / (2*10*DT*100)))

    def test_outside_motor_whitelist_has_no_effect_when_firing_every_sample(self):
        spikes = block(80)
        spikes[-1, 0] = 1
        outside = spikes.copy()
        outside[:, 3:] = 1
        a, _ = self.consume(FiniteMotorTranslator(manifest()), spikes)
        b, _ = self.consume(FiniteMotorTranslator(manifest()), outside)
        np.testing.assert_array_equal(a, b)

    def test_bool_spikes_equivalent_to_numeric_counts(self):
        spikes = block(80, bool)
        spikes[-1, 0] = True
        spikes[-20, 1] = True
        a, _ = self.consume(FiniteMotorTranslator(manifest()), spikes)
        b, _ = self.consume(FiniteMotorTranslator(manifest()), spikes.astype(float))
        np.testing.assert_array_equal(a, b)

    def test_nan_and_infinite_spikes_rejected_even_outside_whitelist(self):
        for value in (np.nan, np.inf, -np.inf):
            for neuron in (0, 4):
                with self.subTest(value=value, neuron=neuron):
                    spikes = block(1)
                    spikes[0, neuron] = value
                    with self.assertRaises((ValueError, TypeError)):
                        self.consume(FiniteMotorTranslator(manifest()), spikes)

    def test_weight_shape_is_unique_nonempty_pools_by_five(self):
        for shape in ((5,), (4, 5), (3, 5), (2, 4), (2, 6), (0, 5), (2, 5, 1)):
            with self.subTest(shape=shape):
                with self.assertRaises((ValueError, TypeError)):
                    FiniteMotorTranslator(manifest(), np.zeros(shape))

    def test_weights_must_be_finite_in_zero_to_four(self):
        for value in (-.00001, 4.00001, np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                weights = np.zeros((2, 5))
                weights[0, 0] = value
                with self.assertRaises((ValueError, TypeError)):
                    FiniteMotorTranslator(manifest(), weights)
        # Both inclusive endpoints are legal; weighting is clipped, not normalized.
        spikes = np.ones((80, 5))
        for value, expected in ((0., 0.), (4., 1.)):
            ctrl, _ = self.consume(FiniteMotorTranslator(
                manifest(), np.full((2, 5), value)), spikes)
            np.testing.assert_array_equal(ctrl, [expected, expected, expected, 0])

    def test_disconnect_immediately_clears_output_and_all_history(self):
        out = FiniteMotorTranslator(manifest())
        self.consume(out, np.ones((80, 5)))
        ctrl, _ = self.consume(out, np.ones((1, 5)), 80*DT, connected=False)
        np.testing.assert_array_equal(ctrl, np.zeros(4))
        ctrl, _ = self.consume(out, block(1), 81*DT)
        np.testing.assert_array_equal(ctrl, np.zeros(4))

    def test_reset_clears_history_and_timing(self):
        out = FiniteMotorTranslator(manifest())
        self.consume(out, np.ones((80, 5)))
        out.reset()
        ctrl, _ = self.consume(out, block(1))
        np.testing.assert_array_equal(ctrl, np.zeros(4))

    def test_chunked_consumption_matches_single_block_reference(self):
        spikes = block(173)
        rng = np.random.default_rng(42)
        spikes[:, :3] = (rng.random((173, 3)) < .025)
        out = FiniteMotorTranslator(manifest())
        offset = 0
        for n in (1, 4, 5, 10, 20, 40, 80, 13):
            ctrl, _ = self.consume(out, spikes[offset:offset+n], offset*DT)
            offset += n
            np.testing.assert_allclose(ctrl, reference(spikes[:offset], manifest()),
                                       rtol=1e-13, atol=1e-15)
        self.assertEqual(offset, len(spikes))

    def test_wrong_sample_dt_or_duration_rejected(self):
        cases = [dict(spike_dt_s=.001, start_s=0., end_s=.001),
                 dict(spike_dt_s=DT, start_s=0., end_s=2*DT),
                 dict(spike_dt_s=DT, start_s=DT, end_s=0.),
                 dict(spike_dt_s=DT, start_s=np.nan, end_s=DT),
                 dict(spike_dt_s=DT, start_s=0., end_s=np.inf)]
        for timing in cases:
            with self.subTest(timing=timing):
                with self.assertRaises((ValueError, TypeError)):
                    FiniteMotorTranslator(manifest()).consume(block(1), **timing)

    def test_replay_overlap_and_gap_rejected_without_changing_history(self):
        for invalid_start in (0., DT, 3*DT):
            with self.subTest(start=invalid_start):
                out = FiniteMotorTranslator(manifest())
                spikes = block(2)
                spikes[0, 0] = 1
                self.consume(out, spikes)
                with self.assertRaises((ValueError, TypeError)):
                    self.consume(out, np.ones((2, 5)), invalid_start)
                ctrl, _ = self.consume(out, block(1), 2*DT)
                np.testing.assert_allclose(ctrl, reference(
                    np.concatenate([spikes, block(1)]), manifest()))

    def test_persistence_roundtrip_preserves_weights_and_mapping(self):
        m = manifest()
        weights = np.array([[.1, .2, .3, .4, .5], [.6, .7, .8, .9, 1.]])
        spikes = block(80)
        spikes[-1, 0] = 1
        spikes[-18, 1] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'translator.npz'
            out = FiniteMotorTranslator(m, weights)
            out.save(path)
            restored = FiniteMotorTranslator.load(m, path)
            expected, _ = self.consume(out, spikes)
            actual, _ = self.consume(restored, spikes)
            np.testing.assert_array_equal(actual, expected)

    def test_persistence_rejects_mismatched_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'translator.npz'
            FiniteMotorTranslator(manifest()).save(path)
            for change in ('pool', 'neurons', 'actuators'):
                with self.subTest(change=change):
                    m = manifest()
                    if change == 'pool':
                        m['rows'][2]['local_indices'] = [3]
                    elif change == 'neurons':
                        m['n_neurons'] = 6
                    else:
                        m['n_actuators'] = 5
                        m['rows'].append(dict(actuator_index=4, local_indices=[]))
                    with self.assertRaises((ValueError, TypeError)):
                        FiniteMotorTranslator.load(m, path)

    def test_no_mutation_of_caller_manifest_weights_or_spikes(self):
        m = manifest()
        original_manifest = copy.deepcopy(m)
        weights = np.tile(DEFAULT_WEIGHTS, (2, 1))
        original_weights = weights.copy()
        spikes = block(80)
        spikes[-1, 0] = 1
        original_spikes = spikes.copy()
        spikes.setflags(write=False)
        weights.setflags(write=False)
        out = FiniteMotorTranslator(m, weights)
        self.consume(out, spikes)
        out.reset()
        self.assertEqual(m, original_manifest)
        np.testing.assert_array_equal(weights, original_weights)
        np.testing.assert_array_equal(spikes, original_spikes)

    def test_caller_array_changes_cannot_modify_internal_weights(self):
        weights = np.tile(DEFAULT_WEIGHTS, (2, 1))
        original = weights.copy()
        out = FiniteMotorTranslator(manifest(), weights)
        weights[:] = 0
        spikes = block(1)
        spikes[0, 0] = 1
        ctrl, _ = self.consume(out, spikes)
        np.testing.assert_allclose(ctrl, reference(spikes, manifest(), original))


if __name__ == '__main__':
    unittest.main(verbosity=2)
