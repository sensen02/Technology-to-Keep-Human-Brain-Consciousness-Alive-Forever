"""Numerical and structural selftests, NOT physiological validation.

Run: /run/media/sensen/Data2/cell_wound_prototype/venv/bin/python run_da_protocol_selftest.py

What a pass means: the measured protocol timing is reproduced exactly, the
model's amounts and concentrations are internally consistent in explicit units,
the finite pool depletes and refills by the closed-form recurrence, amplitude-
only (pool) changes leave t1/2 invariant while clearance changes move it, and
the acceptance region of the qualitative pattern is non-empty and bounded.
What a pass does NOT mean: no parameter here is a measurement except the
protocol timing and the qualitative target pattern (PMC9897283); nothing is
validated against the source study's curves, which could not be extracted.
"""
import unittest

import numpy as np

from engine.da_protocol import (
    DEFAULT_DECLINE_THRESHOLD, DEFAULT_T_HALF_TOLERANCE, DAKinetics,
    NM_PER_UM3_PER_PMOL, ProtocolSpec, amount_pmol_to_concentration_nM,
    build_registry, concentration_nM_to_amount_pmol, curve_features,
    measure_trace_features, pattern_verdict, release_sequence, simulate,
    sweep_clearance, sweep_pool_refill_efficacy, sweep_shape_coupling,
)
from engine.params import ILLUSTRATIVE, MEASURED, ProvenanceError


class ProtocolTimingTests(unittest.TestCase):
    def test_protocol_timing_exactness(self):
        p = ProtocolSpec()
        self.assertEqual(p.n_stimulations, 6)
        self.assertEqual(p.ach_dose_pmol, 0.2)
        self.assertEqual(p.inter_stimulus_interval_s, 600.0)
        self.assertEqual(p.recording_duration_s, 60.0)
        self.assertEqual(p.representative_trace_s, 15.0)
        self.assertEqual(p.pre_bolus_baseline_s, 5.0)
        self.assertEqual(p.post_dissection_rest_s, 1200.0)
        self.assertEqual(p.bolus_times_s, (5.0, 605.0, 1205.0, 1805.0, 2405.0, 3005.0))
        self.assertEqual(list(np.diff(p.bolus_times_s)), [600.0] * 5)
        self.assertEqual(p.record_starts_s, (0.0, 600.0, 1200.0, 1800.0, 2400.0, 3000.0))
        self.assertEqual(p.record_ends_s, (60.0, 660.0, 1260.0, 1860.0, 2460.0, 3060.0))
        self.assertEqual(p.session_end_s, 3060.0)
        self.assertEqual(list(np.diff(p.record_starts_s)), [600.0] * 5)
        for start, end in zip(p.record_starts_s, p.record_ends_s):
            self.assertEqual(end - start, p.recording_duration_s)
        for bolus, start in zip(p.bolus_times_s, p.record_starts_s):
            self.assertEqual(bolus - start, 5.0)
        report = p.timing_report()
        self.assertEqual(report["n_stimulations"], 6)
        self.assertTrue(report["bolus_spacing_all_exactly_600_s"])
        self.assertTrue(report["first_record_allows_5_s_baseline_before_bolus"])
        self.assertEqual(report["session_span_s"], 3060.0)
        self.assertIn("PMC9897283", report["provenance"])
        # 1 min of recording sampled at 0.02 s spans exactly the record window.
        m = p.n_record_samples(0.02)
        t = np.linspace(p.record_starts_s[0], p.record_ends_s[0], m)
        self.assertEqual(t[0], 0.0)
        self.assertEqual(t[-1], 60.0)
        self.assertEqual(m, 3001)
        self.assertEqual(p.n_record_samples(15.0) * 1, 5)
        # the 15 s representative trace spans 5 s of baseline + 10 s of event
        self.assertEqual(p.representative_ends_s[0], 15.0)
        self.assertEqual(p.representative_ends_s[0] - p.record_starts_s[0], 15.0)

    def test_simulated_trace_samples_the_measured_windows(self):
        p = ProtocolSpec()
        r = simulate(p, DAKinetics(), sample_dt_s=0.05, coarse_dt_s=0.5)
        self.assertEqual(r.record_nM.shape, (6, 1201))
        self.assertEqual(r.representative_nM.shape, (6, 1201))
        self.assertEqual(r.bolus_times_s.tolist(), list(p.bolus_times_s))
        # the bolus sits 5 s into each record, so the record starts before it
        for i in range(6):
            start = r.record_time_s[i, 0]
            self.assertAlmostEqual(start, p.bolus_times_s[i] - 5.0, places=12)
            self.assertAlmostEqual(r.record_time_s[i, -1],
                                   p.bolus_times_s[i] + 55.0, places=12)
            pre = r.record_time_s[i] < p.bolus_times_s[i]
            self.assertIn(int(pre.sum()), (100, 101))      # 5 s of baseline
            self.assertLess(float(r.record_nM[i][pre].max()), 1e-40)
        # timeline reaches from the end of the rest period to the session end
        self.assertAlmostEqual(r.timeline_time_s[0], -1200.0, places=12)
        self.assertAlmostEqual(r.timeline_time_s[-1], p.session_end_s, places=12)
        self.assertTrue(np.all(r.timeline_nM >= 0.0))
        self.assertEqual(int(r.timeline_is_record.sum()), 6 * 121)


class UnitsAndLedgerTests(unittest.TestCase):
    def test_pmol_to_concentration_dimensional_check(self):
        self.assertEqual(NM_PER_UM3_PER_PMOL, 1e12)
        # 1 pmol uniformly in 1 um^3 = 1e12 nM (explicit mol / litre arithmetic)
        self.assertLess(abs(amount_pmol_to_concentration_nM(1.0, 1.0) / 1e12 - 1.0),
                        1e-12)
        # 1 pmol in 1 nL (1e6 um^3) = 1e6 nM = 1 mM; 1 pmol in 1 uL = 1 uM
        self.assertLess(abs(amount_pmol_to_concentration_nM(1.0, 1e6) / 1e6 - 1.0),
                        1e-12)
        self.assertLess(abs(amount_pmol_to_concentration_nM(1.0, 1e9) / 1e3 - 1.0),
                        1e-12)
        for amount, volume in ((0.2, 1e8), (3.7, 4.2e6), (0.0, 1.0)):
            conc = amount_pmol_to_concentration_nM(amount, volume)
            back = concentration_nM_to_amount_pmol(conc, volume)
            self.assertAlmostEqual(back, amount, places=12)
        # independent route: nM = pmol/V[um^3] * 1e12
        kin = DAKinetics()
        self.assertLess(abs(kin.conversion_report()["nM_per_pmol_in_this_volume"]
                            / (1e12 / kin.voxel_volume_um3) - 1.0), 1e-12)
        # the model's own ledger closes in pmol, which requires the conversion
        # to be consistent between the release and clearance integrals
        r = simulate(ProtocolSpec(), kin, sample_dt_s=0.05)
        self.assertLess(abs(r.ledger["extracellular_ledger_residual_pmol"]),
                        1e-12 * max(1e-30, r.ledger["released_total_pmol"]))
        # clearance integral check: an instantaneous bolus of A pmol is fully
        # recovered by k*integral(C)dt*V, and A*1e12/V is the peak in nM
        kin2 = DAKinetics(release_tau_s=0.0, clearance_tau_s=4.0, noise_std_nM=0.0)
        res = release_sequence(ProtocolSpec(), kin2)
        self.assertGreater(float(res["released_pmol"][0]), 0.0)
        core = res["ephemeral"]
        q_expected = res["released_pmol"][0] * 1e12 / kin2.voxel_volume_um3
        self.assertAlmostEqual(float(core["q_nM"][0]), q_expected, places=9)
        self.assertAlmostEqual(float(core["peak_nM"][0]), q_expected, places=6)

    def test_amount_conservation_and_positivity(self):
        p = ProtocolSpec()
        for kin in (DAKinetics(),
                    DAKinetics(pool_capacity_pmol=0.03, refill_tau_s=1800.0),
                    DAKinetics(clearance_tau_s=1.25),
                    DAKinetics(pool_capacity_pmol=0.05, stimulus_efficacy=5.0,
                               refill_tau_s=700.0)):
            r = simulate(p, kin, sample_dt_s=0.05)
            scale = max(1e-18, abs(r.ledger["released_total_pmol"]))
            self.assertLess(abs(r.ledger["pool_ledger_residual_pmol"]), 1e-14 * scale)
            self.assertLess(abs(r.ledger["extracellular_ledger_residual_pmol"]),
                            1e-12 * scale)
            self.assertTrue(np.all(r.released_pmol >= 0.0))
            self.assertTrue(np.all(r.pool_after_pmol >= 0.0))
            self.assertTrue(np.all(r.refill_drawn_pmol >= 0.0))
            self.assertTrue(np.all(r.released_pmol <= r.pool_before_pmol + 1e-15))
            self.assertTrue(np.all(r.measured_peak_nM >= 0.0))
            self.assertTrue(np.all(r.record_nM >= 0.0))
            self.assertTrue(np.all(np.isfinite(r.record_nM)))
            # released = cleared + still extracellular at session end
            self.assertAlmostEqual(
                r.ledger["released_total_pmol"],
                r.ledger["cleared_total_pmol"]
                + r.ledger["extracellular_at_session_end_pmol"], places=12)


class FinitePoolTests(unittest.TestCase):
    def test_finite_pool_depletion_monotonicity(self):
        p = ProtocolSpec()
        kin = DAKinetics(pool_capacity_pmol=10.0, refill_tau_s=1e12,
                         stimulus_efficacy=0.05)
        res = release_sequence(p, kin)
        released = res["released_pmol"]
        pool = res["pool_before_pmol"]
        self.assertTrue(np.all(np.diff(released) < 0.0))
        self.assertTrue(np.all(np.diff(pool) < 0.0))
        self.assertLess(abs(float(res["refill_drawn_pmol"].sum())), 1e-9)
        self.assertAlmostEqual(
            float(released.sum()),
            kin.pool_capacity_pmol - float(res["ephemeral"]["pool_final_pmol"])
            + float(res["refill_drawn_pmol"].sum()), delta=1e-12)
        self.assertAlmostEqual(float(res["pool_ledger_residual_pmol"]), 0.0,
                               delta=1e-12)
        self.assertTrue(res["declines_monotonically"])
        # the release caps at the remaining pool instead of going negative
        capped = release_sequence(p, DAKinetics(pool_capacity_pmol=0.05,
                                                stimulus_efficacy=1000.0,
                                                refill_tau_s=1e12))
        self.assertAlmostEqual(float(capped["released_pmol"][0]), 0.05, places=15)
        # the pool is emptied, then refills only by the residual
        # 1 - exp(-600/1e12) of capacity, so later releases are ~1e-11 pmol
        self.assertLess(float(capped["pool_before_pmol"][1]), 1e-9)
        self.assertLess(float(np.max(capped["released_pmol"][1:])), 1e-9)
        self.assertTrue(np.all(capped["released_pmol"] >= 0.0))
        r = simulate(p, DAKinetics(pool_capacity_pmol=0.05, stimulus_efficacy=1000.0,
                                   refill_tau_s=1e12), sample_dt_s=0.05)
        self.assertTrue(bool(r.capped_by_pool[0]))
        self.assertTrue(np.all(r.requested_pmol >= r.released_pmol - 1e-15))

    def test_refilling_recovery_between_stimuli(self):
        p = ProtocolSpec()
        # closed form: deficit erased per interval = 1 - exp(-ISI/tau)
        for tau in (60.0, 300.0, 1800.0):
            kin = DAKinetics(refill_tau_s=tau)
            expected = 1.0 - np.exp(-p.inter_stimulus_interval_s / tau)
            self.assertAlmostEqual(kin.refill_recovery_per_interval(p), expected,
                                   places=15)
            self.assertAlmostEqual(kin.refill_tau_over_isi(p), tau / 600.0, places=15)
        # a pool emptied at every stimulus recovers to exactly 1-exp(-ISI/tau)
        # of capacity, so A2/A1 equals that factor (total-depletion regime)
        for tau in (300.0, 600.0):
            kin = DAKinetics(pool_capacity_pmol=0.003, stimulus_efficacy=0.05,
                             refill_tau_s=tau)
            res = release_sequence(p, kin)
            self.assertAlmostEqual(float(res["released_pmol"][1]
                                         / res["released_pmol"][0]),
                                   1.0 - np.exp(-600.0 / tau), places=12)
            self.assertTrue(np.allclose(res["released_pmol"][1:],
                                        res["released_pmol"][1], rtol=1e-12))
        # fast refilling (>90 % recovered) leaves release essentially flat even
        # for a very small pool: pool size alone is NOT sufficient for a decline
        small_fast = simulate(p, DAKinetics(pool_capacity_pmol=0.003,
                                            refill_tau_s=30.0), sample_dt_s=0.05)
        self.assertGreater(small_fast.release_ratio_last_first, 0.9999)
        self.assertFalse(pattern_verdict(small_fast)["release_declines_enough"])
        self.assertTrue(pattern_verdict(small_fast)["t_half_approximately_constant"])


class PatternTests(unittest.TestCase):
    def test_pool_only_change_keeps_t_half_clearance_change_moves_it(self):
        p = ProtocolSpec()
        base = DAKinetics()
        ref = simulate(p, base, sample_dt_s=0.05)
        for kin in (DAKinetics(pool_capacity_pmol=0.003),
                    DAKinetics(pool_capacity_pmol=0.03),
                    DAKinetics(pool_capacity_pmol=10.0),
                    DAKinetics(stimulus_efficacy=0.5),
                    DAKinetics(pool_capacity_pmol=0.03, refill_tau_s=1800.0)):
            other = simulate(p, kin, sample_dt_s=0.05)
            np.testing.assert_allclose(other.t_half_s, ref.t_half_s, rtol=1e-12)
            np.testing.assert_allclose(other.measured_t_half_s,
                                       ref.measured_t_half_s, rtol=1e-12)
            self.assertLess(other.t_half_relative_spread, 1e-12)
            # whenever the release is not capped by the pool, the FIRST stimulus
            # is identical to the reference (every scenario starts with a full
            # pool), so a release decline can only appear from stimulus 2 on
            if (kin.stimulus_efficacy == base.stimulus_efficacy
                    and kin.stimulus_efficacy * p.ach_dose_pmol <= kin.pool_capacity_pmol):
                self.assertAlmostEqual(float(other.peak_nM[0]),
                                       float(ref.peak_nM[0]), delta=1e-9)
        # the amplitude really does change once the pool limits the release
        # a LARGE pool with fast refilling is just as flat as a small one: the
        # decline needs both a small pool relative to the release size AND
        # refilling that is slow compared with the 600 s inter-stimulus interval
        for kin, expect_change in ((DAKinetics(pool_capacity_pmol=10.0), False),
                                   (DAKinetics(stimulus_efficacy=0.5), True),
                                   (DAKinetics(pool_capacity_pmol=0.003), True),
                                   (DAKinetics(pool_capacity_pmol=0.03), False),
                                   (DAKinetics(pool_capacity_pmol=0.03,
                                               refill_tau_s=1800.0), True)):
            other = simulate(p, kin, sample_dt_s=0.05)
            same = abs(float(other.released_pmol[-1]) - float(ref.released_pmol[-1])) \
                < 1e-6 * float(ref.released_pmol[-1])
            self.assertEqual(same, not expect_change)
        for kin in (DAKinetics(clearance_tau_s=1.25),
                    DAKinetics(clearance_tau_s=2.5),
                    DAKinetics(clearance_tau_s=20.0)):
            other = simulate(p, kin, sample_dt_s=0.05)
            shift = abs(other.measured_t_half_s[0] - ref.measured_t_half_s[0]) \
                / ref.measured_t_half_s[0]
            self.assertGreater(shift, 0.25)
            self.assertLess(other.t_half_relative_spread, 1e-9)

    def test_clearance_change_cannot_produce_the_aged_release_decline(self):
        p = ProtocolSpec()
        sweep = sweep_clearance(p, DAKinetics())
        rows = {r["clearance_tau_s"]: r for r in sweep["rows"]}
        ratios = [r["release_ratio_last_first"] for r in sweep["rows"]]
        self.assertLess(max(ratios) - min(ratios), 1e-12)
        self.assertAlmostEqual(ratios[0], 1.0, delta=1e-9)
        halves = [r["t_half_s"][0] for r in sweep["rows"]]
        self.assertGreater(max(halves) / min(halves), 10.0)
        # monotone in clearance time constant, both within and between rows
        for r in sweep["rows"]:
            self.assertLess(r["t_half_relative_spread"], 1e-12)
            self.assertGreater(r["t_half_s"][0], 0.0)
        self.assertLess(halves[0], halves[-1])
        self.assertLess(rows[1.25]["peak_nM_first"], rows[40.0]["peak_nM_first"])
        # the aged (pool) hypothesis does decline, and keeps the young t1/2
        young = simulate(p, DAKinetics(), sample_dt_s=0.05)
        aged_pool = simulate(p, DAKinetics(pool_capacity_pmol=0.03,
                                           refill_tau_s=1800.0), sample_dt_s=0.05)
        aged_clear = simulate(p, DAKinetics(clearance_tau_s=1.25), sample_dt_s=0.05)
        self.assertLess(aged_pool.release_ratio_last_first,
                        DEFAULT_DECLINE_THRESHOLD)
        self.assertAlmostEqual(aged_pool.measured_t_half_s[0],
                               young.measured_t_half_s[0], places=6)
        self.assertAlmostEqual(aged_clear.release_ratio_last_first, 1.0, delta=1e-9)
        self.assertGreater(abs(aged_clear.measured_t_half_s[0]
                               - young.measured_t_half_s[0])
                           / young.measured_t_half_s[0], 0.5)
        verdict_young = pattern_verdict(young)
        verdict_pool = pattern_verdict(aged_pool)
        verdict_clear = pattern_verdict(aged_clear)
        self.assertFalse(verdict_young["release_declines_enough"])
        self.assertTrue(verdict_young["t_half_approximately_constant"])
        self.assertTrue(verdict_pool["reproduces_measured_pattern"])
        self.assertFalse(verdict_clear["release_declines_enough"])
        self.assertFalse(verdict_clear["reproduces_measured_pattern"])
        self.assertEqual(verdict_pool["criteria"]["decline_threshold_ratio"],
                         DEFAULT_DECLINE_THRESHOLD)
        self.assertEqual(verdict_pool["criteria"]["t_half_relative_spread_tolerance"],
                         DEFAULT_T_HALF_TOLERANCE)

    def test_region_exists_and_known_failure_modes(self):
        p = ProtocolSpec()
        base = DAKinetics()
        sweep = sweep_pool_refill_efficacy(
            p, base,
            pool_sizes_pmol=(0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
            refill_taus_s=(5.0, 15.0, 60.0, 300.0, 600.0, 1800.0, 5400.0, 1e5),
            efficacies=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2))
        self.assertEqual(sweep["n_combinations"], 384)
        self.assertTrue(sweep["region_found"])
        self.assertGreater(sweep["n_accepted"], 0)
        self.assertLess(sweep["n_accepted"], sweep["n_combinations"])
        ax = sweep["axes"]
        pools, refills, effs = ax["pool_sizes_pmol"], ax["refill_taus_s"], ax["efficacies"]

        def cell(eff, tau, pool):
            return sweep["acceptance_grid"][effs.index(eff)][refills.index(tau)][pools.index(pool)]

        def ratio(eff, tau, pool):
            return sweep["decline_ratio_grid"][effs.index(eff)][refills.index(tau)][pools.index(pool)]

        self.assertTrue(cell(0.05, 1800.0, 0.03))     # inside the region
        self.assertFalse(cell(0.05, 1800.0, 1.0))     # pool too large: no decline
        self.assertLess(ratio(0.05, 1800.0, 1.0), 1.0)
        self.assertGreater(ratio(0.05, 1800.0, 1.0), 0.9)
        self.assertFalse(cell(0.05, 60.0, 0.003))     # refill too fast: no decline
        self.assertGreater(ratio(0.05, 60.0, 0.003), 0.999)
        self.assertFalse(cell(0.05, 1800.0, 10.0))    # huge pool: no decline
        # failure modes in aggregate: fast refilling never declines, and no
        # pool size in the grid declines at the largest pools
        fast = [sweep["acceptance_grid"][ie][it]
                for ie in range(len(effs))
                for it, tau in enumerate(refills) if tau <= 60.0]
        self.assertFalse(any(any(row) for row in fast))
        huge = [sweep["acceptance_grid"][ie][it][ip]
                for ie in range(len(effs)) for it in range(len(refills))
                for ip, pool in enumerate(pools) if pool >= 3.0]
        self.assertFalse(any(huge))
        # t1/2 spread is ~0 everywhere in this grid (amplitude-only pool effect)
        self.assertLess(max(abs(x) for row in sweep["t_half_relative_spread_grid"]
                            for r2 in row for x in r2), 1e-9)
        # boundary report is consistent with the grid it summarises
        boundary = sweep["boundaries_largest_pool_still_declining"]["efficacy_0.05"]
        self.assertEqual(boundary["refill_tau_1800s"]["max_pool_size_with_decline_and_invariance"],
                         0.03)
        self.assertIsNone(boundary["refill_tau_30s"]
                          if "refill_tau_30s" in boundary else
                          boundary["refill_tau_60s"]["max_pool_size_with_decline_and_invariance"])
        # analytic area check: decline ratio in the total-depletion regime is
        # exactly the refill recovery fraction, as derived above
        self.assertAlmostEqual(ratio(0.05, 600.0, 0.003), 1.0 - np.exp(-1.0),
                               places=10)

    def test_shape_coupling_is_where_invariance_breaks(self):
        p = ProtocolSpec()
        aged = DAKinetics(pool_capacity_pmol=0.03, refill_tau_s=1800.0)
        sweep = sweep_shape_coupling(p, aged, couplings=(0.0, 0.25, 0.5, 1.0, 2.0))
        spreads = {r["release_shape_coupling"]: r["t_half_relative_spread"]
                   for r in sweep["rows"]}
        self.assertLess(spreads[0.0], 1e-12)
        self.assertLess(spreads[0.25], DEFAULT_T_HALF_TOLERANCE)
        self.assertGreater(spreads[1.0], DEFAULT_T_HALF_TOLERANCE)
        self.assertGreater(spreads[1.0], spreads[0.5])
        self.assertGreater(spreads[2.0], spreads[1.0])
        self.assertTrue(all(r["release_ratio_last_first"] < DEFAULT_DECLINE_THRESHOLD
                            for r in sweep["rows"]))


class MeasurementTests(unittest.TestCase):
    def test_sampled_measurement_agrees_with_analytic_curve(self):
        p = ProtocolSpec()
        r = simulate(p, DAKinetics(), sample_dt_s=0.01)
        for i in range(6):
            self.assertLess(abs(r.measured_t_half_s[i] - r.t_half_s[i])
                            / r.t_half_s[i], 0.01)
            self.assertLess(abs(r.measured_peak_nM[i] - r.peak_nM[i])
                            / r.peak_nM[i], 0.01)
        feats = measure_trace_features(r.record_time_s[0], r.record_nM[0])
        self.assertAlmostEqual(feats["peak_nM"], float(r.measured_peak_nM[0]),
                               places=12)
        self.assertAlmostEqual(feats["t_half_s"], float(r.measured_t_half_s[0]),
                               places=12)
        # the sampled half decay must not be below the clearance-only bound
        self.assertGreater(feats["t_half_s"], 0.0)
        self.assertLess(feats["t_half_s"], float(np.log(2) * 5.0) * 1.5)
        # analytic features are consistent when called directly
        peak, tpk, half_abs, thalf = curve_features(0.0, 100.0, 0.5, 0.2,
                                                    pre_s=5.0, c_start_nM=0.0)
        self.assertGreater(peak, 0.0)
        self.assertGreater(tpk, 0.0)
        self.assertGreater(thalf, 0.0)
        self.assertAlmostEqual(half_abs, tpk + thalf, places=12)

    def test_determinism_and_seeded_replay(self):
        p = ProtocolSpec()
        noisy = DAKinetics(noise_std_nM=2.0)
        a = simulate(p, noisy, sample_dt_s=0.05, seed=7)
        b = simulate(p, noisy, sample_dt_s=0.05, seed=7)
        c = simulate(p, noisy, sample_dt_s=0.05, seed=8)
        np.testing.assert_array_equal(a.record_nM, b.record_nM)
        np.testing.assert_array_equal(a.timeline_nM, b.timeline_nM)
        self.assertEqual(a.seed, 7)
        self.assertFalse(np.array_equal(a.record_nM, c.record_nM))
        self.assertGreater(float(np.max(np.abs(a.record_nM - c.record_nM))), 0.0)
        # noise is observation-only: state, amounts and analytic t1/2 identical
        clean = simulate(p, DAKinetics(), sample_dt_s=0.05)
        np.testing.assert_allclose(a.released_pmol, clean.released_pmol, rtol=0)
        np.testing.assert_allclose(a.pool_before_pmol, clean.pool_before_pmol, rtol=0)
        np.testing.assert_allclose(a.t_half_s, clean.t_half_s, rtol=1e-12)
        np.testing.assert_allclose(a.ledger["released_total_pmol"],
                                   clean.ledger["released_total_pmol"], rtol=1e-14)
        # seed=None uses the documented fixed default seed, so replay is stable
        d = simulate(p, noisy, sample_dt_s=0.05, seed=None)
        e = simulate(p, noisy, sample_dt_s=0.05, seed=None)
        np.testing.assert_array_equal(d.record_nM, e.record_nM)
        self.assertEqual(d.seed, 0)
        # without noise the seed is irrelevant
        f1 = simulate(p, DAKinetics(), sample_dt_s=0.05, seed=1)
        f2 = simulate(p, DAKinetics(), sample_dt_s=0.05, seed=999)
        np.testing.assert_array_equal(f1.record_nM, f2.record_nM)

    def test_invalid_inputs_are_rejected(self):
        for kwargs in (dict(n_stimulations=0), dict(n_stimulations=2.5),
                       dict(ach_dose_pmol=-0.2), dict(ach_dose_pmol=float('nan')),
                       dict(inter_stimulus_interval_s=0.0),
                       dict(recording_duration_s=0.0),
                       dict(representative_trace_s=-1.0),
                       dict(pre_bolus_baseline_s=-1.0),
                       dict(pre_bolus_baseline_s=61.0),
                       dict(representative_trace_s=61.0),
                       dict(post_dissection_rest_s=-5.0)):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                ProtocolSpec(**kwargs)
        for kwargs in (dict(pool_capacity_pmol=0.0),
                       dict(pool_capacity_pmol=-1.0),
                       dict(refill_tau_s=0.0),
                       dict(stimulus_efficacy=-0.1),
                       dict(release_tau_s=-0.5),
                       dict(clearance_tau_s=0.0),
                       dict(clearance_tau_s=float('inf')),
                       dict(voxel_volume_um3=0.0),
                       dict(voxel_volume_um3=-1.0),
                       dict(release_shape_coupling=-1.0),
                       dict(noise_std_nM=-1.0),
                       dict(label='')):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                DAKinetics(**kwargs)
        p = ProtocolSpec()
        with self.assertRaises(ValueError):
            simulate(p, DAKinetics(), sample_dt_s=0.0)
        with self.assertRaises(ValueError):
            simulate(p, DAKinetics(), sample_dt_s=-0.1)
        with self.assertRaises(ValueError):
            simulate(p, DAKinetics(), coarse_dt_s=0.0)
        with self.assertRaises(ValueError):
            simulate(p, DAKinetics(), seed=1.5)
        with self.assertRaises(ValueError):
            amount_pmol_to_concentration_nM(1.0, 0.0)
        with self.assertRaises(ValueError):
            amount_pmol_to_concentration_nM(-1.0, 1.0)
        with self.assertRaises(ValueError):
            concentration_nM_to_amount_pmol(-1.0, 1.0)
        with self.assertRaises(ValueError):
            measure_trace_features([0.0, 1.0], [1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            measure_trace_features([0.0, 1.0, 0.5], [1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            measure_trace_features([0.0, 1.0, 2.0], [1.0, -2.0, 3.0])
        with self.assertRaises(ValueError):
            measure_trace_features([0.0, 1.0, float('nan')], [1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            sweep_pool_refill_efficacy(p, DAKinetics(), pool_sizes_pmol=())
        with self.assertRaises(ValueError):
            sweep_pool_refill_efficacy(p, DAKinetics(), refill_taus_s=())
        with self.assertRaises(ValueError):
            sweep_clearance(p, DAKinetics(), clearance_taus_s=())
        with self.assertRaises(ValueError):
            ProtocolSpec(recording_duration_s=60.0, representative_trace_s=60.0,
                         pre_bolus_baseline_s=5.0).n_record_samples(0.0)

    def test_registry_provenance_is_enforced(self):
        reg = build_registry()
        measured = reg.measured_names()
        self.assertEqual(sorted(measured),
                         ["ach_dose_pmol", "bolus_offset_within_record_s",
                          "inter_stimulus_interval_s", "n_stimulations",
                          "post_dissection_rest_s", "recording_duration_s",
                          "representative_trace_s"])
        for name in measured:
            self.assertIn("PMC9897283", reg.snapshot()[name]["source"])
            self.assertEqual(reg.snapshot()[name]["provenance"], MEASURED)
        self.assertTrue(reg.require_measured(measured))
        with self.assertRaises(ProvenanceError):
            reg.require_measured(["clearance_tau_s"])
        summary = reg.summary()
        self.assertEqual(summary[MEASURED], 7)
        self.assertGreaterEqual(summary[ILLUSTRATIVE], 8)
        self.assertGreater(summary["_fraction_illustrative_or_assumed"], 0.4)
        for name in ("pool_capacity_pmol", "refill_tau_s", "stimulus_efficacy",
                     "release_tau_s", "clearance_tau_s", "voxel_volume_um3",
                     "release_shape_coupling", "noise_std_nM"):
            self.assertEqual(reg.snapshot()[name]["provenance"], ILLUSTRATIVE)
        # a measured-looking parameter with no citation cannot be created
        with self.assertRaises(ProvenanceError):
            reg.define("bogus_measured", 1.0, "s", MEASURED)


if __name__ == '__main__':
    unittest.main(verbosity=2)
