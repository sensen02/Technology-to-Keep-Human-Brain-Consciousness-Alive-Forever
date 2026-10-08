"""Selftests for the Round-5 REAL-DATA comparison (run_da_real_comparison.py).

Run:  /run/media/sensen/Data2/cell_wound_prototype/venv/bin/python run_da_real_comparison_selftest.py

What a pass means: the digitised figure readings load and are validated; the
model reproduces the MEASURED protocol timing through this pipeline; the
residual machinery is correct against hand-checked cases and against a
synthetic fixture where the model's own output is fed back as "data" (so the
residual must be ~0); the bounded grid search is deterministic, reproducible
with a fixed seed and genuinely bounded; the structural claims about the
group-invariant stimulation-1 peak, the pool-capping route and the cross-region
ordering conflict hold on the real readings; and the non-identifiability report
is reproducible.

What a pass does NOT mean: no test here validates the model physiologically.
The measured values are figure readings with ~0.03 uM uncertainty, every model
kinetic constant is illustrative, and a good residual would not identify a
mechanism.

A JSON copy of the exact test counts is written next to the other Round-5
outputs: outputs/brain_isolation/da_real_comparison_selftest.json
"""

import json
import math
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

import run_da_real_comparison as R
from engine.da_protocol import DAKinetics, ProtocolSpec, release_sequence, simulate

OUT_DIR = R.OUT_DIR
SELFTEST_JSON = OUT_DIR / "da_real_comparison_selftest.json"


# ---------------------------------------------------------------------------
# shared fixtures
# ---------------------------------------------------------------------------
def small_grid():
    """A small but full-featured bounded grid for the fast tests."""
    return R.Grid(efficacy=(0.1, 0.2, 0.35, 0.6),
                  release_tau_s=(0.2, 0.5, 1.0),
                  clearance_tau_s=(1.0, 5.0, 12.0),
                  pool_capacity_pmol=(0.02, 0.05, 0.1, 0.3, 1.0),
                  refill_tau_s=(60.0, 300.0, 1200.0, 6000.0),
                  efficacy_multiplier=(0.5, 1.0, 2.0),
                  clearance_multiplier=(0.5, 1.0, 2.0))


def targets_for(data, region):
    return {region: {g: data.regions[region].measured_uM[g] for g in R.GROUPS}}


def synthetic_targets(model_peaks, region="central_complex"):
    return {region: {g: np.asarray(model_peaks[g], dtype=float) for g in R.GROUPS}}


# ---------------------------------------------------------------------------
class DataLoadingTests(unittest.TestCase):
    def test_digitised_json_loads_with_expected_shape_and_keys(self):
        data = R.load_digitised_data()
        self.assertEqual(sorted(data.regions.keys()), sorted(R.REGION_KEYS.keys()))
        self.assertEqual(len(R.GROUPS), 4)
        for name, region in data.regions.items():
            self.assertEqual(sorted(region.measured_uM.keys()), sorted(R.GROUPS))
            for g in R.GROUPS:
                self.assertEqual(region.measured_uM[g].shape, (R.N_STIMULATIONS,))
                self.assertTrue(np.all(np.isfinite(region.measured_uM[g])))
                self.assertTrue(np.all(region.measured_uM[g] >= 0.0))
            self.assertEqual(region.stimulation_number, (1, 2, 3, 4, 5, 6))
            self.assertEqual(region.json_key, R.REGION_KEYS[name])
        self.assertIn("PMC9897283", data.pmcid)
        self.assertIn("Figure 5", data.figure)
        self.assertGreaterEqual(len(data.image_urls), 2)
        for url in data.image_urls:
            self.assertTrue(url.startswith("http"))

    def test_recorded_values_match_the_digitised_file(self):
        data = R.load_digitised_data()
        self.assertEqual(list(data.regions["central_complex"].measured_uM["control_1"]),
                         [0.41, 0.35, 0.33, 0.31, 0.31, 0.30])
        self.assertEqual(list(data.regions["central_complex"].measured_uM["control_45"]),
                         [0.57, 0.44, 0.33, 0.28, 0.25, 0.23])
        self.assertEqual(list(data.regions["central_complex"].measured_uM["parkin_45"]),
                         [0.37, 0.30, 0.25, 0.21, 0.17, 0.155])
        self.assertEqual(list(data.regions["mushroom_body_heel"].measured_uM["parkin_1"]),
                         [0.71, 0.60, 0.56, 0.53, 0.47, 0.46])
        self.assertEqual(list(data.regions["mushroom_body_heel"].measured_uM["control_45"]),
                         [0.43, 0.38, 0.30, 0.26, 0.23, 0.20])

    def test_reading_uncertainty_sem_and_honesty_fields(self):
        data = R.load_digitised_data()
        self.assertEqual(data.reading_uncertainty_uM, R.READING_UNCERTAINTY_UM)
        cc = data.regions["central_complex"]
        for g in R.GROUPS:
            self.assertIsNotNone(cc.sem_uM[g], f"{g}: central complex SEM missing")
            self.assertEqual(cc.sem_uM[g].shape, (R.N_STIMULATIONS,))
            self.assertTrue(np.all(cc.sem_uM[g] > 0.0))
            # the published SEM is wider than the reading uncertainty somewhere
        self.assertTrue(any(float(np.max(cc.sem_uM[g])) > R.READING_UNCERTAINTY_UM
                            for g in R.GROUPS))
        # the second region has no SEM block in the digitised file: reported, not invented
        self.assertTrue(all(data.regions["mushroom_body_heel"].sem_uM[g] is None
                            for g in R.GROUPS))
        honesty = " | ".join(data.honesty)
        self.assertIn("VISUAL READINGS", honesty)
        self.assertRegex(honesty, "NOT raw experimental data|not raw")
        self.assertIn("PMC9897283", data.pmcid)
        self.assertIn("Dumitrescu", data.citation)

    def test_malformed_data_is_rejected(self):
        raw = json.loads(Path(R.DIGITISED_JSON).read_text())
        cases = 0

        def reject(mutate, label):
            nonlocal cases
            bad = json.loads(json.dumps(raw))
            mutate(bad)
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
                json.dump(bad, fh)
                path = fh.name
            try:
                with self.assertRaises(ValueError, msg=label):
                    R.load_digitised_data(path)
            finally:
                Path(path).unlink()
            cases += 1

        def drop_region(b):
            del b["figure_6G_absolute_uM_mushroom_body_heel"]

        def drop_group(b):
            del b["figure_5G_absolute_uM"]["groups"]["parkin_45"]

        def short_curve(b):
            b["figure_5G_absolute_uM"]["groups"]["control_1"] = [0.41, 0.35]

        def nan_value(b):
            b["figure_5G_absolute_uM"]["groups"]["control_1"][2] = float("nan")

        def negative(b):
            b["figure_5G_absolute_uM"]["groups"]["control_1"][2] = -0.2

        def bad_uncertainty(b):
            b["reading_method"]["uncertainty_uM"] = -0.03

        def changed_uncertainty(b):
            b["reading_method"]["uncertainty_uM"] = 0.5

        def short_sem(b):
            b["figure_5G_absolute_uM"]["sem_group_readings_approx"]["control_1"] = [0.04]

        def extra_group(b):
            b["figure_5G_absolute_uM"]["groups"]["control_99"] = [0.1] * 6

        def no_honesty(b):
            b["what_this_is"] = "an accurate table of measured values"

        for mutate, label in ((drop_region, "missing region"),
                              (drop_group, "missing group"),
                              (short_curve, "wrong number of stimulations"),
                              (nan_value, "non-finite value"),
                              (negative, "negative concentration"),
                              (bad_uncertainty, "negative uncertainty"),
                              (changed_uncertainty, "uncertainty no longer 0.03"),
                              (short_sem, "SEM of the wrong length"),
                              (extra_group, "unexpected group"),
                              (no_honesty, "file no longer says these are readings")):
            reject(mutate, label)
        # a JSON file whose top level is not an object
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump([1, 2, 3], fh)
            path = fh.name
        try:
            with self.assertRaises(ValueError):
                R.load_digitised_data(path)
        finally:
            Path(path).unlink()
        cases += 1
        with self.assertRaises(ValueError):
            R.load_digitised_data(Path("/nonexistent/does_not_exist.json"))
        cases += 1
        self.assertEqual(cases, 12)

    def test_protocol_parsed_from_the_data_matches_the_engine(self):
        data = R.load_digitised_data()
        p = R.protocol_from_digitised(data)
        self.assertIsInstance(p, ProtocolSpec)
        self.assertEqual(p.n_stimulations, 6)
        self.assertEqual(p.ach_dose_pmol, 0.2)
        self.assertEqual(p.inter_stimulus_interval_s, 600.0)
        self.assertEqual(p.recording_duration_s, 60.0)
        self.assertEqual(p.representative_trace_s, 15.0)
        self.assertEqual(p.pre_bolus_baseline_s, 5.0)
        self.assertEqual(p.post_dissection_rest_s, 1200.0)
        rep = R.protocol_report(p)
        self.assertTrue(rep["bolus_spacing_all_exactly_600_s"])
        self.assertTrue(rep["ach_is_the_stimulus_da_is_the_modelled_transmitter"])
        self.assertIn("measured", rep["source_of_timing"])

    def test_protocol_mismatch_between_file_and_engine_is_rejected(self):
        data = R.load_digitised_data()
        data.measured_protocol_json = dict(data.measured_protocol_json)
        data.measured_protocol_json["interval_s"] = 300.0
        with self.assertRaises(ValueError):
            R.protocol_from_digitised(data)
        data2 = R.load_digitised_data()
        data2.measured_protocol_json = {}
        with self.assertRaises(ValueError):
            R.protocol_from_digitised(data2)


# ---------------------------------------------------------------------------
class ProtocolThroughPipelineTests(unittest.TestCase):
    def test_pipeline_reproduces_the_measured_timing(self):
        data = R.load_digitised_data()
        p = R.protocol_from_digitised(data)
        self.assertEqual(p.bolus_times_s, (5.0, 605.0, 1205.0, 1805.0, 2405.0, 3005.0))
        self.assertEqual(p.record_starts_s, (0.0, 600.0, 1200.0, 1800.0, 2400.0, 3000.0))
        self.assertEqual(p.record_ends_s, (60.0, 660.0, 1260.0, 1860.0, 2460.0, 3060.0))
        self.assertEqual(p.session_end_s, 3060.0)
        for b, s in zip(p.bolus_times_s, p.record_starts_s):
            self.assertAlmostEqual(b - s, 5.0, places=12)
        # the pipeline's own model prediction has exactly 6 stimulations, in uM
        params = R.ModelParams(efficacy=0.25, release_tau_s=0.5, clearance_tau_s=5.0,
                               pool_capacity_pmol={g: 0.1 for g in R.GROUPS},
                               refill_tau_s={g: 300.0 for g in R.GROUPS})
        peaks = params.predicted_uM(p)
        for g in R.GROUPS:
            self.assertEqual(np.asarray(peaks[g]).shape, (6,))
            self.assertTrue(np.all(np.asarray(peaks[g]) > 0.0))
            self.assertLess(float(np.asarray(peaks[g])[0]),
                            float(np.asarray(peaks[g])[0]) + 1e-12)
        sim = simulate(p, params.group_kinetics("control_1"), sample_dt_s=0.05)
        self.assertEqual(sim.record_nM.shape[0], 6)
        for i in range(6):
            self.assertAlmostEqual(sim.record_time_s[i, 0],
                                   p.bolus_times_s[i] - 5.0, places=12)
        self.assertAlmostEqual(sim.timeline_time_s[0], -1200.0, places=12)

    def test_fast_evaluator_matches_the_engine_sampled_trace(self):
        data = R.load_digitised_data()
        p = R.protocol_from_digitised(data)
        params = R.ModelParams(efficacy=0.3, release_tau_s=0.5, clearance_tau_s=4.0,
                               pool_capacity_pmol={g: 0.08 for g in R.GROUPS},
                               refill_tau_s={g: 400.0 for g in R.GROUPS})
        fast = params.predicted_uM(p)["control_1"]
        sim = simulate(p, params.group_kinetics("control_1"), sample_dt_s=0.05)
        sampled_uM = np.max(sim.record_nM, axis=1) * R.NM_TO_UM
        # the sampled maximum can only be below the analytic peak, by at most one sample
        self.assertTrue(np.all(sampled_uM <= fast + 1e-12))
        self.assertLess(float(np.max(np.abs(sampled_uM - fast) / fast)), 0.02)

    def test_model_is_noise_free_and_seed_independent(self):
        p = ProtocolSpec()
        kin = DAKinetics(noise_std_nM=0.0)
        a = simulate(p, kin, sample_dt_s=0.2, seed=1)
        b = simulate(p, kin, sample_dt_s=0.2, seed=99999)
        np.testing.assert_array_equal(a.record_nM, b.record_nM)


# ---------------------------------------------------------------------------
class ResidualTests(unittest.TestCase):
    def test_residual_hand_checked_case(self):
        model = np.array([0.40, 0.30])
        measured = np.array([0.41, 0.36])
        res = model - measured
        np.testing.assert_allclose(res, [-0.01, -0.06], rtol=0, atol=1e-15)
        rms = float(np.sqrt(np.mean(res ** 2)))
        self.assertAlmostEqual(rms, math.sqrt((0.0001 + 0.0036) / 2.0), places=15)
        self.assertAlmostEqual(rms, 0.04301162633, places=8)
        n_over = int(np.sum(np.abs(res) > R.READING_UNCERTAINTY_UM))
        self.assertEqual(n_over, 1)
        sem = np.array([0.02, 0.05])
        n_over_sem = int(np.sum(np.abs(res) > sem))
        self.assertEqual(n_over_sem, 1)

    def test_residuals_of_a_real_fit_match_a_hand_computation(self):
        data = R.load_digitised_data()
        p = R.protocol_from_digitised(data)
        g = small_grid()
        f = R.fit(g, "A", targets_for(data, "central_complex"), protocol=p, seed=3)
        for region in f.regions:
            for grp in R.GROUPS:
                hand = f.model_uM[region][grp] - f.measured_uM[region][grp]
                np.testing.assert_allclose(f.residuals_uM[region][grp], hand,
                                           rtol=0, atol=0)
                hand_rms = float(np.sqrt(np.mean(hand ** 2)))
                self.assertAlmostEqual(f.rms_per_group_uM[f"{region}:{grp}"],
                                       hand_rms, places=15)
        all_res = np.concatenate([f.residuals_uM[r][grp] for r in f.regions
                                  for grp in R.GROUPS])
        self.assertAlmostEqual(f.rms_overall_uM,
                               float(np.sqrt(np.mean(all_res ** 2))), places=15)
        self.assertAlmostEqual(f.max_abs_residual_uM,
                               float(np.max(np.abs(all_res))), places=15)

    def test_units_and_table_consistency(self):
        p = ProtocolSpec()
        params = R.ModelParams(efficacy=0.25, release_tau_s=0.5, clearance_tau_s=5.0,
                               pool_capacity_pmol={g: 0.1 for g in R.GROUPS},
                               refill_tau_s={g: 300.0 for g in R.GROUPS})
        engine_nM = params.engine_predicted_uM(p, "control_1") / R.NM_TO_UM
        self.assertAlmostEqual(float(engine_nM[0]) / 1000.0,
                               float(params.predicted_uM(p)["control_1"][0]), places=9)
        data = R.load_digitised_data()
        pj = R.protocol_from_digitised(data)
        f = R.fit(small_grid(), "A", targets_for(data, "central_complex"),
                  protocol=pj, seed=0)
        table = f.per_stimulus_table()["central_complex"]
        for grp in R.GROUPS:
            for i, row in enumerate(table[grp]):
                self.assertAlmostEqual(row["residual_uM"],
                                       float(f.residuals_uM["central_complex"][grp][i]),
                                       places=15)
                self.assertEqual(row["stimulation"], i + 1)
                self.assertEqual(row["exceeds_reading_uncertainty"],
                                 bool(abs(row["residual_uM"]) > R.READING_UNCERTAINTY_UM))

    def test_exceedance_counts_agree_with_the_table(self):
        data = R.load_digitised_data()
        p = R.protocol_from_digitised(data)
        f = R.attach_sem(R.fit(small_grid(), "A", targets_for(data, "central_complex"),
                               protocol=p, seed=0), data)
        counts = f.exceedance_counts()
        manual_over = sum(
            1 for r in f.regions for grp in R.GROUPS for i in range(R.N_STIMULATIONS)
            if abs(float(f.residuals_uM[r][grp][i])) > R.READING_UNCERTAINTY_UM)
        self.assertEqual(counts["exceed_reading_uncertainty"], manual_over)
        self.assertEqual(counts["total_points"],
                         len(f.regions) * len(R.GROUPS) * R.N_STIMULATIONS)
        manual_sem = 0
        for r in f.regions:
            for grp in R.GROUPS:
                sem = f.sem_uM[r][grp]
                if sem is None:
                    continue
                manual_sem += sum(1 for i in range(R.N_STIMULATIONS)
                                  if abs(float(f.residuals_uM[r][grp][i])) > float(sem[i]))
        self.assertEqual(counts["exceed_published_sem"], manual_sem)


# ---------------------------------------------------------------------------
class SearchBoundsAndDeterminismTests(unittest.TestCase):
    def test_search_is_deterministic_with_a_fixed_seed(self):
        data = R.load_digitised_data()
        p = R.protocol_from_digitised(data)
        g = small_grid()
        t = targets_for(data, "central_complex")
        f1 = R.fit(g, "A", t, protocol=p, seed=7)
        f2 = R.fit(g, "A", t, protocol=p, seed=7)
        self.assertEqual(f1.rms_overall_uM, f2.rms_overall_uM)
        self.assertEqual(f1.n_model_curve_evaluations, f2.n_model_curve_evaluations)
        self.assertEqual(f1.n_sse_combinations, f2.n_sse_combinations)
        self.assertEqual(f1.params.to_dict(p), f2.params.to_dict(p))
        for grp in R.GROUPS:
            np.testing.assert_array_equal(f1.model_uM["central_complex"][grp],
                                          f2.model_uM["central_complex"][grp])

    def test_search_result_is_seed_invariant(self):
        data = R.load_digitised_data()
        p = R.protocol_from_digitised(data)
        g = small_grid()
        t = targets_for(data, "central_complex")
        base = R.fit(g, "A", t, protocol=p, seed=0)
        for seed in (1, 12345):
            other = R.fit(g, "A", t, protocol=p, seed=seed)
            self.assertAlmostEqual(base.rms_overall_uM, other.rms_overall_uM, places=12)
            self.assertAlmostEqual(base.params.efficacy, other.params.efficacy, places=9)
            self.assertEqual(base.seed, 0)
            self.assertEqual(other.seed, seed)

    def test_search_is_bounded_and_counts_are_exact(self):
        g = small_grid()
        p = ProtocolSpec()
        data = R.load_digitised_data()
        f = R.fit(g, "A", targets_for(data, "central_complex"), protocol=p, seed=0)
        expected = (len(g.efficacy) * len(g.release_tau_s) * len(g.clearance_tau_s)
                    * len(g.pool_capacity_pmol) * len(g.refill_tau_s) * len(R.GROUPS))
        self.assertEqual(f.n_model_curve_evaluations, expected)
        self.assertEqual(f.n_model_curve_evaluations, 720 * 4)
        for grp in R.GROUPS:
            self.assertGreaterEqual(f.params.pool_capacity_pmol[grp],
                                    g.bounds["pool_capacity_pmol"][0] * 0.99)
            self.assertLessEqual(f.params.pool_capacity_pmol[grp],
                                 g.bounds["pool_capacity_pmol"][1] * 1.01)
            self.assertGreaterEqual(f.params.refill_tau_s[grp],
                                    g.bounds["refill_tau_s"][0] * 0.99)
            self.assertLessEqual(f.params.refill_tau_s[grp],
                                 g.bounds["refill_tau_s"][1] * 1.01)
        self.assertGreaterEqual(f.params.efficacy, g.bounds["efficacy"][0] * 0.99)
        self.assertLessEqual(f.params.release_tau_s, g.bounds["release_tau_s"][1] * 1.01)
        # arms B and C add exactly one per-group axis each
        self.assertEqual(g.n_subgrid_points("A"), 20)
        self.assertEqual(g.n_subgrid_points("B"), 60)
        self.assertEqual(g.n_subgrid_points("C"), 60)

    def test_oversized_search_is_rejected_by_the_cap(self):
        data = R.load_digitised_data()
        p = R.protocol_from_digitised(data)
        with self.assertRaises(ValueError) as ctx:
            R.fit(small_grid(), "A", targets_for(data, "central_complex"), protocol=p,
                  seed=0, max_evaluations=100)
        self.assertIn("above the cap", str(ctx.exception))

    def test_invalid_grids_are_rejected(self):
        bad = [dict(efficacy=(), release_tau_s=(0.5,), clearance_tau_s=(5.0,),
                    pool_capacity_pmol=(0.1,), refill_tau_s=(300.0,)),
               dict(efficacy=(0.2, 0.2), release_tau_s=(0.5,), clearance_tau_s=(5.0,),
                    pool_capacity_pmol=(0.1,), refill_tau_s=(300.0,)),
               dict(efficacy=(0.2,), release_tau_s=(0.0,), clearance_tau_s=(5.0,),
                    pool_capacity_pmol=(0.1,), refill_tau_s=(300.0,)),
               dict(efficacy=(-0.2,), release_tau_s=(0.5,), clearance_tau_s=(5.0,),
                    pool_capacity_pmol=(0.1,), refill_tau_s=(300.0,)),
               dict(efficacy=(float("nan"),), release_tau_s=(0.5,),
                    clearance_tau_s=(5.0,), pool_capacity_pmol=(0.1,),
                    refill_tau_s=(300.0,)),
               dict(efficacy=(0.2,), release_tau_s=(0.5,), clearance_tau_s=(5.0,),
                    pool_capacity_pmol=(0.0,), refill_tau_s=(300.0,)),
               dict(efficacy=(0.2,), release_tau_s=(0.5,), clearance_tau_s=(5.0,),
                    pool_capacity_pmol=(0.1,), refill_tau_s=(300.0,),
                    voxel_volume_um3=0.0)]
        for kwargs in bad:
            with self.assertRaises(ValueError, msg=str(kwargs)):
                R.Grid(**kwargs)
        self.assertEqual(len(bad), 7)

    def test_invalid_fit_inputs_are_rejected(self):
        p = ProtocolSpec()
        g = small_grid()
        good = {g_: np.array([0.4, 0.35, 0.33, 0.31, 0.31, 0.30]) for g_ in R.GROUPS}
        with self.assertRaises(ValueError):
            R.fit(g, "Z", {"central_complex": good}, protocol=p)
        with self.assertRaises(ValueError):
            R.fit(g, "A", None, protocol=p)
        short = dict(good)
        short["control_1"] = np.array([0.4, 0.35])
        with self.assertRaises(ValueError):
            R.fit(g, "A", {"central_complex": short}, protocol=p)
        nan = dict(good)
        nan["parkin_1"] = np.array([0.4, 0.35, float("nan"), 0.31, 0.31, 0.30])
        with self.assertRaises(ValueError):
            R.fit(g, "A", {"central_complex": nan}, protocol=p)
        missing = {k: v for k, v in good.items() if k != "parkin_45"}
        with self.assertRaises(ValueError):
            R.fit(g, "A", {"central_complex": missing}, protocol=p)
        with self.assertRaises(ValueError):
            R.fit(g, "A", {"central_complex": good},
                  region_scales={"central_complex": ()}, protocol=p)
        with self.assertRaises(ValueError):
            R.subgrid_peaks_uM((0.0,), (300.0,), 0.2, 0.8, p)
        with self.assertRaises(ValueError):
            R.subgrid_peaks_uM((-1.0,), (300.0,), 0.2, 0.8, p)
        with self.assertRaises(ValueError):
            R.pool_release_sequence(0.0, 300.0, 0.2, p)
        with self.assertRaises(ValueError):
            R.pool_release_sequence(0.1, 0.0, 0.2, p)
        with self.assertRaises(ValueError):
            R.peak_shape_factor(0.5, 0.0)


# ---------------------------------------------------------------------------
class SyntheticFixtureTests(unittest.TestCase):
    def test_model_output_fed_back_as_data_gives_zero_residual(self):
        p = ProtocolSpec()
        g = small_grid()
        truth = R.ModelParams(efficacy=0.2, release_tau_s=0.5, clearance_tau_s=5.0,
                              pool_capacity_pmol={gg: 0.05 for gg in R.GROUPS},
                              refill_tau_s={gg: 1200.0 for gg in R.GROUPS})
        # every group shares the same parameters, so the synthetic data are exactly
        # reproducible by an arm-A configuration that IS inside the grid
        data_like = truth.predicted_uM(p)
        f = R.fit(g, "A", synthetic_targets(data_like), protocol=p, seed=0)
        self.assertLess(f.rms_overall_uM, 1e-12, "residual path is broken")
        self.assertLess(f.max_abs_residual_uM, 1e-12)
        self.assertAlmostEqual(f.params.efficacy, truth.efficacy, places=9)
        self.assertAlmostEqual(f.params.release_tau_s, truth.release_tau_s, places=9)
        self.assertAlmostEqual(f.params.clearance_tau_s, truth.clearance_tau_s, places=9)
        for gg in R.GROUPS:
            self.assertAlmostEqual(f.params.pool_capacity_pmol[gg],
                                   truth.pool_capacity_pmol[gg], places=12)
            self.assertAlmostEqual(f.params.refill_tau_s[gg], truth.refill_tau_s[gg],
                                   places=9)

    def test_synthetic_fixture_with_different_groups_recovers_the_schedule(self):
        p = ProtocolSpec()
        g = small_grid()
        truth = R.ModelParams(
            efficacy=0.35, release_tau_s=1.0, clearance_tau_s=5.0,
            pool_capacity_pmol={"control_1": 1.0, "parkin_1": 1.0,
                                "control_45": 0.1, "parkin_45": 0.05},
            refill_tau_s={"control_1": 300.0, "parkin_1": 300.0,
                          "control_45": 6000.0, "parkin_45": 1200.0})
        f = R.fit(g, "A", synthetic_targets(truth.predicted_uM(p)), protocol=p, seed=0)
        self.assertLess(f.rms_overall_uM, 1e-12)
        for gg in R.GROUPS:
            self.assertAlmostEqual(f.params.pool_capacity_pmol[gg],
                                   truth.pool_capacity_pmol[gg], places=9)
            self.assertAlmostEqual(f.params.refill_tau_s[gg],
                                   truth.refill_tau_s[gg], places=6)

    def test_synthetic_elevated_curve_needs_more_than_pool_or_refill(self):
        """Mimics the real situation: one group's whole curve is scaled up.

        The synthetic "measured" data are the model's own output with the
        control_45 curve multiplied by 1.5 (an increase in the stimulus->release
        efficacy).  Arm A (shared kinetics, only pool/refill per group) cannot
        reproduce it, with or without pool capping; arm B, which carries one extra
        labelled per-group degree of freedom, recovers it exactly.  This pins down
        that the residual path really detects the pattern seen in the real data.
        """
        p = ProtocolSpec()
        truth = R.ModelParams(efficacy=0.2, release_tau_s=0.5, clearance_tau_s=5.0,
                              pool_capacity_pmol={gg: 0.5 for gg in R.GROUPS},
                              refill_tau_s={gg: 300.0 for gg in R.GROUPS})
        base = truth.predicted_uM(p)
        data_like = {gg: np.array(base[gg]) for gg in R.GROUPS}
        data_like["control_45"] = data_like["control_45"] * 1.5
        self.assertGreater(float(data_like["control_45"][0]),
                           float(data_like["control_1"][0]))

        g_a = R.Grid(efficacy=(0.2, 0.3, 0.5), release_tau_s=(0.5,),
                     clearance_tau_s=(5.0,),
                     pool_capacity_pmol=(0.05, 0.1, 0.3, 0.5, 0.75, 1.0),
                     refill_tau_s=(300.0,), efficacy_multiplier=(1.0, 1.5),
                     clearance_multiplier=(1.0,))
        tgt = synthetic_targets(data_like)

        # arm A with capping disallowed: the four stimulation-1 values are forced equal
        f_nocap = R.fit(g_a, "A", tgt, protocol=p, seed=0, exclude_capping=True)
        pred1 = {gg: round(float(f_nocap.model_uM["central_complex"][gg][0]), 12)
                 for gg in R.GROUPS}
        self.assertEqual(len(set(pred1.values())), 1)
        self.assertGreater(f_nocap.max_abs_residual_uM, 0.05)

        # arm A with capping allowed: capping cannot fake a proportionally scaled curve
        f_cap = R.fit(g_a, "A", tgt, protocol=p, seed=0)
        self.assertGreater(f_cap.rms_overall_uM, R.READING_UNCERTAINTY_UM)

        # arm B (one labelled per-group efficacy multiplier) recovers it exactly
        f_armb = R.fit(g_a, "B", tgt, protocol=p, seed=0)
        self.assertLess(f_armb.rms_overall_uM, 1e-12)
        self.assertAlmostEqual(f_armb.params.efficacy_multiplier["control_45"], 1.5,
                               places=12)
        # the recovering configuration must ALSO scale that group's pool: in this
        # model, scaling the efficacy alone deepens the fractional depletion
        # (phi = efficacy*dose/R0), so the two effects are not equivalent
        self.assertAlmostEqual(f_armb.params.pool_capacity_pmol["control_45"], 0.75,
                               places=12)
        self.assertAlmostEqual(f_armb.params.pool_capacity_pmol["control_1"], 0.5,
                               places=12)
        for gg in ("control_1", "parkin_1", "parkin_45"):
            self.assertAlmostEqual(f_armb.params.efficacy_multiplier[gg], 1.0,
                                   places=12)


# ---------------------------------------------------------------------------
class CriticalTestTests(unittest.TestCase):
    def test_stim1_peak_is_group_invariant_when_uncapped(self):
        p = ProtocolSpec()
        params = R.ModelParams(efficacy=0.3, release_tau_s=0.5, clearance_tau_s=5.0,
                               pool_capacity_pmol={gg: 0.2 for gg in R.GROUPS},
                               refill_tau_s={gg: 300.0 for gg in R.GROUPS})
        for R0 in (0.06, 0.1, 0.2, 1.0, 5.0):        # all above efficacy*dose = 0.06
            for refill in (20.0, 600.0, 20000.0):
                params.pool_capacity_pmol = {gg: R0 for gg in R.GROUPS}
                params.refill_tau_s = {gg: refill for gg in R.GROUPS}
                peaks = params.predicted_uM(p)
                firsts = [float(peaks[gg][0]) for gg in R.GROUPS]
                for v in firsts[1:]:
                    self.assertAlmostEqual(v, firsts[0], places=12)
                self.assertAlmostEqual(firsts[0],
                                       R.stim1_peak_uM(0.3, R0, 0.5, 5.0, p),
                                       places=12)

    def test_capping_only_lowers_stim1_and_forces_step_then_flat(self):
        p = ProtocolSpec()
        params = R.ModelParams(efficacy=0.3, release_tau_s=0.5, clearance_tau_s=5.0,
                               pool_capacity_pmol={gg: 0.02 for gg in R.GROUPS},
                               refill_tau_s={gg: 600.0 for gg in R.GROUPS})
        peaks = params.predicted_uM(p)["control_1"]
        ceil = R.stim1_peak_uM(0.3, 0.2, 0.5, 5.0, p)     # uncapped ceiling
        self.assertLess(float(peaks[0]), ceil)
        g = math.exp(-p.inter_stimulus_interval_s / 600.0)
        ratio = 1.0 - g
        for i in range(1, R.N_STIMULATIONS):
            self.assertAlmostEqual(float(peaks[i] / peaks[0]), ratio, places=12)

    def test_critical_test_counts_on_the_real_readings(self):
        data = R.load_digitised_data()
        p = R.protocol_from_digitised(data)
        g = small_grid()
        cc = R.critical_test(g, p, data, "central_complex", "control_45", "control_1")
        self.assertEqual(cc["measured_stim1_uM"]["control_45"], 0.57)
        self.assertEqual(cc["measured_stim1_uM"]["control_1"], 0.41)
        self.assertAlmostEqual(cc["elevation_in_reading_uncertainty_units"],
                               0.16 / R.READING_UNCERTAINTY_UM, places=12)
        floor = cc["analytic_floor_uM"]["FOR_THE_ELEVATED_PAIR_if_both_groups_uncapped"]
        self.assertAlmostEqual(floor["rms_over_the_two_groups_uM"], 0.08, places=12)
        forced = cc["analytic_floor_uM"][
            "IF_NO_GROUP_IS_POOL_CAPPED_all_four_stim1_values_are_forced_equal"]
        hand = math.sqrt(np.mean([(float(np.mean([data.regions["central_complex"]
                                                   .measured_uM[gg][0]
                                                   for gg in R.GROUPS]))
                                   - float(data.regions["central_complex"]
                                           .measured_uM[gg][0])) ** 2
                                  for gg in R.GROUPS]))
        self.assertAlmostEqual(forced["rms_over_the_four_stim1_points_uM"], hand,
                               places=12)
        four = [float(data.regions["central_complex"].measured_uM[gg][0])
                for gg in R.GROUPS]
        mu = sum(four) / 4.0
        self.assertAlmostEqual(forced["rms_over_the_four_stim1_points_uM"],
                               math.sqrt(sum((mu - v) ** 2 for v in four) / 4.0),
                               places=12)
        self.assertAlmostEqual(forced["rms_over_the_four_stim1_points_uM"], 0.07595,
                               places=5)
        self.assertGreater(cc["grid_stim1_pairs_with_elevation"], 0)
        self.assertTrue(cc["every_elevating_pair_requires_pool_capping_of_the_reference_group"])
        self.assertEqual(cc["grid_stim1_pairs_enumerated"],
                         g.n_shared_points("A") * len(g.pool_capacity_pmol) ** 2)
        heel = R.critical_test(g, p, data, "mushroom_body_heel", "parkin_1", "control_1")
        self.assertEqual(heel["measured_stim1_uM"]["parkin_1"], 0.71)
        self.assertAlmostEqual(
            heel["analytic_floor_uM"]["FOR_THE_ELEVATED_PAIR_if_both_groups_uncapped"]
            ["rms_over_the_two_groups_uM"], 0.095, places=12)

    def test_capped_group_cost_is_quantified(self):
        data = R.load_digitised_data()
        p = R.protocol_from_digitised(data)
        rep = R.capped_cost_report(small_grid(), p, data, "central_complex", "control_1")
        self.assertGreater(rep["rms_floor_for_a_capped_group_uM"], 0.0)
        best = rep["best_capped_fit"]
        # a capped group's stim-2 / stim-1 ratio is exactly 1-exp(-600/tau)
        self.assertAlmostEqual(best["stim2_over_stim1"],
                               1.0 - math.exp(-600.0 / best["refill_tau_s"]), places=12)
        # and its plateau is inconsistent with the measured late stimulations
        self.assertGreater(best["best_possible_max_abs_residual_uM"],
                           R.READING_UNCERTAINTY_UM)

    def test_panel_H_normalisation_removes_the_stim1_information(self):
        data = R.load_digitised_data()
        p = R.protocol_from_digitised(data)
        g = small_grid()
        f = R.fit(g, "A", targets_for(data, "central_complex"), protocol=p, seed=0)
        rep = R.normalised_report(f, data, "central_complex")
        for grp in R.GROUPS:
            self.assertAlmostEqual(rep["per_group"][grp]["model_normalised"][0], 1.0,
                                   places=12)
            self.assertAlmostEqual(rep["per_group"][grp]["measured_normalised"][0], 1.0,
                                   places=12)
        self.assertIn("CANNOT test", rep["note_first_point"])
        self.assertIn("mean of ratios", rep["normalisation"])


# ---------------------------------------------------------------------------
class NonIdentifiabilityTests(unittest.TestCase):
    def test_efficacy_and_volume_are_exactly_degenerate(self):
        p = ProtocolSpec()
        g = small_grid()
        out = R.degeneracy_scan(g, p)
        blk = out["efficacy_volume_pool_scaling_degeneracy"]
        self.assertLess(blk["max_deviation_uM"], 1e-15)
        ratios = {round(pp["efficacy_over_volume"], 20) for pp in blk["exact_family"]}
        self.assertEqual(len(ratios), 1)
        phis = {round(pp["release_fraction_phi"], 12) for pp in blk["exact_family"]}
        self.assertEqual(len(phis), 1)
        # scaling ONLY efficacy and volume (leaving the pools) does change the curve,
        # which is why the volume is fixed rather than fitted
        self.assertGreater(blk["if_only_efficacy_and_volume_are_scaled"]
                           ["max_deviation_uM"], 1e-6)

    def test_release_and_clearance_enter_only_through_one_scalar(self):
        """From peak amplitudes alone only peak_shape(tau_rel, tau_clr) is seen.

        The mapping (release_tau_s, clearance_tau_s) -> peak-shape factor is
        many-to-one, and two configurations with the same factor predict identical
        peaks while having clearly different trace half-decay times -- which is why
        the digitised peak amplitudes cannot pin either time constant.
        """
        p = ProtocolSpec()
        g = small_grid()
        out = R.degeneracy_scan(g, p)
        table = out["release_and_clearance_degeneracy"]["peak_shape_table"]
        self.assertEqual(len(table), len(g.release_tau_s) * len(g.clearance_tau_s))
        # search a fine tau_clr grid for a pair matching a reference shape factor
        base_rel, base_clr = 0.5, 5.0
        ps_ref = R.peak_shape_factor(base_rel, base_clr)
        found = None
        for other_rel in (0.2, 1.0, 2.0):
            for tau_clr in np.geomspace(0.2, 200.0, 4000):
                if abs(R.peak_shape_factor(other_rel, float(tau_clr)) - ps_ref) <= 1e-4 * ps_ref:
                    found = (other_rel, float(tau_clr))
                    break
            if found:
                break
        self.assertIsNotNone(found, "no colliding peak-shape pair found")
        other_rel, other_clr = found
        a = R.ModelParams(efficacy=0.2, release_tau_s=base_rel, clearance_tau_s=base_clr,
                          pool_capacity_pmol={gg: 0.1 for gg in R.GROUPS},
                          refill_tau_s={gg: 300.0 for gg in R.GROUPS})
        b = a.copy()
        b.release_tau_s, b.clearance_tau_s = other_rel, other_clr
        ya = a.predicted_uM(p)["control_1"]
        yb = b.predicted_uM(p)["control_1"]
        self.assertLess(float(np.max(np.abs(ya - yb) / ya)), 1e-3,
                        "colliding shape factors must give identical peak curves")
        # yet the recorded trace half-decay differs a lot: amplitudes cannot see it
        eng_a = release_sequence(p, a.group_kinetics("control_1"))["ephemeral"]
        eng_b = release_sequence(p, b.group_kinetics("control_1"))["ephemeral"]
        th_a, th_b = float(eng_a["t_half_s"][0]), float(eng_b["t_half_s"][0])
        self.assertGreater(abs(th_a - th_b) / th_a, 0.10)
        self.assertGreaterEqual(
            out["release_and_clearance_degeneracy"]
            ["n_pairs_colliding_within_0.1_percent"], 0)

    def test_exact_time_scale_invariance_of_the_peak_amplitudes(self):
        """Scaling both time constants together leaves every peak identical."""
        p = ProtocolSpec()
        g = small_grid()
        out = R.degeneracy_scan(g, p)
        rows = out["release_and_clearance_degeneracy"]["exact_time_scale_invariance"]
        self.assertEqual(len(rows), 5)
        for row in rows:
            self.assertLess(row["relative_peak_shape_difference"], 1e-15)
            self.assertLess(row["max_abs_curve_difference_uM"], 1e-15)
        halves = [row["t_half_s_first_stimulus"] for row in rows]
        # k = 0.25 .. 4.0 -> the half-decay spans a factor of 16 while the peak
        # amplitude is identical, i.e. peaks cannot see the absolute time scale
        self.assertAlmostEqual(max(halves) / min(halves), 16.0, places=6)
        for row in rows:
            self.assertAlmostEqual(row["peak_nM_first_stimulus"],
                                   rows[2]["peak_nM_first_stimulus"], places=9)
        self.assertGreaterEqual(
            out["release_and_clearance_degeneracy"]
            ["n_pairs_with_identical_peak_shape"], 1)

    def test_near_optimal_set_on_the_real_readings(self):
        data = R.load_digitised_data()
        p = R.protocol_from_digitised(data)
        g = small_grid()
        out = R.near_optimal_set(g, "A", p, targets_for(data, "central_complex"),
                                 {"central_complex": (1.0,)}, band_uM=0.01, seed=0)
        self.assertEqual(out["n_configurations_scanned"], g.n_shared_points("A"))
        self.assertEqual(out["n_within_reading_uncertainty"], 0)
        self.assertGreater(out["best_max_abs_residual_uM"], R.READING_UNCERTAINTY_UM)
        self.assertGreater(out["n_within_band"], 1)
        self.assertIn("indistinguishable", " ".join(out["summary"]))

    def test_band_members_span_a_wide_parameter_range(self):
        data = R.load_digitised_data()
        p = R.protocol_from_digitised(data)
        g = R.Grid(efficacy=(0.05, 0.1, 0.2, 0.4, 0.8),
                   release_tau_s=(0.2, 0.5, 1.0, 2.0),
                   clearance_tau_s=(1.0, 5.0, 20.0),
                   pool_capacity_pmol=(0.03, 0.05, 0.1, 0.3, 1.0),
                   refill_tau_s=(100.0, 300.0, 1200.0, 6000.0))
        out = R.near_optimal_set(g, "A", p, targets_for(data, "central_complex"),
                                 {"central_complex": (1.0,)}, band_uM=0.02, seed=0)
        self.assertGreater(out["n_within_band"], 1)
        spans = out["ranges_within_band"]
        self.assertGreater(spans["clearance_tau_s"]["ratio_max_over_min"], 1.5)
        self.assertGreater(spans["refill_tau_s"]["ratio_max_over_min"], 1.5)


# ---------------------------------------------------------------------------
class CrossRegionTests(unittest.TestCase):
    def test_measured_ordering_flip_is_real(self):
        data = R.load_digitised_data()
        p = R.protocol_from_digitised(data)
        g = small_grid()
        out = R.cross_region_test({}, {}, {}, g, p, data)
        self.assertTrue(out["ordering_conflict_confirmed_in_the_measured_data"])
        self.assertTrue(out["central_complex_parkin45_below_control45_all_6"])
        self.assertTrue(out["mushroom_body_heel_parkin45_not_below_control45_all_6"])
        self.assertEqual(
            out["mushroom_body_heel_stimulations_with_parkin45_strictly_above"], 5)
        self.assertEqual(
            out["mushroom_body_heel_stimulations_with_the_two_groups_equal"], 1)
        self.assertTrue(out["ordering_conflict_is_not_a_reading_artefact"])
        self.assertGreater(out["smallest_absolute_central_complex_ordering_gap_uM"],
                           R.READING_UNCERTAINTY_UM)
        self.assertGreater(out["heel_ordering_gap_at_stim1_uM"],
                           R.READING_UNCERTAINTY_UM)
        for got, want in zip(out["central_complex_parkin45_minus_control45_uM"],
                             [-0.2, -0.14, -0.08, -0.07, -0.08, -0.075]):
            self.assertAlmostEqual(got, want, places=12)
        for got, want in zip(out["mushroom_body_heel_parkin45_minus_control45_uM"],
                             [0.09, 0.04, 0.04, 0.02, 0.0, 0.01]):
            self.assertAlmostEqual(got, want, places=12)
        self.assertEqual(out["stim1_R0_pairs_matching_BOTH_orderings"], 0)
        self.assertEqual(out["stim1_R0_pairs_matching_central_complex_ordering"]
                         + out["stim1_R0_pairs_matching_mushroom_body_heel_ordering"],
                         out["stim1_R0_pairs_enumerated"])

    def test_model_ordering_is_region_invariant(self):
        p = ProtocolSpec()
        g = small_grid()
        R0 = {"control_1": 0.03, "parkin_1": 0.05, "control_45": 0.02,
              "parkin_45": 0.06}          # all below efficacy*dose = 0.06 -> capped
        params = R.ModelParams(efficacy=0.3, release_tau_s=0.5, clearance_tau_s=5.0,
                               pool_capacity_pmol=dict(R0),
                               refill_tau_s={gg: 300.0 for gg in R.GROUPS})
        for scale in (0.3, 1.0, 2.7):
            peaks = params.predicted_uM(p, scale=scale)
            firsts = {gg: float(peaks[gg][0]) for gg in R.GROUPS}
            self.assertEqual(sorted(firsts, key=lambda gg: firsts[gg]),
                             ["control_45", "control_1", "parkin_1", "parkin_45"])
        # with every group uncapped, all four stimulation-1 peaks are identical
        params.pool_capacity_pmol = {gg: 0.5 for gg in R.GROUPS}
        peaks = params.predicted_uM(p)
        firsts = [float(peaks[gg][0]) for gg in R.GROUPS]
        for v in firsts[1:]:
            self.assertAlmostEqual(v, firsts[0], places=12)
        # a capping comparison: two R0 values, one capped, ordering is R0-monotone
        for eff in (0.2, 0.5, 1.0):        # efficacy*dose = 0.04, 0.10, 0.20
            lo = R.stim1_peak_uM(eff, 0.02, 0.5, 5.0, p)    # pooled-capped
            hi = R.stim1_peak_uM(eff, 0.5, 0.5, 5.0, p)     # uncapped
            self.assertLess(lo, hi)
            self.assertAlmostEqual(
                lo, min(eff * p.ach_dose_pmol, 0.02) * 1e9 / R.VOXEL_VOLUME_UM3
                * R.peak_shape_factor(0.5, 5.0), places=15)

    def test_joint_shared_kinetics_costs_more_than_either_single_region(self):
        data = R.load_digitised_data()
        p = R.protocol_from_digitised(data)
        g = small_grid()
        tgt = {r: {gg: data.regions[r].measured_uM[gg] for gg in R.GROUPS}
               for r in R.REGION_KEYS}
        cc = R.fit(g, "A", {"central_complex": tgt["central_complex"]},
                   protocol=p, seed=0)
        hh = R.fit(g, "A", {"mushroom_body_heel": tgt["mushroom_body_heel"]},
                   protocol=p, seed=0)
        joint = R.fit(g, "A", tgt, {"central_complex": (1.0,),
                                    "mushroom_body_heel": (0.5, 1.0, 2.0)},
                      protocol=p, seed=0)
        joint_cc = float(np.sqrt(np.mean(np.concatenate(
            [joint.residuals_uM["central_complex"][gg] ** 2 for gg in R.GROUPS]))))
        joint_hh = float(np.sqrt(np.mean(np.concatenate(
            [joint.residuals_uM["mushroom_body_heel"][gg] ** 2 for gg in R.GROUPS]))))
        self.assertGreaterEqual(joint_cc, cc.rms_overall_uM - 1e-12)
        self.assertGreaterEqual(joint_hh, hh.rms_overall_uM - 1e-12)
        self.assertIn("mushroom_body_heel", joint.scales)
        # a per-region scale cannot repair the ordering conflict: the joint fit is
        # still far outside the reading uncertainty
        self.assertGreater(joint.rms_overall_uM, R.READING_UNCERTAINTY_UM)


# ---------------------------------------------------------------------------
class EngineAgreementTests(unittest.TestCase):
    def test_fast_evaluator_matches_the_engine(self):
        p = ProtocolSpec()
        out = R.self_test_against_engine(p, small_grid(), n=16, seed=0)
        self.assertTrue(out["passed"], out["worst_abs_difference_uM"])
        self.assertLess(out["worst_abs_difference_uM"], 1e-9)
        self.assertEqual(out["n_configurations"], 16)

    def test_engine_agreement_at_a_fitted_parameter_point(self):
        data = R.load_digitised_data()
        p = R.protocol_from_digitised(data)
        f = R.fit(small_grid(), "A", targets_for(data, "central_complex"),
                  protocol=p, seed=0)
        worst = 0.0
        for gg in R.GROUPS:
            eng = f.params.engine_predicted_uM(p, gg)
            worst = max(worst, float(np.max(np.abs(eng - f.model_uM["central_complex"][gg]))))
        self.assertLess(worst, 1e-9, "fast evaluator disagrees with the engine")

    def test_carry_over_between_stimulations_is_negligible(self):
        p = ProtocolSpec()
        for tau_clr in (0.5, 5.0, 20.0):
            kin = DAKinetics(pool_capacity_pmol=0.2, stimulus_efficacy=0.3 * 0.2 / 0.2,
                             refill_tau_s=300.0, release_tau_s=0.5,
                             clearance_tau_s=tau_clr)
            core = release_sequence(p, kin)["ephemeral"]
            for i in range(1, R.N_STIMULATIONS):
                self.assertLess(float(core["c_start_nM"][i]) * R.NM_TO_UM, 1e-9)


# ---------------------------------------------------------------------------
class ReportAndOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = R.load_digitised_data()
        cls.protocol = R.protocol_from_digitised(cls.data)
        cls.report = R.build_report(cls.data, cls.protocol, small_grid(), seed=0,
                                    verbose=False)

    def test_report_is_json_serialisable_with_required_keys(self):
        payload = {k: v for k, v in self.report.items() if not k.startswith("_")}
        text = json.dumps(R._jsonable(payload))
        self.assertGreater(len(text), 1000)
        for key in ("what_this_is", "honesty", "data_provenance", "measured_protocol",
                    "search", "fits", "critical_test", "non_identifiability",
                    "cross_region", "panel_H_normalised", "residuals_exceeding_uncertainty",
                    "headline_findings", "run_stats"):
            self.assertIn(key, payload)
        for forbidden in ("NaN", "Infinity"):
            self.assertNotIn(forbidden, text)

    def test_report_contains_every_honesty_requirement(self):
        h = self.report["honesty"]
        self.assertTrue(h["measured_points_are_figure_readings_not_raw_data"])
        self.assertIn("NOT raw experimental data", h["measured_points_statement"])
        self.assertTrue(h["a_good_fit_does_not_identify_a_mechanism"])
        self.assertTrue(h["no_mechanism_is_claimed"])
        self.assertTrue(h["no_consciousness_viability_medical_claim"])
        self.assertTrue(h["ach_is_the_stimulus_da_is_the_modelled_transmitter"])
        self.assertIn("measured", h["measured_protocol_timing_only"])
        self.assertGreaterEqual(len(h["what_a_real_fit_would_still_require"]), 6)
        blob = " ".join(h["what_a_real_fit_would_still_require"]).lower()
        for topic in ("raw per-fly", "clearance", "volume", "receptor", "heel",
                      "half-decay"):
            self.assertIn(topic, blob)
        # every residual beyond an uncertainty is listed explicitly
        self.assertEqual(len(self.report["residuals_exceeding_uncertainty"]),
                         self.report["n_residuals_exceeding_uncertainty"])
        for row in self.report["residuals_exceeding_uncertainty"][:5]:
            self.assertTrue(row["exceeds_reading_uncertainty"]
                            or row["exceeds_published_sem"])
        # the search accounting is exact and bounded
        for arm in R.ARMS:
            entry = self.report["search"]["per_arm"][arm]
            self.assertEqual(entry["n_model_curve_evaluations"],
                             entry["n_shared_points"] * entry["n_subgrid_points_per_group"]
                             * len(R.GROUPS))
        self.assertEqual(self.report["search"]["seed"], 0)

    def test_traces_are_written_with_the_expected_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = R.save_traces(self.report, Path(tmp) / "t.npz", sample_dt_s=0.5)
            z = np.load(path)
            for region in R.REGION_KEYS:
                for gg in R.GROUPS:
                    self.assertIn(f"{region}__measured_uM__{gg}", z)
                    self.assertIn(f"{region}__model_armA_uM__{gg}", z)
                    self.assertIn(f"{region}__residual_armA_uM__{gg}", z)
                    self.assertEqual(z[f"{region}__measured_uM__{gg}"].shape, (6,))
                    self.assertEqual(
                        z[f"trace__{region}__armA__{gg}__uM"].shape[0], 6)
            self.assertIn("grid__pool_capacity_pmol", z)
            self.assertEqual(z["stimulation_number"].tolist(), [1, 2, 3, 4, 5, 6])

    def test_figure_is_written_and_layout_checks_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            written = R.save_outputs(self.report, Path(tmp), figure=True,
                                     traces=False, dpi=70)
            self.assertTrue(Path(written["figure"]).exists())
            checks = written["figure_layout_checks"]
            self.assertEqual(checks["clipped"], [],
                             f"layout problems: {checks['clipped']}")
            self.assertTrue(checks["suptitle"]["fully_inside_canvas"])
            self.assertTrue(checks["footnote"]["fully_inside_canvas"])
            for name, panel in checks["panels"].items():
                self.assertFalse(panel["footnote_overlaps_axes"], name)
                self.assertEqual(panel["data_markers_covered_by_legend"], 0, name)
            self.assertTrue(Path(written["report"]).exists())
            data = json.loads(Path(written["report"]).read_text())
            self.assertIn("headline_findings", data)


# ---------------------------------------------------------------------------
class SelfTestHarnessTests(unittest.TestCase):
    def test_helper_tables_are_consistent(self):
        """The grid's declared counts agree with explicit enumeration."""
        g = small_grid()
        for arm in R.ARMS:
            shared = list(g.shared_points(arm))
            sub = list(g.subgrid_points(arm))
            self.assertEqual(len(shared), g.n_shared_points(arm))
            self.assertEqual(len(sub), g.n_subgrid_points(arm))
            for d in shared:
                for k, v in d.items():
                    self.assertIn(k, g.axis_lengths())
                    self.assertGreaterEqual(v, g.bounds[k][0] - 1e-15)
                    self.assertLessEqual(v, g.bounds[k][1] + 1e-15)
            for d in sub:
                for k, v in d.items():
                    self.assertGreaterEqual(v, g.bounds[k][0] - 1e-15)
                    self.assertLessEqual(v, g.bounds[k][1] + 1e-15)

    def test_peak_shape_factor_is_monotone_in_clearance(self):
        p = ProtocolSpec()
        prev = -1.0
        for tau_clr in (0.5, 1.0, 2.0, 5.0, 12.0, 20.0, 1e4):
            ps = R.peak_shape_factor(0.5, tau_clr)
            self.assertGreater(ps, prev)
            self.assertLessEqual(ps, 1.0 + 1e-12)
            prev = ps


# ---------------------------------------------------------------------------
def run_and_report():
    loader = unittest.TestLoader()
    import sys as _sys
    suite = loader.loadTestsFromModule(_sys.modules[__name__])
    n_tests = suite.countTestCases()
    # collect the names BEFORE running: unittest consumes the suite as it runs
    names = sorted(t.id().split(".")[-2] + "." + t.id().split(".")[-1]
                   for t in _iter_tests(suite))
    runner = unittest.TextTestRunner(verbosity=2)
    t0 = time.perf_counter()
    result = runner.run(suite)
    wall = time.perf_counter() - t0
    assert len(names) == n_tests, (len(names), n_tests)
    summary = {
        "n_tests": n_tests,
        "n_run": result.testsRun,
        "n_passed": result.testsRun - len(result.failures) - len(result.errors),
        "n_failures": len(result.failures),
        "n_errors": len(result.errors),
        "n_skipped": len(result.skipped),
        "n_expected_failures": len(result.expectedFailures),
        "wall_clock_s": wall,
        "peak_rss_mb": R.peak_memory_mb(),
        "test_names": names,
        "what_this_selftest_covers": [
            "the digitised figure readings load, are validated, and malformed copies "
            "are rejected",
            "the model reproduces the MEASURED protocol timing through this pipeline",
            "residual and RMS arithmetic is correct against hand-checked cases and a "
            "synthetic fixture whose residual must be ~0",
            "the bounded grid search is deterministic, reproducible with a fixed "
            "seed, seed-order-invariant in its result, and refused when it would "
            "exceed its evaluation cap",
            "the structural claims (group-invariant stimulation-1 peak, capping as "
            "the only route to elevation, the cross-region ordering conflict, the "
            "panel-H normalisation blind spot) hold",
            "the non-identifiability report is reproducible and the report carries "
            "every mandated honesty statement",
        ],
        "what_a_pass_does_NOT_mean": [
            "no test here validates the model physiologically",
            "the measured values are figure readings with ~0.03 uM uncertainty",
            "every model kinetic constant is illustrative; a good residual would not "
            "identify a mechanism",
            "no consciousness, viability or medical claim is made or supported",
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SELFTEST_JSON.write_text(json.dumps(R._jsonable(summary), indent=2))
    print(f"\ncounts: {summary['n_run']} tests run, {summary['n_passed']} passed, "
          f"{summary['n_failures']} failures, {summary['n_errors']} errors, "
          f"{summary['n_skipped']} skipped")
    print(f"wall clock {wall:.1f} s; peak RSS {summary['peak_rss_mb']:.0f} MiB")
    print(f"wrote {SELFTEST_JSON}")
    return 0 if result.wasSuccessful() else 1


def _iter_tests(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _iter_tests(item)
        elif hasattr(item, "id"):
            yield item


if __name__ == "__main__":
    raise SystemExit(run_and_report())
