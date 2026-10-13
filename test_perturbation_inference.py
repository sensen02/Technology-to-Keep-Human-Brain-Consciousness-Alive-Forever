"""Synthetic endpoint/inference regressions; no physics or connectome runs."""
import unittest

from run_neural_muscle_perturbation import endpoint_eligibility, infer_h19


def rows(advantages_ms=(10, 15), silence_s=.020):
    return [{"weight_fraction": fraction, "arms": {
        "normal": {"escape_time_s": silence_s + advantage / 1000},
        "spike_output_silence": {"escape_time_s": silence_s}}}
        for fraction, advantage in zip((0, .5), advantages_ms)]


def evaluate(data, **kwargs):
    return infer_h19(data, .015, push_was_applied=kwargs.get("applied", True),
                     lateral_changed=kwargs.get("changed", True))


class PerturbationInferenceTests(unittest.TestCase):
    def test_pre_push_escape_cannot_be_rescued_by_lateral_motion(self):
        result = evaluate(rows(silence_s=.0149))
        self.assertFalse(result["interpretable"])
        self.assertIsNone(result["supported"])
        self.assertEqual(result["verdict"], "not_interpretable")
        self.assertEqual(result["endpoint_eligibility"][1]["arms"]
                         ["spike_output_silence"]["reason"],
                         "escape_at_or_before_push_onset")

    def test_boundary_and_right_censoring(self):
        self.assertFalse(endpoint_eligibility(.015, .015)["eligible"])
        self.assertTrue(endpoint_eligibility(.0151, .015)["eligible"])
        self.assertEqual(endpoint_eligibility(None, .015)["reason"],
                         "escape_not_observed_right_censored")
        data = rows()
        data[1]["arms"]["normal"]["escape_time_s"] = None
        self.assertIsNone(evaluate(data)["supported"])

    def test_flat_or_decreasing_only_rejects_shape_hypothesis(self):
        for advantages in ((10, 10), (10, 5)):
            result = evaluate(rows(advantages))
            self.assertTrue(result["interpretable"])
            self.assertFalse(result["supported"])
            self.assertEqual(result["verdict"], "H19_not_supported")
            self.assertIn("neither proves nor excludes feedback", result["interpretation_scope"])

    def test_eligible_growing_advantage_supports_only_h19(self):
        result = evaluate(rows())
        self.assertTrue(result["supported"])
        self.assertAlmostEqual(result["advantage_gain_ratio"], 1.5)

    def test_probe_and_design_checks_gate_support(self):
        self.assertIsNone(evaluate(rows(), applied=False)["supported"])
        self.assertIsNone(evaluate(rows(), changed=False)["supported"])
        self.assertIsNone(evaluate(list(reversed(rows())))["supported"])
        self.assertIsNone(evaluate([])["supported"])
        self.assertIsNone(evaluate(rows((0, 5)))["supported"])

    def test_invalid_times_do_not_support_h19(self):
        data = rows()
        data[1]["arms"]["normal"]["escape_time_s"] = float("nan")
        self.assertIsNone(evaluate(data)["supported"])
        with self.assertRaises(ValueError):
            endpoint_eligibility(.020, float("nan"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
