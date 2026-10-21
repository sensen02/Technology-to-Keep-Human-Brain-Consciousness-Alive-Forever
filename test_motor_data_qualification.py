"""Synthetic schema/engineering tests only: no support or behavior success."""
import unittest

import numpy as np

from engine.embodied.motor_data_qualification import qualify_behavior_support


def synthetic_schema_fixture(**overrides):
    """An eligible engineering fixture, NOT measured physical support evidence."""
    fields = dict(sample_edges_s=[.05, .30, .55], foot_normal=[8., 8.],
                  effective_nonfoot=[0, 0], root_height=[1.7, 1.7],
                  tilt_rad=[.35, .35], model_weight=10., initial_root_height=2.)
    fields.update(overrides)
    return fields


class MotorDataQualificationTests(unittest.TestCase):
    def test_threshold_equalities_are_eligible_engineering_only(self):
        result = qualify_behavior_support(**synthetic_schema_fixture())
        self.assertEqual(set(result), {"eligible", "reasons", "measured", "scope", "behavior_success"})
        self.assertIs(result["eligible"], True)
        self.assertEqual(result["reasons"], [])
        self.assertEqual(result["scope"], "engineering_preflight_only")
        self.assertIs(result["behavior_success"], False)
        measured = result["measured"]
        self.assertEqual(measured["time_weighted_foot_normal_ratio"], .8)
        self.assertEqual(measured["effective_nonfoot_duration_s"], 0.)
        self.assertEqual(measured["min_root_height_ratio"], .85)
        self.assertEqual(measured["max_tilt_rad"], .35)
        self.assertEqual(measured["duration_s"], .5)
        self.assertEqual(measured["sample_count"], 2)

    def test_time_weighting_not_sample_mean(self):
        # Arithmetic mean is 1, but interval-weighted support is only .4.
        result = qualify_behavior_support(**synthetic_schema_fixture(
            sample_edges_s=[.05, .15, .55], foot_normal=[20., 0.]))
        self.assertFalse(result["eligible"])
        self.assertAlmostEqual(result["measured"]["time_weighted_foot_normal_ratio"], .4)
        # Arithmetic mean is .5, but long supported interval yields .9.
        result = qualify_behavior_support(**synthetic_schema_fixture(
            sample_edges_s=[.05, .50, .55], foot_normal=[10., 0.]))
        self.assertTrue(result["eligible"])
        self.assertAlmostEqual(result["measured"]["time_weighted_foot_normal_ratio"], .9)

    def test_each_threshold_has_strict_failure(self):
        cases = [
            ({"foot_normal": [np.nextafter(8., 0.), np.nextafter(8., 0.)]},
             "time_weighted_foot_normal_ratio_below_0.8"),
            ({"effective_nonfoot": [0, 1]}, "effective_nonfoot_duration_nonzero"),
            ({"root_height": [1.7, np.nextafter(1.7, 0.)]}, "min_root_height_ratio_below_0.85"),
            ({"tilt_rad": [np.nextafter(.35, 1.), 0.]}, "max_tilt_rad_above_0.35"),
        ]
        for overrides, reason in cases:
            with self.subTest(reason=reason):
                result = qualify_behavior_support(**synthetic_schema_fixture(**overrides))
                self.assertFalse(result["eligible"])
                self.assertEqual(result["reasons"], [reason])
                self.assertIs(result["behavior_success"], False)

    def test_all_failures_reported_and_extrema_not_averages(self):
        result = qualify_behavior_support(**synthetic_schema_fixture(
            foot_normal=[0., 0.], effective_nonfoot=[1, 1],
            root_height=[2., 1.], tilt_rad=[0., .36]))
        self.assertEqual(len(result["reasons"]), 4)
        self.assertEqual(result["measured"]["effective_nonfoot_duration_s"], .5)
        self.assertEqual(result["measured"]["min_root_height_ratio"], .5)

    def test_even_tiny_nonfoot_interval_fails_exact_zero(self):
        result = qualify_behavior_support(**synthetic_schema_fixture(
            sample_edges_s=[.05, np.nextafter(.05, 1.), .55], effective_nonfoot=[1, 0]))
        self.assertFalse(result["eligible"])
        self.assertGreater(result["measured"]["effective_nonfoot_duration_s"], 0)

    def test_truncation_extension_and_shift_rejected(self):
        for edges in ([.05, .30, .54], [.06, .30, .55], [0., .30, .55],
                      [.05, .30, .60], [.051, .30, .551], [.05 + 2e-12, .30, .55],
                      [.05, .30, .55 - 2e-12]):
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                qualify_behavior_support(**synthetic_schema_fixture(sample_edges_s=edges))

    def test_only_endpoint_tolerance_permitted(self):
        result = qualify_behavior_support(**synthetic_schema_fixture(
            sample_edges_s=[.05 + .5e-12, .30, .55 - .5e-12], foot_normal=[10., 10.]))
        self.assertTrue(result["eligible"])

    def test_edges_must_be_nonempty_shared_and_strictly_increasing(self):
        for edges in ([], [.05], [.05, .30, .30, .55], [.05, .40, .30, .55],
                      [[.05, .30, .55]], [.05, .55]):
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                qualify_behavior_support(**synthetic_schema_fixture(sample_edges_s=edges))
        # N=1 is valid with matching interval values.
        result = qualify_behavior_support(**synthetic_schema_fixture(
            sample_edges_s=[.05, .55], foot_normal=[8.], effective_nonfoot=[0],
            root_height=[1.7], tilt_rad=[.35]))
        self.assertTrue(result["eligible"])

    def test_missing_fields_and_claimed_pass_flags_rejected(self):
        for name in synthetic_schema_fixture():
            fields = synthetic_schema_fixture()
            del fields[name]
            with self.subTest(missing=name), self.assertRaises(ValueError):
                qualify_behavior_support(**fields)
        for name in ("pass", "eligible", "behavior_success"):
            with self.subTest(untrusted=name), self.assertRaises(ValueError):
                qualify_behavior_support(**synthetic_schema_fixture(**{name: True}))

    def test_nonfinite_in_every_field_fails_closed(self):
        for name, value in synthetic_schema_fixture().items():
            for bad in (np.nan, np.inf, -np.inf):
                modified = [bad, *value[1:]] if isinstance(value, list) else bad
                with self.subTest(field=name, bad=bad), self.assertRaises(ValueError):
                    qualify_behavior_support(**synthetic_schema_fixture(**{name: modified}))

    def test_nonpositive_or_nonscalar_normalizers_rejected(self):
        for name in ("model_weight", "initial_root_height"):
            for value in (0., -1., [1.], True, None):
                with self.subTest(field=name, value=value), self.assertRaises(ValueError):
                    qualify_behavior_support(**synthetic_schema_fixture(**{name: value}))

    def test_malformed_sample_arrays_rejected(self):
        for name in ("foot_normal", "effective_nonfoot", "root_height", "tilt_rad"):
            for value in (None, [], [1.], [1., 1., 1.], [[1., 1.]], ["1", "1"], [1j, 1j]):
                with self.subTest(field=name, value=value), self.assertRaises(ValueError):
                    qualify_behavior_support(**synthetic_schema_fixture(**{name: value}))
        for value in ([.5, 0], [-1, 1], [2, 0]):
            with self.subTest(binary=value), self.assertRaises(ValueError):
                qualify_behavior_support(**synthetic_schema_fixture(effective_nonfoot=value))
        for name in ("foot_normal", "tilt_rad"):
            with self.subTest(negative=name), self.assertRaises(ValueError):
                qualify_behavior_support(**synthetic_schema_fixture(**{name: [-1., 0.]}))

    def test_derived_overflow_fails_closed(self):
        for fields in (dict(model_weight=1e-308, foot_normal=[1e308, 1e308]),
                       dict(initial_root_height=1e-308, root_height=[1e308, 1e308])):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                qualify_behavior_support(**synthetic_schema_fixture(**fields))

    def test_cli_eligible_rejected_malformed_and_no_overwrite(self):
        import contextlib
        import io
        import json
        from pathlib import Path
        import tempfile
        from qualify_motor_training_data import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, output = root/'support.npz', root/'result.json'
            args = ['--support-data', str(data), '--output', str(output)]
            np.savez(data, **synthetic_schema_fixture())
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            result = json.loads(output.read_text())
            self.assertFalse(result['behavioral_optimization_authorized'])
            self.assertFalse(result['behavior_success'])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            args[-1] = str(root/'rejected.json')
            np.savez(data, **synthetic_schema_fixture(root_height=[1., 1.], effective_nonfoot=[1, 1]))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 2)
            self.assertFalse(json.loads(Path(args[-1]).read_text())['eligible'])
            args[-1] = str(root/'malformed.json')
            np.savez(data, **synthetic_schema_fixture(sample_edges_s=[.1, .3, .55]))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            self.assertFalse(Path(args[-1]).exists())

    def test_numpy_arrays_accepted_without_mutation(self):
        fields = {name: np.array(value) for name, value in synthetic_schema_fixture().items()}
        originals = {name: value.copy() for name, value in fields.items()}
        self.assertTrue(qualify_behavior_support(**fields)["eligible"])
        for name, original in originals.items():
            np.testing.assert_array_equal(fields[name], original)


if __name__ == "__main__":
    unittest.main()
