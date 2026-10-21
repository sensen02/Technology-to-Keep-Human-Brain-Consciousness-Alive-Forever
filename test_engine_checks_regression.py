"""Engineering regressions for CheckSuite boundaries, not scientific validation."""
import unittest

from engine.checks import CheckSuite, ChecksFailed


class CheckSuiteBoundaryRegression(unittest.TestCase):
    def run_one(self, output, tolerance=None, stop_on_error=False):
        suite = CheckSuite('boundary').add(
            'probe', 'invariant', lambda: output, tolerance=tolerance)
        return suite.run(stop_on_error=stop_on_error)[0]

    def test_missing_or_none_tuple_tolerance_uses_fallback(self):
        for output in ((0.0,), (0.0, None), (0.0, None, 'detail')):
            for default in (None, 0.0, 1e-6):
                with self.subTest(output=output, default=default):
                    result = self.run_one(output, default, stop_on_error=True)
                    self.assertTrue(result.passed)
                    self.assertEqual(result.tolerance,
                                     1e-12 if default is None else default)
                    self.assertEqual(result.detail,
                                     'detail' if len(output) > 2 else '')

    def test_fallback_tolerance_is_used_for_comparison(self):
        for output in (2e-12, (2e-12,), (2e-12, None)):
            with self.subTest(output=output):
                self.assertFalse(self.run_one(output).passed)
                self.assertTrue(self.run_one(output, 3e-12).passed)

    def test_valid_scalar_and_explicit_tuple_outputs_remain_compatible(self):
        cases = ((0.0, None, 0.0, 1e-12, True, ''),
                 (1.0, 1.0, 1.0, 1.0, True, ''),
                 (2.0, 1.0, 2.0, 1.0, False, ''),
                 ((1.0, 1.0, 'ok'), 0.0, 1.0, 1.0, True, 'ok'),
                 (('0.25', '0.5', 7), None, 0.25, 0.5, True, '7'),
                 ('0.25', '0.5', 0.25, 0.5, True, ''))
        for output, default, error, tolerance, passed, detail in cases:
            with self.subTest(output=output, default=default):
                result = self.run_one(output, default, stop_on_error=True)
                self.assertEqual((result.error, result.tolerance,
                                  result.passed, result.detail),
                                 (error, tolerance, passed, detail))

    def test_invalid_errors_fail_for_scalar_and_tuple_outputs(self):
        for error in (-1.0, float('nan'), float('inf'), -float('inf')):
            for output in (error, (error, 0.0), (error, float('inf'))):
                with self.subTest(output=output):
                    result = self.run_one(output)
                    self.assertFalse(result.passed)
                    self.assertIn('error must be finite and nonnegative', result.detail)
                    with self.assertRaisesRegex(ValueError, 'error must be'):
                        self.run_one(output, stop_on_error=True)

    def test_invalid_tolerances_fail_for_explicit_and_fallback_values(self):
        for tolerance in (-1.0, float('nan'), float('inf'), -float('inf')):
            for output, default in (((0.0, tolerance), None),
                                    (0.0, tolerance),
                                    ((0.0,), tolerance),
                                    ((0.0, None), tolerance)):
                with self.subTest(output=output, default=default):
                    result = self.run_one(output, default)
                    self.assertFalse(result.passed)
                    self.assertIn('tolerance must be finite and nonnegative', result.detail)
                    with self.assertRaisesRegex(ValueError, 'tolerance must be'):
                        self.run_one(output, default, stop_on_error=True)

    def test_explicit_valid_tolerance_overrides_unused_invalid_default(self):
        self.assertTrue(self.run_one((0.0, 0.0), float('inf')).passed)

    def test_failure_does_not_stop_other_checks_by_default(self):
        suite = CheckSuite().invariant('invalid', lambda: -1.0)
        suite.invariant('valid', lambda: 0.0)
        results = suite.run()
        self.assertEqual([r.passed for r in results], [False, True])
        with self.assertRaises(ChecksFailed):
            suite.require_all_passed(results)

    def test_empty_suite_never_satisfies_required_checks(self):
        suite = CheckSuite('empty')
        self.assertFalse(suite.summary()['all_passed'])
        with self.assertRaisesRegex(ChecksFailed, 'no checks'):
            suite.require_all_passed()
        suite.invariant('otherwise_valid', lambda: 0.0)
        self.assertFalse(suite.summary([])['all_passed'])
        with self.assertRaisesRegex(ChecksFailed, 'no checks'):
            suite.require_all_passed([])

    def test_nonempty_require_all_passed_matches_summary(self):
        for error in (0.0, 1.0):
            with self.subTest(error=error):
                suite = CheckSuite().invariant('probe', lambda: error)
                results = suite.run()
                if suite.summary(results)['all_passed']:
                    self.assertTrue(suite.require_all_passed(results))
                else:
                    with self.assertRaises(ChecksFailed):
                        suite.require_all_passed(results)


if __name__ == '__main__':
    unittest.main()
