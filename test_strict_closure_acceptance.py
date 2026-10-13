import unittest
from run_strict_closure_acceptance import evaluate
class StrictAcceptanceTests(unittest.TestCase):
    def test_interface_pass_cannot_hide_floor_or_fall_or_uncalibrated_translation(self):
        r=evaluate(True,True,False,False)
        self.assertEqual(r['status'],'NOT_ACCEPTED');self.assertEqual(len(r['failed_checks']),3)
    def test_each_unmet_condition_fails_closed(self):
        for args in [(False,False,True,True),(True,True,True,True),(True,False,False,True),(True,False,True,False)]:
            with self.subTest(args=args):self.assertEqual(evaluate(*args)['status'],'NOT_ACCEPTED')
    def test_hypothetical_all_pass_not_current_claim(self):
        self.assertEqual(evaluate(True,False,True,True)['status'],'PASS')
if __name__=='__main__':unittest.main()
