"""Prevent unknown provenance or historical controlled demos becoming autonomy PASS."""
import unittest
from engine.embodied.autonomy_contract import assess_autonomy, REQUIRED_FALSE, REQUIRED_TRUE

class ContractTests(unittest.TestCase):
    def test_missing_is_failure(self):
        self.assertEqual(assess_autonomy({})['autonomous_goal_status'],'NOT_ACHIEVED')
    def test_each_bypass_rejected(self):
        m={**dict.fromkeys(REQUIRED_FALSE,False),**dict.fromkeys(REQUIRED_TRUE,True)}
        for key in REQUIRED_FALSE:
            with self.subTest(key=key):
                bad=dict(m);bad[key]=True
                self.assertEqual(assess_autonomy(bad)['autonomous_goal_status'],'NOT_ACHIEVED')
    def test_unvalidated_translation_rejected(self):
        m={**dict.fromkeys(REQUIRED_FALSE,False),**dict.fromkeys(REQUIRED_TRUE,True)}
        for key in REQUIRED_TRUE:
            with self.subTest(key=key):
                bad=dict(m);bad[key]=False
                self.assertEqual(assess_autonomy(bad)['autonomous_goal_status'],'NOT_ACHIEVED')
    def test_engineering_pass_irrelevant(self):
        self.assertEqual(assess_autonomy({'engineering_check_status':'PASS'})['autonomous_goal_status'],'NOT_ACHIEVED')

if __name__=='__main__':unittest.main()
