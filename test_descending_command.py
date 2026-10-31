"""No-physics tests of identity-restricted transient neuronal command."""
import unittest,numpy as np
from engine.embodied.descending_command import DescendingCommand,DNP09_ROOTS,command_input
class TestDescendingCommand(unittest.TestCase):
 def built(self):return dict(n=4,tier=dict(root_ids=np.array([42,*DNP09_ROOTS,43])))
 def test_identity_and_withdrawal(self):
  for t in [0.,.0495]:
   c,r=command_input(self.built(),DescendingCommand(.3,.05),t);np.testing.assert_array_equal(c,[0,.3,.3,0]);self.assertEqual(r['local_indices'],[1,2])
  for t in [.05,.25]:self.assertFalse(command_input(self.built(),DescendingCommand(.3,.05),t)[0].any())
 def test_zero_control(self):self.assertFalse(command_input(self.built(),DescendingCommand(0.,.05),0.)[0].any())
 def test_invalid(self):
  with self.assertRaises(ValueError):command_input(self.built(),DescendingCommand(-1,.05),0.)
  with self.assertRaises(ValueError):command_input(dict(n=1,tier=dict(root_ids=np.array([42]))),DescendingCommand(),0.)
if __name__=='__main__':unittest.main(verbosity=2)
