import inspect,textwrap,unittest
import numpy as np
from engine.neural_cond import ConductanceNetwork
class Tests(unittest.TestCase):
 def test_zero_matches_old_update(self):
  src=textwrap.dedent(inspect.getsource(ConductanceNetwork.step)).replace('p.threshold_mV + self.threshold_offset_mV','p.threshold_mV');ns={'np':np};exec(src,ns)
  a=ConductanceNetwork(4,dt_ms=.5);b=ConductanceNetwork(4,dt_ms=.5,threshold_offset_mV=np.zeros(4))
  for k in range(80):
   current=np.array([0.,.12,.3,-.02])*(1+(k%3)*.1);np.testing.assert_array_equal(a.step(current),ns['step'](b,current))
   for name in ['v','ge','gi','adaptation','refractory_until_ms','spike_count','_ring']:np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
 def test_only_selected_and_units(self):
  off=np.array([0.,-2.,0.]);net=ConductanceNetwork(3,dt_ms=.5,threshold_offset_mV=off);off[:]=0
  net.v[:]=-51.;event=net.step(0.)
  # Force an unambiguous threshold crossing after membrane integration.
  net=ConductanceNetwork(3,dt_ms=.5,threshold_offset_mV=[0,-2,0]);net.v[:]=-50.5
  np.testing.assert_array_equal(net.step(0.),[False,True,False]);np.testing.assert_array_equal(net.threshold_offset_mV,[0,-2,0]);self.assertEqual(net.p.threshold_mV-2,-52.)
 def test_invalid(self):
  for x in [[0], [np.nan,0],[-20,0]]:
   with self.assertRaises(ValueError):ConductanceNetwork(2,threshold_offset_mV=x)
if __name__=='__main__':unittest.main()
