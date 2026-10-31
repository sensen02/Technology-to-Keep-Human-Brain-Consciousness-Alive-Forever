import unittest,numpy as np
from engine.embodied.motor_cadence_schedule import motor_windows
from engine.embodied.motor_output import MotorOutput
class ClockTest(unittest.TestCase):
 def test_partition_and_hold(self):
  for dt in (.005,.001):
   w=motor_windows(.05,motor_dt_s=dt);ticks=[t for x in w for t in range(x.start_tick,x.end_tick)]
   self.assertEqual(ticks,list(range(100)))
   self.assertEqual(sorted(set(x.sensory_index for x in w)),list(range(10)))
   self.assertTrue(all(x.apply_source_end_tick is None or x.apply_source_end_tick<=x.start_tick for x in w))
 def test_filter_partition_invariant(self):
  manifest=dict(n_neurons=2,n_actuators=1,rows=[dict(actuator_index=0,local_indices=[0,1])]);sp=np.zeros((100,2),bool);sp[[0,9,10,22,81],0]=True
  endpoints=[]
  for size in (10,2):
   decoder=MotorOutput(manifest);counts=0;outputs={}
   for k in range(0,100,size):
    ctrl,tr=decoder.consume(sp[k:k+size],spike_dt_s=.0005,start_s=k*.0005,end_s=(k+size)*.0005);counts+=tr['pool_spike_counts'][0];outputs[k+size]=ctrl.copy()
   self.assertEqual(counts,5);endpoints.append(outputs)
  for k,v in endpoints[0].items():np.testing.assert_array_equal(v,endpoints[1][k])
 def test_duplicate_rejected(self):
  m=dict(n_neurons=1,n_actuators=1,rows=[dict(actuator_index=0,local_indices=[0])]);d=MotorOutput(m);d.consume(np.zeros((2,1)),spike_dt_s=.0005,start_s=0,end_s=.001)
  with self.assertRaises(ValueError):d.consume(np.zeros((2,1)),spike_dt_s=.0005,start_s=0,end_s=.001)
if __name__=='__main__':unittest.main()

