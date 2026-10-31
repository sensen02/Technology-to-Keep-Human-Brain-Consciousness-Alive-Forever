"""No physical or full-network simulation: actual split episode on deterministic stubs."""
import unittest,types
from unittest.mock import patch
import numpy as np
import run_neural_muscle_closure as c
from run_command1ms_single import recording_decoder
from engine.embodied.motor_output import MotorOutput
class Preflight(unittest.TestCase):
 def test_per_tick_partition_state(self):
  m=dict(n_neurons=2,n_actuators=1,rows=[dict(actuator_index=0,local_indices=[0,1])]);sp=np.zeros((100,2),bool);sp[[0,9,10,22,81],0]=1
  histories=[]
  for block in (2,10):
   d,_=recording_decoder(m);reference=MotorOutput(m)
   for k in range(0,100,block):
    got,tr=d.consume(sp[k:k+block],spike_dt_s=.0005,start_s=k*.0005,end_s=(k+block)*.0005)
    expected,_=reference.consume(sp[k:k+block],spike_dt_s=.0005,start_s=k*.0005,end_s=(k+block)*.0005)
    np.testing.assert_array_equal(got,expected)
   histories.append(d.tick_rates)
  np.testing.assert_array_equal(histories[0],histories[1])
 def test_silent_zero_and_completed_body_hold(self):
  m=types.SimpleNamespace(nu=1,opt=types.SimpleNamespace(timestep=.0000125),body=lambda name:types.SimpleNamespace(id=0))
  manifest=dict(n_neurons=1,n_actuators=1,rows=[dict(actuator_index=0,local_indices=[0])]);built=dict(n=1,edges=dict(We=np.zeros((1,1)),Wi=np.zeros((1,1))));seen=[];currents=[]
  class Data:
   def __init__(self,m):self.qpos=np.zeros(1);self.qvel=np.zeros(1);self.act=np.zeros(1);self.ctrl=np.zeros(1);self.xpos=np.array([[0.,0.,1.]]);self.time=0.;self.warning=[]
  class Net:
   def __init__(self,*a,**kw):self.t_ms=0.;self.steps=0
   def set_connectivity(self,*a):pass
   def step(self,current_nA):currents.append(current_nA.copy());self.steps+=1;self.t_ms=self.steps*.5;return np.zeros(1,bool)
  class Encoder:
   def __init__(self,*a):pass
   def encode(self,q):return np.array([q]),{}
  def observe(m,d,*a):seen.append(d.time);return round(d.time/.005)+1
  def step(m,d):d.time+=.0000125
  import mujoco
  for dt in (.005,.001):
   seen.clear();currents.clear();ev={}
   with patch.object(mujoco,'MjData',Data),patch.object(mujoco,'mj_forward',lambda *a:None),patch.object(mujoco,'mj_step',step),patch.object(c,'ConductanceNetwork',Net),patch.object(c,'ChannelSensoryEncoder',Encoder),patch.object(c,'observe_quantities',observe),patch.object(c,'resolve_sensory_reference',lambda *a:np.zeros(1)):
    _,_,cmd,sp=c._episode_split_motor(m,np.zeros(1),{}, {},built,manifest,{'lf':{'bristle_touch':[]}},'normal',.015,replay=None,push=None,external_touch=False,disabled_channels=(),evidence=ev,motor_translator=None,body_observer=None,body_dt_s=.0000125,motor_dt_s=dt)
   np.testing.assert_allclose(seen,[0,.005,.010],atol=1e-14,rtol=0);np.testing.assert_array_equal(np.array(currents).ravel(),np.repeat([1,2,3],10));self.assertFalse(cmd.any());self.assertFalse(sp.any());self.assertFalse(ev['motor_decoded_next_controls'].any())
if __name__=='__main__':unittest.main()
