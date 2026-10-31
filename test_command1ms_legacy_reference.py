"""Supplemental deterministic legacy5ms pending law against actual split schedule."""
import unittest,types
from unittest.mock import patch
import numpy as np
import mujoco
import run_neural_muscle_closure as c
from engine.embodied.motor_output import MotorOutput
class LegacyReference(unittest.TestCase):
 def test_nonzero_default5ms_matches_historical_pending_law(self):
  m=types.SimpleNamespace(nu=1,opt=types.SimpleNamespace(timestep=.0000125),body=lambda name:types.SimpleNamespace(id=0))
  manifest=dict(n_neurons=1,n_actuators=1,rows=[dict(actuator_index=0,local_indices=[0])]);built=dict(n=1,edges=dict(We=np.zeros((1,1)),Wi=np.zeros((1,1))));body=[]
  sp=np.zeros((40,1),bool);sp[[0,9,10,22],0]=1
  class Data:
   def __init__(self,m):self.qpos=np.zeros(1);self.qvel=np.zeros(1);self.act=np.zeros(1);self.ctrl=np.zeros(1);self.xpos=np.array([[0.,0.,1.]]);self.time=0.;self.warning=[]
  class Net:
   def __init__(self,*a,**kw):self.steps=0;self.t_ms=0
   def set_connectivity(self,*a):pass
   def step(self,current_nA):result=sp[self.steps];self.steps+=1;self.t_ms=self.steps*.5;return result
  class Encoder:
   def __init__(self,*a):pass
   def encode(self,q):return np.ones(1),{}
  def step(m,d):d.time+=.0000125;body.append(d.ctrl.copy())
  ev={}
  with patch.object(mujoco,'MjData',Data),patch.object(mujoco,'mj_forward',lambda *a:None),patch.object(mujoco,'mj_step',step),patch.object(c,'ConductanceNetwork',Net),patch.object(c,'ChannelSensoryEncoder',Encoder),patch.object(c,'observe_quantities',return_value={}),patch.object(c,'resolve_sensory_reference',lambda *a:np.zeros(1)):
   _,_,commands,counts=c._episode_split_motor(m,np.zeros(1),{}, {},built,manifest,{'lf':{'bristle_touch':[]}},'normal',.02,replay=None,push=None,external_touch=False,disabled_channels=(),evidence=ev,motor_translator=None,body_observer=None,body_dt_s=.0000125,motor_dt_s=.005)
  # Historical episode law: apply previous pending, run10 samples, consume at endpoint.
  decoder=MotorOutput(manifest);pending=np.zeros(1);old_commands=[];old_decoded=[]
  for k in range(4):
   old_commands.append(pending.copy());pending,_=decoder.consume(sp[k*10:(k+1)*10],spike_dt_s=.0005,start_s=k*.005,end_s=(k+1)*.005);old_decoded.append(pending.copy())
  np.testing.assert_array_equal(commands,old_commands);np.testing.assert_array_equal(ev['motor_decoded_next_controls'],old_decoded);np.testing.assert_array_equal(counts,sp.reshape(4,10,1).sum(1));np.testing.assert_array_equal(body,np.repeat(old_commands,400,axis=0));self.assertTrue(commands[1:].any())
if __name__=='__main__':unittest.main()
