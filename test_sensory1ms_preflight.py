"""Actual derived episode on stubs, zero physics/full neural simulation."""
import unittest,types
from unittest.mock import patch
import numpy as np,mujoco
import run_neural_muscle_closure as c
from run_sensory1ms_single import derive
class Preflight(unittest.TestCase):
 def trial(self,refresh,constant=False,legacy=False):
  m=types.SimpleNamespace(nu=1,actuator_ctrlrange=np.array([[.0001,1.]]),actuator_ctrllimited=np.array([True]),opt=types.SimpleNamespace(timestep=.0000125),body=lambda name:types.SimpleNamespace(id=0));built=dict(n=1,edges=dict(We=np.zeros((1,1)),Wi=np.zeros((1,1))));manifest=dict(n_neurons=1,n_actuators=1,rows=[dict(actuator_index=0,local_indices=[0])]);seen=[];body=[];inputs=[]
  class Data:
   def __init__(self,m):self.qpos=np.zeros(1);self.qvel=np.zeros(1);self.act=np.zeros(1);self.ctrl=np.zeros(1);self.xpos=np.array([[0.,0.,1.]]);self.time=0.;self.warning=[];self.xfrc_applied=np.zeros((1,6));self.actuator_force=np.zeros(1)
  class Net:
   def __init__(self,*a,**kw):self.steps=0;self.t_ms=0.
   def set_connectivity(self,*a):pass
   def step(self,current_nA):inputs.append(current_nA.copy());self.steps+=1;self.t_ms=self.steps*.5;return np.array([self.steps in (1,10,11,23)])
  class Encoder:
   def __init__(self,*a):pass
   def encode(self,q):return np.array([q['lf'].value]),{}
  def observe(m,d,*a):seen.append(d.time);return {'lf':types.SimpleNamespace(value=1. if constant else round(d.time/.001)+1)}
  def step(m,d):d.time+=.0000125;body.append(d.ctrl.copy())
  log=[];cur=[];ev={}
  with patch.object(mujoco,'MjData',Data),patch.object(mujoco,'mj_forward',lambda *a:None),patch.object(mujoco,'mj_step',step),patch.object(c,'ConductanceNetwork',Net),patch.object(c,'ChannelSensoryEncoder',Encoder),patch.object(c,'observe_quantities',observe),patch.object(c,'resolve_sensory_reference',lambda *a:np.zeros(1)):
   fn=c.episode if legacy else derive(refresh,log,cur)[0]
   rec,tr,cmd,sp=fn(m,np.zeros(1),{}, {},built,manifest,{'lf':{'bristle_touch':[]}},'normal',.015,stride=1,full_detail_intervals=0,push=None,external_touch=False,evidence=ev,body_observer=None,body_dt_s=.0000125)
  return dict(log=log,currents=cur,inputs=inputs,body=body,cmd=cmd,sp=sp,ev=ev,seen=seen)
 def test_default_actual_derivation_equals_legacy_fixed_events(self):
  a=self.trial(10);b=self.trial(10,legacy=True)
  for key in ('cmd','sp','body','inputs'):np.testing.assert_array_equal(a[key],b[key])
  np.testing.assert_array_equal(a['ev']['decoded_next_controls'],b['ev']['decoded_next_controls']);self.assertTrue(a['cmd'][1:].any())
 def test_constant_input_no_added_drive_or_consumption(self):
  a=self.trial(10,True);b=self.trial(2,True)
  for key in ('cmd','sp','body','inputs'):np.testing.assert_array_equal(a[key],b[key])
  np.testing.assert_array_equal(a['ev']['decoded_next_controls'],b['ev']['decoded_next_controls']);np.testing.assert_array_equal(b['inputs'],np.ones((30,1)))
 def test_completed_body1ms_and_unchanged_command_hold(self):
  a=self.trial(2);np.testing.assert_allclose(a['seen'],np.arange(15)*.001,rtol=0,atol=1e-13);np.testing.assert_array_equal(np.array(a['inputs']).ravel(),np.repeat(np.arange(15)+1,2));np.testing.assert_array_equal(a['body'],np.repeat(a['cmd'],400,axis=0));self.assertEqual(len(a['inputs']),30)
if __name__=='__main__':unittest.main()
