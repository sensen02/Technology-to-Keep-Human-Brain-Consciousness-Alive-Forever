"""Parent actual zero-Core model isolation and causal released-frame ABI checks."""
import argparse,sys,ast
from runtime import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-resource-released',action='store_true');a=p.parse_args()
 if not a.parent_resource_released:print('CODE_ONLY no model load');return
 cpu1();frozen=verify();previous=sys.modules.get('model');visual,motor,vp,mp=load_policies()
 import numpy as np
 import torch
 camera=module('_visualbody_preflight_camera',VIS/'camera.py');checks={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('visualbody preflight '+k)
 check('basename_model_restored',sys.modules.get('model') is previous);check('no_physical_backend',not any(k in sys.modules for k in ('physics','mujoco','student_wave','flygym')))
 check('all_weights_frozen',all(not p.requires_grad for m in (visual,motor) for p in m.parameters()));check('288neural_and16cache',motor.identity()['state_dims']==[128,64,64] and visual.identity()['hidden']==32 and visual.identity()['ordered_feature']==16)
 with np.load(VIS/'frame_library_v1/RGB.npz',allow_pickle=False) as z:actual=[z['rgb'][i].copy() for i in (0,2,3)]
 q=camera.PendingVision();log=[]
 for k in range(17):
  t=k*.005
  if k%4==0:q.capture(t,actual[(k//4)%3])
  r=q.query(t)
  check('delay_valid'+str(k),r['valid']==(k>=4))
  if k>=4:check('delay_trueimage'+str(k),np.array_equal(r['image'],actual[((k//4)-1)%3]) and r['capture_time_s']<=t-.02+1e-10)
 with torch.no_grad():
  raw=torch.from_numpy(actual[0][None,None]);feat=visual.encode(raw)[0];repeat=visual.encode(raw)[0];h=visual.initial_state(1);turn,latent,hn=visual.step(feat,h);cmd=project(turn)
 check('actualbatch1CNNrepeat_exact',torch.equal(feat,repeat));check('feature_GRU_command_shapes',feat.shape==(1,16) and hn.shape==(1,32) and cmd.shape==(1,2));check('neural_projection',torch.equal(cmd[:,0],torch.relu(turn)/3) and torch.equal(cmd[:,1],turn))
 check('capturedvisibility_clock_boundaries',frame_target('V_left_pulse',.6)=='left' and frame_target('V_left_pulse',3.6)=='off' and frame_target('V_left_to_right',4.8)=='right')
 for name in ('runtime.py','runner.py','audit.py','summary.py','prepare.py','preflight.py','orchestrate.py'):ast.parse((ROOT/name).read_text())
 check('actual_executables_AST',True);verify();save(ROOT/'PREFLIGHT.json',dict(all_passed=True,checks=checks,Core_constructions=0,physical_steps=0,model_loads=2,training_updates=0,actual_frame_encoding_checks=True,sourcefreeze_sha256=sha(ROOT/'FROZEN.json')));print(json.dumps(dict(all_passed=True,checks=len(checks),Core_constructions=0)))
if __name__=='__main__':main()
