"""ParentactualzeroCore sameABI plusfixed-actor/rawrelease control distinction."""
from adapter import namespace
from runtime import *
def extra():
 import numpy as np
 camera=module('_body2_control_preflight_camera',VIS/'camera.py')
 with np.load(VIS/'frame_library_v2/RGB.npz',allow_pickle=False) as z:left=z['rgb'][2].copy();right=z['rgb'][3].copy()
 obj=object.__new__(CameraStream);obj.condition='V_flip_fixedframe';obj.queue=camera.PendingVision();obj.frozen_actor=None
 obj.queue.capture(10.56,left);obj.queue.capture(10.58,left);obj.queue.capture(10.60,right);first=obj.actor_query(10.60)
 for t in (10.605,10.610,10.615):obj.actor_query(t)
 obj.queue.capture(10.62,right);later=obj.actor_query(10.62)
 checks=dict(control_freezes_genuine_releasedleft=first['actor_frame_frozen'] and np.array_equal(first['image'],left) and abs(first['capture_time_s']-10.58)<=1e-10,rawrelease_advanceswhileactorfixed=later['raw_capture_id']!=first['raw_capture_id'] and np.array_equal(later['image'],left) and abs(later['capture_time_s']-10.58)<=1e-10,control_scene_switchnotheadfollowing=frame_target('V_flip_fixedframe',4.8)=='right' and frame_target('V_left_occluded',8.)=='left')
 if not all(checks.values()):raise ValueError('newcontrolzeroCorepreflightfailed')
 return checks

def main():
 ns=namespace('preflight.py');original=ns['save']
 def save_with_controls(path,value):
  if Path(path).name=='PREFLIGHT.json':value['checks'].update(extra());value['all_passed']=all(value['checks'].values());value['controls_checked_zeroCore']=True
  original(path,value)
 ns['save']=save_with_controls;ns['main']()
if __name__=='__main__':main()
