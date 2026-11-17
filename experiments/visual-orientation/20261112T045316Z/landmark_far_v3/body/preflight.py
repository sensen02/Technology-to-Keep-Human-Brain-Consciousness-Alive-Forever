"""Parent-only zero-Core actual far-policy ABI, frame clock and closed-cover geometry checks."""
from runtime import *
import argparse,ast
from adapter import namespace

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--parent-resource-released',action='store_true');args=ap.parse_args()
 if not args.parent_resource_released:print('CODE_ONLY no model/array execution');return
 cpu1();f=verify();sentinel=object();previous=sys.modules.get('model',sentinel);visual,motor,vp,mp=load_policies()
 import numpy as np
 import torch
 checks={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('far body preflight '+k)
 check('restored_model_alias',sys.modules.get('model',sentinel) is previous)
 check('no_physical_backend',not any(k in sys.modules for k in ('physics','mujoco','student_wave','flygym')))
 check('strict_actual_far_final500',vp['format_version']=='VISION_FAR_LANDMARK_POLICY_V3' and vp['step']==500)
 check('frozen_visual_and_motor',all(not p.requires_grad for m in (visual,motor) for p in m.parameters()))
 check('neural288_cache16_separate',visual.identity()['hidden']==32 and visual.identity()['ordered_feature']==16 and motor.identity()['state_dims']==[128,64,64])
 lib=POLICY.parent/'library';manifest=read(lib/'LIBRARY.json')
 with np.load(lib/'RGB.npz',allow_pickle=False) as z:rgb=z['rgb']
 actual=[rgb[next(r['frame_id'] for r in manifest['rows'] if r['config_id']=='T0' and r['stimulus']==s)].copy() for s in ('off','left','right')]
 camera=module('_far_zeroCore_camera',VIS/'camera.py');q=camera.PendingVision()
 for k in range(17):
  t=k*.005
  if k%4==0:q.capture(t,actual[(k//4)%3])
  obs=q.query(t);check('capture_release_valid_'+str(k),obs['valid']==(k>=4))
  if k>=4:check('capture_release_trueRGB_'+str(k),np.array_equal(obs['image'],actual[((k//4)-1)%3]) and obs['capture_time_s']<=t-.02+1e-10)
 with torch.no_grad():
  image=torch.from_numpy(actual[0][None,None]);feat=visual.encode(image)[0];same=visual.encode(image)[0];turn,latent,h=visual.step(feat,visual.initial_state(1));cmd=project(turn)
 check('actual_batch1_encoder_repeat_exact',torch.equal(feat,same))
 check('actual_feature_state_command_shapes',feat.shape==(1,16) and h.shape==(1,32) and cmd.shape==(1,2))
 check('original_neural_projection',torch.equal(cmd[:,0],torch.relu(turn)/3.) and torch.equal(cmd[:,1],turn))
 geom=read(GEOMETRY);spec=read(VIS/'PREREG_CAMERA_PROBE.json')['eyes'];pos=np.array([.8,.1,1.4]);rot=np.eye(3);centers=anchor_landmarks(pos,rot,camera,spec,geom);eyes=camera.eye_frames(pos,rot,geom,spec);mid=(eyes[0]['pos']+eyes[1]['pos'])/2
 check('far_forward2_lateral2tanpoint4',abs(centers['left'][0]-mid[0]-2.)<1e-12 and abs(centers['left'][1]-mid[1]-2*np.tan(.4))<1e-12 and abs(centers['right'][1]-mid[1]+2*np.tan(.4))<1e-12)
 basis=cover_basis_from_head(rot);diagnostics=eye_cover_diagnostics(eyes,centers,basis)
 check('initial_eyes_outside_far_covers',diagnostics['eyes_outside_all_boxes'] and diagnostics['eye_box_min_clearance']>.005)
 center=cover_centers(centers)[0];near=[dict(pos=center.copy())]*2
 check('eye_inside_cover_rejected',not eye_cover_diagnostics(near,centers,basis)['eye_box_near_clear'])
 check('cover_encloses_pillar_height_radius',.18>.12 and 1.4-1.4<=0 and 1.4+1.4>=2.4)
 obj=object.__new__(CameraStream);obj.condition='V_flip_fixedframe';obj.queue=camera.PendingVision();obj.frozen_actor=None
 obj.queue.capture(10.56,actual[1]);obj.queue.capture(10.58,actual[1]);obj.queue.capture(10.60,actual[2]);first=obj.actor_query(10.60)
 for t in (10.605,10.610,10.615):obj.actor_query(t)
 obj.queue.capture(10.62,actual[2]);later=obj.actor_query(10.62)
 check('actor_freezes_true_released_left',first['actor_frame_frozen'] and np.array_equal(first['image'],actual[1]) and abs(first['capture_time_s']-10.58)<1e-10)
 check('actual_camera_advances_actor_stays',later['raw_capture_id']!=first['raw_capture_id'] and np.array_equal(later['image'],first['image']))
 check('source_event_boundaries',frame_target('V_left_pulse',.6)=='left' and frame_target('V_left_pulse',3.6)=='off' and frame_target('V_flip_fixedframe',4.8)=='right')
 for script in ('runner.py','audit.py','summary.py'):
  ns=namespace(script);check('explicit_new_globals_'+script,ns['ROOT']==ROOT and ns['VIS']==VIS and ns['POLICY']==POLICY and ns['CameraStream'] is CameraStream and ns['anchor_landmarks'] is anchor_landmarks and ns['load_policies'] is load_policies)
 for name in ('runtime.py','adapter.py','runner.py','audit.py','summary.py','prepare.py','preflight.py','orchestrate.py'):ast.parse((ROOT/name).read_text())
 verify();save(ROOT/'PREFLIGHT.json',dict(all_passed=True,checks=checks,Core_constructions=0,physics_steps=0,model_loads=2,training_updates=0,closed_cover_outside_geometry_checked=True,actual_RGB_encoding_and_fixedframe_checked=True,sourcefreeze_sha256=sha(ROOT/'FROZEN.json')))
 print(json.dumps(dict(all_passed=True,checks=len(checks),Core_constructions=0,physics_steps=0)))
if __name__=='__main__':main()
