"""Parent actual neural/RGBqueue/sourcequeue preflight; zeroCore/zero render/zero physics."""
import argparse
from runtime import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-resource-released',action='store_true');a=p.parse_args()
 if not a.parent_resource_released:print('CODE_ONLY no neural execution');return
 cpu1();verify();sentinel=object();prior={k:sys.modules.get(k,sentinel) for k in ('model','common')};visual,bridge,motor,vp,bp,mp=load_policies()
 import torch,numpy as np
 checks={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('Visualneck0Core preflight '+k)
 check('all_model_common_aliases_restored',all(sys.modules.get(k,sentinel) is v for k,v in prior.items()));check('no_physical_backend',not any(k in sys.modules for k in ('physics','mujoco','student_wave','flygym')))
 check('actual_final300_bound',bp['step']==300 and bp['format_version']==FORMAT);check('all_three_models_frozen',all(not p.requires_grad for m in (visual,bridge,motor) for p in m.parameters()));check('state304_separatecacheandports',flat(motor.initial_state(1),visual.initial_state(1),torch.zeros((1,16),dtype=torch.float64)).shape==(1,304))
 cg=camera_runtime();check('camera_close_artifact_newROOT',cg['CameraStream'].close.__globals__['ROOT']==ROOT);check('camera_init_capture_helpers_newROOT',cg['_init'].__globals__['ROOT']==cg['_capture'].__globals__['ROOT']==ROOT);check('optical_proxy_globals_newROOT',cg['_OpticalProxy'].render_eye_pair.__globals__['ROOT']==ROOT)
 camera=module('_visualneck_preflight_pendingcamera',VIS/'camera.py')
 with np.load(V7/'body/trials/V_empty/capture_rgb.npz',allow_pickle=False) as z:rgb=z['rgb'][0].copy()
 with np.load(NEUTRAL/'neural_decisions.npz',allow_pickle=False) as z:x=z['X238'][0].copy()
 class Actor:pass
 actor=Actor();actor.queue=camera.PendingVision();actor.condition='V_empty';actor.frozen_actor=None;actor.actor_query=types.MethodType(cg['CameraStream'].actor_query,actor);actor.queue.capture(0.,rgb)
 for ms in (0,5,10,15):check('first_invalid_query_'+str(ms),not actor_observation(actor,ms*.001)['valid'])
 actor.queue.capture(.02,rgb);released=actor_observation(actor,.02);check('genuine_RGB_release_after20ms',released['valid'] and released['capture_time_s']==0 and released['release_time_s']==.02 and np.array_equal(released['image'],rgb))
 with torch.no_grad():
  features=visual.encode(torch.from_numpy(rgb[None,None]))[0];vraw,latent,vh=visual.step(features,visual.initial_state(1));check('frozen_actualRGB_16_32_scalar_shapes',features.shape==(1,16) and vh.shape==(1,32) and vraw.shape==(1,));check('bounded_continuous_visualscalar',bool(torch.isfinite(vraw).all()) and bool((vraw.abs()<=1).all()))
  dn,h=bridge.brain_step(torch.zeros((1,1),dtype=torch.float64),vraw.reshape(1,1),torch.zeros((1,16),dtype=torch.float64));turn=bridge.decoder(dn)[:,0];cmd=project(turn);mean,states=motor.step(torch.tensor(x[None],dtype=torch.float64),cmd,motor.initial_state(1));check('actual_neural_304_ABI',dn.shape==(1,2) and h.shape==(1,16) and mean.shape==(1,42) and flat(states,vh,h).shape==(1,304));check('motor_project_only_decoder',torch.equal(cmd[:,0:1],torch.relu(cmd[:,1:2])/3.))
 loop=BrainNeckLoop(bridge,'V_empty');first=loop.decision(x,torch.zeros((1,),dtype=torch.float64),0)
 for _ in range(5):loop.advance()
 second=loop.decision(x,torch.zeros((1,),dtype=torch.float64),5);check('actual_AN_source2ms_release5ms_consumption',second['AN_capture_step']==0 and second['AN_release_step']==2 and np.array_equal(first['AN_encoded'],second['AN_held']));check('warm_brain_DN_motorcmd_zero',np.max(abs(second['brain_hidden_after']))==np.max(abs(second['DN_encoded']))==np.max(abs(second['command']))==0)
 check('same_111mask',bridge.allowed_raw_indices.tolist()==list(range(42))+list(range(84,126))+list(range(168,195)));check('cut_camera_condition_only_scheduler',camera_condition('V_left_DN_cut')==camera_condition('V_left_AN_cut')=='V_left_pulse');check('source_start_final_unchanged',loop.transport.verify_final()['source_files_unchanged']);check('no_Core_or_renderer_constructed',not any(k in sys.modules for k in ('physics','mujoco','student_wave','flygym')));verify()
 save(ROOT/'PREFLIGHT.json',dict(all_passed=True,checks=checks,state_total=304,neural_records=3080,Core_constructions=0,render_calls=0,physics_steps=0,model_loads=3,actual_visual_scalar=float(vraw[0]),visual_checkpoint_sha256=VISION_SHA,brain_checkpoint_sha256=read(ROOT/'INPUT_BINDINGS.json')['brain_sha256'],neck_source_binding_sha256=BINDING_SHA,camera_close_output_root=str(ROOT),sourcefreeze_sha256=sha(ROOT/'FROZEN.json')));print(json.dumps(dict(all_passed=True,checks=len(checks),Core_constructions=0,physics_steps=0)))
if __name__=='__main__':main()
