"""Latest080 actual saved physics replay. Visual-only opacity/camera changes; never integrate."""
from pathlib import Path
import os,json,hashlib,subprocess,time
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from flygym.compose.fly.musculoskeletal import _load_mjcf
S=Path('/home/sensen/Desktop/engine修复验收/s5-compatible-stance-handover/20261028T080000Z')
O=Path('/home/sensen/Desktop/engine修复验收/s5-latest-handover-visual/20261028T084000Z')
V=Path('/home/sensen/Desktop/S5_LATEST_PHYSICAL_WARMUP_NEURAL_RESULT.mp4')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else str(v))+'\n')
def main():
 assert not O.exists() and not V.exists();O.mkdir(parents=True);start=time.time();sources=[Path(__file__),S/'MODEL.xml',S/'BODY.npz',S/'HANDOVER_BEFORE.npz'];hashes={str(p):sha(p) for p in sources}
 dump(O/'PREREGISTRATION.json',dict(source_hashes=hashes,source_trial='080 actual candidate-pose handover, not058 baseline',physical_duration_s=1.45,handover_s=1.2,display_fps=30,display_frames=435,slowdown=10,scope='Offline saved-state replay, no body integration or network. Native71mesh opacity made1, fixed 3/4 main camera with top inset; camera tracks saved root, never translate body.',criteria='Every frame nearest saved state, all435 receipts verified, full decode, all71 mesh opaque,4 inspected keyframes; outcome remains FAILED neural support.'))
 z=dict(np.load(S/'BODY.npz'));t=z['time'];assert len(t)==1451;m=_load_mjcf(str(S/'MODEL.xml')).compile();d=mujoco.MjData(m);assert(m.nq,m.nv,m.nu,m.nmesh)==(49,48,90,71)
 mesh=np.flatnonzero(m.geom_type==mujoco.mjtGeom.mjGEOM_MESH);assert len(mesh)==71
 physics={name:np.array(getattr(m,name)).copy() for name in ['body_mass','body_inertia','body_pos','body_quat','geom_pos','geom_quat','geom_size','geom_friction','geom_solref','jnt_range','jnt_stiffness','dof_damping','actuator_gainprm','actuator_biasprm','actuator_dynprm','actuator_gear','tendon_range']}
 oldrgba=m.geom_rgba.copy();m.geom_rgba[mesh,3]=1;assert np.all(m.geom_rgba[mesh,3]==1)
 dump(O/'VISUAL_ONLY_LEDGER.json',dict(source_model_sha256=sha(S/'MODEL.xml'),mesh_count=71,mesh_geometries=[m.geom(int(g)).name for g in mesh],old_rgba=oldrgba[mesh],new_rgba=m.geom_rgba[mesh],physical_arrays_checked=list(physics),only_changes='71mesh alpha1, offscreen buffer dimensions; camera and annotations',physical_body_translation_added=False))
 m.vis.global_.offwidth=max(1024,m.vis.global_.offwidth);m.vis.global_.offheight=max(768,m.vis.global_.offheight)
 opt=mujoco.MjvOption();opt.geomgroup[:]=1;opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON]=False
 renderer=mujoco.Renderer(m,height=768,width=1024);small=mujoco.Renderer(m,height=280,width=280)
 cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.distance=9;cam.azimuth=135;cam.elevation=-30
 top=mujoco.MjvCamera();top.type=mujoco.mjtCamera.mjCAMERA_FREE;top.distance=8;top.azimuth=90;top.elevation=-90
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20);tiny=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
 targets=np.linspace(t[0],t[-1],435);indices=np.array([int(np.argmin(abs(t-a))) for a in targets]);byindex={int(i):int(j) for j,i in enumerate(indices)};assert len(byindex)==435
 receipts=[];sides=[];keyindices={0,1000,1200,1215};thorax=m.body('Thorax').id;writer=imageio.get_writer(V,fps=30,codec='libx264',quality=8,macro_block_size=1);counts=dict(model_load=1,forward=0,physics_steps=0,neural_steps=0,rendered_video_frames=0,keyframes=0)
 def draw(i):
  center=d.qpos[:3].copy();center[0]-=.35;cam.lookat[:]=center;top.lookat[:]=center
  renderer.update_scene(d,camera=cam,scene_option=opt);im=Image.fromarray(renderer.render().copy());small.update_scene(d,camera=top,scene_option=opt);im.paste(Image.fromarray(small.render().copy()),(734,111));dr=ImageDraw.Draw(im);assisted=t[i]<1.2-1e-9;color=(155,80,0) if assisted else (150,0,0)
  phase=('ALGORITHM ASSISTED: WALK' if t[i]<1-1e-9 else 'ALGORITHM ASSISTED: SETTLE / HOLD') if assisted else 'NEURAL ONLY: FAILED SUPPORT (no algorithm assistance)'
  dr.rectangle((0,0,1023,105),fill='white');dr.text((10,5),phase,font=font,fill=color);dr.text((10,38),f'Physical t={t[i]:.4f}s | 10x slow | handover=1.2000s',font=font,fill='black');dr.text((10,70),'Replay of actual saved080 physics; full71mesh body; camera tracking only',font=tiny,fill='black')
  dr.rectangle((734,111,1013,139),fill='white');dr.text((740,115),'TOP VIEW (same saved state)',font=tiny,fill='black');dr.rectangle((0,697,1023,767),fill='white');dr.text((10,701),f'Foot contacts {int(z["foot_contact"][i].sum())}/6 | nonfoot contacts {int(z["nonfoot_count"][i])}',font=font,fill='black');dr.text((10,735),'Actual warmup 0-1.2s; neural1.2-1.45s. Autonomous walking NOT achieved.',font=tiny,fill=color)
  return np.asarray(im)
 try:
  for i in range(len(t)):
   d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];d.act[:]=z['activation'][i];d.ctrl[:]=z['command'][i];d.time=float(t[i]);mujoco.mj_forward(m,d);counts['forward']+=1;np.testing.assert_allclose(d.xpos[thorax],z['thorax'][i],atol=1e-10,rtol=0)
   R=d.xmat[thorax].reshape(3,3);vals=[]
   for leg in ['LF','RF']:
    sign=1 if leg=='LF' else -1;vv=[sign*float((R.T@(d.xpos[b]-d.xpos[thorax]))[1]) for b in range(m.nbody) if m.body(b).name.startswith(leg) and any(s in m.body(b).name for s in ['Coxa','Trochanter','Femur','Tibia','Tarsus'])];vals.append(min(vv))
   sides.append(dict(saved_index=i,time=float(t[i]),front_leg_skeletal_side_min=vals))
   if i in byindex:
    frame=byindex[i];arr=draw(i);writer.append_data(arr);counts['rendered_video_frames']+=1;receipts.append(dict(frame=frame,saved_index=i,time=float(t[i]),phase='algorithm_assisted' if t[i]<1.2-1e-9 else 'neural_only',qpos_sha256=hashlib.sha256(z['qpos'][i].tobytes()).hexdigest(),qvel_sha256=hashlib.sha256(z['qvel'][i].tobytes()).hexdigest(),activation_sha256=hashlib.sha256(z['activation'][i].tobytes()).hexdigest()))
   if i in keyindices:imageio.imwrite(O/f'KEYFRAME_SAVED_{i:04d}.png',draw(i));counts['keyframes']+=1
 finally:writer.close();renderer.close();small.close()
 for name,value in physics.items():np.testing.assert_array_equal(value,np.array(getattr(m,name)))
 assert counts['rendered_video_frames']==435 and counts['forward']==1451
 for r in receipts:
  for key,field in [('qpos','qpos'),('qvel','qvel'),('activation','activation')]:assert r[field+'_sha256']==hashlib.sha256(z[key][r['saved_index']].tobytes()).hexdigest()
 probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration,size','-of','json',str(V)],capture_output=True,text=True);assert probe.returncode==0;info=json.loads(probe.stdout);assert int(info['streams'][0]['nb_frames'])==435
 decode=subprocess.run(['ffmpeg','-v','error','-i',str(V),'-f','null','-'],capture_output=True,text=True);assert decode.returncode==0 and not decode.stderr
 assert all(sha(p)==h for p,h in hashes.items());dump(O/'FRAME_RECEIPTS.json',receipts);dump(O/'FRONT_SKELETAL_SIDE_CHECK.json',dict(scope='All1451 saved frames, front leg segment body origins in thorax frame; not full mesh clearance certificate',states=sides,minimum_over_saved_states=np.min([a['front_leg_skeletal_side_min'] for a in sides],axis=0)))
 dump(O/'RENDER_EXECUTION.json',dict(counts=counts,exit_code=0,runtime_s=time.time()-start,video=str(V),video_sha256=sha(V),source_hashes=hashes,all71_meshes_opaque=True,physical_arrays_unchanged=True,all435_receipts_verified=True,full_decode_passed=True,probe=info,first_neural_nonfoot_s=.011,autonomous_walking_achieved=False))
 (O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(dict(video=str(V),counts=counts,images=[str(O/f'KEYFRAME_SAVED_{i:04d}.png') for i in sorted(keyindices)])),flush=True)
if __name__=='__main__':main()
