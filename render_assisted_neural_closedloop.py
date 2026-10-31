"""Render the support-assisted closed-loop trial: full body, phases labelled."""
from pathlib import Path
import json,subprocess,hashlib
import numpy as np,mujoco,imageio
from PIL import Image,ImageDraw,ImageFont
from flygym.compose.fly.musculoskeletal import _load_mjcf
O=Path('/home/sensen/Desktop/engine修复验收/s5-assisted-neural-closedloop/20261028T104000Z')
V=Path('/home/sensen/Desktop/S5_ASSISTED_SUPPORT_NEURAL_CLOSEDLOOP.mp4')
LEGS=('LF','LM','LH','RF','RM','RH')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 src=O/'MODEL.xml';hashes={str(src):sha(src),str(O/'BODY.npz'):sha(O/'BODY.npz')}
 m=_load_mjcf(str(src)).compile()
 m.vis.global_.offwidth=1024;m.vis.global_.offheight=768
 d=mujoco.MjData(m);nm=m.nmesh;assert nm==71
 opt=mujoco.MjvOption()
 for flag in ('mjVIS_CONTACTPOINT','mjVIS_CONTACTFORCE','mjVIS_TENDON','mjVIS_SITE','mjVIS_JOINT','mjVIS_ACTUATOR','mjVIS_AUTOCONNECT'):
  try:opt.flags[getattr(mujoco.mjtVisFlag,flag)]=False
  except AttributeError:pass
 # Full-body visibility: make every fly geom opaque (visual only; physics arrays untouched).
 for g in range(m.ngeom):
  if m.geom_bodyid[g]!=0:m.geom_rgba[g,3]=1.0
 ren=mujoco.Renderer(m,height=768,width=1024);small=mujoco.Renderer(m,height=280,width=280)
 cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.distance=9;cam.azimuth=135;cam.elevation=-30
 top=mujoco.MjvCamera();top.type=mujoco.mjtCamera.mjCAMERA_FREE;top.distance=8;top.azimuth=90;top.elevation=-90
 try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20);tiny=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
 except Exception:font=tiny=ImageFont.load_default()
 z=np.load(O/'BODY.npz');t=z['time'];n=435;targets=np.linspace(t[0],t[-1],n);idx=np.array([int(np.argmin(abs(t-x))) for x in targets])
 thorax=m.body('Thorax').id
 visual_changes=dict(fly_geom_alpha=1.0,tendon_site_joint_hidden=True)
 counts=dict(model_load=1,forward=0,body_step=0,neural_step=0,rendered_frames=0);receipts=[]
 writer=imageio.get_writer(V,fps=30,codec='libx264',quality=8,macro_block_size=1)
 try:
  for k,i in enumerate(idx):
   d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];d.act[:]=z['activation'][i];d.ctrl[:]=z['command'][i];d.time=float(t[i])
   mujoco.mj_forward(m,d);counts['forward']+=1
   c=d.qpos[:3].copy();c[0]-=.35;cam.lookat[:]=c;top.lookat[:]=c
   ren.update_scene(d,camera=cam,scene_option=opt);im=Image.fromarray(ren.render().copy())
   small.update_scene(d,camera=top,scene_option=opt);im.paste(Image.fromarray(small.render().copy()),(734,111))
   dr=ImageDraw.Draw(im);warm=t[i]<1.2-1e-9
   title='ALGORITHM SUPPORT + LIVE NEURAL LOOP' if warm else 'ALGORITHM SUPPORT + LIVE NEURAL LOOP (post-handover)'
   dr.rectangle((0,0,1023,105),fill='white');dr.text((10,5),title,font=font,fill=(155,80,0))
   dr.text((10,38),f'Physical t={t[i]:.4f}s | 10x slow | handover=1.2000s',font=font,fill='black')
   dr.text((10,70),'Replay of actual saved physics; 71 meshes visible; camera tracking only',font=tiny,fill='black')
   dr.rectangle((734,111,1013,139),fill='white');dr.text((740,115),'TOP VIEW (same state)',font=tiny,fill='black')
   dr.rectangle((0,697,1023,767),fill='white')
   dr.text((10,701),f'Feet on ground {int(z["foot_contact"][i].sum())}/6 | nonfoot contacts {int(z["nonfoot_count"][i])} | leg assist {float(np.linalg.norm(z["assist"][i,6:])):.1f}',font=font,fill='black')
   dr.text((10,735),'HYBRID: the algorithm is holding the body up. Neural self-support NOT achieved.',font=tiny,fill=(150,0,0))
   writer.append_data(np.asarray(im));counts['rendered_frames']+=1
   receipts.append(dict(frame=k,saved_index=int(i),time=float(t[i]),phase='assisted_prewarm' if warm else 'assisted_neural_closedloop',
                        qpos_sha256=hashlib.sha256(z['qpos'][i].tobytes()).hexdigest(),activation_sha256=hashlib.sha256(z['activation'][i].tobytes()).hexdigest()))
 finally:writer.close();ren.close();small.close()
 assert counts['rendered_frames']==n
 for rcp in receipts:
  i=rcp['saved_index'];assert rcp['qpos_sha256']==hashlib.sha256(z['qpos'][i].tobytes()).hexdigest();assert rcp['activation_sha256']==hashlib.sha256(z['activation'][i].tobytes()).hexdigest()
 probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,width,height','-of','json',str(V)],capture_output=True,text=True);assert probe.returncode==0
 info=json.loads(probe.stdout);assert int(info['streams'][0]['nb_frames'])==n
 dec=subprocess.run(['ffmpeg','-v','error','-i',str(V),'-f','null','-'],capture_output=True,text=True);assert dec.returncode==0 and not dec.stderr
 assert all(sha(Path(p))==h for p,h in hashes.items())
 # representative stills
 for k in (0,180,360,434):
  (O/f'STILL_{k:04d}.png').write_bytes(subprocess.run(['ffmpeg','-v','error','-i',str(V),'-vf',f'select=eq(n\\,{k})','-vframes','1','-f','image2pipe','-vcodec','png','-'],capture_output=True).stdout)
 (O/'FRAME_RECEIPTS.json').write_text(json.dumps(receipts,indent=2)+'\n')
 (O/'RENDER_EXECUTION.json').write_text(json.dumps(dict(counts=counts,visual_changes=visual_changes,video=str(V),video_sha256=sha(V),ffprobe=info,note='Offline replay of saved physics: no mj_step, no new neural steps. Every fly geom is forced opaque and tendon/site/joint decorations are hidden; physical arrays are not modified.'),indent=2)+'\n')
 print('RENDER_COMPLETE',V,flush=True)
if __name__=='__main__':main()
