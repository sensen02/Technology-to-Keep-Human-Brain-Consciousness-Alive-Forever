"""Render only saved states, with explicit physical prewarm/handover phases; never integrate."""
from pathlib import Path
import argparse, os, json, hashlib, subprocess
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from flygym.compose.fly.musculoskeletal import _load_mjcf

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('directory');ap.add_argument('--video',required=True);ap.add_argument('--model',default='MODEL.xml');ap.add_argument('--handoff',type=float);ap.add_argument('--slowdown',type=float,default=5);args=ap.parse_args()
 o=Path(args.directory);video=Path(args.video);assert not video.exists(), 'Do not overwrite a delivered movie'
 assert args.slowdown>0
 source=o/'BODY.npz';mp=o/args.model
 hashes={str(p):sha(p) for p in [source,mp,Path(__file__)]}
 with np.load(source) as f:z={k:f[k] for k in f.files}
 for target,legacy in [('thorax_position','thorax'),('com_position','com'),('nonfoot_floor_count','nonfoot_count')]:
  if target not in z and legacy in z:z[target]=z[legacy]
 t=z['time'];assert len(t)>1 and np.all(np.diff(t)>0)
 m=_load_mjcf(str(mp)).compile();d=mujoco.MjData(m);assert m.nq==49 and m.nv==48 and m.nu==90
 mesh=np.flatnonzero(m.geom_type==mujoco.mjtGeom.mjGEOM_MESH);assert len(mesh)==71
 m.geom_rgba[mesh,3]=1;m.vis.global_.offwidth=max(1024,m.vis.global_.offwidth);m.vis.global_.offheight=max(768,m.vis.global_.offheight)
 opt=mujoco.MjvOption();opt.geomgroup[:]=1;opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON]=False
 cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
 root=z['qpos'][:,:3];cam.lookat[:]=[(root[:,0].min()+root[:,0].max())/2,(root[:,1].min()+root[:,1].max())/2,1]
 cam.distance=9+float(np.max(np.ptp(root[:,:2],axis=0)));cam.azimuth=140;cam.elevation=-25
 renderer=mujoco.Renderer(m,height=768,width=1024)
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21)
 duration=float(t[-1]-t[0]);n=max(2,int(round(duration*args.slowdown*30)));receipts=[]
 writer=imageio.get_writer(video,fps=30,codec='libx264',quality=8,macro_block_size=1)
 targets=np.linspace(t[0],t[-1],n)
 sample_frames={0,n//4,n//2,n-1}
 if args.handoff is not None:sample_frames.add(int(np.argmin(abs(targets-args.handoff))))
 try:
  for frame,target in enumerate(targets):
   i=int(np.searchsorted(t,target));i=min(i,len(t)-1)
   if i>0 and abs(t[i-1]-target)<abs(t[i]-target):i-=1
   d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];d.act[:]=z['activation'][i];d.ctrl[:]=z['command'][i];d.time=float(t[i]);mujoco.mj_forward(m,d)
   if 'thorax_position' in z:np.testing.assert_allclose(d.xpos[m.body('Thorax').id],z['thorax_position'][i],rtol=0,atol=1e-10)
   renderer.update_scene(d,camera=cam,scene_option=opt);im=Image.fromarray(renderer.render().copy());dr=ImageDraw.Draw(im)
   assisted=args.handoff is None or d.time<args.handoff
   phase='ALGORITHM-ASSISTED PHYSICS | NOT AUTONOMOUS' if assisted else 'NEURAL ONLY | ALL ALGORITHM ASSISTANCE OFF'
   color=(155,80,0) if assisted else (0,70,155)
   dr.rectangle((0,0,1023,105),fill='white');dr.text((10,5),phase,font=font,fill=color)
   dr.text((10,39),f'Physical time {d.time:.4f} s | {args.slowdown:g}x SLOW | full body',font=font,fill='black')
   dr.text((10,73),'Actual saved physics states; no scripted body translation',font=font,fill='black')
   dr.rectangle((0,696,1023,767),fill='white')
   feet=int(z['foot_contact'][i].sum()) if 'foot_contact' in z else -1
   nf=int(z['nonfoot_floor_count'][i]) if 'nonfoot_floor_count' in z else -1
   dr.text((10,701),f'Feet contacting {feet}/6 | nonfoot contacts {nf}',font=font,fill='black')
   caption='Warmup prototype; walking NOT CERTIFIED' if args.handoff is None else f'Handover t={args.handoff:.4f} s; outcome verified separately'
   dr.text((10,734),caption,font=font,fill=color)
   arr=np.asarray(im);writer.append_data(arr)
   receipts.append(dict(frame=frame,saved_index=i,time=float(t[i]),phase='algorithm_assisted' if assisted else 'neural_only',qpos_sha256=hashlib.sha256(z['qpos'][i].tobytes()).hexdigest()))
   if frame in sample_frames:imageio.imwrite(o/f'WALK_VIDEO_FRAME_{frame:04d}.png',arr)
 finally:writer.close();renderer.close()
 probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration,size','-of','json',str(video)],capture_output=True,text=True);assert probe.returncode==0
 info=json.loads(probe.stdout);assert int(info['streams'][0]['nb_frames'])==n
 decode=subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],capture_output=True,text=True);assert decode.returncode==0 and not decode.stderr
 assert all(sha(p)==h for p,h in hashes.items())
 (o/'WALK_VIDEO_RECEIPTS.json').write_text(json.dumps(receipts,indent=2)+'\n')
 (o/'WALK_VIDEO_VERIFICATION.json').write_text(json.dumps(dict(video=str(video),video_sha256=sha(video),source_hashes=hashes,probe=info,frames=n,full_decode_passed=True,slowdown=args.slowdown,handoff=args.handoff,model_load=1,forward=n,physics_steps=0,neural_steps=0,all71_meshes_opaque=True,phase_labels_explicit=True),indent=2)+'\n')
 print(video,flush=True)
if __name__=='__main__':main()
