"""Saved-only full-body slow video of the one integrated repaired rollout."""
from pathlib import Path
import os,json,hashlib,subprocess
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from flygym.compose.fly.musculoskeletal import _load_mjcf
from run_one_second_motor_pair import dump,sha
O=Path('/home/sensen/Desktop/engine修复验收/s4-integrated-repaired-closure/20261028T032000Z')
D=Path('/home/sensen/Desktop/S4_INTEGRATED_REPAIRS_FULLBODY_SLOW.mp4')
def main():
 archive=np.load(O/'BODY.npz');needed=['time','qpos','qvel','activation','command','thorax_position','com_position','foot_contact','nonfoot_floor_count'];z={k:archive[k] for k in needed};archive.close();model=O/'INTEGRATED_MODEL.xml';hashes={str(p):sha(p) for p in [model,O/'BODY.npz',Path(__file__)]};m=_load_mjcf(str(model)).compile();mesh=np.flatnonzero(m.geom_type==mujoco.mjtGeom.mjGEOM_MESH);assert len(mesh)==71;m.geom_rgba[mesh,3]=1;m.vis.global_.offwidth=max(1024,m.vis.global_.offwidth);m.vis.global_.offheight=max(768,m.vis.global_.offheight)
 opt=mujoco.MjvOption();opt.geomgroup[:]=1;opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON]=False
 renderer=mujoco.Renderer(m,height=768,width=1024);cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
 # Fixed world camera fitted to root excursion with a full-fly margin, never pose tracking.
 root=z['qpos'][:,:3];cam.lookat[:]=[(root[:,0].min()+root[:,0].max())/2,(root[:,1].min()+root[:,1].max())/2,1.];cam.distance=max(9.,9.+float(np.ptp(root[:,:2],axis=0).max()));cam.azimuth=120;cam.elevation=-22
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21);big=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',25)
 duration=float(z['time'][-1]);frames=max(2,int(round(duration*20*30)));receipts=[];snapshots=[]
 writer=imageio.get_writer(D,fps=30,codec='libx264',quality=8,macro_block_size=1)
 try:
  for f,target in enumerate(np.linspace(0,duration,frames)):
   i=int(np.argmin(abs(z['time']-target)));d=mujoco.MjData(m);d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];d.act[:]=z['activation'][i];d.ctrl[:]=z['command'][i];d.time=float(z['time'][i]);mujoco.mj_forward(m,d);np.testing.assert_allclose(d.xpos[m.body('Thorax').id],z['thorax_position'][i],rtol=0,atol=1e-12)
   renderer.update_scene(d,camera=cam,scene_option=opt);im=Image.fromarray(renderer.render().copy());draw=ImageDraw.Draw(im);draw.rectangle((0,0,1023,99),fill=(250,250,250));draw.text((12,5),'UPDATED COMPOSITE | LIVE NEURAL CLOSED LOOP',font=big,fill='black');draw.text((12,39),f'Physical t={d.time*1000:8.3f} ms | 20x SLOW | full body',font=font,fill='black');draw.text((12,70),'No external support / scripted control / pose correction',font=font,fill=(170,0,0))
   draw.rectangle((0,696,1023,767),fill=(250,250,250));draw.text((12,701),f'COM height {z["com_position"][i,2]:.4f} | feet contacting {int(z["foot_contact"][i].sum())}/6',font=font,fill='black');draw.text((12,734),f'Nonfoot contacts {int(z["nonfoot_floor_count"][i])} | support/standing NOT CERTIFIED',font=font,fill=(170,0,0));arr=np.asarray(im);writer.append_data(arr)
   receipts.append(dict(frame=f,saved_index=i,physical_time_s=float(z['time'][i]),qpos_sha256=hashlib.sha256(z['qpos'][i].tobytes()).hexdigest()))
   if f in [0,frames//10,frames//2,frames-1]:imageio.imwrite(O/f'VIDEO_FRAME_{f:03d}.png',arr);snapshots.append(f)
 finally:writer.close();renderer.close()
 dump(O/'VIDEO_FRAME_RECEIPTS.json',receipts)
 probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration,size','-of','json',str(D)],capture_output=True,text=True);assert probe.returncode==0;info=json.loads(probe.stdout);stream=info['streams'][0];assert int(stream['nb_frames'])==frames and stream['codec_name']=='h264'
 decode=subprocess.run(['ffmpeg','-v','error','-i',str(D),'-f','null','-'],capture_output=True,text=True);assert decode.returncode==0 and not decode.stderr
 assert all(sha(Path(k))==v for k,v in hashes.items())
 dump(O/'VIDEO_VERIFICATION.json',dict(video=str(D),sha256=sha(D),probe=info,all_frames_decoded=True,frames=frames,fps=30,slowdown=20,physical_duration_s=duration,model_load=1,forward=frames,body_step=0,neural_step=0,mesh_count=71,all_meshes_opaque=True,geomgroup_all_visible=True,tendon_overlay_hidden=True,camera=dict(lookat=cam.lookat.copy(),distance=cam.distance,azimuth=cam.azimuth,elevation=cam.elevation),source_hashes=hashes,snapshots=snapshots,exit_code=0));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print('VIDEO',D,flush=True)
if __name__=='__main__':main()
