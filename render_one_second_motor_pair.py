"""Render only same-run saved state arrays, opaque full mesh visual derivative."""
from pathlib import Path
import os,json,hashlib
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco,imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import engine.canonical_physics as cp
D=Path('/home/sensen/Desktop/engine修复验收/s2-one-second-motor-pair/20261027T153000Z')
def main():
 arms=('normal','motor_cut');z={a:np.load(D/a/'BODY_FLOOR_VERIFIED.npz') for a in arms};m=cp.load_model();mesh=np.flatnonzero(m.geom_type==mujoco.mjtGeom.mjGEOM_MESH);assert len(mesh)==71
 before=m.geom_rgba.copy();m.geom_rgba[:,3]=1
 renderer=mujoco.Renderer(m,height=480,width=640);cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.lookat[:]=[0,0,1];cam.distance=12;cam.azimuth=120;cam.elevation=-25
 fig,axs=plt.subplots(2,2,figsize=(13,9))
 for a in arms:
  s=z[a];t=s['time'];axs[0,0].plot(t,s['thorax_position'][:,2],label=a);axs[0,1].plot(t,s['command'].sum(1),label=a);axs[1,0].plot(t,s['activation'].max(1),label=a);axs[1,1].plot(t,s['foot_normal_force'].sum(1),label=a)
 for ax,title in zip(axs.flat,['Thorax height / model units','Applied command sum / native floor retained','Maximum activation / NOT zeroed at cut','Mapped foot normal force / model units']):ax.set_title(title);ax.set_xlabel('seconds');ax.legend();ax.grid(alpha=.2)
 fig.suptitle('Fresh 1s physical pair; cut15ms; native floor retained; NOT walking');fig.tight_layout();fig.savefig(D/'COMPARISON.png',dpi=140);plt.close(fig)
 frames=[];receipt=[]
 for target in np.linspace(0,1,31):
  panels=[]
  for a in arms:
   s=z[a];i=int(np.argmin(abs(s['time']-target)));d=mujoco.MjData(m);d.qpos[:]=s['qpos'][i];d.qvel[:]=s['qvel'][i];d.act[:]=s['activation'][i];d.ctrl[:]=s['command'][i];mujoco.mj_forward(m,d)
   np.testing.assert_allclose(d.xpos[m.body('Thorax').id],s['thorax_position'][i],atol=1e-12)
   renderer.update_scene(d,camera=cam);panels.append(renderer.render().copy());receipt.append(dict(arm=a,index=i,physical_time=float(s['time'][i]),qpos_sha256=hashlib.sha256(s['qpos'][i].tobytes()).hexdigest()))
  image=Image.fromarray(np.concatenate(panels,axis=1));draw=ImageDraw.Draw(image);draw.rectangle((0,0,1280,50),fill='white');draw.text((10,10),f'Normal (left) | Motor output cut15ms (right) | actual t={target:.3f}s | 1s trajectory, no prewarm / NOT WALKING',fill='black');frames.append(np.asarray(image))
 renderer.close();imageio.mimsave(D/'PAIR.mp4',frames,fps=30,macro_block_size=1);imageio.imwrite(D/'VIDEO_MONTAGE.png',np.concatenate([frames[i] for i in (0,15,30)],axis=0))
 reader=imageio.get_reader(D/'PAIR.mp4');decoded=[f for f in reader];reader.close();assert len(decoded)==31
 errors=[float(np.mean(np.abs(f.astype(float)-g.astype(float)))) for f,g in zip(frames,decoded)];assert max(errors)<20
 out=dict(all_71_meshes_opaque=True,mesh_alpha_before=sorted(set(before[mesh,3].tolist())),xml_unchanged=True,physics_step=0,neural_step=0,render_forward=62,frame_receipt=receipt,video_decoded_frames=31,max_lossy_frame_mean_error=max(errors),source_only=True)
 (D/'RENDER_VERIFICATION.json').write_text(json.dumps(out,indent=2))
 for name,target in [('COMPARISON.png','S2_ONE_SECOND_PAIR.png'),('PAIR.mp4','S2_ONE_SECOND_PAIR.mp4')]:Path('/home/sensen/Desktop',target).write_bytes((D/name).read_bytes())
 print('render+video decode verification exit0')
if __name__=='__main__':main()
