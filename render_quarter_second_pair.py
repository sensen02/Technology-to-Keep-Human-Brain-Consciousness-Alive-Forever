"""Saved-only full-body release response figures/video; no physical step."""
from pathlib import Path
import os,json,hashlib
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_quarter_second_pair import OUT
from run_one_second_motor_pair import dump,sha
import engine.canonical_physics as cp

def main():
 status=json.loads((OUT/'STATUS.json').read_text());assert all(s['error'] is None and s['prefix_exact'] for s in status.values())
 arms=('normal','motor_cut');z={a:np.load(OUT/a/'BODY.npz') for a in arms};summary={};fig,axs=plt.subplots(3,2,figsize=(14,12))
 for a in arms:
  s=z[a];t=s['time'];vz=s['com_velocity'][:,2];non=s['nonfoot_floor_count'];nz=np.flatnonzero(non>0);last=t>=.20
  summary[a]=dict(first_nonfoot_floor_s=None if not len(nz) else float(t[nz[0]]),height_initial=float(s['thorax_position'][0,2]),height_min=float(s['thorax_position'][:,2].min()),height_final=float(s['thorax_position'][-1,2]),com_height_initial=float(s['com_position'][0,2]),com_height_min=float(s['com_position'][:,2].min()),com_height_final=float(s['com_position'][-1,2]),com_vz_final=float(vz[-1]),last50ms_com_vz_minmax=[float(vz[last].min()),float(vz[last].max())],last50ms_nonfoot_samples=int((non[last]>0).sum()),last50ms_samples=int(last.sum()),last50ms_foot_contact_leg_minmax=[int(s['foot_contact'][last].sum(1).min()),int(s['foot_contact'][last].sum(1).max())],max_activation=float(s['activation'].max()),warnings=int(s['warnings'].max()),stability='NOT_JUDGED',walking='NOT_ESTABLISHED')
  axs[0,0].plot(t,s['thorax_position'][:,2],label=a);axs[0,1].plot(t,s['com_position'][:,2],label=a);axs[1,0].plot(t,vz,label=a);axs[1,1].plot(t,2*np.arccos(np.clip(abs(s['qpos'][:,3]),0,1)),label=a);axs[2,0].plot(t,s['foot_contact'].sum(1),label=a+' foot legs');axs[2,0].plot(t,non,label=a+' nonfoot contacts',alpha=.65);axs[2,1].plot(t,s['ground_force_world'][:,2],label=a)
 for ax,title in zip(axs.flat,['Thorax height / model units','COM height / model units','COM vertical speed / model units/s','Root quaternion rotation from identity / rad','True floor contacts (not support certification)','World ground vertical force / model units']):ax.set_title(title);ax.set_xlabel('seconds');ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.suptitle('250ms fresh closure: release, sinking and ground contacts / NOT walking');fig.tight_layout();fig.savefig(OUT/'RELEASE_COMPARISON.png',dpi=140);plt.close(fig)
 m=cp.load_model();mesh=np.flatnonzero(m.geom_type==mujoco.mjtGeom.mjGEOM_MESH);assert len(mesh)==71;m.geom_rgba[:,3]=1
 renderer=mujoco.Renderer(m,height=480,width=640);cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.lookat[:]=[0,0,1];cam.distance=12;cam.azimuth=120;cam.elevation=-25;frames=[];receipts=[]
 for target in np.linspace(0,.25,31):
  panels=[]
  for a in arms:
   s=z[a];i=int(np.argmin(abs(s['time']-target)));d=mujoco.MjData(m);d.qpos[:]=s['qpos'][i];d.qvel[:]=s['qvel'][i];d.act[:]=s['activation'][i];d.ctrl[:]=s['command'][i];mujoco.mj_forward(m,d);np.testing.assert_allclose(d.xpos[m.body('Thorax').id],s['thorax_position'][i],atol=1e-12)
   renderer.update_scene(d,camera=cam);panels.append(renderer.render().copy());receipts.append(dict(arm=a,index=i,time=float(s['time'][i]),qpos_sha256=hashlib.sha256(s['qpos'][i].tobytes()).hexdigest()))
  image=Image.fromarray(np.concatenate(panels,axis=1));dr=ImageDraw.Draw(image);dr.rectangle((0,0,1280,44),fill='white');dr.text((8,10),f'Normal left / cut15ms right | physical t~{target:.4f}s | 250ms saved trajectory, 4x slow | NOT WALKING',fill='black');frames.append(np.asarray(image))
 renderer.close();imageio.mimsave(OUT/'RELEASE_PAIR.mp4',frames,fps=30,macro_block_size=1);reader=imageio.get_reader(OUT/'RELEASE_PAIR.mp4');decoded=[f for f in reader];reader.close();assert len(decoded)==31;errors=[float(np.mean(abs(a.astype(float)-b.astype(float)))) for a,b in zip(frames,decoded)];assert max(errors)<20
 imageio.imwrite(OUT/'VIDEO_MONTAGE.png',np.concatenate([frames[i] for i in (0,6,18,30)],axis=0));dump(OUT/'RENDER_CHECK.json',dict(mesh_count=71,opaque_meshes=True,physics_step=0,neural_step=0,forward=62,decoded_frames=31,max_lossy_error=max(errors),receipts=receipts,model_xml_unchanged=True));dump(OUT/'RESPONSE_SUMMARY.json',summary)
 for name,target in [('RELEASE_COMPARISON.png','S2_RELEASE_250MS.png'),('RELEASE_PAIR.mp4','S2_RELEASE_250MS.mp4')]:Path('/home/sensen/Desktop',target).write_bytes((OUT/name).read_bytes())
 (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(summary));print('render/saved response exit0')
if __name__=='__main__':main()
