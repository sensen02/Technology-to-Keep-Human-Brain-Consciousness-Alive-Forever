from scope import *
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
items=[];sources={};fig,axes=plt.subplots(5,4,figsize=(16,13))
for row,f in enumerate(FAMILIES):
 pair=[]
 for mode in ('teacher','student'):
  folder=HERE/'HELD'/f'{f}_2_{mode}';m=mujoco.MjModel.from_binary_path(str(folder/'BODY.mjb'));d=mujoco.MjData(m)
  with np.load(folder/'TRACE.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
  r=json.loads((folder/'RESULT.json').read_text());m.mat_reflectance[:]=0;renderer=mujoco.Renderer(m,height=240,width=400);pair.append((m,d,a,renderer,r))
  for n in ('TRACE.npz','BODY.mjb','RESULT.json'):sources[str(folder/n)]=sha(folder/n)
  for col,t in enumerate((.45,1.30)):
   i=round(t/DT)-1;d.qpos[:]=a['qpos'][i];mujoco.mj_kinematics(m,d);cam=mujoco.MjvCamera();cam.lookat[:]=[*a['position'][i,:2],.85];cam.distance=8.;cam.azimuth=105.;cam.elevation=-23.
   renderer.update_scene(d,cam);j=col*2+(mode=='student');axes[row,j].imshow(renderer.render());axes[row,j].axis('off');axes[row,j].set_title(f'{f} {mode} t={t:.2f}s',fontsize=9)
 items.append(pair)
fig.suptitle('Formal paired physical runs: finite-angle pivot and active braking. Saved native poses.')
fig.tight_layout();fig.savefig(HERE/'PIVOT_STOP_STATES.png',dpi=130);plt.close(fig)
video=HERE/'PIVOT_STOP.mp4';writer=imageio.get_writer(str(video),fps=25,codec='libx264',quality=7,macro_block_size=16);count=0
try:
 for i in range(0,20000,50):
  rows=[]
  for f,pair in zip(FAMILIES,items):
   columns=[]
   for mode,(m,d,a,renderer,r) in zip(('teacher','student'),pair):
    d.qpos[:]=a['qpos'][i];mujoco.mj_kinematics(m,d);cam=mujoco.MjvCamera();cam.lookat[:]=[*a['position'][i,:2],.85];cam.distance=8.;cam.azimuth=105.;cam.elevation=-23.
    renderer.update_scene(d,cam);im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,400,52),fill=(20,20,20));draw.text((5,2),f'{f} | {mode} | absolute {r["absolute_pass"]}',font=font,fill='white');draw.text((5,25),f't={(i+1)*DT:.4f}s  playback0.125x',font=font,fill='white');columns.append(np.asarray(im))
   rows.append(np.concatenate(columns,axis=1))
  writer.append_data(np.concatenate(rows,axis=0));count+=1
finally:
 writer.close()
 for pair in items:
  for m,d,a,renderer,r in pair:renderer.close()
fig,axes=plt.subplots(2,2,figsize=(13,9))
for f in FAMILIES:
 for mode,style in (('teacher','--'),('student','-')):
  folder=HERE/'HELD'/f'{f}_2_{mode}'
  with np.load(folder/'TRACE.npz',allow_pickle=False) as z:
   t=(np.arange(len(z['qpos']))+1)*DT
   if f.startswith('pivot'):
    axes[0,0].plot(t,np.rad2deg(np.unwrap(z['yaw'])),style,label=f'{f} {mode}')
    axes[0,1].plot(t,np.linalg.norm(z['position'][:,:2]-z['initial_position'][:2],axis=1),style,label=f'{f} {mode}')
   else:
    speed=np.linalg.norm(np.diff(z['position'][:,:2],axis=0)/DT,axis=1);smooth=np.convolve(speed,np.ones(501)/501,mode='valid')
    axes[1,0].plot(t[250:-251],smooth,style,label=f'{f} {mode}')
    axes[1,1].plot(t,np.cumsum(z['normal_load'].sum(1))*0+z['position'][:,2],style,label=f'{f} {mode}')
axes[0,0].set(ylabel='Completed physical yaw (degrees)',xlabel='Time (s)');axes[0,0].legend(fontsize=7)
axes[0,1].set(ylabel='XY distance from start (mm)',xlabel='Time (s)');axes[0,1].axhline(2.25,ls=':',color='k')
axes[1,0].set(ylabel='50ms mean horizontal speed (mm/s)',xlabel='Time (s)');axes[1,0].axvline(.834,color='k',ls=':');axes[1,0].legend(fontsize=7)
axes[1,1].set(ylabel='Thorax height: active braking (mm)',xlabel='Time (s)')
for ax in axes.flat:ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(HERE/'PIVOT_STOP_METRICS.png',dpi=140);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,7))
for f in ('pivot_left','pivot_right'):
 for mode,style in (('teacher','--'),('student','-')):
  with np.load(HERE/'DEV'/f'{f}_0_{mode}'/'TRACE.npz',allow_pickle=False) as a:
   xy=a['position'][:,:2]-a['initial_position'][:2];ax.plot(xy[:,0],xy[:,1],style,label=f'{f} high-rate {mode}')
ax.add_patch(plt.Circle((0,0),2.25,fill=False,ls=':',color='k'));ax.set(xlabel='X from initial (mm)',ylabel='Y from initial (mm)',title='Retained long-pivot failures: same teacher/student translation limitation');ax.axis('equal');ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(HERE/'LONG_PIVOT_LIMITATION.png',dpi=140);plt.close(fig)
write(HERE/'MEDIA.json',dict(video=str(video),sha256=sha(video),frames=count,fps=25,duration_s=16,playback=.125,source_sha256=sources,physical_steps=0,neural_ticks=0,render_only_change='floor reflectance zero, no poses/geometry modified'))
print('rendered',count,'same-run paired frames')
