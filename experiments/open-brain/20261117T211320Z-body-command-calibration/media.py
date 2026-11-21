"""Side-by-side teacher/student rendering from the same recorded formal runs."""
from scope import *
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
families=('arc_right','stride_right','back','speed');font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
bodies=[];sources={};fig,axes=plt.subplots(4,4,figsize=(16,11))
for row,family in enumerate(families):
 pair=[]
 for mode in ('teacher','student'):
  folder=HERE/'HELD'/f'{family}_2_{mode}';m=mujoco.MjModel.from_binary_path(str(folder/'BODY.mjb'));d=mujoco.MjData(m)
  with np.load(folder/'TRACE.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
  m.mat_reflectance[:]=0;renderer=mujoco.Renderer(m,height=320,width=480);pair.append((m,d,a,renderer))
  for name in ('TRACE.npz','BODY.mjb','RESULT.json'):sources[str(folder/name)]=sha(folder/name)
  for j,t in enumerate((.90,1.85)):
   tick=round(t/DT)-1;d.qpos[:]=a['qpos'][tick];mujoco.mj_kinematics(m,d);cam=mujoco.MjvCamera();cam.lookat[:]=[a['position'][tick,0],a['position'][tick,1],.9];cam.distance=8.;cam.azimuth=105.;cam.elevation=-23.
   renderer.update_scene(d,cam);col=j*2+(mode=='student');axes[row,col].imshow(renderer.render());axes[row,col].axis('off');axes[row,col].set_title(f'{family} | {mode} | {t:.2f}s',fontsize=9)
 bodies.append(pair)
fig.suptitle('Formal unseen-phase paired physical runs. Native saved states; local CPG declared.')
fig.tight_layout();fig.savefig(HERE/'COMMAND_STATES.png',dpi=140);plt.close(fig)
video=HERE/'TEACHER_STUDENT_COMMANDS.mp4';writer=imageio.get_writer(str(video),fps=25,codec='libx264',quality=7,macro_block_size=16);count=0
try:
 for tick in range(0,20000,50):
  panels=[]
  for family,pair in zip(families,bodies):
   line=[]
   for mode,(m,d,a,renderer) in zip(('teacher','student'),pair):
    d.qpos[:]=a['qpos'][tick];mujoco.mj_kinematics(m,d);cam=mujoco.MjvCamera();cam.lookat[:]=[a['position'][tick,0],a['position'][tick,1],.9];cam.distance=8.;cam.azimuth=105.;cam.elevation=-23.
    renderer.update_scene(d,cam);im=Image.fromarray(renderer.render());dr=ImageDraw.Draw(im);dr.rectangle((0,0,480,57),fill=(20,20,20));dr.text((6,3),f'{family} | {mode}',font=font,fill='white');dr.text((6,27),f't={(tick+1)*DT:.4f}s | 0.125x',font=font,fill='white');line.append(np.asarray(im))
   panels.append(np.concatenate(line,axis=1))
  writer.append_data(np.concatenate(panels,axis=0));count+=1
finally:
 writer.close()
 for pair in bodies:
  for m,d,a,renderer in pair:renderer.close()
fig,axes=plt.subplots(2,2,figsize=(13,9))
for family in FAMILIES+('a01_left','a01_right'):
 group='HELD_A01' if family.startswith('a01') else 'HELD'
 for mode,style in (('teacher','--'),('student','-')):
  folder=HERE/group/f'{family}_2_{mode}'
  with np.load(folder/'TRACE.npz',allow_pickle=False) as z:
   t=(np.arange(len(z['position']))+1)*DT
   if family.startswith(('arc','a01')):axes[0,0].plot(t,np.rad2deg(np.unwrap(z['yaw'])),style,label=f'{family} {mode}')
   elif family.startswith('stride'):
    result=json.loads((folder/'RESULT.json').read_text());axes[0,1].plot(np.arange(6),result['stride_by_leg'],style,marker='o',label=f'{family} {mode}')
   elif family=='back':axes[1,0].plot(t,z['position'][:,0],style,label=mode)
   else:
    pos=z['position'][:,:2];yaw=z['yaw'];v=np.sum(np.diff(pos,axis=0)*np.c_[np.cos(yaw[:-1]),np.sin(yaw[:-1])],axis=1)/DT
    smooth=np.convolve(v,np.ones(501)/501,mode='valid');axes[1,1].plot(t[250:-251],smooth,style,label=mode)
axes[0,0].set(ylabel='Yaw (degrees)',xlabel='Time (s)');axes[0,0].legend(fontsize=6,ncol=2)
axes[0,1].set(ylabel='Median cycle fore-aft excursion (mm)',xlabel='Leg: LF LM LH RF RM RH');axes[0,1].legend(fontsize=7)
axes[1,0].set(ylabel='Thorax x: backward motion (mm)',xlabel='Time (s)');axes[1,0].legend()
axes[1,1].set(ylabel='50ms mean signed speed (mm/s)',xlabel='Time (s)');axes[1,1].legend()
for ax in axes.flat:ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(HERE/'COMMAND_METRICS.png',dpi=150);plt.close(fig)
write(HERE/'MEDIA.json',dict(video=str(video),sha256=sha(video),frames=count,fps=25,expected_duration_s=16.,playback=.125,
 source_sha256=sources,physical_steps=0,neural_ticks=0,render_only_change='floor reflectance zero; geometry and recorded poses unchanged'))
print('Rendered',count,'frames paired formal body runs')
