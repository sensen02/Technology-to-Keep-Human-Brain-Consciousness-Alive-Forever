from scope import *
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
items=[];sources={};fig,axes=plt.subplots(3,2,figsize=(12,10))
for row,f in enumerate(('stop60','stop74a','stop74b')):
 panels=[]
 for group,variant,label in (('HELD',2,'simultaneous off'),('SEQUENCED',7,'brake off +200ms')):
  meta=json.loads((HERE/'COMMANDS'/f'{f}_{variant}'/'RESULT.json').read_text())
  with np.load(HERE/'COMMANDS'/f'{f}_{variant}'/'TRACE.npz',allow_pickle=False) as z:
   t=(np.arange(20000)+1)*DT-meta['off']*DT;axes[row,0].plot(t,z['command'][:,0],label=label)
  for mode,style in (('teacher','--'),('student','-')):
   folder=HERE/group/f'{f}_{variant}_{mode}';m=mujoco.MjModel.from_binary_path(str(folder/'BODY.mjb'));d=mujoco.MjData(m)
   with np.load(folder/'TRACE.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
   m.mat_reflectance[:]=0;renderer=mujoco.Renderer(m,height=240,width=400);panels.append((m,d,a,renderer,meta,f'{f} {label} {mode}'))
   start=meta['off'];pos=a['position'][start:,:2];distance=np.linalg.norm(pos-pos[0],axis=1);axes[row,1].plot(np.arange(len(pos))*DT,distance,style,label=f'{label} {mode}')
   for n in ('TRACE.npz','BODY.mjb'):sources[str(folder/n)]=sha(folder/n)
 items.append(panels);axes[row,0].set(title=f'{f}: actual postneck drive',xlim=(-.05,.45),ylabel='Left-side drive',xlabel='Time from forward-off (s)');axes[row,1].set(title=f'{f}: post-off displacement',ylabel='Distance (mm)',xlabel='Time from forward-off (s)');axes[row,1].axhline(.01,ls=':',color='k')
 for ax in axes[row]:ax.grid(alpha=.2);ax.legend(fontsize=6)
fig.tight_layout();fig.savefig(HERE/'RELEASE_SEQUENCE_METRICS.png',dpi=140);plt.close(fig)
writer=imageio.get_writer(str(HERE/'RELEASE_SEQUENCE.mp4'),fps=25,codec='libx264',quality=7,macro_block_size=16);count=0
try:
 for tick in range(0,20000,50):
  rows=[]
  for panels in items:
   columns=[]
   for m,d,a,renderer,meta,label in panels:
    d.qpos[:]=a['qpos'][tick];mujoco.mj_kinematics(m,d);cam=mujoco.MjvCamera();cam.lookat[:]=[*a['position'][tick,:2],.85];cam.distance=8;cam.azimuth=105;cam.elevation=-23
    renderer.update_scene(d,cam);im=Image.fromarray(renderer.render());dr=ImageDraw.Draw(im);dr.rectangle((0,0,400,48),fill=(20,20,20));dr.text((5,2),label,font=font,fill='white');dr.text((5,24),f't={(tick+1)*DT:.4f}s | .125x',font=font,fill='white');columns.append(np.asarray(im))
   rows.append(np.concatenate(columns,axis=1))
  writer.append_data(np.concatenate(rows,axis=0));count+=1
finally:
 writer.close()
 for panels in items:
  for m,d,a,r,meta,label in panels:r.close()
write(HERE/'SEQUENCE_MEDIA.json',dict(frames=count,fps=25,duration_s=16,playback=.125,width=1600,height=720,
 source_sha256=sources,video_sha256=sha(HERE/'RELEASE_SEQUENCE.mp4'),physical_steps=0,neural_ticks=0,
 note='Unmatched phase/timing examples compare profiles; not a one-variable matched-body causality proof. Only floor reflection removed, no pose/geometry change.'))
print('Rendered sequenced release comparison',count)
