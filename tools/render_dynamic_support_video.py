"""Render saved trajectories to slow-motion comparison video.

Pure replay: qpos is read from BODY.npz and pushed through mj_forward only. No mj_step,
no network, no added animation, so every frame comes from the same simulation that was
scored. Optical-only repair makes all 71 body meshes visible; physics arrays are hashed
before and after to prove the model is otherwise untouched.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import sys,json,hashlib,argparse
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont

O=Path(os.environ.get('VIDEO_OUT','/home/sensen/Desktop/engine修复验收/s5-dynamic-support-video/20261103T023000Z'))
RUNS={
 'baseline_standard':dict(model=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z/RUN/MODEL.xml'),
  body=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z/RUN/BODY.npz'),
  label='BASELINE (no training)',t_release=1.3),
 'dynamic_training_best':dict(model=Path('/home/sensen/Desktop/engine修复验收/s5-dynamic-support-training/20261103T002606Z/VALIDATION_R1/CAND0_A_release200ms/RUN/MODEL.xml'),
  body=Path('/home/sensen/Desktop/engine修复验收/s5-dynamic-support-training/20261103T002606Z/VALIDATION_R1/CAND0_A_release200ms/RUN/BODY.npz'),
  label='DYNAMIC-SUPPORT TRAINED (best)',t_release=1.3),
}
W,H=640,480
FPS=25
WEIGHT=24.446354755878005
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';FONTB='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def load(model_path):
 from flygym.compose.fly.musculoskeletal import _load_mjcf
 model=_load_mjcf(str(model_path)).compile()
 physics_keys=[k for k in dir(model) if not k.startswith('_') and isinstance(getattr(model,k,None),np.ndarray)
               and not k.startswith(('vis','light','mat_','tex_')) and k!='geom_rgba']
 before={k:array_sha(getattr(model,k)) for k in physics_keys}
 meshids=np.flatnonzero(model.geom_type==mujoco.mjtGeom.mjGEOM_MESH)
 inv=[]
 for gid in meshids:
  inv.append(dict(geom=model.geom(int(gid)).name,body=model.body(int(model.geom_bodyid[gid])).name,old_rgba=model.geom_rgba[gid].copy().tolist()))
  model.geom_rgba[gid,3]=1.0
  inv[-1]['new_rgba']=model.geom_rgba[gid].tolist()
 assert len(meshids)==71 and np.all(model.geom_rgba[meshids,3]==1.0)
 after={k:array_sha(getattr(model,k)) for k in physics_keys}
 assert before==after,'optical repair changed physics arrays'
 model.vis.global_.offwidth=max(int(model.vis.global_.offwidth),W);model.vis.global_.offheight=max(int(model.vis.global_.offheight),H)
 return model,inv,dict(mesh_count=int(len(meshids)),visible=int(np.count_nonzero(model.geom_rgba[meshids,3]>0)),
  previously_invisible=int(sum(r['old_rgba'][3]==0 for r in inv)),physics_arrays_unchanged=True,geoms=inv)

def stats(body,t0,t1):
 z=body;t=z['time'];m=(t>=t0)&(t<=t1)
 foot=z['foot_force'][:,:,2][m].sum(1)/WEIGHT;nf=z['nonfoot_force'][:,2][m]
 total=foot+nf/WEIGHT
 return foot,nf,total

def draw(frame,lines,bar=None):
 base=Image.fromarray(frame).convert('RGBA')
 ov=Image.new('RGBA',base.size,(0,0,0,0));d=ImageDraw.Draw(ov)
 f=ImageFont.truetype(FONT,19);fb=ImageFont.truetype(FONTB,23)
 hdr=14+sum(30 if k=='title' else 25 for _,k in lines)+8
 d.rectangle([0,0,base.size[0],hdr],fill=(0,0,0,150))
 if bar:d.rectangle([0,base.size[1]-58,base.size[0],base.size[1]],fill=(0,0,0,150))
 im=Image.alpha_composite(base,ov);d=ImageDraw.Draw(im)
 y=10
 for txt,kind in lines:
  d.text((12,y),txt,font=(fb if kind=='title' else f),fill=(255,240,120) if kind=='title' else (238,238,238))
  y+=30 if kind=='title' else 25
 if bar:
  x0,y0,w,h,frac,lab,color=bar
  d.rectangle([x0,y0,x0+w,y0+h],outline=(210,210,210))
  d.rectangle([x0,y0,x0+int(w*max(0.,min(1.4,frac))/1.4),y0+h],fill=color)
  d.text((x0,y0-22),lab,font=f,fill=(238,238,238))
 return np.asarray(im.convert('RGB'))

CAMS={'A':dict(distance=7.0,lookat=1.05,elevation=-26.),'B':dict(distance=5.6,lookat=0.85,elevation=-24.)}
def render_one(key,t_common,camkey='A'):
 """One run at a time: a single GL context, closed before the next run starts."""
 v=RUNS[key];model,inv,invsum=load(v['model'])
 body=dict(np.load(v['body']))
 Path(O/f'FULLBODY_INVENTORY_{key}.json').write_text(json.dumps(invsum,ensure_ascii=False,indent=2)+'\n')
 ren=mujoco.Renderer(model,height=H,width=W)
 cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam)
 opt=mujoco.MjvOption();mujoco.mjv_defaultOption(opt);opt.sitegroup[:]=0
 for fl in ('mjVIS_TENDON','mjVIS_ACTUATOR','mjVIS_JOINT'):opt.flags[getattr(mujoco.mjtVisFlag,fl)]=0
 opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]=0;opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]=0
 d=mujoco.MjData(model)
 frames=[];replay_err=0.
 for t_raw in t_common:
  t=float(np.clip(t_raw,body['time'][0],body['time'][-1]))
  idx=int(np.searchsorted(body['time'],t))
  d.qpos[:]=body['qpos'][idx];d.act[:]=body['activation'][idx];d.qvel[:]=0.
  mujoco.mj_forward(model,d)
  # replay fidelity: the saved COM height must be reproduced from qpos alone
  thorax=model.body('Thorax').id if any(model.body(i).name=='Thorax' for i in range(model.nbody)) else 0
  replay_err=max(replay_err,abs(float(d.subtree_com[thorax,2])-float(body['com'][idx,2])))
  cp=CAMS[camkey];cam.lookat[:]=[0.,0.,cp['lookat']];cam.distance=cp['distance'];cam.azimuth=125.;cam.elevation=cp['elevation']
  ren.update_scene(d,camera=cam,scene_option=opt)
  foot=float(body['foot_force'][idx,:,2].sum()/WEIGHT);nf=float(body['nonfoot_force'][idx,2]/WEIGHT)
  frames.append((ren.render().copy(),t,foot,nf,float(body['com'][idx,2]),int((body['foot_force'][idx,:,2]>1e-9).sum())))
 ren.close()
 return frames,replay_err

def compose(frames_by_run,out_path,speed_label,prefix,title,hold=1):
 panels=[]
 for i in range(len(frames_by_run['baseline_standard'])):
  parts=[]
  for key in RUNS:
   img,t,foot,nf,com,bearing=frames_by_run[key][i]
   v=RUNS[key]
   phase='ALGORITHM ASSIST' if t<v['t_release'] else 'NEURAL ONLY (assist removed)'
   lines=[(v['label'],'title'),(f't={t:.4f}s   {speed_label}','n'),(phase,'n'),
          (f'foot {foot:.3f}W   non-foot {nf:.3f}W','n'),
          (f'feet down {bearing}/6   height {com:.3f}','n')]
   col=[(80,200,120) if foot>=0.8 else ((230,190,80) if foot>=0.4 else (225,90,90))]
   parts.append(draw(img,lines,bar=(30,H-40,W-60,13,foot,'foot-only support / weight  (target 0.8)',col[0])))
  panels.append(np.concatenate(parts,axis=0))
 panels=[f for f in panels for _ in range(hold)]
 vid=O/out_path
 imageio.mimsave(vid,panels,fps=FPS,quality=8)
 sel=[0,len(panels)//3,2*len(panels)//3,len(panels)-1]
 sheet=np.concatenate([np.concatenate([panels[i] for i in sel[:2]],axis=1),np.concatenate([panels[i] for i in sel[2:]],axis=1)],axis=0)
 imageio.imwrite(O/f'{prefix}_SHEET.png',sheet)
 print(json.dumps(dict(video=str(vid),frames=len(panels),fps=25,slow_factor=speed_label,title=title),ensure_ascii=False))
 return vid,len(panels)

CLIPS=[
 dict(name='C1_WHOLE_TRIAL',video='C1_WHOLE_TRIAL_16x_slow.mp4',t0=0.0,t1=1.5,step=0.0025,hold=1,cam='A',
      title='whole trial: algorithm assist then neural-only release'),
 dict(name='C2_RELEASE_80x',video='C2_RELEASE_80x_slow.mp4',t0=1.20,t1=1.40,step=0.0005,hold=1,cam='B',
      title='every native sample around the handover'),
 dict(name='C3_FIRST60MS_160x',video='C3_FIRST_60MS_160x_slow.mp4',t0=1.30,t1=1.36,step=0.0005,hold=2,cam='B',
      title='the first 60 ms of neural-only control, two frames per sample'),
]
def main():
 O.mkdir(parents=True,exist_ok=True)
 meta={}
 for c in CLIPS:
  step=c['step'];t=np.arange(c['t0'],c['t1']+step*0.5,step)
  slow=(1.0/FPS*c['hold'])/step
  lab=f'{step*1000:.2f} ms/frame x{c["hold"]} @ {FPS}fps = {slow:.0f}x slower'
  fr={};err={}
  for key in RUNS:fr[key],err[key]=render_one(key,t,c['cam'])
  vid,n=compose(fr,c['video'],lab,c['name'],c['title'],hold=c['hold'])
  meta[c['name']]=dict(video=str(vid),frames=n,fps=FPS,sim_span_s=[c['t0'],c['t1']],
   sim_step_s=step,frame_hold=c['hold'],slow_factor_vs_realtime=round(slow,1),
   replay_com_error=max(err.values()),definition='slow factor = (frame display time) / (simulation time advanced per frame)')
  print(json.dumps(meta[c['name']],ensure_ascii=False))
 meta['provenance']=dict(model_sha256={k:sha(v['model']) for k,v in RUNS.items()},
  body_sha256={k:sha(v['body']) for k,v in RUNS.items()},
  replay='qpos/act read from BODY.npz, mj_forward only, qvel zeroed because it is not drawn; no mj_step, no neural step, no added animation',
  optical_repair='all 71 mesh geoms forced opaque; physics arrays hashed identical before and after',
  phases='algorithm assist until 1.3 s then neural-only; labels state the phase per frame',
  honesty='not walking: after withdrawal the body is carried mostly by its abdomen (about 72-79 percent of ground reaction); foot support is 0.03-0.25 W of body weight',
  correction='the previous release labelled slow factors as simulation-ms-per-frame (250x/2500x), which actually correspond to only 4x and 80x slower than real time; this version states the real factor')
 for k in RUNS:
  pass
 (O/'VIDEO_META.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:meta[k] for k in meta if k!='provenance'},ensure_ascii=False,indent=1)[:900])
if __name__=='__main__':main()
