"""Render actual saved full-body states, without generating any motion."""
import os,sys,json,hashlib
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import numpy as np
import mujoco as mj
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from incremental_actuator import load_full_model

def main():
 source=Path(sys.argv[1]);out=Path(sys.argv[2]);title=sys.argv[3] if len(sys.argv)>3 else 'ANN + incremental actuator'
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 before=sha(source)
 with np.load(source) as z:b={k:z[k] for k in z.files}
 m=load_full_model();m.vis.global_.offwidth=max(m.vis.global_.offwidth,960);m.vis.global_.offheight=max(m.vis.global_.offheight,640)
 assert sum(m.geom_type==mj.mjtGeom.mjGEOM_MESH)==71
 renderer=mj.Renderer(m,height=640,width=960);d=mj.MjData(m);cam=mj.MjvCamera();mj.mjv_defaultCamera(cam);cam.distance=8;cam.azimuth=125;cam.elevation=-28
 opt=mj.MjvOption();mj.mjv_defaultOption(opt);opt.sitegroup[:]=0
 for name in ['mjVIS_TENDON','mjVIS_ACTUATOR','mjVIS_JOINT']:opt.flags[getattr(mj.mjtVisFlag,name)]=0
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21);fps=30;slow=4;mapping=[];maxerr=0.;sample_written=False
 times=np.arange(b['time'][0],b['time'][-1]+1e-9,1/(fps*slow))
 with imageio.get_writer(out,fps=fps,codec='libx264',quality=8,macro_block_size=16) as writer:
  for fi,t in enumerate(times):
   ix=int(np.argmin(abs(b['time']-t)));actual=float(b['time'][ix]);d.qpos[:]=b['qpos'][ix];d.qvel[:]=b['qvel'][ix];d.ctrl[:]=b['ctrl'][ix];mj.mj_forward(m,d)
   err=float(abs(d.subtree_com[m.body('Thorax').id]-b['com'][ix]).max());maxerr=max(maxerr,err)
   cam.lookat[:]=[d.qpos[0],d.qpos[1],.9];renderer.update_scene(d,camera=cam,scene_option=opt)
   im=Image.fromarray(renderer.render());dr=ImageDraw.Draw(im);dr.rectangle([0,0,960,100],fill=(18,18,18));dr.text((12,8),title,font=font,fill='white');phase='TEACHER WARMUP' if actual<1 else 'LEARNED ANN + INCREMENTAL ACTUATOR'
   dr.text((12,38),f't={actual:.3f}s | 4x slow | {phase}',font=font,fill=(255,220,90));dr.text((12,68),'Free body | normal gravity | no root support | not a walking claim',font=font,fill='white')
   frame=np.asarray(im);writer.append_data(frame);mapping.append(dict(frame=fi,requested_time=float(t),source_row=ix,actual_time=actual))
   if actual>=1 and not sample_written:imageio.imwrite(out.with_suffix('.png'),frame);sample_written=True
 renderer.close();assert sha(source)==before and maxerr<1e-12
 reader=imageio.get_reader(out);decoded=reader.count_frames();assert decoded==len(mapping);reader.get_data(decoded//2);reader.close()
 result=dict(importance='重要辅助',source=str(source),source_sha256=before,frames=decoded,fps=fps,slow_factor=slow,visible_meshes=71,source_unchanged=True,replay_com_max_error=maxerr,visual_inspection='not performed; model image capability must be checked separately',mapping=mapping)
 out.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='mapping'}))
if __name__=='__main__':main()
