"""Saved states only: full-body 40x slow comparison; never step/neural execution."""
from pathlib import Path
import os,json,hashlib,sys,time
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
import engine.canonical_physics as cp
P=Path(__file__).parent
O=Path('/home/sensen/Desktop/engine修复验收/s4-neural-gain-render/20261028T012000Z')
D=Path('/home/sensen/Desktop/S4_NEURAL_GAIN_FULLBODY_SLOW_COMPARISON.mp4')
H=Path('/home/sensen/Desktop/engine修复验收/s2-release-250ms/20261027T161000Z/normal/BODY.npz')
N=Path('/home/sensen/Desktop/engine修复验收/s4-dynamic-neural-gain/20261028T011000Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def main():
 O.mkdir(parents=True,exist_ok=False)
 paths=[H,N/'train_01/BODY.npz',N/'train_02/BODY.npz'];names=['Historical neural baseline','train_01 | best foot share / full score','train_02 | best shared dynamic score']
 src=paths+[Path(cp.CANONICAL_XML),Path(__file__)];hashes={str(p):sha(p) for p in src}
 dump(O/'PREREGISTRATION.json',dict(inputs_sha256=hashes,saved_only=True,frames=300,fps=30,slowdown=40,physical_duration_s=.25,selection='index round(frame*20000/299), original saved sample only; 250ms displayed over 10s, no repeats/end hold/interpolation',camera=dict(lookat=[0,0,1],distance=9,azimuth=120,elevation=-22),visual_only='all71mesh alpha=1; all geomgroups enabled; framebuffer720x600 minimum; nonmesh rgba unchanged; no XML edits',step=0,neural=0))
 z=[np.load(p) for p in paths];assert all(a['qpos'].shape==(20001,49) for a in z)
 m=cp.load_model();mesh=np.flatnonzero(m.geom_type==mujoco.mjtGeom.mjGEOM_MESH);assert len(mesh)==71
 before=m.geom_rgba.copy();m.geom_rgba[mesh,3]=1.;assert np.all(m.geom_rgba[mesh,3]==1)
 dump(O/'VISUAL_DIFF.json',dict(mesh_count=71,all_meshes_opaque=True,alpha_changed_geom_indices=np.flatnonzero(before[:,3]!=m.geom_rgba[:,3]).tolist(),physical_parameters_changed=False))
 m.vis.global_.offheight=max(600,m.vis.global_.offheight);m.vis.global_.offwidth=max(720,m.vis.global_.offwidth)
 option=mujoco.MjvOption();option.geomgroup[:]=1
 renderer=mujoco.Renderer(m,height=600,width=720);cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.lookat[:]=[0,0,1];cam.distance=9;cam.azimuth=120;cam.elevation=-22
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);big=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21)
 writer=imageio.get_writer(D,fps=30,codec='libx264',quality=8,macro_block_size=1);receipts=[];count=0;begin=time.time();samples=[]
 try:
  for f in range(300):
   idx=int(round(f*20000/299));panels=[]
   for j,a in enumerate(z):
    d=mujoco.MjData(m);d.qpos[:]=a['qpos'][idx];d.qvel[:]=a['qvel'][idx];d.act[:]=a['activation'][idx];d.ctrl[:]=a['command'][idx];d.time=float(a['time'][idx]);mujoco.mj_forward(m,d);count+=1
    np.testing.assert_allclose(d.xpos[m.body('Thorax').id],a['thorax_position'][idx],atol=1e-12,rtol=0)
    renderer.update_scene(d,camera=cam,scene_option=option);panels.append(renderer.render().copy())
    receipts.append(dict(frame=f,panel=j,saved_index=idx,physical_time_s=float(a['time'][idx]),qpos_sha256=hashlib.sha256(a['qpos'][idx].tobytes()).hexdigest()))
   im=Image.fromarray(np.concatenate(panels,axis=1));draw=ImageDraw.Draw(im)
   for j,a in enumerate(z):
    x=j*720;draw.rectangle((x,0,x+719,95),fill=(250,250,250));draw.text((x+10,5),names[j],font=big,fill='black');draw.text((x+10,34),f'Physical t={a["time"][idx]*1000:7.3f} ms | 40x SLOW',font=font,fill='black');draw.text((x+10,62),'Full body | neural closed loop | NOT STANDING',font=font,fill=(180,0,0))
    draw.rectangle((x,537,x+719,599),fill=(250,250,250));draw.text((x+10,542),f'COM height {a["com_position"][idx,2]:.4f} model units',font=font,fill='black');draw.text((x+10,570),f'Foot-contact legs {int(a["foot_contact"][idx].sum())}/6 | Nonfoot contacts {int(a["nonfoot_floor_count"][idx])}',font=font,fill=(180,0,0) if a['nonfoot_floor_count'][idx] else 'black')
   arr=np.asarray(im);writer.append_data(arr)
   if f in [0,45,90,299]:
    imageio.imwrite(O/f'FRAME_{f:03d}.png',arr);samples.append(arr)
 finally:writer.close();renderer.close()
 dump(O/'FRAME_RECEIPTS.json',receipts)
 assert all(sha(Path(k))==v for k,v in hashes.items())
 dump(O/'EXECUTION.json',dict(command=sys.executable+' '+str(Path(__file__)),exit_code=0,model_load=1,forward=count,body_step=0,neural_step=0,decoder_replay=0,frames=300,inputs_unchanged=True,elapsed_s=time.time()-begin,video=str(D)))
 (O/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 print('VIDEO',D,flush=True);print('Forward',count,'step0 neural0',flush=True)
if __name__=='__main__':main()
