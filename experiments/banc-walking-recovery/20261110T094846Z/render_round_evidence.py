"""Full-body actual teacher and neural comparison states; no physics step."""
import json
from pathlib import Path
import numpy as np
import mujoco as mj
from PIL import Image,ImageDraw
from physics import Core
from identity import HERE,atomic_json,sha256


def main():
    root=HERE.parent
    cases=[('Smooth physical teacher',root/'20261110T093347Z/early_unload/TRACE_1MS.npz',6.0),
           ('Normalized FF neural failure',root/'20261110T094554Z/EVAL_FF_NORM/trace_1ms.npz',7.3),
           ('Normalized GRU neural result',root/'20261110T094554Z/EVAL_GRU_NORM/trace_1ms.npz',7.3)]
    recovery=HERE/'EVAL_FF_RECOVERY/trace_1ms.npz'
    if recovery.exists():cases.append(('Recovery FF neural result',recovery,7.3))
    c=Core(adhesion=False);renderer=mj.Renderer(c.plant,height=480,width=640)
    camera=mj.MjvCamera();camera.type=mj.mjtCamera.mjCAMERA_FREE
    camera.distance=8.;camera.azimuth=140.;camera.elevation=-25.
    opt=mj.MjvOption();opt.flags[mj.mjtVisFlag.mjVIS_TENDON]=False
    sheet=Image.new('RGB',(1280,960),'white');evidence=[]
    for n,(label,path,t) in enumerate(cases):
        with np.load(path,allow_pickle=False) as z:
            times=z['time'];i=int(np.searchsorted(times,t));qpos=z['qpos'][i];qvel=z['qvel'][i];com=z['com'][i];nf=float(z['nonfoot'][i])
        c.d.qpos[:]=qpos;c.d.qvel[:]=qvel;mj.mj_forward(c.plant,c.d);camera.lookat[:]=com
        renderer.update_scene(c.d,camera=camera,scene_option=opt)
        img=Image.fromarray(renderer.render());draw=ImageDraw.Draw(img)
        draw.rectangle((0,0,640,58),fill='white')
        draw.text((5,4),label+' / ACTUAL SAVED STATE',fill='black')
        draw.text((5,22),f't={times[i]:.3f}s; COM={com[2]:.3f}; nonfoot={nf:.3f} native',fill='black')
        draw.text((5,40),'Teacher evidence is separate from neural autonomous success.',fill='black')
        sheet.paste(img,((n%2)*640,(n//2)*480))
        evidence.append(dict(label=label,path=str(path),sha256=sha256(path),sample=i,time_s=float(times[i])))
    renderer.close();output=HERE/'ROUND2_ACTUAL_STATES.png';sheet.save(output)
    atomic_json(HERE/'ROUND2_VISUAL_EVIDENCE.json',dict(output=str(output),evidence=evidence,
      note='saved actual full physical states, opaque71meshes; tendonoverlayhidden forrenderonly; not new sim or final successfulhandover'))
    print(str(output))

if __name__=='__main__':main()
