"""Read saved physical states for actual fullbody evidence; never integrate motion."""
from pathlib import Path
import json
import numpy as np
import mujoco as mj
from PIL import Image,ImageDraw
from physics import Core
from identity import HERE,atomic_json,sha256


def main():
    root=HERE.parent
    cases=[('Half-student recovery ANN',root/'20261110T102045Z/EVAL_HALF_RECOVERY',7.3),
       ('Burst recovery ANN',root/'20261110T103413Z/EVAL_FF_BURST',7.3),
       ('Paired target-label ANN',root/'20261110T104706Z/EVAL_FF_FEEDBACK',7.3),
       ('Trainable inhibitory ANN',HERE/'EVAL_FF_INHIBITORY',7.3)]
    c=Core(adhesion=False);renderer=mj.Renderer(c.plant,height=480,width=640)
    cam=mj.MjvCamera();cam.type=mj.mjtCamera.mjCAMERA_FREE
    cam.distance=8.;cam.azimuth=140.;cam.elevation=-25.
    option=mj.MjvOption();option.flags[mj.mjtVisFlag.mjVIS_TENDON]=False
    sheet=Image.new('RGB',(1280,960),'white');evidence=[]
    for n,(label,directory,t) in enumerate(cases):
        path=directory/'trace_1ms.npz';report=json.loads((directory/'RESULT.json').read_text())
        with np.load(path,allow_pickle=False) as z:
            times=z['time'];i=int(np.searchsorted(times,t));q=z['qpos'][i];v=z['qvel'][i];com=z['com'][i];nf=float(z['nonfoot'][i])
        c.d.qpos[:]=q;c.d.qvel[:]=v;mj.mj_forward(c.plant,c.d);cam.lookat[:]=com
        renderer.update_scene(c.d,camera=cam,scene_option=option)
        image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image)
        draw.rectangle((0,0,640,59),fill='white')
        draw.text((5,4),label+' / ACTUAL SAVED PHYSICAL STATE',fill='black')
        draw.text((5,22),f't={times[i]:.3f}s; COM={com[2]:.3f}; nonfoot={nf:.3f} native',fill='black')
        draw.text((5,40),f'Whole autonomous-window gates: {report["autonomous"]["all_gates"]}; not final acceptance.',fill='black')
        sheet.paste(image,((n%2)*640,(n//2)*480))
        evidence.append(dict(label=label,trace=str(path),sha256=sha256(path),index=i,time_s=float(times[i]),
            whole_run_gates=report['autonomous']['all_gates']))
    renderer.close();sheet.save(HERE/'ROUND3_ACTUAL_STATES.png')
    atomic_json(HERE/'VISUAL_EVIDENCE.json',dict(evidence=evidence,body_meshes=71,interpretation='saved physical trajectory states only; render hides tendonoverlay but changes no physics'))
    print(str(HERE/'ROUND3_ACTUAL_STATES.png'))

if __name__=='__main__':main()
