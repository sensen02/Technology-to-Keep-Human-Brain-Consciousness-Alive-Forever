"""Decoded video and failed-handover state evidence. No physical integration."""
from pathlib import Path
import json
import numpy as np
import mujoco as mj
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from physics import Core
from identity import HERE,atomic_json,sha256


def main():
    video=HERE/'TEACHER_VIDEO/ALGORITHM_TEACHER_NOT_NEURAL.mp4'
    reader=imageio.get_reader(video)
    sheet=Image.new('RGB',(1280,1440),'white')
    chosen=[20,60,100,140,180,220]
    for n,index in enumerate(chosen):
        frame=Image.fromarray(reader.get_data(index)).crop((0,0,640,480))
        sheet.paste(frame,((n%2)*640,(n//2)*480))
    reader.close();sheet.save(HERE/'TEACHER_VIDEO/decoded_six_phase_sheet.png')
    c=Core(adhesion=False);renderer=mj.Renderer(c.plant,height=480,width=640)
    camera=mj.MjvCamera();camera.type=mj.mjtCamera.mjCAMERA_FREE
    camera.distance=8.;camera.azimuth=140.;camera.elevation=-25.
    opt=mj.MjvOption();opt.flags[mj.mjtVisFlag.mjVIS_TENDON]=False
    evidence=[]
    for folder in ['EVAL_FF238_HANDOVER','EVAL_GRU238_HANDOVER']:
        p=HERE/folder/'trace_1ms.npz'
        with np.load(p,allow_pickle=False) as z:
            # Show actual neural-stage failure after 1s settle+4.8s teacher prefix.
            index=int(np.searchsorted(z['time'],7.0))
            c.d.qpos[:]=z['qpos'][index];c.d.qvel[:]=z['qvel'][index]
            mj.mj_forward(c.plant,c.d);camera.lookat[:]=z['com'][index]
            renderer.update_scene(c.d,camera=camera,scene_option=opt)
            image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image)
            draw.rectangle((0,0,640,61),fill='white')
            draw.text((5,4),folder+' / ACTUAL FAILURE STATE',fill='black')
            draw.text((5,22),f't={z["time"][index]:.3f}s; neural stage starts5.8s; source={int(z["control_source"][index])}',fill='black')
            draw.text((5,40),f'COM={z["com"][index,2]:.3f}, nonfoot={z["nonfoot"][index]:.3f} native; NOT walking',fill='black')
            out=HERE/folder/'failure_actual_state.png';image.save(out)
            evidence.append(dict(path=str(out),trace=str(p),trace_sha256=sha256(p),index=index,time_s=float(z['time'][index])))
    renderer.close()
    atomic_json(HERE/'VISUAL_EVIDENCE.json',dict(teacher_video=str(video),teacher_sha256=sha256(video),
         decoded_video_frame_indices=chosen,teacher_sheet=str(HERE/'TEACHER_VIDEO/decoded_six_phase_sheet.png'),
         neural_failure_images=evidence,note='physical saved states only; video frames decoded, no new body motion'))
    print('Decoded six teacher gait phases and two actual neural failure images')

if __name__=='__main__':main()
