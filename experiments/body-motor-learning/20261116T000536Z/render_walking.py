"""Render the saved continuous stand–walk–stop trajectory only."""
import argparse
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from walking_bench import stand, read


def render(source,out):
    out.mkdir()
    result=read(source/"RESULT.json")
    path=source/"TRACE.npz"
    assert stand.sha(path)==result["trace_sha256"]
    with np.load(path,allow_pickle=False) as a:
        qpos,qvel,com=a["qpos"],a["qvel"],a["com"]
    def forbidden(*args,**kwargs):
        raise RuntimeError("Only saved configurations may be used")
    mj.mj_step=mj.mj_step1=mj.mj_step2=forbidden
    core=stand.physics.Core(adhesion=False)
    camera=mj.MjvCamera()
    camera.type=mj.mjtCamera.mjCAMERA_FREE
    camera.azimuth=132
    camera.elevation=-9
    camera.distance=6.4
    camera.lookat[:]=[*qpos[:,:2].mean(0),.92]
    opt=mj.MjvOption()
    font="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    large=ImageFont.truetype(font,24)
    small=ImageFont.truetype(font,17)
    video=out/"站立-起步-直走-停稳.mp4"
    all_ms=len(qpos)-1
    indices=list(range(0,len(qpos),20))
    if all_ms!=indices[-1]:
        indices.append(all_ms)
    spots={0,500,2400,5200,9800,10200,all_ms}
    frames=[]
    with mj.Renderer(core.plant,height=380,width=640) as renderer,imageio.get_writer(
            video,fps=30,codec="libx264",quality=8,macro_block_size=2,ffmpeg_log_level="error") as writer:
        for k in indices:
            core.d.qpos[:]=qpos[k]
            core.d.qvel[:]=qvel[k]
            mj.mj_forward(core.plant,core.d)
            np.testing.assert_array_equal(core.d.qpos,qpos[k])
            renderer.update_scene(core.d,camera=camera,scene_option=opt)
            image=Image.new("RGB",(640,478),(245,245,245))
            image.paste(Image.fromarray(renderer.render()),(0,73))
            draw=ImageDraw.Draw(image)
            stage="静立" if k<500 else "直走" if k<500+result["walk_ms"] else "停步并站稳"
            label="联合检查通过" if result["joint_sampled_pass"] else "诊断｜本次联合检查未通过"
            draw.text((12,5),f"平地连续试验 · {stage}",font=large,fill=(30,40,47))
            draw.text((12,43),f"{k/1000:.2f} s · 约 1.67 倍慢放 · {label}",font=small,fill=(38,55,52))
            draw.text((12,454),f"同次实测轨迹 · 前向位移 {com[k,0]-com[0,0]:+.3f} 原生单位",
                      font=small,fill=(50,59,65))
            writer.append_data(np.asarray(image))
            if k in spots:
                frames.append(image.resize((320,239)))
    montage=Image.new("RGB",(960,239*((len(frames)+2)//3)),(245,245,245))
    for j,image in enumerate(frames):
        montage.paste(image,(320*(j%3),239*(j//3)))
    montage.save(out/"关键帧.png")
    ff= subprocess.run(["ffprobe","-v","error","-count_frames","-select_streams","v:0",
        "-show_entries","stream=nb_read_frames","-of","default=noprint_wrappers=1:nokey=1",str(video)],
        capture_output=True,check=True,text=True)
    assert int(ff.stdout)==len(indices)
    assert stand.sha(path)==result["trace_sha256"]
    stand.put(out/"VIDEO.json",dict(video=str(video),sha256=stand.sha(video),frames=len(indices),fps=30,
               saved_frame_indices=indices,source=str(source),trace_sha256=result["trace_sha256"],extra_integrations=0))


if __name__=="__main__":
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("source",type=Path)
    a.add_argument("output",type=Path)
    args=a.parse_args()
    render(args.source.resolve(),args.output.resolve())
