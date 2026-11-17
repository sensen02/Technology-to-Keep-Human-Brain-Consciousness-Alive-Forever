"""Render only saved native righting frames and visually review the transitions."""
import argparse
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from search import HERE, read, stand

FONT="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def draw_record(core, renderer, source, output, caption):
    report=read(source/"RESULT.json")
    src=source/"TRACE.npz"
    digest=stand.sha(src)
    assert digest==report["trace_sha256"]
    with np.load(src,allow_pickle=False) as z:
        q,v=z["qpos_ms"],z["qvel_ms"]
        branch=z["branches"] if "branches" in z else None
    xs=list(range(0,len(q),10))
    if xs[-1]!=len(q)-1:
        xs.append(len(q)-1)
    cam=mj.MjvCamera()
    cam.type=mj.mjtCamera.mjCAMERA_FREE
    cam.azimuth=152
    cam.elevation=-12
    cam.distance=6.5+float(np.max(np.ptp(q[:,:2],axis=0)))*.3
    cam.lookat[:]=[*q[:,:2].mean(0),.85]
    opt=mj.MjvOption()
    title=ImageFont.truetype(FONT,25)
    font=ImageFont.truetype(FONT,17)
    stage="无记忆身体网络" if branch is not None else "搜索教师接续站立控制"
    target=output/f"{report['seed']}.mp4"
    checked=[]
    with imageio.get_writer(target,fps=30,codec="libx264",quality=8,macro_block_size=2,
                            ffmpeg_log_level="error") as writer:
        for i in xs:
            core.d.qpos[:]=q[i]
            core.d.qvel[:]=v[i]
            mj.mj_forward(core.plant,core.d)
            assert np.array_equal(core.d.qpos,q[i])
            renderer.update_scene(core.d,camera=cam,scene_option=opt)
            canvas=Image.new("RGB",(640,494),(245,245,245))
            canvas.paste(Image.fromarray(renderer.render()),(0,80))
            draw=ImageDraw.Draw(canvas)
            tilt=np.rad2deg(np.arccos(np.clip(core.d.xmat[core.th,8],-1.,1.)))
            draw.text((14,7),f"{caption} · {stage}",font=title,fill=(30,35,40))
            draw.text((14,48),f"仿真 {i/1000:.2f} s  ·  躯干倾角 {tilt:.1f}°  ·  约 3.3 倍慢放",
                      font=font,fill=(36,52,55))
            draw.text((14,463),"同次物理轨迹  ·  每帧仅重建保存状态",font=font,fill=(48,55,60))
            writer.append_data(np.asarray(canvas))
            select= i==0 or i==xs[-1] or i==xs[len(xs)//2]
            if select:
                checked.append(canvas.copy().resize((320,247)))
    montage=Image.new("RGB",(960,247),(245,245,245))
    for index,frame in enumerate(checked):
        montage.paste(frame,(320*index,0))
    montage.save(output/f"{report['seed']}.png")
    raw=subprocess.run(["ffprobe","-v","error","-count_frames","-select_streams","v:0",
        "-show_entries","stream=nb_read_frames","-of","default=noprint_wrappers=1:nokey=1",str(target)],
        capture_output=True,check=True,text=True)
    if int(raw.stdout)!=len(xs):
        raise RuntimeError("Decoded frame count does not match saved frames")
    if stand.sha(src)!=digest:
        raise RuntimeError("Source changed")
    return dict(video=str(target),trace_sha256=digest,video_sha256=stand.sha(target),
                frame_count=len(xs),fps=30,source_indices=xs)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("output",type=Path)
    p.add_argument("side",type=Path)
    p.add_argument("supine",type=Path)
    args=p.parse_args()
    args.output.mkdir()
    def forbidden(*args,**kwargs):
        raise RuntimeError("Rendering never integrates")
    mj.mj_step=mj.mj_step1=mj.mj_step2=forbidden
    core=stand.physics.Core(adhesion=False)
    with mj.Renderer(core.plant,height=380,width=640) as renderer:
        rows=[draw_record(core,renderer,d,args.output,t) for d,t in
              ((args.side,"侧倒初态"),(args.supine,"近仰卧初态"))]
    stand.put(args.output/"MANIFEST.json",dict(traces=rows,physics_steps=0))


if __name__=="__main__":
    main()
