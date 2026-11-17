"""Render two saved turn trials from their own synchronized physical traces."""
import argparse
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from turning import read, stand


def render(a,b,out,show_exit_slip=False):
    out.mkdir()
    results=[read(p/"RESULT.json") for p in (a,b)]
    data=[]
    for p,r in zip((a,b),results,strict=True):
        assert stand.sha(p/"TRACE.npz")==r["trace_sha256"]
        with np.load(p/"TRACE.npz",allow_pickle=False) as z:
            data.append({k:z[k] for k in ("qpos","qvel","yaw","com")})
    if len(data[0]["qpos"])!=len(data[1]["qpos"]):
        raise ValueError("Two simultaneous panels require equal recorded lengths")
    core=stand.physics.Core(adhesion=False)
    def forbidden(*args,**kw):
        raise RuntimeError("No physical steps in the video renderer")
    mj.mj_step=mj.mj_step1=mj.mj_step2=forbidden
    cameras=[]
    opt=mj.MjvOption()
    for d in data:
        camera=mj.MjvCamera()
        camera.type=mj.mjtCamera.mjCAMERA_FREE
        camera.azimuth=138
        camera.elevation=-51
        camera.distance=6.4+float(np.ptp(d["com"][:,:2],axis=0).max())*.4
        camera.lookat[:]=[*d["com"][:,:2].mean(0),.85]
        cameras.append(camera)
    font="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    big=ImageFont.truetype(font,27)
    little=ImageFont.truetype(font,17)
    names=("左转","右转")
    turn_kind="原地转向" if results[0]["mode"].startswith("pivot") else "行走弧线转向"
    ms=len(data[0]["qpos"])-1
    frames=list(range(0,ms+1,20))
    if frames[-1]!=ms:
        frames.append(ms)
    key_indices=[min(frames,key=lambda i:abs(i-t)) for t in (0,results[0]["bounds"][0],
                    sum(results[0]["bounds"][:2])//2,results[0]["bounds"][1],ms)]
    video=out/f"{turn_kind}左右对照.mp4"
    montage_frames=[]
    yaws=[np.unwrap(d["yaw"]) for d in data]
    with mj.Renderer(core.plant,height=350,width=480) as renderer,imageio.get_writer(
        video,fps=30,codec="libx264",quality=8,macro_block_size=2,ffmpeg_log_level="error") as writer:
        for j in frames:
            image=Image.new("RGB",(960,452),(245,245,245))
            draw=ImageDraw.Draw(image)
            draw.text((14,4),f"{turn_kind} · 同次身体物理轨迹",font=big,fill=(37,40,46))
            draw.text((14,37),f"仿真 {j*.001:.2f} 秒 · 1.67 倍慢放",font=little,fill=(42,51,56))
            for i,(d,r,yaw,camera) in enumerate(zip(data,results,yaws,cameras,strict=True)):
                core.d.qpos[:]=d["qpos"][j]
                core.d.qvel[:]=d["qvel"][j]
                mj.mj_forward(core.plant,core.d)
                np.testing.assert_array_equal(core.d.qpos,d["qpos"][j])
                renderer.update_scene(core.d,camera=camera,scene_option=opt)
                image.paste(Image.fromarray(renderer.render()),(480*i,65))
                start_ms,end_ms,exit_ms,_=r["bounds"]
                phase="转向" if start_ms<=j<end_ms else "直走" if 500<=j<start_ms or end_ms<=j<exit_ms else "站立／停步"
                angle=float(np.rad2deg(yaw[j]-yaw[start_ms])) if j>=start_ms else 0.
                label="工程检查通过" if r["joint_pass"] else "诊断｜本回合未全过"
                if show_exit_slip and end_ms<=j<exit_ms:
                    label="推进通过｜滑移偏高"
                draw.text((480*i+12,423),f"{names[i]} · {phase} · 相对转角 {angle:+.1f}° · {label}",
                          font=little,fill=(30,48,57))
            writer.append_data(np.asarray(image))
            if j in key_indices:
                montage_frames.append(image.resize((480,226)))
    montage=Image.new("RGB",(960,226*((len(montage_frames)+1)//2)),(245,245,245))
    for i,im in enumerate(montage_frames):
        montage.paste(im,(i%2*480,i//2*226))
    montage.save(out/"关键帧.png")
    command=["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries","stream=nb_read_frames",
             "-of","default=noprint_wrappers=1:nokey=1",str(video)]
    assert int(subprocess.check_output(command,text=True))==len(frames)
    stand.put(out/"MANIFEST.json",dict(video=str(video.resolve()),file_sha256=stand.sha(video),decoded_frames=len(frames),
              fps=30,indices=frames,sources=[str(p.resolve()) for p in (a,b)],physics_steps=0))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("left",type=Path)
    p.add_argument("right",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("--exit-slip-diagnostic",action="store_true")
    a=p.parse_args()
    render(a.left,a.right,a.out,a.exit_slip_diagnostic)
