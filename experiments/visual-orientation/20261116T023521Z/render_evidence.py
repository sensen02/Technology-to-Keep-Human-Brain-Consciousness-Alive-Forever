"""Render saved visual closed-loop trials, including the exact published eyes."""
import argparse
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bench_runtime import turn
from optics import add_decor


def render(run, output):
    result = turn.read(run/"RESULT.json")
    assert turn.stand.sha(run/"TRACE.npz") == result["trace_sha256"]
    assert turn.stand.sha(run/"EYES.npz") == result["eyes_sha256"]
    with np.load(run/"TRACE.npz") as saved:
        qpos, qvel, com, yaw = [saved[k] for k in ("qpos","qvel","com","yaw")]
    with np.load(run/"EYES.npz") as saved:
        rgb, labels, commands, errors, goals = [saved[k] for k in ("rgb","labels","commands","bearing","goal_world")]
    assert len(qpos) == len(goals)
    case = result["case"]
    body = turn.stand.physics.Core(adhesion=False)
    start = case["onset_ms"]
    font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    title = ImageFont.truetype(font, 25)
    text = ImageFont.truetype(font, 17)
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.azimuth = 136
    camera.elevation = -45
    camera.distance = 6.1
    camera.lookat[:] = [*com[0,:2], .9]
    yaw = np.rad2deg(np.unwrap(yaw)-yaw[start])
    output.mkdir()
    movie = output / "视觉颈部身体同步.mp4"
    phases = {0, start//20*20, result["actual_ms"]//40*20, result["actual_ms"]}
    keys = []

    def forbidden(*a, **b):
        raise RuntimeError("Renderer cannot advance the physics")
    mj.mj_step = mj.mj_step1 = mj.mj_step2 = forbidden
    indices = list(range(0, result["actual_ms"]+1,20))
    if indices[-1] != result["actual_ms"]:
        indices.append(result["actual_ms"])
    samples = []
    labels_ = ("右转","停止","左转")
    with mj.Renderer(body.plant, height=420,width=640) as renderer, imageio.get_writer(
        movie,fps=30,codec="libx264",quality=8,macro_block_size=2,ffmpeg_log_level="error",
    ) as writer:
        for index in indices:
            body.d.qpos[:] = qpos[index]
            body.d.qvel[:] = qvel[index]
            mj.mj_forward(body.plant,body.d)
            np.testing.assert_array_equal(body.d.qpos,qpos[index])
            renderer.update_scene(body.d,camera=camera)
            if np.isfinite(goals[index]).all():
                add_decor(renderer.scene,mj,mj.mjtGeom.mjGEOM_CYLINDER,[case["width"],1.09,0.],
                          goals[index],np.eye(3),case["color"])
            board = Image.new("RGB",(960,520),(244,245,246))
            board.paste(Image.fromarray(renderer.render()),(0,55))
            pen=ImageDraw.Draw(board)
            pen.text((14,8),"视觉定向 · 实际脑—颈部—身体闭环",font=title,fill=(19,34,43))
            pen.text((648,54),"脑收到的延迟双眼图",font=text,fill=(31,55,62))
            command=commands[min(index//5,len(commands)-1)]
            cap=int(command[1])
            if cap>=0:
                j=cap//20
                assert labels[j,0]==cap and cap+20<=index
                for eye,column in enumerate(("左眼","右眼")):
                    dx=648+152*eye
                    panel=Image.fromarray(rgb[j,eye]).resize((142,142),Image.Resampling.NEAREST)
                    board.paste(panel,(dx,98))
                    pen.text((dx,244),column,font=text,fill=(31,55,62))
            lines=[f"脑意图  {labels_[int(command[2])+1]}",
                   f"身体译码  {labels_[int(command[3])+1]}",
                   f"胸部相对转角  {yaw[index]:+.1f}°",
                   f"目标相对视线  {errors[index]:+.1f}°",
                   f"接收条件  {result['profile']}",
                   f"仿真时刻  {index/1000:.2f} 秒"]
            for k,line in enumerate(lines):
                pen.text((650,278+28*k),line,font=text,fill=(31,55,62))
            pen.text((12,484),"原始物理轨迹回放 · 1.67 倍慢放",font=text,fill=(31,55,62))
            writer.append_data(np.asarray(board))
            if index in phases:
                samples.append(board.resize((480,260)))
                keys.append(index)
    montage=Image.new("RGB",(960,260*((len(samples)+1)//2)),(244,245,246))
    for i,im in enumerate(samples):
        montage.paste(im,(480*(i%2),260*(i//2)))
    montage.save(output/"关键帧.png")
    count=subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0",
                                   "-show_entries","stream=nb_read_frames","-of","csv=p=0",str(movie)],text=True)
    assert int(count)==len(indices)
    turn.stand.put(output/"MEDIA.json",dict(movie=str(movie),sha256=turn.stand.sha(movie),
                   frames=len(indices),key_indices=keys,source=str(run),physics_steps=0))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    render(args.run.resolve(),args.output.resolve())
