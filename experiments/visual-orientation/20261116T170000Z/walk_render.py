"""Render archived visual and body outputs without integrating physics."""
import argparse
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from walk_pilot import turn
from optics import add_decor


def render(run, output):
    result = turn.read(run/"WALK_RESULT.json")
    for name in ("TRACE.npz","WALK_EYES.npz","WALK_COMMANDS.npz"):
        assert turn.stand.sha(run/name) == result["files"][name]
    with np.load(run/"TRACE.npz") as z:
        qpos, qvel, com = [z[k] for k in ("qpos","qvel","com")]
    with np.load(run/"WALK_EYES.npz") as z:
        images, decisions = [z[k] for k in ("rgb", "decisions")]
    with np.load(run/"WALK_COMMANDS.npz") as z:
        commands = z["commands"]
    case = result["case"]
    onset = case["onset"]
    end = onset + case["walk_ms"]
    last = len(qpos) - 1
    indices = sorted(set(range(0,last+1,20)) | {last})
    keys = {0, onset+300, (onset+end)//40*20, last}
    body = turn.stand.physics.Core(adhesion=False)
    body.d.qpos[:] = qpos[onset]
    mj.mj_forward(body.plant, body.d)
    head = body.plant.body("Head").id
    rotation = body.d.xmat[head].reshape(3,3)
    bearing = np.arctan2(rotation[1,0],rotation[0,0])+np.deg2rad(case["bearing"])
    target = body.d.xpos[head] + case["distance"]*np.array([np.cos(bearing),np.sin(bearing),0.])
    target[2] = 1.25

    def forbid_step(*args, **kwargs):
        raise RuntimeError("Rendering must preserve archived physics.")
    mj.mj_step = mj.mj_step1 = mj.mj_step2 = forbid_step
    output.mkdir()
    font_file = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    title = ImageFont.truetype(font_file, 24)
    font = ImageFont.truetype(font_file, 17)
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.azimuth = 138
    camera.elevation = -38
    camera.distance = 7.5
    movie = output/"视觉起步与停稳.mp4"
    thumbs = []
    with mj.Renderer(body.plant,height=420,width=640) as renderer, imageio.get_writer(
        movie,fps=30,codec="libx264",quality=8,macro_block_size=2,ffmpeg_log_level="error"
    ) as writer:
        for t in indices:
            body.d.qpos[:], body.d.qvel[:] = qpos[t], qvel[t]
            mj.mj_forward(body.plant,body.d)
            np.testing.assert_array_equal(body.d.qpos,qpos[t])
            camera.lookat[:] = [*com[t,:2], .9]
            renderer.update_scene(body.d,camera=camera)
            if onset <= t < end and case["mode"] != "off":
                add_decor(renderer.scene,mj,mj.mjtGeom.mjGEOM_CYLINDER,
                          [case["width"],1.09,0.],target,np.eye(3),(1.,.045,.035,1.))
            panel = Image.new("RGB",(960,510),(243,246,246))
            panel.paste(Image.fromarray(renderer.render()),(0,55))
            draw = ImageDraw.Draw(panel)
            draw.text((14,8),"视觉起步、直走与停稳",font=title,fill="#19343a")
            draw.text((650,55),"人工脑实际收到的双眼图",font=font,fill="#19343a")
            row = commands[min(len(commands)-1,t//5)]
            capture = int(row[3])
            if capture >= 0:
                i = capture//20
                assert decisions[i,1] == capture and capture+20 <= t
                for eye in range(2):
                    panel.paste(Image.fromarray(images[i,eye]).resize((141,141),Image.Resampling.NEAREST),(649+152*eye,96))
            labels = ("停", "推进")
            for i,text in enumerate([f"人工脑意图  {labels[row[1]]}",f"DNg100 到达后译码  {labels[row[2]]}",
                                     f"质心前进  {com[t,0]-com[0,0]:.3f}",f"仿真时刻  {t*.001:.2f} 秒"]):
                draw.text((650,266+32*i),text,font=font,fill="#19343a")
            writer.append_data(np.asarray(panel))
            if t in keys:
                thumbs.append(panel.resize((480,255)))
    montage = Image.new("RGB",(960,255*((len(thumbs)+1)//2)),(243,246,246))
    for i,image in enumerate(thumbs):
        montage.paste(image,(480*(i%2),255*(i//2)))
    montage.save(output/"关键帧.png")
    count = subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0",
                                     "-show_entries","stream=nb_read_frames","-of","csv=p=0",str(movie)], text=True)
    assert int(count) == len(indices)
    turn.stand.put(output/"MEDIA.json",dict(trajectory=str(run),movie_sha256=turn.stand.sha(movie),
                  frames=len(indices),original_physics_steps=0,key_times_ms=sorted(keys)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("output",type=Path)
    a = p.parse_args()
    render(a.run.resolve(),a.output.resolve())
