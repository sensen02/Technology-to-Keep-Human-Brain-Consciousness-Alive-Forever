"""Render synchronized four-arm playback directly from audited trajectories."""
import subprocess

import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from render_probes import ARMS,FONT,HERE,PHASES,build_scene,read,stan

body,_,_ = build_scene(None)
camera = mj.MjvCamera()
camera.type = mj.mjtCamera.mjCAMERA_FREE
camera.azimuth = 155
camera.elevation = -45
camera.distance = 6.1
traces = []
source_hashes = {}
for name, display in ARMS:
    path = HERE / f"FLAT_{name.upper()}" / "TRACE.npz"
    source_hashes[name] = stan.sha(path)
    assert source_hashes[name] == read(path.parent/"RESULT.json")["trace_files"]["TRACE.npz"]
    with np.load(path) as trace:
        traces.append((display,*(trace[k] for k in ("qpos","qvel","com","yaw"))))
font = ImageFont.truetype(FONT,17)
movie = HERE/"转向探针对照回放.mp4"
indices = sorted({*range(0,16101,100),*PHASES})
with mj.Renderer(body.plant,width=400,height=234) as renderer, imageio.get_writer(
        movie,fps=10,codec="libx264",quality=8,macro_block_size=2,ffmpeg_log_level="error") as writer:
    for ms in indices:
        canvas = Image.new("RGB",(800,540),"#e8efec")
        pen = ImageDraw.Draw(canvas)
        for i,(display,position,velocity,com,yaw) in enumerate(traces):
            left,top = 400*(i%2),270*(i//2)
            body.d.qpos[:] = position[ms]
            body.d.qvel[:] = velocity[ms]
            mj.mj_forward(body.plant,body.d)
            np.testing.assert_array_equal(body.d.qpos,position[ms])
            camera.lookat[:] = [*traces[0][3][ms,:2], .86]
            renderer.update_scene(body.d,camera=camera)
            canvas.paste(Image.fromarray(renderer.render()),(left,top+36))
            interval_yaw = np.rad2deg(yaw[ms]-yaw[PHASES[0]])
            pen.text((left+8,top+7),f"{display} · {ms*.001:.3f}s · 相对发令前转角 {interval_yaw:+.2f}°",
                     font=font,fill="#183936")
        writer.append_data(np.asarray(canvas))
count = subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0",
                                 "-show_entries","stream=nb_read_frames","-of","csv=p=0",str(movie)],text=True)
assert int(count)==len(indices)
stan.put(HERE/"VIDEO.json",dict(sources=source_hashes,physics_steps=0,frames=len(indices),sha256=stan.sha(movie)))
