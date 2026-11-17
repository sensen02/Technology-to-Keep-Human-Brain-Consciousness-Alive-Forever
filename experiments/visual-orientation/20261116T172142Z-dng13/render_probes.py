"""Create two saved-state views per matched physical steering probe."""
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from probe import HERE, STIM_FROM, STIM_TO, WARM, build_scene, read, stan

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
PHASES = (WARM+STIM_FROM,WARM+STIM_TO)
ARMS = (("off","零活动"),("dna","左 DNa02"),("dng","左 DNg13"),("both","同时发送"))


def main():
    audit = read(HERE/"STEERING_AUDIT.json")
    body,_,_ = build_scene(None)
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.azimuth = 155
    camera.elevation = -45
    camera.distance = 6.1
    sheet = Image.new("RGB",(800,4*276),"#e8efec")
    pen = ImageDraw.Draw(sheet)
    text = ImageFont.truetype(FONT,18)
    sources = {}
    with mj.Renderer(body.plant,width=400,height=234) as renderer:
        for row,(name,display) in enumerate(ARMS):
            folder = HERE/f"FLAT_{name.upper()}"
            trace = folder/"TRACE.npz"
            assert stan.sha(trace)==read(folder/"RESULT.json")["trace_files"]["TRACE.npz"]
            sources[name]=stan.sha(trace)
            with np.load(trace,allow_pickle=False) as data:
                qpos,qvel,com = [data[k] for k in ("qpos","qvel","com")]
                yaw = np.rad2deg(np.unwrap(data["yaw"])-data["yaw"][PHASES[0]])
            for col,phase in enumerate(PHASES):
                phase = min(phase,len(qpos)-1)
                body.d.qpos[:] = qpos[phase]
                body.d.qvel[:] = qvel[phase]
                mj.mj_forward(body.plant,body.d)
                np.testing.assert_array_equal(body.d.qpos,qpos[phase])
                if col==0 and row==0:
                    camera.lookat[:] = [com[phase,0]+.55,com[phase,1],.86]
                renderer.update_scene(body.d,camera=camera)
                left,top=400*col,276*row
                sheet.paste(Image.fromarray(renderer.render()),(left,top+36))
                pen.text((left+8,top+7),f"{display} · {phase*.001:.3f}s · 实测转角 {yaw[phase]:+.2f}°",font=text,fill="#183936")
    output=HERE/"转向探针对照关键帧.png"
    sheet.save(output)
    stan.put(HERE/"RENDER.json",dict(sources=sources,physics_steps=0,frames=8,sha256=stan.sha(output)))


if __name__=="__main__":
    main()
