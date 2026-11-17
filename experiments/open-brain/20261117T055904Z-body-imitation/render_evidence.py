"""Render saved physical states and plot the exact native trajectory."""
import argparse
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent


def image(renderer, model, data, index, poses, native):
    data.qpos[:] = poses[index]
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = native[index*10, 1:4]
    camera.distance, camera.azimuth, camera.elevation = 7.5, 132, -21
    renderer.update_scene(data, camera=camera)
    return renderer.render().copy()


def main(target, out):
    model=mujoco.MjModel.from_binary_path(str(HERE/'BODY.mjb'))
    data=mujoco.MjData(model)
    with np.load(target/'TRACE.npz',allow_pickle=False) as z:
        native=z['trace']; poses=z['qpos']
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
    out.mkdir()
    with mujoco.Renderer(model,height=360,width=560) as renderer:
        for a,index in zip(axes.flat[:5],(80,520,780,1170,1570),strict=True):
            a.imshow(image(renderer,model,data,index,poses,native))
            a.set_title(f'{index/1000:.3f} s • saved physics')
            a.axis('off')
        a=axes.flat[5]; a.plot(native[:,0], native[:,1]-native[0,1], label='Forward (mm)')
        a.plot(native[:,0], native[:,6], label='VNC-derived drive'); a.set_xlabel('Seconds')
        a.legend(frameon=False)
        fig.savefig(out/'REVIEW.png',dpi=150)
        frames=list(range(0,len(poses),20))
        cmd=['ffmpeg','-nostdin','-loglevel','error','-f','rawvideo','-pixel_format','rgb24',
             '-video_size','560x360','-framerate','30','-i','pipe:0','-c:v','libx264','-pix_fmt','yuv420p',str(out/'WALK_AND_STAND.mp4')]
        font=ImageFont.load_default(size=16)
        process=subprocess.Popen(cmd,stdin=subprocess.PIPE)
        try:
            for index in frames:
                pict=Image.fromarray(image(renderer,model,data,index,poses,native))
                draw=ImageDraw.Draw(pict)
                draw.rectangle((0,0,560,34),fill=(18,25,35))
                draw.text((9,7),f'{target.name} | {index/1000:.3f} s | playback 0.60x',fill='white',font=font)
                process.stdin.write(np.asarray(pict).tobytes())
        finally:
            process.stdin.close()
        assert process.wait()==0
    print(len(frames),'rendered frames',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('target',type=Path); p.add_argument('out',type=Path)
    args=p.parse_args()
    main(args.target,args.out)
