"""Render the actual approach and eye images from the recorded states."""
import argparse
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from approach_runtime import prepare, turn, sha


def render(folder, output):
    source=turn.read(folder/"RESULT.json")
    assert all(sha(folder/name)==value for name,value in source["files"].items())
    with np.load(folder/"TRACE.npz") as z:
        pose={key:z[key] for key in ("qpos","qvel","com","head")}
    with np.load(folder/"OBS.npz") as z:
        data={key:z[key] for key in ("decisions","eyes","stamps","goal")}
    block_path=folder/"OBSTACLE.json"
    block=turn.read(block_path) if block_path.is_file() else None
    body,_=prepare(output,"nominal")
    cam=mj.MjvCamera()
    cam.type=mj.mjtCamera.mjCAMERA_FREE
    cam.azimuth=135
    cam.elevation=-36
    cam.distance=7.6
    font=ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",18)
    stamps=list(range(0,len(pose["qpos"]),40))
    if stamps[-1]!=len(pose["qpos"])-1:
        stamps.append(len(pose["qpos"])-1)
    moments=[stamps[min(range(len(stamps)),key=lambda i:abs(stamps[i]-t))]
             for t in (200,1500,len(pose["qpos"])//2,len(pose["qpos"])-1)]
    sheet=Image.new("RGB",(1320,808),"white")
    target=output/"视觉趋近与调速.mp4"
    def never(*_a,**_kw):
        raise RuntimeError("Only archived poses may be rendered")
    mj.mj_step = mj.mj_step1 = mj.mj_step2 = never
    with mj.Renderer(body.plant,width=500,height=342) as renderer:
        with imageio.get_writer(target,fps=25,codec="libx264",macro_block_size=2) as video:
            for t in stamps:
                panel=Image.new("RGB",(660,404),"#edf1ef")
                body.d.qpos[:]=pose["qpos"][t]
                body.d.qvel[:]=pose["qvel"][t]
                mj.mj_forward(body.plant,body.d)
                assert np.array_equal(body.d.qpos,pose["qpos"][t])
                cam.lookat[:]=[*pose["com"][0,:2],.9]
                renderer.update_scene(body.d,camera=cam)
                scene=renderer.scene
                if block:
                    mj.mjv_initGeom(scene.geoms[scene.ngeom],mj.mjtGeom.mjGEOM_BOX,
                                   np.asarray(block["half_extent"]),np.asarray(block["center"]),
                                   np.eye(3).ravel(),np.array([.03,.90,.08,1.],dtype=np.float32))
                    scene.ngeom+=1
                if t>=500:
                    mj.mjv_initGeom(scene.geoms[scene.ngeom],mj.mjtGeom.mjGEOM_CYLINDER,
                                   np.array([.15,1.09,0.]),data["goal"],np.eye(3).ravel(),
                                   np.array([1.,.045,.035,1.],dtype=np.float32))
                    scene.ngeom+=1
                panel.paste(Image.fromarray(renderer.render()),(0,62))
                i=np.searchsorted(data["stamps"],t-20,side="right")-1
                if i>=0:
                    for side in (0,1):
                        panel.paste(Image.fromarray(data["eyes"][i,side]).resize((64,64),Image.Resampling.NEAREST),(510+side*70,92))
                stamp=data["decisions"][min(t//5,len(data["decisions"])-1)]
                distance=np.linalg.norm(data["goal"][:2]-pose["head"][t,:2])
                text=ImageDraw.Draw(panel)
                text.text((10,6),f"视觉趋近  {t/1000:.2f} s   实测目标距离 {distance:.3f}",font=font,fill="#263944")
                text.text((10,34),f"颈后前进请求 {stamp[3]:.3f}   转向 {stamp[4]:+.0f}",font=font,fill="#263944")
                text.text((510,66),"人工脑双眼输入",font=font,fill="#263944")
                video.append_data(np.asarray(panel))
                if t in moments:
                    idx=moments.index(t)
                    sheet.paste(panel,((idx%2)*660,(idx//2)*404))
    sheet.save(output/"视觉趋近关键帧.png")
    read=int(subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries","stream=nb_read_frames","-of","csv=p=0",str(target)],text=True).strip())
    assert read==len(stamps)
    turn.stand.put(output/"VIDEO.json",dict(frames=read,source=str(folder),render_physics_steps=0))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    render(a.source.resolve(),a.output.resolve())
