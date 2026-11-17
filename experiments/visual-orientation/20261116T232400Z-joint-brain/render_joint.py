"""Re-render the matched intact and cut physical states, with visible replay probe."""
import argparse
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from touch_runtime import BODY, turn, sha, RADIUS


def render(root, output):
    output.mkdir()
    model = BODY.act_mod.load_full_model(BODY.PAD_XML)
    data = mj.MjData(model)
    sets = []
    for name in ("intact","cut"):
        folder = root/name
        info = turn.read(folder/"RESULT.json")
        assert info["hashes"]["RUN.npz"] == sha(folder/"RUN.npz")
        with np.load(folder/"RUN.npz",allow_pickle=False) as trial:
            sets.append({key:trial[key] for key in ("qpos","qvel","commands","probe","com")})
    length = len(sets[0]["probe"])
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",20)
    small = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",16)
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.distance = 6.5
    cam.azimuth = 138
    cam.elevation = -38
    scene_width, scene_height = 560, 350
    moments = (550,2320,2960,length-1)
    samples = sorted(set(range(0,length,40)).union(moments))
    image_out = output/"触觉上行完整与切断关键帧.png"
    movie = output/"触觉上行停步对照.mp4"
    montage = Image.new("RGB",(2*scene_width,4*(scene_height+70)))
    def never(*_a,**_kw):
        raise RuntimeError("Video may use only the original integrated data")
    mj.mj_step = mj.mj_step1 = mj.mj_step2 = never
    with mj.Renderer(model,width=scene_width,height=scene_height) as renderer:
        with imageio.get_writer(movie,fps=25,codec="libx264",macro_block_size=2) as sink:
            for stamp in samples:
                row = Image.new("RGB",(2*scene_width,scene_height+70),"#eaf0f1")
                for i, track in enumerate(sets):
                    data.qpos[:] = track["qpos"][stamp]
                    data.qvel[:] = track["qvel"][stamp]
                    mj.mj_forward(model,data)
                    np.testing.assert_array_equal(data.qpos,track["qpos"][stamp])
                    cam.lookat[:] = [track["com"][0,0]+.15,track["com"][0,1],.9]
                    renderer.update_scene(data,camera=cam)
                    point = track["probe"][stamp]
                    scene = renderer.scene
                    if point[2] < 3. and scene.ngeom < scene.maxgeom:
                        mj.mjv_initGeom(scene.geoms[scene.ngeom],mj.mjtGeom.mjGEOM_SPHERE,
                                       np.array([RADIUS,0.,0.]),point,np.eye(3).ravel(),
                                       np.array([.95,.38,.06,1.],dtype=np.float32))
                        scene.ngeom += 1
                    row.paste(Image.fromarray(renderer.render()),(i*scene_width,70))
                    label = "上行完整" if i==0 else "上行切断"
                    _, visual, decoded, intention, delivered, _ = track["commands"][stamp//5]
                    contact = "左前足" if decoded==0 else "右前足" if decoded==1 else "无触碰" if decoded==-1 else "未知"
                    move = "继续走" if delivered else "停止"
                    draw = ImageDraw.Draw(row)
                    draw.text((i*scene_width+10,5),f"{label}  {stamp/1000:.3f} s",font=font,fill="#20363b")
                    travelled = track["com"][stamp,0]-track["com"][0,0]
                    draw.text((i*scene_width+10,39),f"侧碰参考读出 {contact}   身体收到 {move}   前进 {travelled:+.3f}",font=small,fill="#20363b")
                sink.append_data(np.asarray(row))
                if stamp in moments:
                    montage.paste(row,(0,moments.index(stamp)*(scene_height+70)))
    montage.save(image_out)
    count = int(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-count_frames",
                  "-show_entries","stream=nb_read_frames","-of","csv=p=0",str(movie)],text=True).strip())
    assert count == len(samples)
    turn.stand.put(output/"VIDEO.json",dict(frames=count,mp4_sha256=sha(movie),png_sha256=sha(image_out),
                  source=str(root.resolve()),physics_steps=0,probe_visible_only_in_replay=True))


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    render(a.run,a.output)
