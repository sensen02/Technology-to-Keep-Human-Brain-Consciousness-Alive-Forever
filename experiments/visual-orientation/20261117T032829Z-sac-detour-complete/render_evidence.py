"""Render trajectory and physical-frame evidence for the demonstrations."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"20261117T022700Z-sac-startup-replay"))
from course import ACTOR_S, BODY, COLOR, WIDTH, sha


FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FAMILY = FontProperties(fname=FONT)
LABELS = {"cal_left":"左绕","cal_right":"右绕"}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("out",type=Path)
    args = cli.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    report = json.loads((HERE/"COMPARISON.json").read_text())
    records = {item["case"]:item for item in report["demonstrations"]}
    fig, axs = plt.subplots(2,2,figsize=(12,8),layout="constrained")
    physical = BODY.act_mod.load_full_model(BODY.PAD_XML)
    data = mj.MjData(physical)
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.azimuth = 131.
    camera.elevation = -50.
    camera.distance = 9.
    canvas = Image.new("RGB",(1470,700),"#f3f5f6")
    captions = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT,18)
    with mj.Renderer(physical,width=480,height=300) as renderer:
        for idx, evidence in enumerate(report["comparisons"][:2]):
            case = evidence["case"]
            trial = Path(evidence["source"])
            meta = json.loads((trial/"RESULT.json").read_text())
            qualified = records[case]
            path = trial/"EPISODE.npz"
            assert sha(path) == evidence["sha256"] == qualified["episode_sha256"]
            with np.load(path,allow_pickle=False) as saved:
                pose, com, goal, reward = (saved[k] for k in ("qpos","com","goal","reward"))
            changed = HERE/"SCORES"/f"{case}.npz"
            assert sha(changed) == evidence["score_sha256"]
            with np.load(changed,allow_pickle=False) as scored:
                new = scored["reward"]
            t = ACTOR_S*np.arange(1,len(reward)+1)
            trajectory, scoring = axs[idx]
            trajectory.plot(*com[:,:2].T,color="#226958",linewidth=1.8)
            trajectory.scatter(*com[0,:2],s=32,c="#465972",marker="o",zorder=3)
            trajectory.scatter(*com[-1,:2],s=32,c="#634479",marker="s",zorder=3)
            trajectory.scatter(*goal[:2],s=100,c="#ad3631",marker="*",zorder=4)
            boxes = meta["geometry"]["boxes"]
            for x,y,_,a,b,_ in boxes:
                trajectory.add_patch(Rectangle((x-a,y-b),2*a,2*b,color="#548970"))
            trajectory.set_aspect("equal",adjustable="datalim")
            trajectory.set_title(f"{LABELS[case]} · 质心轨迹",fontproperties=FAMILY)
            trajectory.set_xlabel("世界 X",fontproperties=FAMILY)
            trajectory.set_ylabel("世界 Y",fontproperties=FAMILY)
            fixed = slice(0,len(t)-1) if meta["terminated"] else slice(None)
            scoring.plot(t[fixed],np.cumsum(reward[fixed]),color="#ac6837",label="旧评分")
            scoring.plot(t[fixed],np.cumsum(new[fixed]),color="#356598",label="质心安全路线评分")
            if qualified["clear_at_s"] is not None:
                scoring.axvline(qualified["clear_at_s"],c="#428372",lw=1.,ls="--",label="全身越箱")
            scoring.set_title("终止事件前的累计评分",fontproperties=FAMILY)
            scoring.set_xlabel("仿真时间 / 秒",fontproperties=FAMILY)
            scoring.legend(prop=FAMILY,frameon=False)
            for axis in axs[idx]:
                axis.grid(alpha=.15)

            clear = round((qualified["clear_at_s"] or len(reward)*ACTOR_S)/ACTOR_S)
            frames = ((min(1600,len(pose)-1),"斜走绕箱"), (clear,"全身越箱"), (len(pose)-1,"回合末"))
            for col,(at,title) in enumerate(frames):
                data.qpos[:] = pose[at]
                mj.mj_forward(physical,data)
                np.testing.assert_array_equal(data.qpos,pose[at])
                anchor = com[max(0,at-1)]
                camera.lookat[:] = [anchor[0]+.3,anchor[1],.7]
                renderer.update_scene(data,camera=camera)
                scene = renderer.scene
                for x,y,z,a,b,c in boxes:
                    mj.mjv_initGeom(scene.geoms[scene.ngeom],mj.mjtGeom.mjGEOM_BOX,np.array([a,b,c]),
                                    np.array([x,y,z]),np.eye(3).ravel(),np.array([.03,.9,.08,1.]))
                    scene.ngeom += 1
                mj.mjv_initGeom(scene.geoms[scene.ngeom],mj.mjtGeom.mjGEOM_CYLINDER,np.array([WIDTH,1.09,0.]),
                                goal,np.eye(3).ravel(),np.asarray(COLOR))
                scene.ngeom += 1
                px, py = col*490, idx*350
                captions.text((px+11,py+11),f"{LABELS[case]} · {title} · {at*ACTOR_S:.2f} s",font=font,fill="#142927")
                canvas.paste(Image.fromarray(renderer.render()),(px+5,py+39))
    fig.savefig(args.out/"左右绕障轨迹和评分.png",dpi=145)
    plt.close(fig)
    canvas.save(args.out/"左右绕障实际姿态.png")


if __name__ == "__main__":
    main()
