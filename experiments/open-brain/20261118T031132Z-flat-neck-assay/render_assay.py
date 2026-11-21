"""Visualize the archived matched traces without altering their states."""
from pathlib import Path
import json

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from schedule import ARMS, SEEDS, CUT_ON, RESTORE, YAW_SAMPLE

HERE = Path(__file__).resolve().parent
LABELS = ("Intact", "Sham", "Sustained block", "Reopen at 0.90 s")
COLORS = ("#263144", "#638390", "#c85c35", "#2b68c2")
DT = .0001


def read_arrays(path, keys):
    with np.load(path, allow_pickle=False) as data:
        return {k:data[k] for k in keys}


def plots():
    fig, axes = plt.subplots(3, 2, figsize=(12, 8.4), gridspec_kw={"width_ratios":[1.08,1]})
    for row,seed in enumerate(SEEDS):
        folder = HERE/"FORMAL"/f"seed{seed}"
        t = np.arange(20000)*DT
        traffic = read_arrays(folder/"NEURAL.npz", ("an_arrived",))["an_arrived"]
        raster, behavior = axes[row]
        for lane,(arm,label,color) in enumerate(zip(ARMS,LABELS,COLORS,strict=True)):
            hits = t[traffic[:,lane,0]]
            raster.vlines(hits, lane-.26, lane+.26, color=color,lw=.7,label=label)
            a = read_arrays(folder/arm/"TRACE.npz", ("yaw",))
            angles = np.rad2deg(np.unwrap(a["yaw"])-a["yaw"][0])
            if lane != 0:
                behavior.plot(t,angles,color=color,lw=1.4,label=label)
            else:
                behavior.plot(t,angles,color=color,lw=.8,ls=":",label=label)
        raster.set(ylim=(3.5,-.5),yticks=np.arange(4),yticklabels=LABELS, xlim=(.31,1.23))
        behavior.set(xlim=(0,2),ylabel="Yaw (deg)")
        behavior.axvline((YAW_SAMPLE+1)*DT,ls=":",lw=.8,color="#555555")
        for ax in (raster,behavior):
            ax.axvline(CUT_ON*DT,lw=.9,color="#444444",ls="--")
            ax.axvline(RESTORE*DT,lw=.9,color="#444444",ls="--")
            ax.grid(axis="x",alpha=.19)
        raster.text(.015,.93,f"Phase {seed}",transform=raster.transAxes,ha="left",
                    color="#464646",fontsize=9)
    axes[0,0].set_title("Actual left AN arrivals after the neck")
    axes[0,1].set_title("Body yaw from the same walk-start state")
    axes[-1,0].set_xlabel("Time (s)")
    axes[-1,1].set_xlabel("Time (s) | primary sample 1.80 s")
    handles,labels=axes[0,1].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc="lower center",frameon=False)
    fig.suptitle("Left ascending axon transmission block | flat matched assay",fontweight="bold")
    fig.tight_layout(rect=(0,.045,1,.95))
    fig.savefig(HERE/"ASSAY_OVERVIEW.png",dpi=155)
    plt.close(fig)


def render():
    phase=11701
    folder=HERE/"FORMAL"/f"seed{phase}"
    model=mujoco.MjModel.from_binary_path(str(folder/"intact.mjb"))
    model.mat_reflectance[:]=0
    state=mujoco.MjData(model)
    renderer=mujoco.Renderer(model,height=224,width=352)
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",13)
    poses=[read_arrays(folder/arm/"TRACE.npz",("qpos","position")) for arm in ARMS]

    def frame(tick):
        row=[]
        for lane,pose in enumerate(poses):
            state.qpos[:]=pose["qpos"][tick]
            mujoco.mj_kinematics(model,state)
            camera=mujoco.MjvCamera()
            camera.lookat[:]=[*pose["position"][tick,:2],.68]
            camera.distance=6.2
            camera.azimuth=115.
            camera.elevation=-32.
            renderer.update_scene(state,camera)
            image=Image.fromarray(renderer.render())
            label=ImageDraw.Draw(image)
            label.rectangle((0,0,352,42),fill="#192029")
            label.rectangle((0,0,4,42),fill=COLORS[lane])
            label.text((9,3),LABELS[lane],fill="white",font=font)
            blocked=tick>=CUT_ON and (lane==2 or (lane==3 and tick<RESTORE))
            status="left AN blocked" if blocked else "left AN open"
            label.text((9,22),f"t={(tick+1)*DT:.3f}s | {status} | 0.25x",fill="#dbe5ef",font=font)
            row.append(np.asarray(image))
        return np.concatenate(row,axis=1)

    try:
        snapshot=np.concatenate([frame(i) for i in (4400,7800,9800,19400)],axis=0)
        Image.fromarray(snapshot).save(HERE/"ASSAY_STATES.png")
        with imageio.get_writer(str(HERE/"MATCHED_ASSAY.mp4"),fps=25,codec="libx264",
                                quality=7,macro_block_size=16) as video:
            for tick in range(0,20000,100):
                video.append_data(frame(tick))
    finally:
        renderer.close()


if __name__=="__main__":
    plots()
    render()
