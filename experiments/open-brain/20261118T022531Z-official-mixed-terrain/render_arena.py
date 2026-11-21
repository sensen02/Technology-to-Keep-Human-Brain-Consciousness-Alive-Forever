"""Render the recorded terrain runs. No simulation or root updates are performed."""
from pathlib import Path
import json

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TRIAL = Path(__file__).resolve().parent
SEEDS = (10901, 11311)
MODES = ("native_full", "reference_boundary", "translated", "repaired", "forward_probe")
LABELS = ("Original brain", "Controlled reference", "Old translation", "Timing repair", "FlyGym teacher | forward")
COLORS = ("#424858", "#267754", "#b54b38", "#205dcb", "#7e4d9c")
TILE_PALETTE = {
    "initial_flat": (.40, .50, .50, 1), "raised_block": (.54, .48, .36, 1),
    "other_block": (.34, .31, .25, 1), "bridge": (.48, .54, .62, 1),
    "gap_bottom": (.21, .24, .28, 1), "exit_flat": (.34, .56, .43, 1),
}
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
BOLD = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
SIZE = (320, 224)


def read(path):
    return json.loads(path.read_text())


def lane_files(seed, mode):
    if mode == "forward_probe":
        folder = TRIAL/"TEACHER"/f"seed{seed}"/mode
        return folder/"BODY.mjb", folder/"TRACE.npz"
    folder = TRIAL/"FORMAL"/f"seed{seed}"
    return folder/f"{mode}.mjb", folder/mode/"TRACE.npz"


def load(path, *keys):
    with np.load(path, allow_pickle=False) as trace:
        return {key: trace[key] for key in keys}


def figures():
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex="col")
    for idx, seed in enumerate(SEEDS):
        for mode, label, color in zip(MODES, LABELS, COLORS, strict=True):
            _, source = lane_files(seed, mode)
            trace = load(source, "position", "yaw", "upright")
            pose = trace["position"]
            time = (np.arange(len(pose))+1) / 10000
            axes[idx,0].plot(time, pose[:,0], label=label, color=color, lw=1.5)
            axes[idx,1].plot(time, pose[:,2], color=color, lw=1.4)
            axes[idx,2].plot(time, np.rad2deg(np.unwrap(trace["yaw"])-trace["yaw"][0]), color=color, lw=1.4)
        axes[idx,0].axhspan(4.935, 8.9, color="#bca467", alpha=.12)
        axes[idx,0].axhspan(9.4, 9.7, color="#294564", alpha=.14)
        axes[idx,0].axhspan(10.7, 11., color="#294564", alpha=.14)
        axes[idx,0].axhline(11., color="#3d765a", lw=1, ls="--")
        axes[idx,1].axhline(.7*.99047366585408, color="#6f6f6f", lw=1, ls="--")
        for ax in axes[idx]:
            ax.axvline(1.5, color="gray", lw=.8, ls=":")
            ax.set_xlim(0, 2)
            ax.grid(alpha=.18)
        axes[idx,0].set_ylabel(f"Phase {seed}")
    for j,title in enumerate(("Thorax X (mm) | exit at 11", "Thorax height (mm)", "Yaw change (degrees)")):
        axes[0,j].set_title(title)
        axes[1,j].set_xlabel("Time (s)")
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(MODES), frameon=False)
    fig.suptitle("FlyGym 2.1.0 MixedTerrainWorld | two matched phases", fontweight="bold")
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.savefig(TRIAL/"ARENA_METRICS.png", dpi=145)
    plt.close(fig)


def renderer(seed, mode, model, tiles):
    mjdata = mujoco.MjData(model)
    ren = mujoco.Renderer(model,height=SIZE[1],width=SIZE[0])
    _, path = lane_files(seed, mode)
    trace = load(path, "qpos", "position")
    return mjdata, ren, trace


def frame(bundle, tick, label, seed, color, fps_factor):
    model, mjdata, ren, trace = bundle
    if tick >= len(trace["qpos"]):
        im = Image.new("RGB", SIZE, color=(26,28,32))
        ImageDraw.Draw(im).text((10,100), "Run stopped on numerical warning", fill="white", font=FONT)
        return np.asarray(im)
    mjdata.qpos[:] = trace["qpos"][tick]
    mujoco.mj_kinematics(model, mjdata)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [trace["position"][tick,0]+.75, trace["position"][tick,1], .30]
    camera.distance = 7.4
    camera.azimuth = 134.
    camera.elevation = -31.
    ren.update_scene(mjdata,camera)
    im=Image.fromarray(ren.render())
    draw=ImageDraw.Draw(im)
    draw.rectangle((0,0,SIZE[0],40),fill=(24,27,32))
    draw.rectangle((0,0,4,40),fill=color)
    draw.text((10,2),label,font=BOLD,fill="#ffffff")
    draw.text((10,22),f"Phase {seed}    t={(tick+1)/10000:.3f}s    {fps_factor}x playback",font=FONT,fill="#d7dee9")
    return np.asarray(im)


def video():
    tiles = read(TRIAL/"FORMAL/seed10901/ARENA.json")["tiles"]
    model_file, _ = lane_files(SEEDS[0],MODES[0])
    model = mujoco.MjModel.from_binary_path(str(model_file))
    for tile in tiles:
        model.geom_rgba[tile["id"]] = TILE_PALETTE.get(tile["terrain_class"],(.42,.43,.44,1))
    model.mat_reflectance[:] = 0
    bundles = [[(model, *renderer(seed,mode,model,tiles)) for mode in MODES] for seed in SEEDS]

    def combine(tick):
        rows = []
        for seed, lanes in zip(SEEDS, bundles, strict=True):
            rows.append(np.concatenate([frame(bundle,tick,label,seed,color,.25)
                                         for bundle,label,color in zip(lanes,LABELS,COLORS,strict=True)],axis=1))
        return np.concatenate(rows,axis=0)

    try:
        observations = [2250, 8500, 17250]
        Image.fromarray(np.concatenate([combine(t) for t in observations],axis=0)).save(TRIAL/"ARENA_FRAMES.png")
        with imageio.get_writer(str(TRIAL/"MIXED_TERRAIN_COMPARISON.mp4"),fps=25,
                                quality=7,codec="libx264",macro_block_size=16) as writer:
            for tick in range(0,20000,100):
                writer.append_data(combine(tick))
    finally:
        for lanes in bundles:
            for _, _, ren, _ in lanes:
                ren.close()


if __name__ == "__main__":
    figures()
    video()
