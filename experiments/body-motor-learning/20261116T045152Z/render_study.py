"""Render one or two saved body trajectories with no dynamics steps."""
import argparse
from pathlib import Path
import subprocess
import sys

import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "20261116T000536Z"))
from walking_bench import read, stand

WIDTH, HEIGHT, FPS = 590, 410, 30
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def load(source, caption):
    summary = read(source / "RESULT.json")
    trace = source / "TRACE.npz"
    assert stand.sha(trace) == summary["trace_sha256"]
    with np.load(trace, allow_pickle=False) as z:
        return dict(source=source, caption=caption, summary=summary, trace=trace,
                    qpos=z["qpos"], qvel=z["qvel"], com=z["com"])


def render(output, runs):
    output.mkdir()
    def forbidden(*args, **kwargs):
        raise RuntimeError("Only saved trajectory configurations may be used")

    mj.mj_step = mj.mj_step1 = mj.mj_step2 = forbidden
    core = stand.physics.Core(adhesion=False)
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.azimuth = 132
    camera.elevation = -9
    camera.distance = 6.4
    camera.lookat[:] = [*runs[0]["qpos"][:, :2].mean(0), .92]
    heading = ImageFont.truetype(FONT, 21)
    detail = ImageFont.truetype(FONT, 16)
    panel_width = WIDTH * len(runs)
    max_ms = max(len(run["qpos"])-1 for run in runs)
    indices = list(range(0, max_ms+1, 20))
    if max_ms != indices[-1]:
        indices.append(max_ms)
    spots = {0, 500, 3700, 3720, 5200, 9800, 10200, max_ms}
    samples = []
    destination = output / "同次闭环回放.mp4"
    with mj.Renderer(core.plant, height=HEIGHT-60, width=WIDTH) as renderer, imageio.get_writer(
            destination, fps=FPS, codec="libx264", quality=8, macro_block_size=2,
            ffmpeg_log_level="error") as writer:
        for ms in indices:
            frame = Image.new("RGB", (panel_width, HEIGHT), (245, 245, 245))
            draw = ImageDraw.Draw(frame)
            for j, run in enumerate(runs):
                k = min(ms, len(run["qpos"])-1)
                pos, vel = run["qpos"][k], run["qvel"][k]
                core.d.qpos[:] = pos
                core.d.qvel[:] = vel
                mj.mj_forward(core.plant, core.d)
                np.testing.assert_array_equal(core.d.qpos, pos)
                renderer.update_scene(core.d, camera=camera)
                x = j*WIDTH
                frame.paste(Image.fromarray(renderer.render()), (x, 58))
                summary = run["summary"]
                walk_ms = summary.get("walk_ms", 9600)
                stage = "静立" if k < 500 else "直走" if k < 500+walk_ms else "停步"
                if ms > k:
                    stage = "轨迹已结束"
                status = "联合门通过" if summary["joint_sampled_pass"] else "联合门未通过"
                if summary.get("which") == "walking" or summary.get("pulse") == "push":
                    onset = summary.get("onset_ms", 3700)
                    duration = summary.get("duration_ms", 20)
                    if onset <= k < onset + duration:
                        stage += " · 侧推施力"
                draw.text((x+12, 4), f'{run["caption"]} · {stage}', font=heading, fill=(30, 40, 47))
                draw.text((x+13, 34), f"{k/1000:.2f} s · {status} · 前进 {run['com'][k,0]-run['com'][0,0]:+.3f}",
                          font=detail, fill=(50, 60, 58))
            writer.append_data(np.asarray(frame))
            if ms in spots:
                samples.append(frame.resize((panel_width//2, HEIGHT//2)))
    montage = Image.new("RGB", (panel_width, HEIGHT//2*((len(samples)+1)//2)), (245, 245, 245))
    for j, frame in enumerate(samples):
        montage.paste(frame, (panel_width//2*(j%2), HEIGHT//2*(j//2)))
    montage.save(output / "关键帧.png")
    decoded = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                              "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(destination)],
                             capture_output=True, check=True, text=True)
    assert int(decoded.stdout.strip()) == len(indices)
    for run in runs:
        assert stand.sha(run["trace"]) == run["summary"]["trace_sha256"]
    stand.put(output / "VIDEO.json", dict(video=str(destination), sha256=stand.sha(destination),
              renderer_sha256=stand.sha(Path(__file__)),
              decoded_frames=len(indices), fps=FPS, milliseconds=indices, extra_integrations=0,
              runs=[dict(source=str(r["source"]), trace_sha256=r["summary"]["trace_sha256"],
                         caption=r["caption"]) for r in runs]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("caption")
    parser.add_argument("--compare", nargs=2, metavar=("SOURCE", "CAPTION"))
    args = parser.parse_args()
    runs = [load(args.source.resolve(), args.caption)]
    if args.compare:
        runs.append(load(Path(args.compare[0]).resolve(), args.compare[1]))
    render(args.output.resolve(), runs)
