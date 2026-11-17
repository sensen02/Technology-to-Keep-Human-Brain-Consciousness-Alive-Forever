"""Render recorded, synchronized gap/control trajectories; no physics integration."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import subprocess
import sys

import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gap_policy import HERE, LEG_ORDER, checksum, read, write

BENCH = HERE.parent / "teacher"
WIDTH, HEIGHT, VISUAL, FPS = 620, 380, 314, 30
RF = LEG_ORDER.index("RF")
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def load_trial(path, caption):
    summary = read(path / "RESULT.json")
    trace = path / "TRACE.npz"
    if checksum(trace) != summary["trace_sha256"]:
        raise ValueError(f"Trace identity differs in {path}")
    with np.load(trace, allow_pickle=False) as stored:
        data = {key: stored[key] for key in ("qpos", "qvel", "loads", "padz", "rf_pad_xyz")}
    return dict(path=path, caption=caption, result=summary, data=data, trace=trace)


def camera(point, azimuth, elevation, distance):
    view = mj.MjvCamera()
    view.type = mj.mjtCamera.mjCAMERA_FREE
    view.azimuth = azimuth
    view.elevation = elevation
    view.distance = distance
    view.lookat[:] = point
    return view


def pose_image(trial, ms, detail=False):
    saved = trial["data"]
    data = trial["mujoco_data"]
    data.qpos[:] = saved["qpos"][ms]
    data.qvel[:] = saved["qvel"][ms]
    mj.mj_forward(trial["model"], data)
    np.testing.assert_array_equal(data.qpos, saved["qpos"][ms])
    renderer = trial["renderer"]
    renderer.update_scene(data, camera=trial["detail_camera"] if detail else trial["wide_camera"])
    return Image.fromarray(renderer.render())


def render(output, trials, event_ms, anchor_role, stationary_only=False):
    sys.path.insert(0, str(BENCH))
    from geometry import make_gap_model
    from walking_bench import W, stand

    def forbidden(*args, **kwargs):
        raise RuntimeError("Rendering cannot integrate dynamics")

    mj.mj_step = mj.mj_step1 = mj.mj_step2 = forbidden
    if len({len(trial["data"]["qpos"]) for trial in trials}) != 1:
        raise ValueError("Paired playback requires the same interval in both arms")
    final = len(trials[0]["data"]["qpos"]) - 1
    loss, bottom, reload = event_ms
    if not 500 < loss <= bottom < reload < final:
        raise ValueError("Check ordered event frame indices against the accepted evaluation")
    shapes = [arm["result"]["scenario"] for arm in trials if arm["result"].get("scenario")]
    if not shapes:
        raise ValueError("At least one trial should include the actual gap model")
    x0, x1, y0, y1 = shapes[-1]["opening"]
    center = np.array([(x0 + x1)/2., (y0 + y1)/2., 0.])
    for trial in trials:
        model = make_gap_model(trial["result"].get("scenario"))
        model.geom_friction[:, 0] = stand.physics.rc.MU
        wide_xy = trial["data"]["qpos"][:, :2].mean(0)
        trial.update(model=model, mujoco_data=mj.MjData(model),
                     wide_camera=camera([*wide_xy, .82], 125, -12, 5.55),
                     detail_camera=camera(center + [0., .16, .18], 250, -36, 1.7))

    near_start = max(0, loss - 70)
    near_end = min(final, reload + 120)
    moments = [(anchor_role + "失载前", near_start), (anchor_role + "实际失载", loss),
               (anchor_role + "右前足最低时刻", bottom), (anchor_role + "重新持续承重", reload),
               ("两侧终段状态", final)]
    indices = sorted(set(range(0, final+1, 20)) | set(range(near_start, near_end+1, 5)) |
                     {ms for _, ms in moments})
    heading = ImageFont.truetype(FONT, 21)
    caption = ImageFont.truetype(FONT, 15)

    def frame(ms, note=None, detail=False, inset=False):
        image = Image.new("RGB", (WIDTH*len(trials), HEIGHT), (245, 247, 246))
        drawing = ImageDraw.Draw(image)
        for col, trial in enumerate(trials):
            x = WIDTH*col
            image.paste(pose_image(trial, ms, detail), (x, HEIGHT-VISUAL))
            if inset:
                zoom = pose_image(trial, ms, detail=True).resize((218, 116), Image.Resampling.LANCZOS)
                image.paste((245, 247, 246), (x + WIDTH - 231, 71, x + WIDTH - 7, 193))
                image.paste(zoom, (x + WIDTH - 228, 74))
            trial_loads = trial["data"]["loads"][ms] / W
            z = trial["data"]["rf_pad_xyz"][ms, 2]
            right = trial_loads[RF]
            rest = trial_loads.sum() - right
            stage = "起始站立" if ms < 500 else "行走" if ms < 10100 else "撤令停稳"
            if stationary_only and ms >= loss:
                stage = "踏空后停稳诊断"
            drawing.text((x+11, 5), f"{trial['caption']} · {note or stage}", font=heading, fill="#203030")
            drawing.text((x+11, 39), f"{ms/1000:.3f} s  右前足 {right:.3f} W  其余足 {rest:.3f} W  足垫高 {z:.3f}",
                         font=caption, fill="#394b48")
        return image

    output.mkdir()
    target = output / "踏空配对回放.mp4"
    montage_path = output / "踏空关键帧.png"
    with ExitStack() as handles:
        for trial in trials:
            trial["renderer"] = handles.enter_context(mj.Renderer(trial["model"], height=VISUAL, width=WIDTH))
        movie = handles.enter_context(imageio.get_writer(target, fps=FPS, codec="libx264", quality=8,
                                                          macro_block_size=2, ffmpeg_log_level="error"))
        for ms in indices:
            movie.append_data(np.asarray(frame(ms, inset=near_start <= ms <= near_end)))
        montage = Image.new("RGB", (WIDTH*len(trials), HEIGHT*len(moments)))
        for i, (note, ms) in enumerate(moments):
            montage.paste(frame(ms, note=note, detail=ms != final), (0, i*HEIGHT))
        montage.save(montage_path)
    decoded = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                              "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(target)],
                             capture_output=True, check=True, text=True)
    if int(decoded.stdout.strip()) != len(indices):
        raise ValueError("Video frame count differs from recorded pose schedule")
    for trial in trials:
        if checksum(trial["trace"]) != trial["result"]["trace_sha256"]:
            raise ValueError("Saved physical trajectory changed during rendering")
    manifest = dict(video=str(target), video_sha256=checksum(target), montage=str(montage_path),
                    montage_sha256=checksum(montage_path), renderer_sha256=checksum(Path(__file__)),
                    actual_integrations=0, fps=FPS, milliseconds=indices, decoded_frames=len(indices),
                    stationary_recovery_diagnostic_only=stationary_only,
                    highlights=moments, runs=[dict(path=str(trial["path"]), caption=trial["caption"],
                                                   trace_sha256=trial["result"]["trace_sha256"],
                                                   scenario=trial["result"]["scenario"])
                                              for trial in trials])
    write(output / "VIDEO.json", manifest)
    print(f"Decoded {len(indices)} stored frames", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--events-ms", type=int, nargs=3, required=True, metavar=("LOSS", "LOWEST", "RELOAD"))
    parser.add_argument("--control-caption", default="冻结步态")
    parser.add_argument("--target-caption", default="学习收腿与补撑")
    parser.add_argument("--anchor-role", default="学习侧")
    parser.add_argument("--stationary-only", action="store_true")
    opts = parser.parse_args()
    render(opts.output.resolve(), [load_trial(opts.control.resolve(), opts.control_caption),
                                   load_trial(opts.student.resolve(), opts.target_caption)], opts.events_ms,
           opts.anchor_role, opts.stationary_only)
