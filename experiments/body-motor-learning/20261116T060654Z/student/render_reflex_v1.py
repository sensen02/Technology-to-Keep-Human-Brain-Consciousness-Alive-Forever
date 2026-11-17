"""Render synchronized body poses from the recorded obstacle and control trials."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import subprocess
import sys

import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from stumble_policy import HERE, checksum, read, write

sys.path.insert(0, str(HERE.parent / "teacher"))
from geometry import make_obstacle_model
from walking_bench import LEGS, W, stand

WIDTH, HEIGHT, FPS = 620, 380, 30
VISUAL = HEIGHT-66
RF = LEGS.index("RF")
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def load(source, caption):
    result = read(source / "RESULT.json")
    trace = source / "TRACE.npz"
    if checksum(trace) != result["trace_sha256"]:
        raise ValueError(f"Trace does not match safety metrics {source}")
    with np.load(trace, allow_pickle=False) as z:
        rows = {key: z[key] for key in ("qpos", "qvel", "com", "loads", "padz", "obstacle")}
    return dict(source=source, caption=caption, result=result, rows=rows)


def camera(look, azimuth, elevation, distance):
    out = mj.MjvCamera()
    out.type = mj.mjtCamera.mjCAMERA_FREE
    out.azimuth = azimuth
    out.elevation = elevation
    out.distance = distance
    out.lookat[:] = look
    return out


def draw_pose(panel, frame_ms, detail=False):
    row = panel["rows"]
    k = min(frame_ms, len(row["qpos"])-1)
    data = panel["data"]
    data.qpos[:] = row["qpos"][k]
    data.qvel[:] = row["qvel"][k]
    mj.mj_forward(panel["model"], data)
    np.testing.assert_array_equal(data.qpos, row["qpos"][k])
    panel["renderer"].update_scene(data, camera=panel["near"] if detail else panel["camera"])
    return Image.fromarray(panel["renderer"].render())


def get_event(run):
    result = run["result"]
    event = result["event"]
    impact = event.get("first_resisting_touch_ms", event.get("obstacle_first_contact_ms"))
    if impact is None:
        return {}
    reload = event.get("first_target_floor_reload_ms")
    span = run["rows"]["padz"][impact:min(len(run["rows"]["padz"]), impact+500), RF]
    peak = int(impact + int(np.argmax(span)))
    return dict(impact=impact, peak=peak, reload=reload)


def render(output, runs):
    def forbidden(*_a, **_k):
        raise RuntimeError("Rendering replays existing qpos only")

    mj.mj_step = mj.mj_step1 = mj.mj_step2 = forbidden
    output.mkdir()
    end = max(len(run["rows"]["qpos"])-1 for run in runs)
    events = [get_event(run) for run in runs]
    impact = min((event["impact"] for event in events if event), default=2000)
    local_end = max((val for event in events for val in event.values() if val is not None), default=impact+350)
    interval = (max(0, impact-55), min(end, max(impact+140, local_end+35)))
    detail_k = list(range(interval[0], interval[1]+1, 5))
    indices = sorted(set(range(0, end+1, 20)) | set(detail_k) |
                     {val for event in events for val in event.values() if val is not None} | {end})
    large = ImageFont.truetype(FONT, 21)
    small = ImageFont.truetype(FONT, 15)
    x0 = runs[0]["rows"]["qpos"][:, :2].mean(0)
    fallback_center = next((run["result"]["scenario"]["center"] for run in runs
                            if run["result"].get("scenario") is not None), [0.85, -0.73, 0.14])
    for run in runs:
        model = make_obstacle_model(run["result"].get("scenario"))
        model.geom_friction[:,0] = stand.physics.rc.MU
        run.update(model=model, data=mj.MjData(model),
                   camera=camera([*x0, .94], 126, -11, 5.9),
                   near=camera([fallback_center[0], fallback_center[1]+.09, fallback_center[2]+.16],
                               239, -15, 1.55))
    last_event = events[-1] or dict(impact=impact, peak=impact+100, reload=None)
    named = [("碰障前", max(0, last_event["impact"]-45)),
             ("第一次触碰", last_event["impact"]),
             ("恢复窗最高抬足", last_event["peak"])]
    if last_event["reload"] is not None:
        named.append(("首次恢复足端承重", last_event["reload"]))
    named.append(("末端站立", end))

    def frame(ms, detail=False, inset=False, title=None):
        canvas = Image.new("RGB", (WIDTH*len(runs), HEIGHT), (246, 247, 245))
        dr = ImageDraw.Draw(canvas)
        for i, run in enumerate(runs):
            x = i*WIDTH
            k = min(ms, len(run["rows"]["qpos"])-1)
            rgb = draw_pose(run, ms, detail)
            canvas.paste(rgb,(x, 65))
            if inset and not detail:
                crop = draw_pose(run, ms, True).resize((222, 113), Image.Resampling.LANCZOS)
                canvas.paste((241,244,241),(x+WIDTH-239,73,x+WIDTH-10,193))
                canvas.paste(crop, (x+WIDTH-235,77))
            y = run["rows"]
            info = run["result"]
            stage = ("起始站立" if k < 500 else "行走" if k < 500+info["walk_ms"] else "撤令站立")
            if k < ms:
                stage = "轨迹停止"
            if title:
                stage = title
            labels = f'{run["caption"]} · {stage}'
            typ = f"{k/1000:.3f} s  右前足地载 {y['loads'][k,RF]/W:.3f} W"
            typ += f"  障碍逆向力 {y['obstacle'][k,3]:.3f} W"
            dr.text((x+12, 4), labels, font=large, fill="#172d2f")
            dr.text((x+12, 38), typ, font=small, fill="#415354")
        return canvas

    destination = output/"绊脚配对回放.mp4"
    montage_destination = output/"绊脚关键帧.png"
    with ExitStack() as stack:
        for run in runs:
            run["renderer"] = stack.enter_context(mj.Renderer(run["model"],height=VISUAL,width=WIDTH))
        writer = stack.enter_context(imageio.get_writer(
            destination, fps=FPS, codec="libx264", quality=8, macro_block_size=2,
            ffmpeg_log_level="error"))
        for ms in indices:
            writer.append_data(np.asarray(frame(ms,inset=interval[0] <= ms <= interval[1])))
        montage = Image.new("RGB",(WIDTH*len(runs), HEIGHT*len(named)), (246,247,245))
        for n, (title, ms) in enumerate(named):
            montage.paste(frame(ms, detail=ms != end, title=title),(0, HEIGHT*n))
        montage.save(montage_destination)
    decoded = subprocess.run(["ffprobe","-v","error","-count_frames","-select_streams","v:0",
        "-show_entries","stream=nb_read_frames","-of","csv=p=0",str(destination)],
        capture_output=True,check=True,text=True)
    assert int(decoded.stdout.strip()) == len(indices)
    for run in runs:
        assert checksum(run["source"]/"TRACE.npz") == run["result"]["trace_sha256"]
    write(output/"VIDEO.json", dict(video=str(destination),video_sha256=checksum(destination),
        montage=str(montage_destination),montage_sha256=checksum(montage_destination),
        renderer_sha256=checksum(Path(__file__)), actual_integrations=0, fps=FPS,
        milliseconds=indices, decoded_frames=len(indices), highlights=named,
        runs=[dict(source=str(run["source"]),caption=run["caption"],scenario=run["result"].get("scenario"),
                   trace_sha256=run["result"]["trace_sha256"]) for run in runs]))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("source",type=Path)
    p.add_argument("caption")
    p.add_argument("--compare",nargs=2,metavar=("SOURCE","CAPTION"),required=True)
    args=p.parse_args()
    render(args.output.resolve(),[load(args.source.resolve(),args.caption),
                                  load(Path(args.compare[0]).resolve(),args.compare[1])])
