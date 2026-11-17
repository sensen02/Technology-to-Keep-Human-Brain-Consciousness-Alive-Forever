"""Render saved multi-state standing trajectories without integrating physics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
RUN = HERE / "RUN"
OUT = RUN / "VIDEO_DERIVED"
BODY = HERE.parents[1] / "banc-walking-recovery/20261110T152714Z"
sys.path.insert(0, str(BODY))
from physics import Core  # noqa: E402


ORDER = ("nominal", "lean_left", "lean_right", "moving_forward",
         "moving_backward", "uneven_feet")
ZH = {
    "nominal": "标称站姿",
    "lean_left": "左倾初态",
    "lean_right": "右倾初态",
    "moving_forward": "前移初态",
    "moving_backward": "后移初态",
    "uneven_feet": "足载不均初态",
}
CASES = (
    ("medium_x", "01_0.5W_纵向_六初态.mp4", "0.5 W 世界 +X（纵向）"),
    ("medium_y", "02_0.5W_侧向_六初态.mp4", "0.5 W 世界 +Y（侧向）"),
    ("strong_x", "03_0.8W_纵向_六初态.mp4", "0.8 W 世界 +X（纵向）"),
    ("strong_y", "04_0.8W_侧向_六初态_全部倒地.mp4", "0.8 W 世界 +Y（侧向）"),
)
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def panel_label(name: str, initial: dict, event: dict) -> tuple[str, str]:
    first = f"{ZH[name]}  初始倾角 {initial[name]['tilt_deg']:.2f}°"
    if event["recovered"]:
        second = f"恢复站稳：{event['recovery_ms']} ms"
    else:
        second = "未恢复（倒地后没有自主翻正）"
    return first, second


def render_case(core: Core, renderer: mj.Renderer, camera: mj.MjvCamera,
                option: mj.MjvOption, result: dict, initial: dict,
                case: str, filename: str, heading: str, source_hashes: dict) -> dict:
    traces = {}
    events = {}
    paths = {}
    for name in ORDER:
        row = result[f"{name}__{case}"]["model"]
        base = RUN / "MATRIX" / row["file"]
        native = base.with_suffix(".native.npz")
        paths[name] = native
        source_hashes[str(native)] = sha256(native)
        traces[name] = load_npz(native)
        events[name] = row["event"]
        if traces[name]["qpos_ms"].shape != (1256, core.plant.nq):
            raise ValueError(f"Unexpected qpos shape: {native}")
    indices = list(range(0, 1256, 10))
    if indices[-1] != 1255:
        indices.append(1255)
    video = OUT / filename
    title_font = ImageFont.truetype(FONT_PATH, 24)
    label_font = ImageFont.truetype(FONT_PATH, 17)
    small_font = ImageFont.truetype(FONT_PATH, 15)
    writer = imageio.get_writer(video, fps=30, codec="libx264", quality=8,
                                 macro_block_size=2, ffmpeg_log_level="error")
    try:
        for index in indices:
            t_ms = index
            canvas = Image.new("RGB", (960, 590), (242, 242, 242))
            draw = ImageDraw.Draw(canvas)
            phase = "施力中" if 20 <= t_ms < 55 else ("施力前" if t_ms < 20 else "撤力后")
            color = (185, 25, 25) if phase == "施力中" else (20, 20, 20)
            draw.text((16, 8), f"{heading}｜人工身体网络（800步权重）｜{t_ms} ms｜{phase}",
                      font=title_font, fill=color)
            draw.text((16, 40), "保存状态回放；每帧仅 mj_forward，物理积分为 0；约 4 倍慢放",
                      font=small_font, fill=(40, 40, 40))
            for slot, name in enumerate(ORDER):
                trace = traces[name]
                core.d.qpos[:] = trace["qpos_ms"][index]
                core.d.qvel[:] = trace["qvel_ms"][index]
                mj.mj_forward(core.plant, core.d)
                if not np.array_equal(core.d.qpos, trace["qpos_ms"][index]):
                    raise RuntimeError("Renderer changed qpos")
                camera.lookat[:] = trace["qpos_ms"][0, :3]
                renderer.update_scene(core.d, camera=camera, scene_option=option)
                picture = Image.fromarray(renderer.render())
                x = (slot % 3) * 320
                y = 70 + (slot // 3) * 260
                canvas.paste(picture, (x, y))
                draw.rectangle((x, y + 210, x + 319, y + 259), fill=(248, 248, 248))
                line1, line2 = panel_label(name, initial, events[name])
                draw.text((x + 7, y + 212), line1, font=label_font, fill=(20, 20, 20))
                line2_color = (15, 105, 25) if events[name]["recovered"] else (180, 20, 20)
                draw.text((x + 7, y + 236), line2, font=small_font, fill=line2_color)
            writer.append_data(np.asarray(canvas))
    finally:
        writer.close()
    return {
        "path": str(video),
        "sha256": sha256(video),
        "frames": len(indices),
        "fps": 30,
        "source_physical_ms": 1255,
        "playback_s": len(indices) / 30,
        "physics_integrations": 0,
        "case": case,
        "all_recovered": all(events[name]["recovered"] for name in ORDER),
        "events": {name: events[name] for name in ORDER},
    }


def fallen_seed_image(core: Core, renderer: mj.Renderer, camera: mj.MjvCamera,
                      option: mj.MjvOption, bank: dict, source_hashes: dict) -> Path:
    title_font = ImageFont.truetype(FONT_PATH, 25)
    label_font = ImageFont.truetype(FONT_PATH, 18)
    canvas = Image.new("RGB", (960, 390), (242, 242, 242))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 8), "翻正训练的两个倒地起点（不是自主翻正结果）", font=title_font,
              fill=(180, 20, 20))
    for slot, name in enumerate(("side_fallen", "near_supine")):
        path = RUN / "RIGHTING_SEEDS_VERIFIED" / f"{name}.npz"
        source_hashes[str(path)] = sha256(path)
        seed = load_npz(path)
        core.d.qpos[:] = seed["qpos"]
        core.d.qvel[:] = seed["qvel"]
        mj.mj_forward(core.plant, core.d)
        camera.lookat[:] = seed["qpos"][:3]
        renderer.update_scene(core.d, camera=camera, scene_option=option)
        picture = Image.fromarray(renderer.render()).resize((480, 315))
        canvas.paste(picture, (slot * 480, 48))
        row = bank["seeds"][name]
        zh = "侧倒" if name == "side_fallen" else "近仰卧"
        draw.rectangle((slot * 480, 330, slot * 480 + 479, 389), fill=(248, 248, 248))
        draw.text((slot * 480 + 10, 333), f"{zh}：{row['tilt_deg']:.2f}°", font=label_font,
                  fill=(20, 20, 20))
        draw.text((slot * 480 + 10, 360), "仅已验证可精确加载；尚无翻正控制轨迹",
                  font=label_font, fill=(180, 20, 20))
    path = OUT / "05_倒地起点_尚未自主翻正.png"
    canvas.save(path)
    return path


def main() -> None:
    OUT.mkdir(exist_ok=False)
    result_path = RUN / "MATRIX/RESULT.json"
    initial_path = RUN / "INITIAL/BANK.json"
    seed_bank_path = RUN / "RIGHTING_SEEDS_VERIFIED/BANK.json"
    result = json.loads(result_path.read_text())['cases']
    initial = json.loads(initial_path.read_text())
    seed_bank = json.loads(seed_bank_path.read_text())
    source_hashes = {str(path): sha256(path) for path in
                     (result_path, initial_path, seed_bank_path, Path(__file__).resolve())}

    # Rendering is deliberately prohibited from calling any MuJoCo integration routine.
    def forbidden(*_args, **_kwargs):
        raise RuntimeError("Physics integration is forbidden in saved-state renderer")

    mj.mj_step = forbidden
    mj.mj_step1 = forbidden
    mj.mj_step2 = forbidden
    core = Core(adhesion=False)
    renderer = mj.Renderer(core.plant, height=210, width=320)
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.distance = 6.2
    camera.azimuth = 140
    camera.elevation = -24
    option = mj.MjvOption()
    option.flags[mj.mjtVisFlag.mjVIS_TENDON] = False
    try:
        videos = [render_case(core, renderer, camera, option, result, initial, *spec,
                              source_hashes) for spec in CASES]
        fallen = fallen_seed_image(core, renderer, camera, option, seed_bank, source_hashes)
    finally:
        renderer.close()

    # Decode every produced video and save three representative decoded frames.
    decoded = {}
    for row in videos:
        path = Path(row["path"])
        reader = imageio.get_reader(path)
        frames = []
        picked = {}
        try:
            for i, pixels in enumerate(reader):
                frames.append(i)
                if i in (0, row["frames"] // 2, row["frames"] - 1):
                    output = OUT / f"{path.stem}_decoded_{i:03d}.png"
                    Image.fromarray(pixels).save(output)
                    picked[str(i)] = {"path": str(output), "sha256": sha256(output)}
        finally:
            reader.close()
        if len(frames) != row["frames"]:
            raise RuntimeError(f"Decoded {len(frames)} != {row['frames']}: {path}")
        decoded[path.name] = picked

    if any(sha256(Path(path)) != digest for path, digest in source_hashes.items()):
        raise RuntimeError("A frozen source changed while rendering")
    manifest = {
        "scope": "derived videos from immutable saved physical states",
        "videos": videos,
        "fallen_seed_image": {"path": str(fallen), "sha256": sha256(fallen)},
        "decoded_keyframes": decoded,
        "source_hashes": source_hashes,
        "source_unchanged": True,
        "physics_integrations": 0,
        "righting_controller_completed": False,
        "righting_status": "two exact fallen-state seeds only; no autonomous righting trajectory",
    }
    (OUT / "VIDEO_MANIFEST.json").write_text(json.dumps(manifest, indent=2,
                                                         ensure_ascii=False) + "\n")
    print(json.dumps({"videos": [row["path"] for row in videos],
                      "fallen": str(fallen), "manifest": str(OUT / 'VIDEO_MANIFEST.json')},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
