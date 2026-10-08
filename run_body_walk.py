"""Milestone 1: does the fly actually walk, and can I check that by looking?

WHAT THIS DELIVERS
------------------
A physical adult-fly body on flat ground, driven by FlyGym's own tripod CPG,
walked for a fixed window on a list of fixed seeds, with:
  * pre-registered engineering gates (declared before the run, not tuned after);
  * per-seed metrics in model units (MM, not metres);
  * recorded truth arrays (thorax pose, joint angles, contact, actuator forces);
  * rendered frames saved as a montage AND a video, which are then opened and
    looked at -- the plan requires long results to be checked visually.

WHAT THIS DOES NOT CLAIM
------------------------
The controller is FlyGym's demo CPG: an ENGINEERING BASELINE.  Nothing here is
connectome-driven, and a body that walks is not evidence about experience.  The
report records this in ``honesty`` and the figure title states it.

Run (BODY environment, separate from the neural venv):
    ./venv_body/bin/python run_body_walk.py --seeds 0,1,2,3,4,5,6,7,8,9 --seconds 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "outputs", "embodied_body")

from engine.embodied import BodyBackend, BodyConfig, select_gl_backend, summarise_walk  # noqa: E402


def walk_once(seed: int, seconds: float, gl, render_every_steps: int,
              cfg_kwargs: dict):
    cfg = BodyConfig(seed=seed, **cfg_kwargs)
    be = BodyBackend(cfg, gl_backend=gl).attach_cpg_baseline()
    n_steps = int(round(seconds / cfg.timestep_s))

    times, thorax, contacts, joints, raw_found = [], [], [], [], []
    renderer = None
    if render_every_steps > 0:
        # set_renderer() takes a CAMERA SPEC (the QUALIFIED name, measured as
        # "nmf/trackcam") and returns the Renderer it built.  Frame sampling is
        # governed by playback_speed/output_fps, so set them explicitly instead
        # of accepting the default (0.2) which would buffer thousands of frames.
        renderer = be.sim.set_renderer(be.camera_spec, camera_res=(240, 320),
                                       playback_speed=1.0, output_fps=25,
                                       buffer_frames=True)

    t0 = time.perf_counter()
    for k in range(n_steps):
        be.step()
        if k % 50 == 0:
            obs = be.observe()
            times.append(obs.time_s)
            thorax.append(obs.thorax_position_mm)
            contacts.append(obs.contact_present)
            raw_found.append(obs.contact_found_raw.copy())
            joints.append(obs.joint_angles_rad)
        if renderer is not None:
            be.sim.render_as_needed()   # honours playback_speed/output_fps
    wall = time.perf_counter() - t0

    frames = []
    if renderer is not None:
        frames = [np.asarray(f).copy()
                  for f in renderer.frames.get(be.camera_spec, [])]
    metrics = summarise_walk(times, thorax, contacts, cfg,
                             contact_found_raw=np.asarray(raw_found))
    metrics["wall_seconds"] = wall
    metrics["sim_seconds"] = float(times[-1])
    metrics["realtime_factor"] = float(times[-1] / wall) if wall > 0 else float("nan")
    metrics["n_steps"] = n_steps
    desc = be.describe()
    be.close()
    return {
        "seed": seed,
        "metrics": metrics,
        "describe": desc,
        "traces": {"times_s": np.asarray(times),
                   "thorax_mm": np.asarray(thorax),
                   "contact_present": np.asarray(contacts),
                   "contact_found_raw": np.asarray(raw_found),
                   "joint_angles_rad": np.asarray(joints)},
        "frames": frames,
    }


def make_montage(frames, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not frames:
        return None
    pick = np.linspace(0, len(frames) - 1, min(6, len(frames))).astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.4), layout="constrained")
    for ax, i in zip(axes.ravel(), pick):
        f = frames[i]
        img = f if f.dtype == np.uint8 else (255 * np.clip(f, 0, 1)).astype("uint8")
        ax.imshow(img)
        ax.set_title(f"frame {i}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[len(pick):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--render-every", type=int, default=5000)
    ap.add_argument("--gl", default=None, help="force a MUJOCO_GL backend")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    os.makedirs(OUT, exist_ok=True)
    t_start = time.perf_counter()

    if args.gl is not None:
        gl, probe = args.gl, [{"backend": args.gl, "ok": True, "forced": True}]
    else:
        gl, probe = select_gl_backend()
    print(f"GL backend: {gl!r}")

    runs, summary = [], []
    for seed in seeds:
        r = walk_once(seed, args.seconds, gl,
                      args.render_every if seed == seeds[0] else 0,
                      {"timestep_s": 1e-4, "spawn_position_mm": (0.0, 0.0, 0.5),
                       "world_half_size_mm": 1000.0})
        runs.append(r)
        m = r["metrics"]
        summary.append({k: m[k] for k in
                        ("duration_s", "horizontal_displacement_mm", "mean_speed_mm_s",
                         "thorax_height_min_mm", "thorax_height_max_mm",
                         "thorax_height_end_mm", "contact_legs_mean",
                         "passed_engineering_gates", "wall_seconds", "realtime_factor")})
        print(f"  seed {seed}: disp {m['horizontal_displacement_mm']:.2f} mm  "
              f"speed {m['mean_speed_mm_s']:.2f} mm/s  "
              f"gate {m['passed_engineering_gates']}  "
              f"wall {m['wall_seconds']:.1f}s ({m['realtime_factor']:.3f}x realtime)")

    n_pass = sum(1 for s in summary if s["passed_engineering_gates"])
    success_rate = n_pass / len(summary) if summary else 0.0

    montage = make_montage(
        runs[0]["frames"], os.path.join(OUT, "body_walk_montage.png"),
        "PHYSICAL FLY WALKING — FlyGym/NeuroMechFly v2 + MuJoCo, driven by FlyGym's "
        "tripod CPG (ENGINEERING BASELINE, NOT the connectome).\n"
        "Model units are MILLIMETRES (gravity -9810 mm/s^2). Not a claim about experience.")

    video = None
    if runs[0]["frames"]:
        try:
            import mediapy
            frames = [f if f.dtype == np.uint8 else (255 * np.clip(f, 0, 1)).astype("uint8")
                      for f in runs[0]["frames"]]
            video = os.path.join(OUT, "body_walk_seed%d.mp4" % seeds[0])
            mediapy.write_video(video, frames, fps=25)
        except Exception as exc:                                  # noqa: BLE001
            video = "unavailable: %s: %s" % (type(exc).__name__, exc)

    trace_path = os.path.join(OUT, "body_walk_traces.npz")
    np.savez_compressed(trace_path, **{f"seed{r['seed']}_{k}": v
                                       for r in runs for k, v in r["traces"].items()})

    report = {
        "milestone": "M1: a physical fly that walks on flat ground",
        "status": "engineering baseline walking; NOT connectome-driven",
        "question": "does the body translate, stay upright, and use its legs, "
                    "verified numerically AND by looking at rendered frames?",
        "honesty": [
            "the controller is FlyGym's demo tripod CPG: an ENGINEERING BASELINE, "
            "not the connectome and not a brain model",
            "a body that walks is NOT evidence about experience, perception or "
            "consciousness, and no such claim is made",
            "model units are millimetres; distances are NOT metres",
            "the engineering gates were declared in code before the run and were "
            "not tuned afterwards",
        ],
        "units": {"length": "mm", "time": "s", "angle": "rad",
                  "speed": "mm/s", "gravity": "mm/s^2 (-9810)"},
        "backend": {"probe": probe, "chosen_gl": gl,
                    "describe": runs[0]["describe"] if runs else None},
        "config": {"seeds": seeds, "seconds": args.seconds,
                   "timestep_s": 1e-4, "render_every_steps": args.render_every},
        "per_seed": summary,
        "n_seeds": len(seeds),
        "n_passed": n_pass,
        "success_rate": success_rate,
        "gate_declared_requirement": ">= 80% of pre-registered seeds pass",
        "passed_milestone_gate": bool(success_rate >= 0.80),
        "figures": {"montage": montage, "video": video, "traces": trace_path},
        "total_wall_seconds": time.perf_counter() - t_start,
    }
    with open(os.path.join(OUT, "body_walk_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(json.dumps({k: report[k] for k in
                      ("n_seeds", "n_passed", "success_rate", "passed_milestone_gate",
                       "total_wall_seconds")}, indent=2))
    print("montage:", montage)
    print("video:", video)
    return 0 if report["passed_milestone_gate"] else 1


if __name__ == "__main__":
    sys.exit(main())
