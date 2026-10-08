"""Smoke-probe the FlyGym 2.1.0 backend before building the embodied runner.

Answers, with evidence rather than assumptions:
  1. do the imports work?
  2. does the locomotion fly + flat-ground world compile and step?
  3. what exactly do the observation getters return (shapes/types)?
  4. does OFFSCREEN RENDERING work, and under which MUJOCO_GL backend?
  5. does the official tripod CPG baseline actually make the fly translate?

Rendering matters because the approved plan requires long results to be checked
by actually looking at frames.  This machine has no CUDA GPU and no libOSMesa, so
the GL backends are probed in order and the working one is recorded, never assumed.

Each backend runs in its own SUBPROCESS because MUJOCO_GL is read when MuJoCo is
imported and cannot be changed inside one interpreter.

Run with the BODY environment:
    ./venv_body/bin/python probe_body_backend.py
"""
from __future__ import annotations

import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs", "embodied_body")
PY = os.path.join(HERE, "venv_body", "bin", "python")

STAGE_SRC = r'''
import json, os, sys, traceback
import numpy as np

OUT = {}

def note(stage, **kw):
    OUT.setdefault("stages", {})[stage] = kw
    print("PROBE_PARTIAL " + json.dumps(OUT, default=str), flush=True)

def main():
    from flygym.compose import ActuatorType, FlatGroundWorld
    from flygym.simulation import Renderer, Simulation
    from flygym.utils.math import Rotation3D
    from flygym_demo.complex_terrain.common import (
        apply_locomotion_action, get_default_locomotion_dof_order, make_locomotion_fly)
    from flygym_demo.complex_terrain.cpg_controller import (
        CPGController, make_tripod_cpg_network)
    from flygym_demo.complex_terrain.preprogrammed import PreprogrammedSteps
    import flygym, mujoco

    OUT["env"] = {"mujoco_gl": os.environ.get("MUJOCO_GL"),
                  "mujoco": mujoco.__version__, "numpy": np.__version__,
                  "flygym_has_version": hasattr(flygym, "__version__")}

    fly = make_locomotion_fly("nmf")
    world = FlatGroundWorld(name="flat_world", half_size=20)
    world.add_fly(fly, spawn_position=[0.0, 0.0, 0.5],
                  spawn_rotation=Rotation3D("quat", [1.0, 0.0, 0.0, 0.0]))
    model, data = world.compile()
    note("build",
         nbody=int(model.nbody), nq=int(model.nq), nv=int(model.nv),
         nu=int(model.nu), ngeom=int(model.ngeom),
         timestep=float(model.opt.timestep), fly_name=fly.name)

    bodysegs = [str(b) for b in fly.get_bodysegs_order()]
    OUT["bodysegs"] = bodysegs
    OUT["thorax_index"] = bodysegs.index("c_thorax") if "c_thorax" in bodysegs else 0

    sim = Simulation(world, timestep=1e-4)
    sim.warmup()
    OUT["sim"] = {"timestep": float(sim.timestep)}

    def shape_of(v):
        if isinstance(v, (tuple, list)):
            return [shape_of(x) for x in v]
        a = np.asarray(v)
        return {"shape": list(a.shape), "dtype": str(a.dtype)}

    def probe(name, fn):
        try:
            OUT.setdefault("observations", {})[name] = shape_of(fn())
        except Exception as exc:
            OUT.setdefault("observations", {})[name] = "ERR %s: %s" % (type(exc).__name__, exc)

    probe("joint_angles", lambda: sim.get_joint_angles(fly.name))
    probe("joint_velocities", lambda: sim.get_joint_velocities(fly.name))
    probe("body_positions", lambda: sim.get_body_positions(fly.name))
    probe("body_rotations", lambda: sim.get_body_rotations(fly.name))
    probe("ground_contact_info", lambda: sim.get_ground_contact_info(fly.name))
    probe("actuator_forces_position",
          lambda: sim.get_actuator_forces(fly.name, ActuatorType.POSITION))
    probe("site_positions", lambda: sim.get_site_positions(fly.name))
    note("observations_done", n=len(OUT.get("observations", {})))

    # ---- CPG-driven stepping: ENGINEERING BASELINE, NOT connectome-driven ----
    order = get_default_locomotion_dof_order()
    ctrl = CPGController(make_tripod_cpg_network(timestep=sim.timestep),
                         PreprogrammedSteps(), output_dof_order=order)
    ti = OUT["thorax_index"]
    p0 = np.asarray(sim.get_body_positions(fly.name))[ti].copy()
    contact_hist = []
    for k in range(5000):
        apply_locomotion_action(sim, fly.name, ctrl.step())
        sim.step()
        if k % 500 == 0:
            c = sim.get_ground_contact_info(fly.name)
            contact_hist.append(int(np.asarray(c[0]).sum()))
    p1 = np.asarray(sim.get_body_positions(fly.name))[ti].copy()
    OUT["cpg_baseline"] = {
        "n_dof": len(order), "sim_time_s": float(sim.time), "n_steps": 5000,
        "thorax_start_m": p0.tolist(), "thorax_end_m": p1.tolist(),
        "horizontal_displacement_mm": float(np.linalg.norm(p1[:2] - p0[:2]) * 1000),
        "thorax_height_end_mm": float(p1[2] * 1000),
        "contact_sums_over_time": contact_hist,
    }
    note("cpg_baseline_done",
         disp_mm=OUT["cpg_baseline"]["horizontal_displacement_mm"])

    # ---- offscreen render (the MUJOCO_GL-dependent part) --------------------
    renderer = Renderer(sim, camera="track")
    sim.set_renderer(renderer)
    for _ in range(5):
        sim.step()
    frame = np.asarray(renderer.render())
    OUT["render"] = {"shape": list(frame.shape), "dtype": str(frame.dtype),
                     "std": float(frame.std()), "mean": float(frame.mean()),
                     "nonblank": bool(frame.std() > 1.0)}
    if OUT["render"]["nonblank"]:
        from PIL import Image
        img = frame if frame.dtype == np.uint8 else (255 * np.clip(frame, 0, 1)).astype("uint8")
        suffix = os.environ.get("MUJOCO_GL") or "default"
        path = os.path.join(os.environ["PROBE_OUT"], "render_%s.png" % suffix)
        Image.fromarray(img).save(path)
        OUT["render"]["saved"] = path
        try:
            eye = np.asarray(sim.get_ommatidia_readouts(fly.name))
            OUT["render"]["ommatidia_shape"] = list(eye.shape)
        except Exception as exc:
            OUT["render"]["ommatidia_err"] = "%s: %s" % (type(exc).__name__, exc)

try:
    main()
except Exception as exc:
    OUT["fatal"] = "%s: %s" % (type(exc).__name__, exc)
    OUT["traceback_tail"] = traceback.format_exc().splitlines()[-10:]
finally:
    print("PROBE_JSON " + json.dumps(OUT, default=str))
'''


def run_backend(gl):
    env = dict(os.environ)
    env["PROBE_OUT"] = OUT
    if gl is None:
        env.pop("MUJOCO_GL", None)
    else:
        env["MUJOCO_GL"] = gl
    proc = subprocess.run([PY, "-c", STAGE_SRC], capture_output=True, text=True,
                          env=env, timeout=2400)
    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith("PROBE_JSON "):
            payload = json.loads(line[len("PROBE_JSON "):])
    if payload is None:
        payload = {"fatal": "no payload",
                   "stderr_tail": proc.stderr.splitlines()[-8:],
                   "returncode": proc.returncode}
    render = payload.get("render") or {}
    verdict = ("renders" if render.get("nonblank")
               else "no_render" if render else "failed_before_render")
    return {"mujoco_gl": gl or "default", "verdict": verdict, "payload": payload}


def main():
    os.makedirs(OUT, exist_ok=True)
    result = {"probes": []}
    for gl in (None, "egl", "osmesa"):
        try:
            r = run_backend(gl)
        except subprocess.TimeoutExpired:
            r = {"mujoco_gl": gl or "default", "verdict": "timeout", "payload": {}}
        result["probes"].append(r)
        print(f"[{r['mujoco_gl']}] {r['verdict']}")
        if r["payload"].get("fatal"):
            print("    fatal:", r["payload"]["fatal"])
        if r["verdict"] == "renders":
            result["chosen_backend"] = r["mujoco_gl"]
            break
    with open(os.path.join(OUT, "body_backend_probe.json"), "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    chosen = result["probes"][-1]["payload"]
    print(json.dumps({k: chosen.get(k) for k in
                      ("build", "observations", "cpg_baseline", "render", "fatal")},
                     indent=2, default=str)[:3000])


if __name__ == "__main__":
    main()
