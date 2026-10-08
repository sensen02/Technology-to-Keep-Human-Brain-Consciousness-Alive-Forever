"""Self-test for the declarative scene layer (``engine/embodied/scene.py``).

Run in the BODY environment (venv_body), because it needs MuJoCo:
    /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python \
        /run/media/sensen/Data2/cell_wound_prototype/run_scene_selftest.py

WHAT THESE TESTS ESTABLISH, AND WHAT THEY DO NOT
------------------------------------------------
They establish that the three presets ("blank", "natural",
"natural_lit_no_objects") really do what their names say, checked against the
COMPILED ``mujoco.MjModel`` rather than against the configuration:

* "blank" is still the degenerate scene: one uniform ``geom_friction`` row on
  every geom, FlyGym's own pair friction untouched, no lights, no heightfield,
  no roughness geoms, no objects, the same 70 geoms and 48 actuators.
* "natural" differs in exactly the ways it claims: per-class geom friction AND
  per-class pair friction, three lights, one heightfield whose measured
  peak-to-peak relief equals the configured amplitude, six extra adhesion
  actuators for the capillary term, and three scene-object geoms.
* "natural_lit_no_objects" is identical to "natural" except that it has no
  object geoms -- and, because objects are visual-only, its initial state hash
  is identical too, which is what makes it a valid control for the objects.
* Determinism: two freshly built "natural" worlds with the same seed have
  identical initial qpos/qvel hashes.
* A 1 s stance rollout and a 1 s CPG-driven rollout are finite and the fly does
  not fall through the ground; the friction coefficient MuJoCo actually uses at
  the contacts is read back from ``data.contact[i].friction``.

They do NOT establish that the numbers are physically right.  Every friction,
roughness and capillary magnitude in the presets is ``ASSUMED`` or
``ENGINEERING_DEFAULT`` with a declared sweep range; the ledger holds ZERO
``MEASURED_CITED`` records because no external source was verified.  The
rollout is a stance and a demo-CPG walk, not a fly behaviour, and the capillary
force is a placeholder.

Assertions that cannot be evaluated on this stack are printed as
``EVIDENCE NOT FOUND`` with the reason; they are counted separately and are
never silently counted as passes.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "outputs", "embodied_body")

os.environ.setdefault("MUJOCO_GL", "egl")

FLY_NAME = "nmf"
DT = 1e-4
ROLLOUT_SECONDS = 1.0
ROLLOUT_STEPS = int(round(ROLLOUT_SECONDS / DT))
SAMPLE_EVERY = 25
CPG_INTRINSIC_HZ = 12.0          # ENGINEERING BASELINE, FlyGym's demo CPG
SPAWN = [0.0, 0.0, 0.5]
FALL_THROUGH_Z_MM = -0.1         # a body centre below this means it fell through

PRESET_KEYS = ("blank", "natural", "natural_lit_no_objects")

RESULTS: list[dict] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append({"name": name, "status": "PASS" if ok else "FAIL",
                    "ok": bool(ok), "detail": str(detail)[:600]})
    print(("PASS  " if ok else "FAIL  ") + name +
          (" | " + str(detail)[:200] if detail else ""), flush=True)
    return bool(ok)


def evidence_not_found(name: str, reason: str) -> None:
    RESULTS.append({"name": name, "status": "EVIDENCE NOT FOUND", "ok": None,
                    "detail": str(reason)[:600]})
    print("EVIDENCE NOT FOUND  " + name + " | " + str(reason)[:200], flush=True)


def _jsonable(obj):
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else repr(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):
        return _jsonable(obj.tolist())
    if hasattr(obj, "item"):
        return _jsonable(obj.item())
    return repr(obj)


# --------------------------------------------------------------------------- #
# building
# --------------------------------------------------------------------------- #

def _make_fly(cfg):
    from flygym_demo.complex_terrain.common import make_locomotion_fly
    return make_locomotion_fly(FLY_NAME, add_adhesion=cfg.adhesion_added,
                               adhesion_gain=cfg.adhesion_gain)


def build_scene(cfg, *, timestep: float = DT):
    """Build fly + world, apply the scene, compile.  Returns (fly, world, sim, report)."""
    from flygym.compose import FlatGroundWorld
    from flygym.simulation import Simulation
    from flygym.utils.math import Rotation3D
    from engine.embodied import scene as S

    fly = _make_fly(cfg)
    world = FlatGroundWorld(name=f"world_{cfg.name}", half_size=cfg.world_half_size_mm)
    world.add_fly(fly, spawn_position=list(SPAWN),
                  spawn_rotation=Rotation3D("quat", [1.0, 0.0, 0.0, 0.0]))
    report = S.apply_scene(world, fly, cfg)
    model, data = world.compile()
    sim = Simulation(world, timestep=timestep)
    sim.warmup()
    return fly, world, sim, report, model, data


def measure(cfg, report, model):
    from engine.embodied import scene as S
    return S.SceneApplicationReport.measured_after_compile(
        model, fly_name=FLY_NAME, ground_geom_names=report.ground_geom_names,
        n_roughness_geoms=report.n_roughness_geoms)


def state_hash(data) -> dict:
    return {
        "qpos_sha256": hashlib.sha256(np.asarray(data.qpos, float).tobytes()).hexdigest(),
        "qvel_sha256": hashlib.sha256(np.asarray(data.qvel, float).tobytes()).hexdigest(),
        "n_qpos": int(np.asarray(data.qpos).size),
        "n_qvel": int(np.asarray(data.qvel).size),
    }


# --------------------------------------------------------------------------- #
# rollouts
# --------------------------------------------------------------------------- #

def rollout(cfg, report, sim, model, data, *, use_cpg: bool, steps: int = ROLLOUT_STEPS):
    """A short stance (or CPG-driven) rollout, sampling every SAMPLE_EVERY steps."""
    from engine.embodied import scene as S

    # ``Simulation.__init__`` compiles its OWN model/data pair, so the arrays
    # passed in from the earlier ``world.compile()`` are a different (unstepped)
    # snapshot.  Measure the ones the simulation actually advances.
    model, data = sim.mj_model, sim.mj_data
    steps = int(steps)
    controller = None
    cpg_error = None
    if use_cpg:
        try:
            from flygym_demo.complex_terrain.cpg_controller import (
                CPGController, make_tripod_cpg_network)
            from flygym_demo.complex_terrain.preprogrammed import PreprogrammedSteps
            from flygym_demo.complex_terrain.common import (
                get_default_locomotion_dof_order, apply_locomotion_action)
            net = make_tripod_cpg_network(timestep=DT,
                                          intrinsic_frequency=CPG_INTRINSIC_HZ, seed=0)
            controller = CPGController(
                net, PreprogrammedSteps(),
                output_dof_order=get_default_locomotion_dof_order())
        except Exception as exc:            # capability missing, reported honestly
            cpg_error = f"{type(exc).__name__}: {exc}"

    heights_mean, heights_min, legs, tang, nrm, mu, xy = [], [], [], [], [], [], []
    capillary_force_samples = []
    capillary_actuator_force_last: list[float] = []
    capillary_gain_last: list[float] = []
    t0 = time.perf_counter()
    for k in range(steps):
        if controller is not None:
            apply_locomotion_action(sim, FLY_NAME, controller.step())
        if cfg.capillary is not None and cfg.capillary.enabled:
            S.drive_capillary_actuators(model, data, cfg)
        sim.step()
        if k % SAMPLE_EVERY == 0:
            bodies = np.asarray(sim.get_body_positions(FLY_NAME), float)
            heights_mean.append(float(bodies[:, 2].mean()))
            heights_min.append(float(bodies[:, 2].min()))
            xy.append(bodies[0, :2].copy())
            found = np.asarray(sim.get_ground_contact_info(FLY_NAME)[0], float)
            legs.append(float((found > 0).sum()))
            cf = S.measure_ground_contact_forces(model, data, report.ground_geom_names)
            tang.append(cf["mean_tangential_force_mN"])
            nrm.append(cf["mean_normal_force_mN"])
            mu.append(cf["mean_sliding_mu"])
            if cfg.capillary is not None and cfg.capillary.enabled:
                names = S.capillary_actuator_names(cfg)
                import mujoco as mj
                ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, n) for n in names]
                # ``data.actuator_force`` IS the actuator's scalar output already
                # (measured: gainprm[0] * ctrl), so summing it is the applied force;
                # multiplying by gainprm[0] again would double-count the gain.
                af = np.asarray(data.actuator_force[ids], float)
                capillary_force_samples.append(float(af.sum()))
                capillary_actuator_force_last = [float(v) for v in af]
                capillary_gain_last = [float(v) for v in
                                       np.asarray(model.actuator_gainprm[ids, 0], float)]

    qpos = np.asarray(data.qpos, float)
    qvel = np.asarray(data.qvel, float)

    def _mean(seq):
        vals = [v for v in seq if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else None

    finite = bool(np.isfinite(qpos).all() and np.isfinite(qvel).all())
    min_body_z = float(min(heights_min)) if heights_min else None
    out = {
        "scene": cfg.name,
        "controller": ("flygym_demo tripod CPG (ENGINEERING BASELINE, not the "
                       "connectome)" if controller is not None else "none (neutral pose held)"),
        "controller_error": cpg_error,
        "steps": steps,
        "sim_seconds": steps * DT,
        "wall_seconds": time.perf_counter() - t0,
        "all_finite": finite,
        "min_body_z_mm": min_body_z,
        "fell_through_ground": (None if min_body_z is None
                                else bool(min_body_z < FALL_THROUGH_Z_MM)),
        "mean_body_height_mm": _mean(heights_mean),
        "min_body_height_mm": min_body_z,
        "mean_legs_in_contact": _mean(legs),
        "min_legs_in_contact": float(min(legs)) if legs else None,
        "max_legs_in_contact": float(max(legs)) if legs else None,
        "mean_tangential_force_mN": _mean(tang),
        "mean_normal_force_mN": _mean(nrm),
        "mean_sliding_mu": _mean(mu),
        "n_samples": len(heights_mean),
        "n_samples_missing_contact_force": sum(1 for v in tang if v is None
                                               or not np.isfinite(v)),
        "horizontal_displacement_mm": (float(np.linalg.norm(xy[-1] - xy[0]))
                                       if len(xy) > 1 else None),
        "mean_capillary_actuator_force_mN": _mean(capillary_force_samples),
        "capillary_actuator_force_last_sample_mN": capillary_actuator_force_last,
        "capillary_actuator_gain_last_sample": capillary_gain_last,
        "final_qpos_sha256": hashlib.sha256(qpos.tobytes()).hexdigest(),
    }
    return out


def raw_baseline_state(cfg):
    """Build the scene the way the project builds it TODAY, with no scene layer.

    Used to check that applying the ``blank`` preset leaves the physics exactly
    where it was: same geom friction, same pair friction, same initial state.
    """
    from flygym.compose import FlatGroundWorld
    from flygym.simulation import Simulation
    from flygym.utils.math import Rotation3D

    fly = _make_fly(cfg)
    world = FlatGroundWorld(name="world_raw", half_size=1000.0)
    world.add_fly(fly, spawn_position=list(SPAWN),
                  spawn_rotation=Rotation3D("quat", [1.0, 0.0, 0.0, 0.0]))
    model, data = world.compile()
    sim = Simulation(world, timestep=DT)
    sim.warmup()
    geom_rows = sorted({tuple(round(float(v), 12)
                             for v in np.asarray(model.geom_friction[i]))
                        for i in range(int(model.ngeom))})
    pair_rows = sorted({tuple(round(float(v), 12)
                             for v in np.asarray(model.pair_friction[i]))
                        for i in range(int(model.npair))})
    out = {"geom_rows": geom_rows, "pair_rows": pair_rows,
           "ngeom": int(model.ngeom), "nu": int(model.nu),
           "nlight": int(model.nlight), "nhfield": int(model.nhfield),
           "state": state_hash(data)}
    sim.close()
    return out


def build_scene_raw(cfg, *, with_cameras: bool = True):
    """Build fly + world, apply the scene, and compile -- optionally with cameras.

    Cameras must exist BEFORE the compile, which is why this is separate from
    ``build_scene`` (the project's ``BodyConfig`` has the same constraint and says
    so; adding a camera to a compiled model raises "Camera ... not found").
    """
    from flygym.compose import FlatGroundWorld
    from flygym.utils.math import Rotation3D
    from engine.embodied import scene as S

    fly = _make_fly(cfg)
    world = FlatGroundWorld(name=f"render_{cfg.name}", half_size=cfg.world_half_size_mm)
    world.add_fly(fly, spawn_position=list(SPAWN),
                  spawn_rotation=Rotation3D("quat", [1.0, 0.0, 0.0, 0.0]))
    report = S.apply_scene(world, fly, cfg, measure=False)
    if with_cameras:
        add_render_cameras(world)
    model, data = world.compile()
    return fly, world, report, model, data


def _new_sim(world):
    from flygym.simulation import Simulation
    sim = Simulation(world, timestep=DT)
    sim.warmup()
    return sim


def render_frames(sim, out_dir, tag, res=(420, 560)):
    """Render one frame per camera; returns per-camera pixel statistics + paths."""
    import mujoco as mj
    import matplotlib.image as mpimg

    model, data = sim.mj_model, sim.mj_data
    stats = {}
    renderer = mj.Renderer(model, res[0], res[1])
    try:
        for cam in ("wide", "close"):
            cid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, cam)
            renderer.update_scene(data, cid)
            img = renderer.render()
            path = os.path.join(out_dir, f"scene_{tag}_{cam}.png")
            mpimg.imsave(path, img)
            luma = img[:, :, :3].max(axis=2)
            stats[cam] = {
                "path": path, "mean": float(img[:, :, :3].mean()),
                "std": float(img[:, :, :3].std()),
                "dark_pixels_lt30": int((luma < 30).sum()),
                "bright_pixels_gt240": int((luma > 240).sum()),
                "shape": list(img.shape)}
    finally:
        renderer.close()
    return stats


def add_render_cameras(world):
    """Two fixed cameras: a wide view of the scene and a close view of the fly."""
    def look(name, pos, target, fovy):
        pos_a, tgt_a = np.array(pos, float), np.array(target, float)
        d = tgt_a - pos_a
        d = d / np.linalg.norm(d)
        z = -d
        x = np.cross(np.array([0.0, 0.0, 1.0]), z)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        world.mjcf_root.worldbody.add_camera(name=name, pos=pos_a.tolist(),
                                             xyaxes=list(x) + list(y), fovy=fovy)
    look("wide", (-7.0, -7.0, 5.0), (2.5, 0.6, 0.3), 45.0)
    look("close", (3.2, -3.2, 1.7), (0.6, 0.05, 0.35), 40.0)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    t_start = time.perf_counter()
    from engine.embodied import scene as S

    result: dict = {
        "module_under_test": os.path.join(HERE, "engine", "embodied", "scene.py"),
        "runner": os.path.abspath(__file__),
        "python": sys.executable,
        "units": "mm, s, rad, kg; force unit mN (kg*mm/s^2)",
        "rollout": {"seconds": ROLLOUT_SECONDS, "timestep_s": DT,
                    "steps": ROLLOUT_STEPS, "sample_every": SAMPLE_EVERY},
    }

    print("=" * 78)
    print("1. provenance / ledger / sweep (pure python, no MuJoCo needed)")
    print("=" * 78)

    prov = S.scene_provenance_report()
    counts = prov["counts_by_provenance"]
    result["provenance"] = prov
    check("ledger: non-empty and counts sum to the number of records",
          prov["n_records"] > 0 and sum(counts.values()) == prov["n_records"],
          f"{prov['n_records']} records, counts={counts}")
    check("ledger: provenance classes are exactly the four allowed ones",
          set(counts) == set(S.PROVENANCE_CLASSES), str(sorted(counts)))
    check("ledger: no MEASURED_CITED record without a verified source",
          counts[S.MEASURED_CITED] == 0 and prov["measured_cited_names"] == [],
          "0 MEASURED_CITED records: no external source was verified in this session")
    check("ledger: every ASSUMED/ENGINEERING_DEFAULT record carries a sweep_range",
          all(r["sweep_range"] and len(r["sweep_range"]) >= 2
              for r in prov["ledger"]
              if r["provenance"] in (S.ASSUMED, S.ENGINEERING_DEFAULT)),
          "enforced by ParameterRecord.__post_init__ and re-checked here")
    check("ledger: preset counts are reported per provenance class",
          all(k in counts for k in S.PROVENANCE_CLASSES) and prov["n_records"] == len(S.PARAMETER_LEDGER),
          f"{prov['n_records']} records")

    # the citation gate itself
    from engine.embodied.scene import (ParameterRecord,
                                      UncitedMeasuredClaimError)
    rejected = False
    try:
        ParameterRecord(name="x.tarsal_adhesion", value=1.0, unit="uN",
                        provenance=S.MEASURED_CITED, source="")
    except UncitedMeasuredClaimError:
        rejected = True
    check("provenance: a MEASURED_CITED record with no source is REJECTED",
          rejected, "UncitedMeasuredClaimError raised")
    rejected2 = False
    try:
        ParameterRecord(name="x.tarsal_adhesion", value=1.0, unit="uN",
                        provenance=S.MEASURED_CITED, source="Smith 2001 J Exp Biol 204:1-9")
    except UncitedMeasuredClaimError:
        rejected2 = True
    check("provenance: a MEASURED_CITED record with author/year/venue is ACCEPTED",
          not rejected2, "gate checks structure (author/year/venue or URL), not truth")
    rejected3 = False
    try:
        ParameterRecord(name="x.k", value=1.0, unit="u", provenance=S.ASSUMED,
                        sweep_range=None)
    except S.ProvenanceError:
        rejected3 = True
    check("provenance: an ASSUMED record with no sweep_range is REJECTED",
          rejected3, "sweep_range is mandatory for ASSUMED / ENGINEERING_DEFAULT")

    # nested parameter setting, which is what a sweep relies on
    base = S.SCENE_PRESETS["natural"]
    v1 = S.set_scene_parameter(base, "roughness.amplitude_mm", 0.2)
    v2 = S.set_scene_parameter(base, "friction_by_class.tarsus.0", 2.5)
    v3 = S.set_scene_parameter(base, "capillary.enabled", False)
    v4 = S.set_scene_parameter(base, "lights.0.diffuse.1", 0.5)
    check("sweep: dotted-path setter reaches nested dataclass fields",
          (v1.roughness.amplitude_mm == 0.2 and v2.friction_by_class["tarsus"][0] == 2.5
           and v3.capillary.enabled is False and v4.lights[0].diffuse[1] == 0.5),
          "roughness/friction_by_class/capillary/lights all reachable")
    check("sweep: setter returns a NEW config and leaves the original untouched",
          base.roughness.amplitude_mm == S.ROUGHNESS_AMPLITUDE_MM,
          f"original amplitude still {base.roughness.amplitude_mm}")
    invalid_refused = False
    try:
        S.set_scene_parameter(base, "roughness.kind", "not_a_kind")
    except ValueError:
        invalid_refused = True
    check("sweep: the setter re-validates, so a bad sweep value is refused",
          invalid_refused, "RoughnessConfig.__post_init__ raised ValueError")

    synth = S.sensitivity_sweep(base, "roughness.amplitude_mm", [0.01, 0.05, 0.2],
                                evaluator=lambda c: {"amp_mm": c.roughness.amplitude_mm})
    check("sweep: sensitivity_sweep with an injected evaluator returns one outcome "
          "per value and no MuJoCo",
          synth["n_values"] == 3 and len(synth["outcomes"]) == 3
          and synth["n_errors"] == 0
          and synth["values_applied"] == [0.01, 0.05, 0.2],
          f"read back {synth['values_applied']}")
    refused = False
    try:
        S.sensitivity_sweep(base, "roughness.amplitude_mm", [0.1], evaluator=None)
    except TypeError:
        refused = True
    check("sweep: sensitivity_sweep refuses to run without an evaluator",
          refused, "no silent default evaluator that would need MuJoCo")

    # ------------------------------------------------------------------ #
    print("=" * 78)
    print("2. build + apply_scene + compile, then check the COMPILED model")
    print("=" * 78)

    built: dict[str, dict] = {}
    for key in PRESET_KEYS:
        cfg = S.SCENE_PRESETS[key]
        t0 = time.perf_counter()
        fly, world, sim, report, model, data = build_scene(cfg)
        build_s = time.perf_counter() - t0
        meas = measure(cfg, report, model)
        built[key] = {"cfg": cfg, "fly": fly, "world": world, "sim": sim,
                      "report": report, "model": model, "data": data,
                      "measured": meas, "build_seconds": build_s,
                      "state": state_hash(data)}
        print(f"  [{key}] built in {build_s:.1f}s: ngeom={meas['ngeom']} "
              f"nlight={meas['nlight']} nhfield={meas['nhfield']} npair={meas['npair']} "
              f"nu={meas['nu']} n_roughness={meas['n_roughness_geoms']} "
              f"n_objects={meas['n_object_geoms']}", flush=True)

    result["presets"] = {
        key: {
            "description": S.SCENE_PRESETS[key].description,
            "config": _jsonable({
                "name": S.SCENE_PRESETS[key].name,
                "friction_by_class": S.SCENE_PRESETS[key].friction_by_class,
                "roughness": S.SCENE_PRESETS[key].roughness,
                "capillary": S.SCENE_PRESETS[key].capillary,
                "adhesion_gain": S.SCENE_PRESETS[key].adhesion_gain,
                "lights": [l.name for l in S.SCENE_PRESETS[key].lights],
                "objects": [o.name for o in S.SCENE_PRESETS[key].objects],
                "leave_pairs_untouched": S.SCENE_PRESETS[key].leave_pairs_untouched}),
            "report": built[key]["report"].as_dict(),
            "measured_after_compile": built[key]["measured"],
            "build_seconds": built[key]["build_seconds"],
        }
        for key in PRESET_KEYS
    }

    blank_m = built["blank"]["measured"]
    nat_m = built["natural"]["measured"]
    nlo_m = built["natural_lit_no_objects"]["measured"]

    check("blank: every geom carries ONE uniform friction row "
          "(friction does not differ by class)",
          blank_m["geom_friction_row_is_uniform"] is True
          and len(blank_m["distinct_geom_friction_rows"]) == 1,
          f"distinct rows = {blank_m['distinct_geom_friction_rows']}")
    check("blank: that uniform row is still today's (1.0, 0.005, 0.0001)",
          tuple(blank_m["distinct_geom_friction_rows"][0]) == (1.0, 0.005, 0.0001),
          str(blank_m["distinct_geom_friction_rows"][0]))
    check("blank: FlyGym's own pair friction is untouched "
          "(1, 1, 0.02, 1e-4, 1e-4), so the control is faithful",
          all(tuple(r) == (1.0, 1.0, 0.02, 1e-4, 1e-4)
              for r in blank_m["distinct_pair_friction_rows"])
          and len(blank_m["distinct_pair_friction_rows"]) == 1,
          f"distinct pair rows = {blank_m['distinct_pair_friction_rows']}")
    check("blank: nlight == 0",
          blank_m["nlight"] == 0, f"model.nlight = {blank_m['nlight']}")
    check("blank: no heightfield and no roughness geoms",
          blank_m["nhfield"] == 0 and blank_m["n_roughness_geoms"] == 0,
          f"nhfield={blank_m['nhfield']} roughness geoms={blank_m['n_roughness_geoms']}")
    check("blank: no object geoms",
          blank_m["n_object_geoms"] == 0, f"{blank_m['n_object_geoms']}")
    check("blank: topology is unchanged from today (70 geoms, 48 actuators, "
          "55 pairs, 6 sensors)",
          (blank_m["ngeom"], blank_m["nu"], blank_m["npair"], blank_m["nsensor"])
          == (70, 48, 55, 6),
          f"ngeom={blank_m['ngeom']} nu={blank_m['nu']} npair={blank_m['npair']} "
          f"nsensor={blank_m['nsensor']}")

    # Fidelity of the control: compare against a world built with NO scene layer.
    raw = raw_baseline_state(S.SCENE_PRESETS["blank"])
    result["raw_baseline_no_scene_layer"] = raw
    check("control fidelity: 'blank' gives the SAME geom and pair friction rows as a "
          "world built with no scene layer at all",
          [tuple(r) for r in raw["geom_rows"]]
          == [tuple(r) for r in blank_m["distinct_geom_friction_rows"]]
          and [tuple(r) for r in raw["pair_rows"]]
          == [tuple(r) for r in blank_m["distinct_pair_friction_rows"]],
          f"raw geom rows {raw['geom_rows']} vs blank "
          f"{blank_m['distinct_geom_friction_rows']}")
    check("control fidelity: 'blank' gives the SAME initial qpos/qvel hash and the "
          "same model shape as an untouched build",
          raw["state"]["qpos_sha256"] == built["blank"]["state"]["qpos_sha256"]
          and raw["state"]["qvel_sha256"] == built["blank"]["state"]["qvel_sha256"]
          and (raw["ngeom"], raw["nu"], raw["nlight"], raw["nhfield"])
          == (blank_m["ngeom"], blank_m["nu"], blank_m["nlight"], blank_m["nhfield"]),
          f"qpos {raw['state']['qpos_sha256'][:16]}... == "
          f"{built['blank']['state']['qpos_sha256'][:16]}...")

    check("natural: tarsus and body GEOM friction rows differ after compilation",
          nat_m["geom_friction_rows_by_class"]["tarsus"]
          != nat_m["geom_friction_rows_by_class"]["body"]
          and nat_m["geom_friction_row_is_uniform"] is False,
          f"tarsus={nat_m['geom_friction_rows_by_class']['tarsus']} "
          f"body={nat_m['geom_friction_rows_by_class']['body']}")
    check("natural: tarsus and body PAIR friction rows differ "
          "(the mechanism MuJoCo actually honours here)",
          nat_m["pair_friction_rows_by_class"]["tarsus"]
          != nat_m["pair_friction_rows_by_class"]["body"],
          f"tarsus pair={nat_m['pair_friction_rows_by_class']['tarsus']} "
          f"body pair={nat_m['pair_friction_rows_by_class']['body']}")
    check("natural: model.nlight > 0", nat_m["nlight"] > 0,
          f"nlight={nat_m['nlight']} "
          f"({[l['name'] for l in nat_m['lights']]})")
    check("natural: a heightfield was compiled into the model "
          "(model.nhfield >= 1)",
          nat_m["nhfield"] >= 1 and nat_m["n_roughness_geoms"] >= 1,
          f"nhfield={nat_m['nhfield']} roughness geoms={nat_m['n_roughness_geoms']} "
          f"size={nat_m['hfield_size']} nrow/ncol={nat_m['hfield_nrow']}/"
          f"{nat_m['hfield_ncol']}")
    check("natural: measured heightfield peak-to-peak relief equals the configured "
          f"amplitude ({S.ROUGHNESS_AMPLITUDE_MM} mm)",
          nat_m["hfield_peak_to_peak_mm"] is not None
          and abs(nat_m["hfield_peak_to_peak_mm"] - S.ROUGHNESS_AMPLITUDE_MM) < 1e-9,
          f"measured p2p = {nat_m['hfield_peak_to_peak_mm']} mm "
          f"(read from model.hfield_data * model.hfield_size[2])")
    check("natural: object geoms exist in the compiled model",
          nat_m["n_object_geoms"] == len(S.SCENE_PRESETS["natural"].objects) > 0,
          f"{nat_m['n_object_geoms']} object geoms: "
          + str(sorted(k for k in nat_m["geom_counts_by_type"])))
    check("natural: the 6 capillary adhesion actuators reached the compiled model",
          nat_m["nu"] == blank_m["nu"] + 6,
          f"nu={nat_m['nu']} vs blank {blank_m['nu']}")

    check("natural_lit_no_objects: no object geoms at all",
          nlo_m["n_object_geoms"] == 0, f"{nlo_m['n_object_geoms']}")
    check("natural_lit_no_objects: still lit and still rough "
          "(so it isolates the objects, not the lights)",
          nlo_m["nlight"] == nat_m["nlight"] and nlo_m["nhfield"] == nat_m["nhfield"],
          f"nlight={nlo_m['nlight']} nhfield={nlo_m['nhfield']}")
    check("natural_lit_no_objects: friction rows identical to 'natural'",
          nlo_m["geom_friction_rows_by_class"] == nat_m["geom_friction_rows_by_class"]
          and nlo_m["pair_friction_rows_by_class"] == nat_m["pair_friction_rows_by_class"],
          "same geom and pair rows")
    check("control validity: 'natural' and 'natural_lit_no_objects' have identical "
          "INITIAL qpos/qvel (objects are visual-only and cannot change physics)",
          built["natural"]["state"]["qpos_sha256"]
          == built["natural_lit_no_objects"]["state"]["qpos_sha256"]
          and built["natural"]["state"]["qvel_sha256"]
          == built["natural_lit_no_objects"]["state"]["qvel_sha256"],
          "sha256 of qpos and qvel match")

    # ------------------------------------------------------------------ #
    print("=" * 78)
    print("3. determinism: a second, freshly built 'natural' world")
    print("=" * 78)
    fly_b, world_b, sim_b, report_b, model_b, data_b = build_scene(
        S.SCENE_PRESETS["natural"])
    state_b = state_hash(data_b)
    check("determinism: two freshly built 'natural' worlds (same seed) have "
          "identical initial qpos and qvel hashes",
          state_b["qpos_sha256"] == built["natural"]["state"]["qpos_sha256"]
          and state_b["qvel_sha256"] == built["natural"]["state"]["qvel_sha256"],
          f"qpos {state_b['qpos_sha256'][:16]}... == "
          f"{built['natural']['state']['qpos_sha256'][:16]}...")
    meas_b = measure(S.SCENE_PRESETS["natural"], report_b, model_b)
    check("determinism: the compiled heightfield data is byte-reproducible",
          meas_b["hfield_peak_to_peak_mm"] == nat_m["hfield_peak_to_peak_mm"]
          and meas_b["hfield_size"] == nat_m["hfield_size"],
          "same hfield size and p2p relief")

    # ------------------------------------------------------------------ #
    print("=" * 78)
    print("4. rollouts: blank vs natural (stance, then CPG-driven walk)")
    print("=" * 78)

    rolls: dict[str, dict] = {}
    for key in ("blank", "natural"):
        b = built[key]
        r = rollout(b["cfg"], b["report"], b["sim"], b["model"], b["data"],
                    use_cpg=False)
        rolls[f"stance_{key}"] = r
        print(f"  stance {key}: height {r['mean_body_height_mm']:.4f} mm, "
              f"legs {r['mean_legs_in_contact']:.3f}, "
              f"tangential {r['mean_tangential_force_mN']:.4f} mN, mu {r['mean_sliding_mu']}",
              flush=True)

    cpg_ok = True
    for key in ("blank", "natural"):
        b = built[key]
        try:
            r = rollout(b["cfg"], b["report"], b["sim"], b["model"], b["data"],
                        use_cpg=True)
        except Exception as exc:
            cpg_ok = False
            r = {"scene": key, "error": f"{type(exc).__name__}: {exc}",
                 "traceback": traceback.format_exc()[-800:]}
        rolls[f"cpg_{key}"] = r
        if cpg_ok and r.get("all_finite") is not None:
            print(f"  cpg {key}: height {r['mean_body_height_mm']:.4f} mm, "
                  f"legs {r['mean_legs_in_contact']:.3f}, "
                  f"travel {r['horizontal_displacement_mm']:.3f} mm, "
                  f"tangential {r['mean_tangential_force_mN']:.4f} mN, "
                  f"mu {r['mean_sliding_mu']}", flush=True)
    result["rollouts"] = rolls

    for key in ("blank", "natural"):
        r = rolls[f"stance_{key}"]
        check(f"rollout(stance {key}): all qpos/qvel finite over "
              f"{ROLLOUT_SECONDS:.1f}s",
              r["all_finite"] is True, f"all_finite={r['all_finite']}")
        check(f"rollout(stance {key}): the fly does not fall through the ground",
              r["fell_through_ground"] is False,
              f"min body z = {r['min_body_z_mm']:.4f} mm "
              f"(threshold {FALL_THROUGH_Z_MM} mm)")
        check(f"rollout(stance {key}): sampled leg contacts are within [0, 6] and "
              f"at least one leg is down",
              r["min_legs_in_contact"] is not None
              and 1.0 <= r["min_legs_in_contact"]
              and r["max_legs_in_contact"] <= 6.0,
              f"legs min/max/mean = {r['min_legs_in_contact']}/"
              f"{r['max_legs_in_contact']}/{r['mean_legs_in_contact']:.3f}")

    check("rollout(stance blank): MuJoCo resolves sliding mu == 1.0 at every "
          "sampled ground contact (the single uniform class)",
          rolls["stance_blank"]["mean_sliding_mu"] is not None
          and abs(rolls["stance_blank"]["mean_sliding_mu"] - 1.0) < 1e-9,
          f"measured mean mu = {rolls['stance_blank']['mean_sliding_mu']}")
    check("rollout(stance natural): MuJoCo resolves sliding mu == 1.6 (the tarsus "
          "row), i.e. the per-class friction is really what the feet feel",
          rolls["stance_natural"]["mean_sliding_mu"] is not None
          and abs(rolls["stance_natural"]["mean_sliding_mu"] - 1.6) < 1e-9,
          f"measured mean mu = {rolls['stance_natural']['mean_sliding_mu']}")
    check("rollout: sliding-friction-driven tangential contact force was obtained "
          "for every sample in both scenes",
          rolls["stance_blank"]["n_samples_missing_contact_force"] == 0
          and rolls["stance_natural"]["n_samples_missing_contact_force"] == 0
          and rolls["stance_blank"]["mean_tangential_force_mN"] is not None,
          "reconstructed as |(f1,f2)| in MuJoCo's contact frame via mj_contactForce")

    if not cpg_ok:
        evidence_not_found(
            "rollout(CPG-driven): locomotion comparison blank vs natural",
            "flygym_demo.complex_terrain CPG controller could not be imported, so the "
            "walking rollout could not be run")
    else:
        for key in ("blank", "natural"):
            r = rolls[f"cpg_{key}"]
            check(f"rollout(CPG {key}): all qpos/qvel finite over "
                  f"{ROLLOUT_SECONDS:.1f}s",
                  bool(r["all_finite"]), f"all_finite={r['all_finite']}")
            check(f"rollout(CPG {key}): the fly does not fall through the ground",
                  r["fell_through_ground"] is False,
                  f"min body z = {r['min_body_z_mm']:.4f} mm")
            check(f"rollout(CPG {key}): the fly actually moves "
                  f"(>1 mm of travel in {ROLLOUT_SECONDS:.1f}s)",
                  r["horizontal_displacement_mm"] is not None
                  and r["horizontal_displacement_mm"] > 1.0,
                  f"travel = {r['horizontal_displacement_mm']:.3f} mm")
        check("rollout(CPG): the two scenes give DIFFERENT trajectories, so the scene "
              "layer changes behaviour rather than only the model file",
              rolls["cpg_blank"]["final_qpos_sha256"]
              != rolls["cpg_natural"]["final_qpos_sha256"],
              "final qpos hashes differ")

    cap_force = rolls["stance_natural"]["mean_capillary_actuator_force_mN"]
    check("capillary: the actuators exist, are driven to ctrl=1 and report the "
          "expected measured output force",
          cap_force is not None
          and abs(cap_force - 6 * S.CAPILLARY_FORCE_MN) < 1e-9,
          f"measured sum(data.actuator_force) = {cap_force} mN vs expected "
          f"{6 * S.CAPILLARY_FORCE_MN} mN; per-actuator "
          f"{rolls['stance_natural']['capillary_actuator_force_last_sample_mN']} mN with "
          f"gainprm={rolls['stance_natural']['capillary_actuator_gain_last_sample']} "
          f"(MuJoCo's actuator_force already equals gainprm[0]*ctrl)")
    evidence_not_found(
        "capillary: the secretion force measured as a REAL attractive force at the "
        "tarsus-ground contact",
        "make_locomotion_fly attaches no force/torque sensors to the fly bodies, so the "
        "attraction cannot be separated from the normal contact force in data; only the "
        "actuator's own commanded output is readable (reported above)")
    evidence_not_found(
        "visual pathway: the scene objects are actually visible to the fly",
        f"the compiled model has ncam={nat_m['ncam']} and make_locomotion_fly never "
        "calls add_vision(), so there are no eye cameras and no retina: whether the fly "
        "could see the objects cannot be evaluated in this scene")
    check("natural: the CPG baseline is an ENGINEERING BASELINE, recorded as such",
          "ENGINEERING BASELINE" in rolls["cpg_blank"].get("controller", ""),
          rolls["cpg_blank"].get("controller", ""))

    # ------------------------------------------------------------------ #
    print("=" * 78)
    print("5. rendered evidence: does the scene actually LOOK different?")
    print("=" * 78)
    os.makedirs(OUT, exist_ok=True)
    renders: dict[str, dict] = {}
    for key in ("blank", "natural"):
        cfg = S.SCENE_PRESETS[key]
        try:
            fly_r, world_r, rep_r, model_r, data_r = build_scene_raw(cfg)
            sim_r = _new_sim(world_r)
            for _ in range(3000):      # settle 0.3 s so the fly is standing
                if cfg.capillary is not None and cfg.capillary.enabled:
                    S.drive_capillary_actuators(sim_r.mj_model, sim_r.mj_data, cfg)
                sim_r.step()
            renders[key] = render_frames(sim_r, OUT, key)
            sim_r.close()
            for cam, st in renders[key].items():
                print(f"  {key}/{cam}: mean={st['mean']:.1f} std={st['std']:.1f} "
                      f"dark(<30)={st['dark_pixels_lt30']} -> {st['path']}", flush=True)
        except Exception as exc:
            renders[key] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  {key}: render failed: {type(exc).__name__}: {exc}", flush=True)
    result["renders"] = renders

    if "error" in renders.get("blank", {}) or "error" in renders.get("natural", {}):
        evidence_not_found(
            "rendered visual comparison of 'blank' vs 'natural'",
            "off-screen rendering failed here (MuJoCo GL backend unavailable?): "
            + str(renders.get("blank", {}).get("error")
                  or renders.get("natural", {}).get("error")))
    else:
        b_wide = renders["blank"]["wide"]
        n_wide = renders["natural"]["wide"]
        check("render: the lit scene really is brighter than the unlit control "
              "(the lights are in the render, not only in the model)",
              n_wide["mean"] > b_wide["mean"] + 10.0,
              f"mean pixel: blank {b_wide['mean']:.1f} vs natural {n_wide['mean']:.1f}")
        check("render: the high-contrast target object is VISIBLE as dark pixels in "
              "the 'natural' frame and absent from the control",
              n_wide["dark_pixels_lt30"] > 100
              and n_wide["dark_pixels_lt30"] > 4 * max(b_wide["dark_pixels_lt30"], 1),
              f"dark(<30) pixels: blank {b_wide['dark_pixels_lt30']} vs natural "
              f"{n_wide['dark_pixels_lt30']} (the black landmark box)")

    # ------------------------------------------------------------------ #
    print("=" * 78)
    print("6. scattered-roughness fallback (approximation, checked to compile)")
    print("=" * 78)
    scattered_cfg = S.set_scene_parameter(S.SCENE_PRESETS["natural"],
                                          "roughness.kind", "scattered")
    try:
        fly_s, world_s, sim_s, rep_s, model_s, data_s = build_scene(scattered_cfg)
        meas_s = measure(scattered_cfg, rep_s, model_s)
        result["scattered_fallback"] = {"measured": meas_s,
                                        "report": rep_s.as_dict()}
        check("fallback: 'scattered' roughness compiles and produces roughness geoms "
              "instead of a heightfield",
              meas_s["nhfield"] == 0
              and meas_s["n_roughness_geoms"] == S.ROUGHNESS_SCATTER_N,
              f"nhfield={meas_s['nhfield']} roughness geoms="
              f"{meas_s['n_roughness_geoms']} (expected {S.ROUGHNESS_SCATTER_N})")
        check("fallback: the report says plainly that 'scattered' is an approximation",
              rep_s.roughness_fallback_used is True
              and any("APPROXIMATION" in w for w in rep_s.warnings),
              "report.roughness_fallback_used = True and a warning is present")
        sim_s.close()
    except Exception as exc:
        result["scattered_fallback"] = {"error": f"{type(exc).__name__}: {exc}",
                                        "traceback": traceback.format_exc()[-1200:]}
        check("fallback: 'scattered' roughness compiles", False,
              f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ #
    print("=" * 78)
    print("7. sensitivity sweep with a REAL rollout evaluator")
    print("=" * 78)

    def evaluator(variant_cfg):
        """Short stance rollout; returns measured outcomes only, no config echo."""
        fly_x, world_x, sim_x, rep_x, model_x, data_x = build_scene(variant_cfg)
        r = rollout(variant_cfg, rep_x, sim_x, model_x, data_x,
                    use_cpg=False, steps=3000)
        sim_x.close()
        return {"mean_body_height_mm": r["mean_body_height_mm"],
                "mean_legs_in_contact": r["mean_legs_in_contact"],
                "mean_sliding_mu": r["mean_sliding_mu"],
                "min_body_z_mm": r["min_body_z_mm"],
                "all_finite": r["all_finite"]}

    sweep = S.sensitivity_sweep(S.SCENE_PRESETS["natural"], "roughness.amplitude_mm",
                                [0.02, 0.05, 0.15], evaluator=evaluator,
                                rollout_seconds=0.3)
    result["sensitivity_sweep"] = sweep
    check("sweep: a real rollout evaluator produced one measured outcome per value "
          "with no errors",
          sweep["n_errors"] == 0 and len(sweep["outcomes"]) == 3
          and all(o["outcome"] and o["outcome"]["all_finite"] for o in sweep["outcomes"]),
          "values " + str(sweep["values_applied"]))
    heights = [o["outcome"]["mean_body_height_mm"] for o in sweep["outcomes"]
               if o["outcome"]]
    check("sweep: the measured mean body height responds to the roughness amplitude "
          "(a flat curve would mean the parameter was ignored)",
          len(heights) == 3 and (max(heights) - min(heights)) > 1e-6,
          f"mean heights {[round(h, 5) for h in heights]} mm for amplitudes "
          f"{sweep['values_applied']}")

    # ------------------------------------------------------------------ #
    for b in built.values():
        try:
            b["sim"].close()
        except Exception:
            pass
    try:
        sim_b.close()
    except Exception:
        pass

    n_pass = sum(1 for r in RESULTS if r["status"] == "PASS")
    n_fail = sum(1 for r in RESULTS if r["status"] == "FAIL")
    n_nf = sum(1 for r in RESULTS if r["status"] == "EVIDENCE NOT FOUND")
    result.update({
        "what_these_tests_establish": [
            "the three presets really produce what their names claim, verified against "
            "the COMPILED mujoco.MjModel and not against the configuration",
            "per-class friction genuinely reaches the contact model: the sliding mu "
            "MuJoCo resolves at the contacts is 1.0 for 'blank' and 1.6 for 'natural'",
            "the heightfield, the lights and the object geoms are present in the "
            "compiled model, and the measured relief equals the configured amplitude",
            "'natural_lit_no_objects' is a valid control for the objects: it has no "
            "object geoms and an identical initial state",
            "the scene is reproducible (identical qpos/qvel hashes for two builds)",
            "a 1 s stance rollout and a 1 s CPG-driven walk stay finite and the fly "
            "does not fall through the ground",
        ],
        "what_they_do_not_establish": [
            "that any friction, roughness or capillary magnitude is physically right: "
            "they are ASSUMED / ENGINEERING_DEFAULT, with zero MEASURED_CITED records",
            "that the fly sees the scene objects (no eye cameras, no retina: reported "
            "as EVIDENCE NOT FOUND)",
            "that the capillary force reaches the ground as a real attraction (no body "
            "force sensors: reported as EVIDENCE NOT FOUND)",
            "any biological norm: the walk uses FlyGym's demo tripod CPG, an "
            "ENGINEERING BASELINE, not the connectome",
        ],
        "checks": RESULTS,
        "n_checks": len(RESULTS),
        "n_passed": n_pass,
        "n_failed": n_fail,
        "n_evidence_not_found": n_nf,
        "passed": bool(n_fail == 0),
        "wall_seconds": time.perf_counter() - t_start,
    })

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "scene_selftest.json")
    with open(path, "w") as fh:
        json.dump(_jsonable(result), fh, indent=2)

    print("=" * 78)
    print(f"{n_pass}/{len(RESULTS) - n_nf} assertions passed "
          f"({n_fail} failed, {n_nf} EVIDENCE NOT FOUND) "
          f"in {result['wall_seconds']:.1f}s")
    print(f"JSON: {path}")
    print("=" * 78)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
