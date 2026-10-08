#!/usr/bin/env python3
"""TACTILE RECORDING: per-leg contact FORCE and contact AREA over a walking episode.

WHAT IS RECORDED, AND FROM WHERE
--------------------------------
Per leg, per frame:

    force_N          the simulator's own contact force, resolved from the contact frame
                     into a normal and a tangential component
    area_m2          from tactile.py, NOT from the simulator (MuJoCo has no contact patch)
    pressure_Pa      F / A, which is only meaningful once an area exists
    penetration_m    the measured overlap depth of the contact, which is the input the area
                     model could have used if it were confident; it is recorded so the choice
                     of a force-driven model can be checked against the geometry that was
                     actually there

Plus the aggregate: duty, impulse, area-time, peak pressure, and the fit of A ~ F^n, whose
EXPONENT says whether the scene is in the smooth-plane (n ~ 2/3) or rough-surface (n ~ 1)
regime.  That exponent is the main scientific content of this file.

WHY THE CONTACT FORCE IS RESOLVED RATHER THAN TAKEN RAW
------------------------------------------------------
``get_ground_contact_info`` returns the force in the CONTACT frame, in which the FIRST axis is
the normal and the other two are tangential.  A raw magnitude would mix a normal load with
scrub, so the normal and tangential parts are separated: the normal part is what presses the
tarsus into the substrate and therefore what sets the contact area, and the tangential part is
what would shear a pad.

Run (BODY environment):
    cd /run/media/sensen/Data2/cell_wound_prototype
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
      venv_body/bin/python run_tactile_record.py --seeds 0,1 --seconds 4
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "outputs" / "tactile"

from tactile import (DEFAULT_TACTILE, TactileConfig, area_from_penetration,  # noqa: E402
                     contact_area_series, pressure_series, tactile_summary)

LEG_NAMES = ("lf", "lm", "lh", "rf", "rm", "rh")

#: Rig conditions.  "bare" is the control, the others are the recording rigs whose masses are
#: now actually in the model (see electrode_payload.RIG_PRESETS and
#: ARENA_FIX_ROUND5.zh-CN.md for how the injection was fixed).
CONDITIONS = ("bare", "electrode_only", "tether", "telemetry", "full")


def measure_tip_geometry(backend) -> dict:
    """The tarsal tip's real size, from the compiled model's geom AABB, in metres.

    MEASURED, and it sets the area model's footprint: ``geom_aabb`` is (xmin, ymin, zmin,
    xmax, ymax, zmax) in the model's units.  The model is in MILLIMETRES (gravity is -9810),
    so the conversion to metres is explicit here rather than implicit at each use.
    """
    import mujoco
    m = backend.model
    names = [str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)) for i in range(m.ngeom)]
    rows = []
    for i, n in enumerate(names):
        if n and n.endswith("_tarsus5"):
            bb = np.asarray(m.geom_aabb[i], dtype=float)      # mm
            size_mm = bb[3:] - bb[:3]
            rows.append({"geom": n, "aabb_mm": bb.tolist(),
                         "size_mm": size_mm.tolist(),
                         "radius_m": float(0.5 * (size_mm[0] + size_mm[1]) * 1e-3),
                         "length_m": float(size_mm[2] * 1e-3)})
    if not rows:
        return {"n": 0}
    rad = float(np.mean([r["radius_m"] for r in rows]))
    length = float(np.mean([r["length_m"] for r in rows]))
    return {"n": len(rows), "per_geom": rows,
            "tip_radius_m": rad, "tip_length_m": length,
            "tip_radius_um": rad * 1e6, "tip_length_um": length * 1e6,
            "source": ("compiled model geom_aabb of the *_tarsus5 meshes, converted from "
                       "millimetres to metres")}


#: Model force units.  The model uses MILLIMETRES as its length unit (gravity is -9810,
#: and the fly's mass is 1.02e-3 kg), so a force of one model unit is kg*mm/s^2 = 1e-3 N =
#: one MILLINEWTON.  Every force this file reports is converted explicitly from model units to
#: newtons here, once, rather than at each use.
MODEL_FORCE_TO_N = 1.0e-3


def _tarsal_forces(be, geom_names, legs=LEG_NAMES) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-leg tarsal contact force from ``mj_contactForce``, in NEWTONS, plus penetration.

    WHY NOT THE FLYGym SENSOR.  ``get_ground_contact_info`` returns a 16-dimension custom
    sensor channel per leg, and MEASURED, its force entries are NOT the contact force: summing
    them gives a total that is 2e5 times the fly's weight and sometimes negative on the normal
    axis, whereas the same total computed from ``mj_contactForce`` (whose 6-vector is
    documented as [normal, tangent1, tangent2, and three torques] in the contact frame) is
    16.3 times the fly's weight and always positive.  The documented API is used, and the
    ratio is reported rather than hidden because it is a property of the CONTROLLER: the CPG
    position actuators push the legs into the substrate far harder than the body weight, which
    is an engineering baseline, not a measurement of fly muscle forces.

    A contact between a tarsus geom and the ground is assigned to its leg; contacts involving
    other body parts are excluded and their count is reported by the caller.
    """
    import mujoco
    fn = np.zeros(len(legs), dtype=float)
    ft = np.zeros(len(legs), dtype=float)
    pen = np.zeros(len(legs), dtype=float)
    f6 = np.zeros(6, dtype=float)
    d = be.data
    for c in range(d.ncon):
        con = d.contact[c]
        n1 = geom_names[int(con.geom1)] or ""
        n2 = geom_names[int(con.geom2)] or ""
        leg = None
        for j, name in enumerate(legs):
            if f"{name}_tarsus" in n1 or f"{name}_tarsus" in n2:
                leg = j
                break
        if leg is None:
            continue
        mujoco.mj_contactForce(be.model, d, c, f6)
        fn[leg] += float(f6[0]) * MODEL_FORCE_TO_N
        ft[leg] += float(math.hypot(f6[1], f6[2])) * MODEL_FORCE_TO_N
        pen[leg] = max(pen[leg], float(-con.dist) * 1e-3)     # mm -> m
    return fn, ft, pen


def run_episode(seed: int, seconds: float, rig: str, dt_obs: float,
                gl: str | None) -> dict:
    from engine.embodied import BodyBackend, BodyConfig
    from electrode_payload import ElectrodePayloadConfig

    payload = None if rig == "bare" else ElectrodePayloadConfig(rig=rig, label=rig)
    cfg = BodyConfig(seed=seed, scene_preset="meadow_grass",
                     add_tracking_camera=False, add_world_camera=False,
                     add_vision=False, electrode_payload=payload)
    be = BodyBackend(cfg, gl_backend=gl).attach_cpg_baseline()
    geometry = measure_tip_geometry(be)
    tcfg = TactileConfig(tip_radius_m=geometry.get("tip_radius_m",
                                                    DEFAULT_TACTILE.tip_radius_m),
                         tip_length_m=geometry.get("tip_length_m",
                                                   DEFAULT_TACTILE.tip_length_m))

    import mujoco
    geom_names = [str(mujoco.mj_id2name(be.model, mujoco.mjtObj.mjOBJ_GEOM, i))
                  for i in range(be.model.ngeom)]

    timestep = cfg.timestep_s
    n_steps = int(round(seconds / timestep))
    obs_period = max(1, int(round(dt_obs / timestep)))

    T, F_all, FN, FT, RAW, POS, PEN = ([] for _ in range(7))
    try:
        for k in range(n_steps):
            be.step()
            if k % obs_period:
                continue
            obs = be.observe()
            T.append(obs.time_s)
            cf = np.asarray(obs.contact_forces, dtype=float)
            F_all.append(cf)
            RAW.append(np.asarray(obs.contact_found_raw, dtype=float).copy())
            POS.append(np.asarray(obs.contact_positions_mm, dtype=float))
            fn, ft, pen = _tarsal_forces(be, geom_names)
            FN.append(fn)
            FT.append(ft)
            PEN.append(pen)
    finally:
        payload_report = be.payload_report
        be.close()

    T = np.asarray(T, dtype=float)
    FN = np.asarray(FN, dtype=float)          # (n, 6) normal force, N
    FT = np.asarray(FT, dtype=float)
    area = contact_area_series(FN, tcfg)
    # THE GEOMETRIC ROUTE IS THE PRIMARY AREA, the force route is the cross-check: see
    # tactile.area_from_penetration for the measurements that forced that ordering.
    PEN_arr = np.asarray(PEN, dtype=float)
    geo = area_from_penetration(PEN_arr, tcfg)
    P = pressure_series(FN, geo["A_flat_m2"])
    dt = float(np.median(np.diff(T))) if len(T) > 1 else float(dt_obs)
    summary = tactile_summary(FN, geo["A_flat_m2"], P, LEG_NAMES, dt)
    summary["area_route"] = ("geometric, from the measured penetration depth "
                             "(tactile.area_from_penetration)")
    summary["area_force_route_capped_fraction"] = area["capped_fraction"]
    # the GRF-to-weight ratio is reported because it is the reason the force route is only a
    # cross-check: measured at 1.4e4 x with this controller
    summary["grf_over_body_weight"] = None
    return {
        "truth": {"time_s": T, "contact_forces_N": np.asarray(F_all, dtype=np.float64),
                  "normal_force_N": FN, "tangential_force_N": FT,
                  "contact_found_raw": np.asarray(RAW, dtype=np.float64),
                  "contact_positions_mm": np.asarray(POS, dtype=np.float64),
                  "penetration_m": np.asarray(PEN, dtype=np.float64)},
        "tactile": {"area_m2": geo["A_flat_m2"], "area_um2": geo["A_flat_m2"] * 1e12,
                    "area_dome_um2": geo["A_dome_m2"] * 1e12,
                    "area_flat_uncapped_um2": geo["A_flat_uncapped_m2"] * 1e12,
                    "area_capped_fraction_geometric": geo["capped_fraction_flat"],
                    "area_force_route_um2": area["A_um2"],
                    "area_force_route_capped_fraction": area["capped_fraction"],
                    "contact_radius_um": geo["a_m"] * 1e6,
                    "area_hertz_um2": area["A_hertz_um2"],
                    "area_uncapped_um2": area["A_uncapped_um2"],
                    "pressure_Pa": P,
                    "capped_fraction": area["capped_fraction"],
                    "cap_um2": area["cap_um2"], "footprint_um2": area["footprint_um2"],
                    "regime_used": area["regime_used"]},
        "geometry": geometry,
        "tactile_config": area["config"],
        "summary": summary,
        "payload": payload_report,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--dt-obs", type=float, default=0.005)
    ap.add_argument("--gl", default="egl")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--allow-missing-scene", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    conds = [s for s in args.conditions.split(",") if s.strip()]
    index = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "seconds": args.seconds, "dt_obs_s": args.dt_obs,
             "seeds": seeds, "conditions": conds,
             "python": sys.version.split()[0], "platform": platform.platform(),
             "episodes": []}
    for seed in seeds:
        for rig in conds:
            t0 = time.perf_counter()
            ep = run_episode(seed, args.seconds, rig, args.dt_obs, args.gl)
            wall = time.perf_counter() - t0
            d = OUT / f"tactile_{rig}_seed{seed}"
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / "tactile.npz",
                                **{f"truth/{k}": v for k, v in ep["truth"].items()},
                                **{f"tactile/{k}": v for k, v in ep["tactile"].items()})
            (d / "summary.json").write_text(json.dumps(
                {"summary": ep["summary"], "geometry": ep["geometry"],
                 "tactile_config": ep["tactile_config"],
                 "payload": ep["payload"]},
                indent=2, sort_keys=True, default=str))
            s = ep["summary"]
            rec = {"seed": seed, "rig": rig, "wall_seconds": wall,
                   "duration_s": s["duration_s"], "n_frames": s["n_frames"],
                   "legs_in_contact_mean": s["total"]["legs_in_contact_mean"],
                   "force_peak_uN": s["total"]["force_peak_uN"],
                   "area_peak_um2": s["total"]["area_peak_um2"],
                   "area_mean_um2": s["total"]["area_mean_um2"],
                   "scaling_exponent": s["scaling"]["exponent"] if s["scaling"].get("ok")
                   else None,
                   "scaling_r2": s["scaling"].get("r2"),
                   "tip_radius_um": ep["geometry"].get("tip_radius_um"),
                   "dir": str(d)}
            index["episodes"].append(rec)
            print(json.dumps(rec, default=str), flush=True)
    index["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (OUT / "tactile_index.json").write_text(json.dumps(index, indent=2, sort_keys=True,
                                                       default=str))
    print(f"wrote {OUT / 'tactile_index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
